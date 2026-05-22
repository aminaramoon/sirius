/*
 * Copyright 2026, Sirius Contributors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "io/uring/uring_reactor.hpp"

#include "driver_types.h"
#include "io/types.hpp"

#include <fcntl.h>
#include <spdlog/spdlog.h>
#include <sys/stat.h>

#include <algorithm>
#include <cassert>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <stdexcept>

namespace sirius::io {

// ---------------------------------------------------------------------------
// uring_reactor
// ---------------------------------------------------------------------------

std::unique_ptr<uring_io_object> uring_reactor::create_io_object(std::string path)
{
  if (!supports(path))
    throw std::runtime_error("uring_reactor::create_io_object: unsupported path: " + path);

  file_descriptor fd{::open(path.c_str(), O_RDONLY)};
  if (!fd)
    throw std::runtime_error("uring_reactor::create_io_object: open failed: " + path + ": " +
                             strerror(errno));

  file_descriptor fd_direct{::open(path.c_str(), O_RDONLY | O_DIRECT)};
  if (!fd_direct)
    throw std::runtime_error("uring_reactor::create_io_object: O_DIRECT open failed: " + path +
                             ": " + strerror(errno));

  auto file_size = size(fd.get());
  return std::make_unique<uring_io_object>(
    std::move(path), std::move(fd), std::move(fd_direct), file_size);
}

size_t uring_reactor::size(int fd)
{
  struct stat st{};
  if (::fstat(fd, &st) != 0)
    throw std::runtime_error("uring_reactor::size: fstat failed: " + std::string(strerror(errno)));
  return static_cast<size_t>(st.st_size);
}

uring_reactor::uring_reactor(cucascade::memory::fixed_size_host_memory_resource& mr,
                             unsigned ring_entries)
  : _bounce_slot_size(mr.get_block_size()),
    // Ring is sized to 2 × NUM_CHUNKS so the worst-case short-read storm
    // (every in-flight op completes short in one reap pass) can be re-queued
    // without ever exhausting SQEs.  `ring_entries` is taken as a *minimum*;
    // callers asking for a larger ring get what they asked for.
    _ring_entries(std::max<unsigned>(ring_entries, 2u * NUM_CHUNKS))
{
  _bounce_storage = mr.allocate_multiple_blocks(NUM_CHUNKS * _bounce_slot_size);
  auto blocks     = _bounce_storage->get_blocks();
  if (blocks.size() < NUM_CHUNKS) {
    throw std::runtime_error(
      "uring_reactor: fixed_size_host_memory_resource returned fewer blocks (" +
      std::to_string(blocks.size()) + ") than required (" + std::to_string(NUM_CHUNKS) + ")");
  }
  for (int i = 0; i < static_cast<int>(NUM_CHUNKS); ++i) {
    _bounce[i].buf     = blocks[i];
    _cb_args[i]        = {this, i, nullptr};
    cudaError_t ev_err = cudaEventCreateWithFlags(&_cb_args[i].event, cudaEventDisableTiming);
    if (ev_err != cudaSuccess) {
      for (int j = 0; j < i; ++j)
        cudaEventDestroy(_cb_args[j].event);
      throw std::runtime_error(std::string("uring_reactor: cudaEventCreateWithFlags failed: ") +
                               cudaGetErrorString(ev_err));
    }
  }

  _worker = std::thread([this] { worker_loop(); });
}

uring_reactor::~uring_reactor()
{
  shutdown();
  for (int i = 0; i < static_cast<int>(NUM_CHUNKS); ++i) {
    if (_cb_args[i].event) cudaEventDestroy(_cb_args[i].event);
  }
}

void uring_reactor::interrupt() { _request_queue.notify(); }

void uring_reactor::shutdown()
{
  if (_worker.joinable()) {
    _stop.store(true, std::memory_order_release);
    interrupt();
    _worker.join();
  }
}

cudf::io::text::byte_range_info uring_reactor::align_to_physical(
  cudf::io::text::byte_range_info logical, size_t file_size)
{
  auto offset    = static_cast<size_t>(logical.offset());
  auto size      = static_cast<size_t>(logical.size());
  size_t a_start = offset & ~(IO_BLOCK_SIZE - 1);
  size_t a_end   = std::min((offset + size + IO_BLOCK_SIZE - 1) & ~(IO_BLOCK_SIZE - 1),
                          (file_size + IO_BLOCK_SIZE - 1) & ~(IO_BLOCK_SIZE - 1));
  return {static_cast<int64_t>(a_start), static_cast<int64_t>(a_end - a_start)};
}

bool uring_reactor::supports(std::string_view path)
{
  std::error_code ec;
  std::filesystem::path p{path};
  return std::filesystem::is_regular_file(p, ec) && !ec;
}

void uring_reactor::enqueue_bulk(std::span<device_read_req_type> batch)
{
  if (batch.empty()) return;

  // Capture ctxs before bulk-enqueue: on failure (OOM in moodycamel) items
  // may already be moved-from, so batch[i].ctx could be null.  The captured
  // shared_ptrs let us drain ctx->pending via chunk_failed so the
  // completion handler still fires instead of stranding readers in
  // wait_while_pending().
  std::vector<std::shared_ptr<request_context>> ctxs;
  ctxs.reserve(batch.size());
  for (auto& r : batch)
    ctxs.push_back(r.ctx);

  if (!_request_queue.try_enqueue_device_bulk(std::make_move_iterator(batch.begin()),
                                              batch.size())) {
    auto e = std::make_exception_ptr(
      std::runtime_error("uring_reactor::enqueue_bulk: queue enqueue failed"));
    for (auto& ctx : ctxs)
      ctx->chunk_failed(e);
  }
}

size_t uring_reactor::host_read(int fd, size_t offset, size_t size, uint8_t* dst)
{
  if (size == 0) return 0;
  // Loop until either the full requested size is read, EOF (n == 0), or a
  // real error. pread on a regular file should only return short on EOF, but
  // we retry defensively against EINTR and any unexpected short-read paths
  // so callers don't have to.
  size_t total = 0;
  while (total < size) {
    ssize_t n = ::pread(fd, dst + total, size - total, static_cast<off_t>(offset + total));
    if (n < 0) {
      if (errno == EINTR) continue;
      throw std::runtime_error("uring_reactor::host_read pread: " + std::string(strerror(errno)));
    }
    if (n == 0) break;  // EOF
    total += static_cast<size_t>(n);
  }
  return total;
}

void uring_reactor::host_read_async(host_read_req_type req)
{
  // Hold the ctx separately so we can drain pending via chunk_failed if the
  // enqueue itself fails (otherwise req.ctx may be moved-from and lost).
  auto ctx = req.ctx;
  if (!_request_queue.try_enqueue_host(std::move(req))) {
    ctx->chunk_failed(std::make_exception_ptr(
      std::runtime_error("uring_reactor::host_read_async: queue enqueue failed")));
  }
}

void uring_reactor::host_enqueue_bulk(std::span<host_read_req_type> batch)
{
  if (batch.empty()) return;

  // See enqueue_bulk() above for the rationale: snapshot ctxs first so a
  // partial-move failure on the moodycamel queue still drains ctx->pending
  // and fires the completion handler instead of deadlocking readers.
  std::vector<std::shared_ptr<request_context>> ctxs;
  ctxs.reserve(batch.size());
  for (auto& r : batch)
    ctxs.push_back(r.ctx);

  if (!_request_queue.try_enqueue_host_bulk(std::make_move_iterator(batch.begin()), batch.size())) {
    auto e = std::make_exception_ptr(
      std::runtime_error("uring_reactor::host_enqueue_bulk: queue enqueue failed"));
    for (auto& ctx : ctxs)
      ctx->chunk_failed(e);
  }
}

void uring_reactor::worker_loop()
{
  unique_ring ring = [this]() -> unique_ring {
#if defined(IORING_SETUP_SINGLE_ISSUER) && defined(IORING_SETUP_DEFER_TASKRUN)
    auto r                   = std::make_unique<io_uring>();
    struct io_uring_params p = {0};
    p.flags |= IORING_SETUP_SINGLE_ISSUER;
    p.flags |= IORING_SETUP_COOP_TASKRUN | IORING_SETUP_DEFER_TASKRUN;
    int rc = io_uring_queue_init_params(_ring_entries, r.get(), &p);
    if (rc == 0) {
      spdlog::debug("uring_device_reactor: ring using SINGLE_ISSUER|DEFER_TASKRUN, entries={}",
                    _ring_entries);
      return unique_ring{r.release()};
    }
    spdlog::debug(
      "uring_device_reactor: SINGLE_ISSUER|DEFER_TASKRUN unsupported "
      "({}), falling back to plain flags",
      strerror(-rc));
#endif
    auto r2 = std::make_unique<io_uring>();
    int rc2 = io_uring_queue_init(_ring_entries, r2.get(), 0);
    if (rc2 < 0)
      throw std::runtime_error("uring_reactor: ring init: " + std::string(strerror(-rc2)));
    spdlog::debug("uring_reactor: ring using plain flags, entries={}", _ring_entries);
    return unique_ring{r2.release()};
  }();

  // Register the managed bounce buffers with io_uring so device_managed reads
  // can use prep_read_fixed — the kernel keeps the pages pinned across all
  // IOs on this ring, avoiding per-IO pin/unmap overhead.  Host and BYO reads
  // use plain prep_read since those buffers are not in this table.
  std::array<iovec, NUM_CHUNKS> iovecs{};
  for (size_t i = 0; i < NUM_CHUNKS; ++i)
    iovecs[i] = {_bounce[i].buf, _bounce_slot_size};
  if (int rc = io_uring_register_buffers(ring.get(), iovecs.data(), NUM_CHUNKS); rc < 0)
    throw std::runtime_error("uring_reactor: io_uring_register_buffers: " +
                             std::string(strerror(-rc)));

  // ---------------------------------------------------------------------------
  // io_slot — flat per-slot state extracted from the incoming request.
  //
  // io_buffer is the buffer io_uring reads into. It comes from one of two
  // sources, distinguished by is_registered:
  //   is_registered == true  → io_buffer points into _bounce[slot] (managed
  //                            device read); SQE uses prep_read_fixed with
  //                            buf_index == slot.  Slot lifetime owns the
  //                            bounce buffer.
  //   is_registered == false → io_buffer is user-provided (host read dst, or
  //                            BYO device bounce); SQE uses prep_read.  Slot
  //                            lifetime does not own the buffer.
  //
  // destination_buf encodes the output kind:
  //   nullptr  → host read (data already in io_buffer; no H2D copy needed)
  //   non-null → device read (H2D copy from io_buffer + user_offset)
  // ---------------------------------------------------------------------------
  struct io_slot {
    int fd{-1};
    void* io_buffer{nullptr};
    size_t io_offset{0};
    size_t io_size{0};
    size_t user_offset{0};
    size_t user_size{0};
    void* destination_buf{nullptr};
    bool is_registered{false};
    cudaStream_t stream{nullptr};
    int device_id{-1};
    size_t bytes_read{0};
    std::shared_ptr<request_context> ctx;
  };
  std::array<io_slot, NUM_CHUNKS> slots{};

  auto update_slot_device = [this](device_read_req_type const& req, int si, io_slot& s) {
    s.fd              = req.handle;
    s.io_offset       = req.file_off;
    s.io_size         = req.io_size;
    s.user_offset     = req.data_off;
    s.user_size       = req.data_size;
    s.destination_buf = req.dst;
    s.stream          = req.stream;
    s.device_id       = req.device_id;
    s.ctx             = req.ctx;
    s.bytes_read      = 0;
    // Managed: use the slot's registered bounce buffer.
    // BYO: caller-provided bounce buffer.
    s.is_registered = (req.bounce == nullptr);
    s.io_buffer     = s.is_registered ? _bounce[si].buf : req.bounce;
    // Defense against upstream contract violations: io_size must fit in the
    // registered slab. Chunks are 1 MiB and slab is sized for that — this
    // should never trip.
    assert(!s.is_registered || s.io_size <= _bounce_slot_size);
  };

  auto update_slot_host = [](host_read_req_type const& req, io_slot& s) {
    s.fd              = req.handle;
    s.io_offset       = req.offset;
    s.io_size         = req.size;
    s.user_offset     = 0;
    s.user_size       = req.size;
    s.destination_buf = nullptr;
    s.io_buffer       = req.dst;
    s.is_registered   = false;
    s.stream          = nullptr;
    s.device_id       = -1;
    s.ctx             = req.ctx;
    s.bytes_read      = 0;
  };

  int inflight = 0;

  // Dequeue directly from _request_queue into io_uring — no intermediate buffer.
  // Acquires slot and SQE first (both non-blocking); if the queue turns out to
  // be empty, releases them and exits.  Pool-exhausted and ring-full cases are
  // handled by the main loop (Park B / Park C).
  auto submit_pending = [&]() {
    int added = 0;
    while (true) {
      int si = _slot_pool.try_acquire();
      if (si < 0) break;  // pool exhausted

      // Dequeue BEFORE acquiring the SQE: io_uring_get_sqe() advances the
      // ring's sqe_tail and the next io_uring_submit() ships everything up
      // to that tail.  If we got an SQE and then found the queue empty, the
      // unprepared SQE would submit as garbage (zero-initialized → NOP with
      // user_data=0 on the first wrap), producing a phantom CQE that
      // reap_cqes would route into slots[0] — corrupting inflight, double-
      // releasing the slot, and dereferencing a moved-from s.ctx.
      auto& s = slots[si];
      device_read_req_type dr;
      host_read_req_type hr;
      if (_request_queue.try_dequeue_device(dr)) {
        update_slot_device(dr, si, s);
      } else if (_request_queue.try_dequeue_host(hr)) {
        update_slot_host(hr, s);
      } else {
        _slot_pool.release(si);
        break;  // queue empty
      }

      io_uring_sqe* sqe = io_uring_get_sqe(ring.get());
      if (!sqe) {
        // Provably unreachable given _ring_entries >= 2 * NUM_CHUNKS: a free
        // slot implies a free SQE.  If this fires the sizing invariant has
        // been violated; we've already moved the request out of the queue
        // (no push-back on moodycamel) so drain the ctx via chunk_failed.
        spdlog::critical(
          "uring_reactor: SQE exhausted in submit_pending after dequeue — "
          "ring sizing invariant violated (slot={})",
          si);
        s.ctx->chunk_failed(std::make_exception_ptr(std::runtime_error(
          "uring_reactor: SQE exhausted in submit_pending — ring sizing invariant violated")));
        _slot_pool.release(si);
        break;
      }

      if (s.is_registered) {
        io_uring_prep_read_fixed(sqe,
                                 s.fd,
                                 s.io_buffer,
                                 static_cast<unsigned>(s.io_size),
                                 static_cast<unsigned long long>(s.io_offset),
                                 si);
      } else {
        io_uring_prep_read(sqe,
                           s.fd,
                           s.io_buffer,
                           static_cast<unsigned>(s.io_size),
                           static_cast<__u64>(s.io_offset));
      }

      io_uring_sqe_set_data64(sqe, static_cast<uint64_t>(si));
      ++inflight;
      ++added;
    }
    if (added > 0) {
      int rc = io_uring_submit(ring.get());
      if (rc < 0) {
        // Fatal: SQEs are queued in the ring but the kernel won't consume
        // them. inflight has already been incremented for each, so without
        // intervention we'd deadlock waiting for completions that can never
        // arrive. There's no clean recovery here — log loudly and abort the
        // worker. Any in-flight ops are abandoned; pending ctxs in the queue
        // are left untouched (callers will time out on their own waits, or
        // shutdown will drain them).
        spdlog::critical(
          "uring_reactor: io_uring_submit failed in submit_pending: {} "
          "(added={}, inflight={})",
          strerror(-rc),
          added,
          inflight);
        throw std::runtime_error("uring_reactor: io_uring_submit failed: " +
                                 std::string(strerror(-rc)));
      }
      if (rc < added) {
        // Partial submit. Extremely rare on a non-IOPOLL/non-SQPOLL ring on
        // modern kernels, but flag it loudly. The un-submitted SQEs are
        // still in the SQ ring and will be picked up by the next submit()
        // call (e.g. from the next reap pass or submit_pending invocation).
        // inflight accounting is still consistent because we never decrement
        // until a CQE arrives.
        spdlog::warn("uring_reactor: short submit in submit_pending: {}/{}", rc, added);
      }
    }
  };

  auto reap_cqes = [&]() {
    std::array<io_uring_cqe*, NUM_CHUNKS> cqes{};
    unsigned n         = io_uring_peek_batch_cqe(ring.get(), cqes.data(), NUM_CHUNKS);
    bool need_resubmit = false;

    // Helper: get an SQE, draining pending retries to the kernel if the ring
    // appears full. With _ring_entries >= 2 * NUM_CHUNKS this is provably
    // sufficient — at most NUM_CHUNKS retries can be queued in one reap pass
    // and a submit() between them frees every slot in the ring.
    auto get_sqe_with_drain = [&]() -> io_uring_sqe* {
      io_uring_sqe* sqe = io_uring_get_sqe(ring.get());
      if (sqe) return sqe;
      if (need_resubmit) {
        int rc = io_uring_submit(ring.get());
        if (rc < 0) {
          spdlog::critical("uring_reactor: io_uring_submit failed in reap_cqes drain: {}",
                           strerror(-rc));
          throw std::runtime_error(
            "uring_reactor: io_uring_submit failed during short-read drain: " +
            std::string(strerror(-rc)));
        }
        need_resubmit = false;
      }
      return io_uring_get_sqe(ring.get());
    };

    for (auto* cqe : std::span{cqes.data(), n}) {
      uint64_t raw = io_uring_cqe_get_data64(cqe);
      assert(raw < NUM_CHUNKS);
      int si  = static_cast<int>(raw);
      int res = cqe->res;
      io_uring_cqe_seen(ring.get(), cqe);
      --inflight;

      auto& s = slots[si];

      if (res < 0) {
        s.ctx->chunk_failed(std::make_exception_ptr(std::runtime_error(strerror(-res))));
        _slot_pool.release(si);
        continue;
      }

      s.bytes_read += static_cast<size_t>(res);
      bool const fully_read = s.bytes_read >= s.io_size;
      bool const eof        = (res == 0);

      if (!fully_read && !eof) {
        io_uring_sqe* sqe = get_sqe_with_drain();
        if (!sqe) {
          // Provably unreachable given _ring_entries >= 2 * NUM_CHUNKS: at
          // worst the SQ ring held NUM_CHUNKS retries from this pass, and
          // get_sqe_with_drain() just flushed all of them. If this fires the
          // sizing invariant has been violated; fail loud rather than deliver
          // a truncated chunk to the caller.
          spdlog::critical(
            "uring_reactor: SQE exhausted on short-read retry after drain. "
            "slot={} bytes_read={}/{}",
            si,
            s.bytes_read,
            s.io_size);
          s.ctx->chunk_failed(std::make_exception_ptr(std::runtime_error(
            "uring_reactor: SQE exhausted on short-read retry — ring sizing invariant violated")));
          _slot_pool.release(si);
          continue;
        }

        size_t remaining   = s.io_size - s.bytes_read;
        auto next_file_off = static_cast<unsigned long long>(s.io_offset + s.bytes_read);
        if (s.is_registered) {
          io_uring_prep_read_fixed(sqe,
                                   s.fd,
                                   static_cast<uint8_t*>(s.io_buffer) + s.bytes_read,
                                   static_cast<unsigned>(remaining),
                                   next_file_off,
                                   si);
        } else {
          io_uring_prep_read(sqe,
                             s.fd,
                             static_cast<uint8_t*>(s.io_buffer) + s.bytes_read,
                             static_cast<unsigned>(remaining),
                             static_cast<__u64>(next_file_off));
        }
        io_uring_sqe_set_data64(sqe, static_cast<uint64_t>(si));
        ++inflight;
        need_resubmit = true;
        spdlog::warn(
          "uring_reactor: short read, retrying. slot={} fd={} io_offset={} "
          "io_size={} bytes_read={} this_rd={}",
          si,
          s.fd,
          s.io_offset,
          s.io_size,
          s.bytes_read,
          res);
        continue;
      }

      // --- Complete the request -----------------------------------------

      if (s.destination_buf == nullptr) {
        // Host read: data already landed in io_buffer (== user dst).
        s.ctx->chunk_done();
        _slot_pool.release(si);
        continue;
      }

      // Device read: compute how many bytes to H2D-copy into destination_buf.
      size_t actual =
        s.bytes_read > s.user_offset ? std::min(s.user_size, s.bytes_read - s.user_offset) : 0;
      void* src_buf = static_cast<uint8_t*>(s.io_buffer) + s.user_offset;

      if (actual == 0 || s.ctx->failed.load(std::memory_order_relaxed)) {
        if (actual == 0)
          spdlog::warn(
            "uring_reactor: chunk completed with no H2D. slot={} fd={} "
            "io_offset={} io_size={} user_offset={} user_size={} "
            "bytes_read={} ctx_failed={}",
            si,
            s.fd,
            s.io_offset,
            s.io_size,
            s.user_offset,
            s.user_size,
            s.bytes_read,
            s.ctx->failed.load(std::memory_order_relaxed));
        s.ctx->chunk_done();
        _slot_pool.release(si);
        continue;
      }

      if (s.device_id >= 0) {
        cudaError_t set_err = cudaSetDevice(s.device_id);
        if (set_err != cudaSuccess) {
          cudaGetLastError();
          s.ctx->chunk_failed(std::make_exception_ptr(std::runtime_error(
            std::string("uring_reactor: cudaSetDevice failed: ") + cudaGetErrorString(set_err))));
          _slot_pool.release(si);
          continue;
        }
      }
      cudaError_t cpy_err =
        cudaMemcpyAsync(s.destination_buf, src_buf, actual, cudaMemcpyHostToDevice, s.stream);
      if (cpy_err != cudaSuccess) {
        cudaGetLastError();
        s.ctx->chunk_failed(std::make_exception_ptr(std::runtime_error(
          std::string("uring_reactor: cudaMemcpyAsync failed: ") + cudaGetErrorString(cpy_err))));
        _slot_pool.release(si);
        continue;
      }

      if (!s.is_registered) {
        // BYO: caller owns the bounce buffer; fire-and-forget.
        // CUDA errors poison the stream and surface on the next stream-sync.
        s.ctx->chunk_done();
        _slot_pool.release(si);
      } else {
        // Managed: stash ctx in cb_arg and record an event on the copy
        // stream.  poll_copy_completions() on the worker thread queries the
        // event each iteration and drives chunk_done/chunk_failed + slot
        // release when the H2D copy has finished.
        _cb_args[si].ctx = std::move(s.ctx);
        auto stream      = s.stream;
        _copying_count.fetch_add(1, std::memory_order_relaxed);
        cudaError_t ev_err = cudaEventRecord(_cb_args[si].event, stream);
        if (ev_err != cudaSuccess) {
          // Event record failed: poll_copy_completions will never see this
          // slot as ready, so roll back here exactly as we would in the
          // callback path — chunk_failed the ctx, unwind _copying_count,
          // and release the slot.
          cudaGetLastError();
          auto ctx = std::move(_cb_args[si].ctx);
          ctx->chunk_failed(std::make_exception_ptr(std::runtime_error(
            std::string("uring_reactor: cudaEventRecord failed: ") + cudaGetErrorString(ev_err))));
          _copying_count.fetch_sub(1, std::memory_order_release);
          _slot_pool.release(si);
        }
        // Otherwise: slot released in poll_copy_completions.
      }
    }
    if (need_resubmit) {
      int rc = io_uring_submit(ring.get());
      if (rc < 0) {
        spdlog::critical("uring_reactor: io_uring_submit failed after reap retries: {}",
                         strerror(-rc));
        throw std::runtime_error("uring_reactor: io_uring_submit failed after reap retries: " +
                                 std::string(strerror(-rc)));
      }
    }
  };

  // ---------------------------------------------------------------------------
  // poll_copy_completions — drain slots whose H2D event has fired.
  //
  // Runs on the worker thread.  Iterates all slots that have a pending ctx,
  // queries their cudaEvent, and on completion calls chunk_done/chunk_failed,
  // resets the ctx, releases the slot back to _slot_pool, and decrements
  // _copying_count.
  // ---------------------------------------------------------------------------
  auto poll_copy_completions = [&]() {
    for (int i = 0; i < static_cast<int>(NUM_CHUNKS); ++i) {
      if (!_cb_args[i].ctx) continue;
      cudaError_t ev_status = cudaEventQuery(_cb_args[i].event);
      if (ev_status == cudaErrorNotReady) continue;
      if (ev_status == cudaSuccess) {
        _cb_args[i].ctx->chunk_done();
      } else {
        cudaGetLastError();
        _cb_args[i].ctx->chunk_failed(std::make_exception_ptr(std::runtime_error(
          std::string("uring_reactor: H2D copy failed: ") + cudaGetErrorString(ev_status))));
      }
      _cb_args[i].ctx.reset();
      _slot_pool.release(i);
      _copying_count.fetch_sub(1, std::memory_order_release);
    }
  };

  // ---------------------------------------------------------------------------
  // Main loop
  //
  // Shutdown: queued-but-not-yet-submitted requests are cancelled immediately
  // via chunk_failed.  Already-submitted io_uring ops (inflight) and pending
  // CUDA copies (_copying_count) are allowed to complete naturally.
  //
  // Park points:
  //  A) queue empty, no in-flight IO, no pending copies: block on the
  //     _request_queue seq counter.  If copies are still in flight, yield
  //     instead so the loop spins back to poll_copy_completions.
  //  B) in-flight IO: wait on io_uring_wait_cqe_timeout; reap releases slots.
  //  C) inflight == 0 but pool exhausted (all slots held by H2D copies):
  //     yield so the loop spins back to poll_copy_completions, which releases
  //     slots as events fire.
  // ---------------------------------------------------------------------------
  static constexpr long SHUTDOWN_POLL_MS = 100;
  auto shutdown_err = std::make_exception_ptr(std::runtime_error("uring_reactor: shutting down"));

  while (true) {
    if (_stop.load(std::memory_order_acquire)) {
      device_read_req_type dr;
      while (_request_queue.try_dequeue_device(dr))
        dr.ctx->chunk_failed(shutdown_err);
      host_read_req_type hr;
      while (_request_queue.try_dequeue_host(hr))
        hr.ctx->chunk_failed(shutdown_err);
      poll_copy_completions();
      if (inflight == 0 && _copying_count.load(std::memory_order_acquire) == 0) break;
    }

    poll_copy_completions();

    // Park A: nothing queued and no io_uring work in flight.
    if (_request_queue.empty() && inflight == 0) {
      if (_copying_count.load(std::memory_order_acquire) == 0) {
        // Truly idle: block until a producer enqueues work.
        uint64_t seq = _request_queue.current_seq();
        if (_request_queue.empty()) _request_queue.wait(seq);
      } else {
        // H2D copies still in flight: yield so we loop back to poll them.
        std::this_thread::yield();
      }
      continue;
    }

    submit_pending();

    if (inflight > 0) {
      io_uring_cqe* tmp = nullptr;
      // Bounded wait so the top-of-loop _stop check is reachable even when
      // no CQE arrives.  SINGLE_ISSUER means we can't post a NOP SQE from
      // interrupt() to unblock a plain wait_cqe; the timeout bounds shutdown
      // latency to SHUTDOWN_POLL_MS.
      __kernel_timespec ts{};
      ts.tv_sec  = SHUTDOWN_POLL_MS / 1000;
      ts.tv_nsec = (SHUTDOWN_POLL_MS % 1000) * 1'000'000L;
      int rc     = io_uring_wait_cqe_timeout(ring.get(), &tmp, &ts);
      if (rc < 0 && rc != -EINTR && rc != -ETIME) {
        spdlog::error("uring_reactor: io_uring_wait_cqe_timeout failed: {}", strerror(-rc));
        break;
      }
      reap_cqes();
      continue;
    }

    // Park C: inflight == 0, queue non-empty, pool exhausted by H2D copies.
    // Yield so the loop spins back to poll_copy_completions, which will
    // release slots as events fire.
    if (!_slot_pool.any_free()) { std::this_thread::yield(); }
  }
}

}  // namespace sirius::io
