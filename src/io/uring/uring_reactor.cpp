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
#include <cstring>
#include <deque>
#include <filesystem>
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
  : _bounce_slot_size(mr.get_block_size()), _ring_entries(ring_entries)
{
  _bounce_storage = mr.allocate_multiple_blocks(NUM_CHUNKS * _bounce_slot_size);
  auto blocks     = _bounce_storage->get_blocks();
  if (blocks.size() < NUM_CHUNKS) {
    throw std::runtime_error(
      "uring_reactor: fixed_size_host_memory_resource returned fewer blocks (" +
      std::to_string(blocks.size()) + ") than required (" + std::to_string(NUM_CHUNKS) + ")");
  }
  for (int i = 0; i < static_cast<int>(NUM_CHUNKS); ++i) {
    _bounce[i].buf = blocks[i];
    _cb_args[i]    = {this, i, nullptr};
  }

  _worker = std::thread([this] { worker_loop(); });
}

uring_reactor::~uring_reactor() { shutdown(); }

void uring_reactor::interrupt()
{
  _wake_seq.fetch_add(1, std::memory_order_release);
  _wake_seq.notify_one();
}

void uring_reactor::shutdown()
{
  if (_worker.joinable()) {
    _stop.store(true, std::memory_order_release);
    interrupt();
    _worker.join();
  }
}

void uring_reactor::cuda_copy_cb(cudaStream_t /*stream*/, cudaError_t status, void* p) noexcept
{
  auto* arg = static_cast<cb_arg*>(p);
  if (status != cudaSuccess) {
    arg->ctx->chunk_failed(std::make_exception_ptr(std::runtime_error(
      std::string("uring_reactor: H2D copy failed: ") + cudaGetErrorString(status))));
  } else {
    arg->ctx->chunk_done();
  }
  arg->ctx.reset();
  arg->self->_copying_count.fetch_sub(1, std::memory_order_release);
  arg->self->_slot_pool.release(arg->slot);
  arg->self->_wake_seq.fetch_add(1, std::memory_order_release);
  arg->self->_wake_seq.notify_one();
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
  // wait_while_loading().
  std::vector<std::shared_ptr<request_context>> ctxs;
  ctxs.reserve(batch.size());
  for (auto& r : batch)
    ctxs.push_back(r.ctx);

  if (!_queue.enqueue_bulk(std::make_move_iterator(batch.begin()), batch.size())) {
    auto e = std::make_exception_ptr(
      std::runtime_error("uring_reactor::enqueue_bulk: queue enqueue failed"));
    for (auto& ctx : ctxs)
      ctx->chunk_failed(e);
    return;
  }

  _wake_seq.fetch_add(1, std::memory_order_release);
  _wake_seq.notify_one();
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
  if (!_host_queue.enqueue(std::move(req))) {
    ctx->chunk_failed(std::make_exception_ptr(
      std::runtime_error("uring_reactor::host_read_async: queue enqueue failed")));
    return;
  }
  _wake_seq.fetch_add(1, std::memory_order_release);
  _wake_seq.notify_one();
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

  if (!_host_queue.enqueue_bulk(std::make_move_iterator(batch.begin()), batch.size())) {
    auto e = std::make_exception_ptr(
      std::runtime_error("uring_reactor::host_enqueue_bulk: queue enqueue failed"));
    for (auto& ctx : ctxs)
      ctx->chunk_failed(e);
    return;
  }

  _wake_seq.fetch_add(1, std::memory_order_release);
  _wake_seq.notify_one();
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
      spdlog::debug("uring_device_reactor: ring using SINGLE_ISSUER|DEFER_TASKRUN");
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
    spdlog::debug("uring_reactor: ring using plain flags");
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
  // Exactly one of registered_bounce_buf / user_host_buf is non-null:
  //   registered_bounce_buf  non-null → managed device read (prep_read_fixed)
  //   user_host_buf          non-null → host read or BYO device read (prep_read)
  //
  // destination_buf encodes the output kind:
  //   nullptr  → host read (data lands in user_host_buf; no H2D copy needed)
  //   non-null → device read (H2D copy required after disk IO completes)
  // ---------------------------------------------------------------------------
  struct io_slot {
    int fd{-1};
    void* registered_bounce_buf{nullptr};
    void* user_host_buf{nullptr};
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
  // registered_bounce_buf is invariant per slot — set once here, never
  // touched by update_slot_* again.
  for (size_t i = 0; i < NUM_CHUNKS; ++i)
    slots[i].registered_bounce_buf = _bounce[i].buf;

  auto update_slot_device = [](device_read_req_type const& req, io_slot& s) {
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
    s.user_host_buf   = req.bounce;  // nullptr for managed, BYO buffer otherwise
    s.is_registered   = (req.bounce == nullptr);
  };

  auto update_slot_host = [](host_read_req_type const& req, io_slot& s) {
    s.fd              = req.handle;
    s.io_offset       = req.offset;
    s.io_size         = req.size;
    s.user_offset     = 0;
    s.user_size       = req.size;
    s.destination_buf = nullptr;
    s.user_host_buf   = req.dst;
    s.is_registered   = false;
    s.stream          = nullptr;
    s.device_id       = -1;
    s.ctx             = req.ctx;
    s.bytes_read      = 0;
  };

  std::deque<device_read_req_type> pending;
  std::deque<host_read_req_type> pending_host;
  int inflight = 0;

  auto drain_queue = [&]() {
    device_read_req_type r;
    while (_queue.try_dequeue(r))
      pending.push_back(std::move(r));
    host_read_req_type hr;
    while (_host_queue.try_dequeue(hr))
      pending_host.push_back(std::move(hr));
  };

  // Submit as many queued requests as possible — up to the available ring
  // SQEs and slot_pool capacity (both bounded by NUM_CHUNKS).
  auto submit_pending = [&]() {
    int added = 0;
    while (!pending.empty() || !pending_host.empty()) {
      int si = _slot_pool.try_acquire();
      if (si < 0) break;  // pool exhausted — wait for completions

      io_uring_sqe* sqe = io_uring_get_sqe(ring.get());
      if (!sqe) {
        _slot_pool.release(si);
        break;  // ring full — submit what we have and wait for CQEs
      }

      auto& s = slots[si];
      if (!pending.empty()) {
        update_slot_device(pending.front(), s);
        pending.pop_front();
      } else {
        update_slot_host(pending_host.front(), s);
        pending_host.pop_front();
      }

      if (s.is_registered) {
        io_uring_prep_read_fixed(sqe,
                                 s.fd,
                                 s.registered_bounce_buf,
                                 static_cast<unsigned>(s.io_size),
                                 static_cast<unsigned long long>(s.io_offset),
                                 si);
      } else {
        io_uring_prep_read(sqe,
                           s.fd,
                           s.user_host_buf,
                           static_cast<unsigned>(s.io_size),
                           static_cast<__u64>(s.io_offset));
      }

      io_uring_sqe_set_data64(sqe, static_cast<uint64_t>(si));
      ++inflight;
      ++added;
    }
    if (added > 0) io_uring_submit(ring.get());
  };

  auto reap_cqes = [&]() {
    std::array<io_uring_cqe*, NUM_CHUNKS> cqes{};
    unsigned n         = io_uring_peek_batch_cqe(ring.get(), cqes.data(), NUM_CHUNKS);
    bool need_resubmit = false;
    for (auto* cqe : std::span{cqes.data(), n}) {
      int si  = static_cast<int>(io_uring_cqe_get_data64(cqe));
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
        io_uring_sqe* sqe = io_uring_get_sqe(ring.get());
        if (sqe) {
          size_t remaining   = s.io_size - s.bytes_read;
          auto next_file_off = static_cast<unsigned long long>(s.io_offset + s.bytes_read);
          if (s.is_registered) {
            io_uring_prep_read_fixed(sqe,
                                     s.fd,
                                     static_cast<uint8_t*>(s.registered_bounce_buf) + s.bytes_read,
                                     static_cast<unsigned>(remaining),
                                     next_file_off,
                                     si);
          } else {
            io_uring_prep_read(sqe,
                               s.fd,
                               static_cast<uint8_t*>(s.user_host_buf) + s.bytes_read,
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
        spdlog::warn("uring_reactor: SQE exhausted on short-read retry, slot={}", si);
        // Fall through and complete with whatever was read.
      }

      // --- Complete the request -----------------------------------------

      if (s.destination_buf == nullptr) {
        // Host read: data already landed in user_host_buf.
        s.ctx->chunk_done();
        _slot_pool.release(si);
        continue;
      }

      // Device read: compute how many bytes to H2D-copy into destination_buf.
      size_t actual =
        s.bytes_read > s.user_offset ? std::min(s.user_size, s.bytes_read - s.user_offset) : 0;
      void* src_buf = s.is_registered
                        ? static_cast<uint8_t*>(s.registered_bounce_buf) + s.user_offset
                        : static_cast<uint8_t*>(s.user_host_buf) + s.user_offset;

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

      if (s.device_id >= 0) cudaSetDevice(s.device_id);
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
        // Managed: stash ctx in cb_arg, register callback.
        // cudaStreamAddCallback fires even if the stream is in an error
        // state — chunk_done/chunk_failed always fires.
        _cb_args[si].ctx = std::move(s.ctx);
        auto stream      = s.stream;
        _copying_count.fetch_add(1, std::memory_order_relaxed);
        cudaStreamAddCallback(stream, cuda_copy_cb, &_cb_args[si], 0);
        // Slot released in cuda_copy_cb.
      }
    }
    if (need_resubmit) io_uring_submit(ring.get());
  };

  // ---------------------------------------------------------------------------
  // Main loop
  //
  // Park points:
  //  A) truly idle (nothing queued, no in-flight IO, no pending CUDA copies):
  //     wait on _wake_seq — bumped by enqueue_bulk / interrupt / cuda_copy_cb.
  //  B) pending items but pool exhausted (all slots in CUDA callback):
  //     wait on _wake_seq — cuda_copy_cb bumps it when a slot is released.
  //  C) in-flight IO: wait on io_uring_wait_cqe_timeout.
  // ---------------------------------------------------------------------------
  while (true) {
    drain_queue();

    if (_stop.load(std::memory_order_acquire)) {
      // Drain any remaining CUDA callbacks before destroying reactor members.
      while (_copying_count.load(std::memory_order_acquire) > 0) {
        uint64_t seq = _wake_seq.load(std::memory_order_acquire);
        if (_copying_count.load(std::memory_order_acquire) == 0) break;
        _wake_seq.wait(seq, std::memory_order_relaxed);
      }
      break;
    }

    bool const have_work = !pending.empty() || !pending_host.empty();

    if (!have_work && inflight == 0 && _copying_count.load(std::memory_order_acquire) == 0) {
      // Park A: truly idle.
      uint64_t seq = _wake_seq.load(std::memory_order_acquire);
      drain_queue();
      if (pending.empty() && pending_host.empty() && inflight == 0 &&
          _copying_count.load(std::memory_order_acquire) == 0 &&
          !_stop.load(std::memory_order_acquire)) {
        _wake_seq.wait(seq, std::memory_order_relaxed);
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
      static constexpr long SHUTDOWN_POLL_MS = 100;
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

    // Park B: have pending work but pool is exhausted (all slots held by CUDA
    // callbacks).  Wait for a callback to release a slot and bump _wake_seq.
    uint64_t seq = _wake_seq.load(std::memory_order_acquire);
    drain_queue();
    if (!pending.empty() || !pending_host.empty()) {
      if (!_slot_pool.any_free() && !_stop.load(std::memory_order_acquire)) {
        _wake_seq.wait(seq, std::memory_order_relaxed);
      }
    }
  }
}

}  // namespace sirius::io
