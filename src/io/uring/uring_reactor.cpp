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

  // ---------------------------------------------------------------------------
  // Per-slot state — covers all three request kinds (host, BYO device,
  // managed device).  The slot pool is the single source of truth for
  // free/in-use status; slots[] is a parallel array of per-slot payloads.
  //
  // host        : read goes directly into host_req.dst; slot released on CQE.
  // device_byo  : read into dev_req.bounce (caller-supplied); fire H2D async
  //               + chunk_done(); release slot immediately.
  // device_managed: read into _bounce[si].buf; fire H2D async + register
  //               cuda_copy_cb; callback calls chunk_done() + releases slot.
  // ---------------------------------------------------------------------------
  enum class slot_kind : uint8_t { host, device_byo, device_managed };
  struct slot_info {
    slot_kind kind{slot_kind::host};
    device_read_req_type dev_req{};
    host_read_req_type host_req{};
    size_t bytes_read{0};
  };
  std::array<slot_info, NUM_CHUNKS> slots{};

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

      slots[si].bytes_read = 0;

      if (!pending.empty()) {
        auto req = std::move(pending.front());
        pending.pop_front();
        slots[si].dev_req = std::move(req);

        if (slots[si].dev_req.bounce != nullptr) {
          slots[si].kind = slot_kind::device_byo;
          io_uring_prep_read(sqe,
                             slots[si].dev_req.handle,
                             slots[si].dev_req.bounce,
                             static_cast<unsigned>(slots[si].dev_req.io_size),
                             static_cast<unsigned long long>(slots[si].dev_req.file_off));
        } else {
          slots[si].kind = slot_kind::device_managed;
          io_uring_prep_read(sqe,
                             slots[si].dev_req.handle,
                             _bounce[si].buf,
                             static_cast<unsigned>(slots[si].dev_req.io_size),
                             static_cast<unsigned long long>(slots[si].dev_req.file_off));
        }
      } else {
        auto req = std::move(pending_host.front());
        pending_host.pop_front();
        slots[si].kind     = slot_kind::host;
        slots[si].host_req = std::move(req);
        io_uring_prep_read(sqe,
                           slots[si].host_req.handle,
                           slots[si].host_req.dst,
                           static_cast<unsigned>(slots[si].host_req.size),
                           static_cast<__u64>(slots[si].host_req.offset));
      }

      io_uring_sqe_set_data64(sqe, static_cast<uint64_t>(si));
      ++inflight;
      ++added;
    }
    if (added > 0) io_uring_submit(ring.get());
  };

  auto reap_cqes = [&]() {
    io_uring_cqe* cqes[NUM_CHUNKS];
    unsigned n         = io_uring_peek_batch_cqe(ring.get(), cqes, NUM_CHUNKS);
    bool need_resubmit = false;
    for (auto* cqe : std::span{cqes, n}) {
      int si  = static_cast<int>(io_uring_cqe_get_data64(cqe));
      int res = cqe->res;
      io_uring_cqe_seen(ring.get(), cqe);
      --inflight;

      auto& s = slots[si];

      if (res < 0) {
        auto e = std::make_exception_ptr(std::runtime_error(strerror(-res)));
        if (s.kind == slot_kind::host)
          s.host_req.ctx->chunk_failed(e);
        else
          s.dev_req.ctx->chunk_failed(e);
        s = {};
        _slot_pool.release(si);
        continue;
      }

      size_t rd = static_cast<size_t>(res);
      s.bytes_read += rd;

      size_t const expected = (s.kind == slot_kind::host) ? s.host_req.size : s.dev_req.io_size;
      bool const fully_read = s.bytes_read >= expected;
      bool const eof        = (rd == 0);

      if (!fully_read && !eof) {
        io_uring_sqe* sqe = io_uring_get_sqe(ring.get());
        if (sqe) {
          // Retry the unread tail into the same buffer at the next-byte offset.
          if (s.kind == slot_kind::host) {
            io_uring_prep_read(sqe,
                               s.host_req.handle,
                               s.host_req.dst + s.bytes_read,
                               static_cast<unsigned>(s.host_req.size - s.bytes_read),
                               static_cast<__u64>(s.host_req.offset + s.bytes_read));
            spdlog::warn(
              "uring_reactor: host short read, retrying. fd={} offset={} size={} "
              "bytes_read={} this_rd={}",
              s.host_req.handle,
              s.host_req.offset,
              s.host_req.size,
              s.bytes_read,
              rd);
          } else if (s.kind == slot_kind::device_byo) {
            io_uring_prep_read(sqe,
                               s.dev_req.handle,
                               s.dev_req.bounce + s.bytes_read,
                               static_cast<unsigned>(s.dev_req.io_size - s.bytes_read),
                               static_cast<__u64>(s.dev_req.file_off + s.bytes_read));
            spdlog::warn(
              "uring_reactor: BYO short read, retrying. fd={} file_off={} io_size={} "
              "bytes_read={} this_rd={}",
              s.dev_req.handle,
              s.dev_req.file_off,
              s.dev_req.io_size,
              s.bytes_read,
              rd);
          } else {
            io_uring_prep_read(sqe,
                               s.dev_req.handle,
                               static_cast<uint8_t*>(_bounce[si].buf) + s.bytes_read,
                               static_cast<unsigned>(s.dev_req.io_size - s.bytes_read),
                               static_cast<__u64>(s.dev_req.file_off + s.bytes_read));
            spdlog::warn(
              "uring_reactor: device short read, retrying. slot={} file_off={} io_size={} "
              "bytes_read={} this_rd={}",
              si,
              s.dev_req.file_off,
              s.dev_req.io_size,
              s.bytes_read,
              rd);
          }
          io_uring_sqe_set_data64(sqe, static_cast<uint64_t>(si));
          ++inflight;
          need_resubmit = true;
          continue;  // slot still in use
        }
        spdlog::warn("uring_reactor: SQE exhausted on short-read retry, slot={}", si);
        // Fall through and complete with whatever was read.
      }

      // --- Complete the request ------------------------------------------

      if (s.kind == slot_kind::host) {
        s.host_req.ctx->chunk_done();
        s = {};
        _slot_pool.release(si);
        continue;
      }

      // Device read (BYO or managed): compute user-visible bytes from the
      // accumulated bytes_read after all retries.
      auto& req = s.dev_req;
      size_t actual =
        s.bytes_read > req.data_off ? std::min(req.data_size, s.bytes_read - req.data_off) : 0;

      void* src_buf = (s.kind == slot_kind::device_byo)
                        ? static_cast<void*>(req.bounce + req.data_off)
                        : static_cast<void*>(static_cast<uint8_t*>(_bounce[si].buf) + req.data_off);

      if (actual == 0 || req.ctx->failed.load(std::memory_order_relaxed)) {
        if (actual == 0)
          spdlog::warn(
            "uring_reactor: chunk completed with no H2D. slot={} file_off={} io_size={} "
            "data_off={} data_size={} bytes_read={} ctx_failed={}",
            si,
            req.file_off,
            req.io_size,
            req.data_off,
            req.data_size,
            s.bytes_read,
            req.ctx->failed.load(std::memory_order_relaxed));
        req.ctx->chunk_done();
        s = {};
        _slot_pool.release(si);
        continue;
      }

      if (req.device_id >= 0) cudaSetDevice(req.device_id);
      cudaError_t cpy_err =
        cudaMemcpyAsync(req.dst, src_buf, actual, cudaMemcpyHostToDevice, req.stream);
      if (cpy_err != cudaSuccess) {
        cudaGetLastError();
        req.ctx->chunk_failed(std::make_exception_ptr(std::runtime_error(
          std::string("uring_reactor: cudaMemcpyAsync failed: ") + cudaGetErrorString(cpy_err))));
        s = {};
        _slot_pool.release(si);
        continue;
      }

      if (s.kind == slot_kind::device_byo) {
        // Fire-and-forget: the caller owns the bounce buffer and the stream.
        // chunk_done fires immediately so the request_context can complete
        // once all chunks are done.  CUDA errors poison the caller's stream
        // and will surface on the next stream-sync.
        req.ctx->chunk_done();
        s = {};
        _slot_pool.release(si);
      } else {
        // Managed slot: the callback owns the slot until the H2D copy
        // completes.  Stash the ctx in cb_arg so the callback can resolve it,
        // then register — cudaStreamAddCallback fires even if the stream is
        // already in an error state.
        _cb_args[si].ctx = req.ctx;
        s                = {};
        _copying_count.fetch_add(1, std::memory_order_relaxed);
        cudaStreamAddCallback(req.stream, cuda_copy_cb, &_cb_args[si], 0);
        // Slot released in cuda_copy_cb; do NOT call _slot_pool.release here.
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
