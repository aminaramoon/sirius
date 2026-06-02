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

#include "cucascade/cuda/event.hpp"
#include "driver_types.h"
#include "io/details/slot_pool.hpp"
#include "io/types.hpp"
#include "io/uring/types.hpp"
#include "util/error_utils.hpp"

#include <rmm/cuda_device.hpp>

#include <absl/cleanup/cleanup.h>
#include <fcntl.h>
#include <spdlog/spdlog.h>
#include <sys/stat.h>

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <memory>
#include <stdexcept>
#include <system_error>
#include <thread>
#include <vector>

namespace sirius::io::uring {

using request_type_ptr             = std::unique_ptr<rx_request>;
using chunk_io_request_type_ptr    = std::unique_ptr<chunked_rx_request>;
static constexpr size_t NUM_CHUNKS = 64;  // max concurrent device reads, i.e. ring size / 2

namespace {

/// True iff @p v is a multiple of IO_BLOCK_SIZE (O_DIRECT page size).
[[nodiscard]] constexpr bool is_block_aligned(size_t v) noexcept
{
  return (v & (static_cast<size_t>(IO_BLOCK_SIZE) - 1)) == 0;
}

struct io_slot {
  using slot_token = slot_pool<NUM_CHUNKS>::token;
  explicit io_slot(int slot_index, uint8_t* internal_buffer)
    : slot_index(slot_index), internal_buffer(internal_buffer)
  {
    assert(internal_buffer && "io_slot: internal_buffer must not be null");
  }

  void register_sqe(io_uring_sqe* sqe)
  {
    assert(sqe);
    auto segment = req->get_remaining_chunk(bytes_read);
    if (use_internal_buffer) {
      io_uring_prep_read_fixed(sqe,
                               req->fd,
                               segment.data(),
                               static_cast<unsigned>(segment.size),
                               static_cast<__u64>(segment.offset),
                               slot_index);
    } else {
      io_uring_prep_read(sqe,
                         req->fd,
                         segment.data(),
                         static_cast<unsigned>(segment.size),
                         static_cast<__u64>(segment.offset));
    }
    io_uring_sqe_set_data64(sqe, static_cast<uint64_t>(slot_index));
  }

  void on_request(chunk_io_request_type_ptr r,
                  slot_token token,
                  cucascade::cuda::cuda_event* cu_event = nullptr)
  {
    req                 = std::move(r);
    use_internal_buffer = req->chunk.data() == nullptr;
    if (use_internal_buffer) { req->chunk.set_data(internal_buffer, true); }
    bytes_read       = 0;
    this->pool_token = std::move(token);
    event            = cu_event;
  }

  void on_error(const typename request_manager::error_type& error,
                std::source_location loc = std::source_location::current())
  {
    req->manager->report_error(error, loc);
    reset();
  }

  void on_complete(std::size_t n_bytes)
  {
    req->manager->chunk_complete(n_bytes);
    reset();
  }

  cudaError_t copy_h2d_async()
  {
    cudaEvent_t copy_event = event ? (use_internal_buffer ? event->get() : nullptr) : nullptr;
    return req->copy_h2d_async(copy_event);
  }

  void reset()
  {
    req.reset();
    pool_token.reset();
    bytes_read = 0;
  }

  slot_token release_slot() noexcept { return std::exchange(pool_token, {}); }

  int slot_index;
  uint8_t* const internal_buffer;
  std::unique_ptr<chunked_rx_request> req;
  bool use_internal_buffer{false};
  size_t bytes_read{0};
  cucascade::cuda::cuda_event* event;
  slot_token pool_token;
};

/// Build one device-read chunk for the O_DIRECT-aligned file window
/// [@p window_off, @p window_off + @p read_size) (read_size <= _bounce_slot_size, with
/// both ends page aligned).
///
/// The reactor reads that whole window into @p host_buf — when @p host_buf is
/// null it stages the read through one of its own internal bounce slots
/// instead.  Once the data lands, the worker H2D-copies just the part of this
/// window that overlaps the request [@p req_offset, @p req_offset + @p req_size)
/// into @p dst at its position within that request.  For the first window the
/// copy offset is the alignment overhang (req_offset - window_off); for the
/// rest it is zero and the dst position advances by the bytes already copied.
[[nodiscard]] chunk_io_request_type_ptr make_device_chunk(int fd,
                                                          size_t window_off,
                                                          size_t read_size,
                                                          uint8_t* host_buf,
                                                          size_t req_offset,
                                                          size_t req_size,
                                                          uint8_t* dst,
                                                          rmm::cuda_stream_view stream,
                                                          int device_id,
                                                          std::shared_ptr<request_manager> manager)
{
  size_t const req_end = req_offset + req_size;
  size_t const data_lo = std::max(req_offset, window_off);
  size_t const data_hi = std::min(req_end, window_off + read_size);

  auto req   = std::make_unique<chunked_rx_request>();
  req->fd    = fd;
  req->chunk = io_object_segment{window_off, read_size, host_buf};

  auto cpy       = std::make_unique<device_cpy_request>();
  cpy->dst       = dst + (data_lo - req_offset);  // where this window lands in dst
  cpy->offset    = data_lo - window_off;          // offset of the wanted data within host_buf
  cpy->size      = data_hi - data_lo;
  cpy->stream    = stream;
  cpy->device_id = device_id;
  req->cpy_req   = std::move(cpy);

  req->manager = std::move(manager);
  return req;
}

/**
 * @brief Custom deleter for @c unique_ring: calls @c io_uring_queue_exit
 *        before freeing the allocation.
 */
struct ring_deleter {
  void operator()(io_uring* r) const noexcept
  {
    io_uring_queue_exit(r);
    delete r;
  }
};
using unique_ring_ptr = std::unique_ptr<io_uring, ring_deleter>;

unique_ring_ptr make_ring(unsigned depth)
{
#if defined(IORING_SETUP_SINGLE_ISSUER) && defined(IORING_SETUP_DEFER_TASKRUN)
  auto r                   = std::make_unique<io_uring>();
  struct io_uring_params p = {0};
  p.flags |= IORING_SETUP_SINGLE_ISSUER;
  p.flags |= IORING_SETUP_COOP_TASKRUN | IORING_SETUP_DEFER_TASKRUN;
  int rc = io_uring_queue_init_params(depth, r.get(), &p);
  if (rc == 0) {
    spdlog::trace("uring_device_reactor: ring using SINGLE_ISSUER|DEFER_TASKRUN, entries={}",
                  depth);
    return unique_ring_ptr{r.release()};
  }
  spdlog::trace(
    "uring_device_reactor: SINGLE_ISSUER|DEFER_TASKRUN unsupported "
    "({}), falling back to plain flags",
    strerror(-rc));
#endif
  auto r2 = std::make_unique<io_uring>();
  int rc2 = io_uring_queue_init(depth, r2.get(), 0);
  if (rc2 < 0) throw std::runtime_error("uring_reactor: ring init: " + std::string(strerror(-rc2)));
  spdlog::trace("uring_reactor: ring using plain flags, entries={}", depth);
  return unique_ring_ptr{r2.release()};
}

struct unique_ring {
  explicit unique_ring(unsigned depth) : ring(make_ring(depth)) {}
  ~unique_ring() noexcept = default;

  [[nodiscard]] io_uring* native_handle() const noexcept { return ring.get(); }

  void register_buffers(std::span<iovec> iovecs)
  {
    if (int rc = io_uring_register_buffers(ring.get(), iovecs.data(), iovecs.size()); rc < 0) {
      throw std::runtime_error("uring_reactor: io_uring_register_buffers: " +
                               std::string(strerror(-rc)));
    }
  }

  [[nodiscard]] int wait_for(std::chrono::milliseconds timeout) const
  {
    io_uring_cqe* tmp = nullptr;
    // Bounded wait so the top-of-loop _stop check is reachable even when
    // no CQE arrives.  SINGLE_ISSUER means we can't post a NOP SQE from
    // interrupt() to unblock a plain wait_cqe; the timeout bounds shutdown
    // latency to SHUTDOWN_POLL_MS.
    __kernel_timespec ts{};
    ts.tv_sec  = timeout.count() / 1000;
    ts.tv_nsec = (timeout.count() % 1000) * 1'000'000L;
    int rc     = io_uring_wait_cqe_timeout(ring.get(), &tmp, &ts);
    if (rc < 0 && rc != -EINTR && rc != -ETIME) { return -rc; }
    return rc;
  }

  [[nodiscard]] int wait() const
  {
    io_uring_cqe* tmp = nullptr;
    int rc            = io_uring_wait_cqe(ring.get(), &tmp);
    if (rc < 0) { return -1; }
    return 0;
  }

  [[nodiscard]] io_uring_sqe* get_sqe() const { return io_uring_get_sqe(ring.get()); }

  [[nodiscard]] io_uring_sqe* get_sqe_with_drain() const { return io_uring_get_sqe(ring.get()); }

  [[nodiscard]] unsigned peek_cqe_batch(std::span<io_uring_cqe*> cqes) const
  {
    return io_uring_peek_batch_cqe(ring.get(), cqes.data(), cqes.size());
  }

  [[nodiscard]] void mark_cqe_seen(io_uring_cqe* cqe) const { io_uring_cqe_seen(ring.get(), cqe); }

  void submit([[maybe_unused]] std::size_t n_added)
  {
    if (int rc = io_uring_submit(ring.get()); rc < 0) {
      throw std::runtime_error("uring_reactor: io_uring_submit: " + std::string(strerror(-rc)));
    }
  }

 private:
  unique_ring_ptr ring;
};

}  // namespace

// ---------------------------------------------------------------------------
// uring_reactor
// ---------------------------------------------------------------------------

uring_reactor::uring_reactor(cucascade::memory::fixed_size_host_memory_resource& mr,
                             std::string_view tname)
  : _config{mr.get_block_size()}, _bounce_slot_size(mr.get_block_size())
{
  _bounce_storage = mr.allocate_multiple_blocks(NUM_CHUNKS * _bounce_slot_size);
  _worker = std::jthread([this](std::stop_token stop_token) { worker_loop(std::move(stop_token)); },
                         _stop_source.get_token());
  if (!tname.empty()) {
    std::string full_name = std::string(tname) + "_worker";
    pthread_setname_np(_worker.native_handle(), full_name.c_str());
  }
}

uring_reactor::~uring_reactor() { shutdown(); }

std::unique_ptr<local_io_object> uring_reactor::create_io_object(std::string path)
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

  auto file_size = size(fd.native_handle());
  return std::make_unique<local_io_object>(
    std::move(path), std::move(fd), std::move(fd_direct), file_size);
}

size_t uring_reactor::size(int fd)
{
  struct stat st{};
  if (::fstat(fd, &st) != 0)
    throw std::runtime_error("uring_reactor::size: fstat failed: " + std::string(strerror(errno)));
  return static_cast<size_t>(st.st_size);
}

request_type_ptr uring_reactor::prep_host_rx_request(const reactor_config_type& cfg,
                                                     const io_object_type& file,
                                                     const io_object_segment& segment)
{
  if (segment.size == 0) { return rx_request::create({}); }

  // Buffered host read straight into the caller's destination buffer (no H2D
  // copy).  When align_for_device is set the read goes through the O_DIRECT fd,
  // so the destination, offset and size must all be page-aligned.
  if (segment.is_device_accessible()) {
    assert(segment.is_odirect_compatible() &&
           "create_host_rx_request: segment must be O_DIRECT compatible when "
           "is_device_accessible is set");
  }

  auto manager = std::make_shared<request_manager>(segment.size, 1);

  int const fd = segment.is_odirect_compatible() ? file.odirect_handle() : file.buffered_handle();

  // The read lands directly in the caller's buffer (no bounce, no H2D copy), so
  // a single request covers the whole range — no chunking needed.
  auto req     = std::make_unique<chunked_rx_request>();
  req->fd      = fd;
  req->chunk   = segment;
  req->manager = std::move(manager);

  std::vector<chunk_io_request_type_ptr> chunks;
  chunks.push_back(std::move(req));
  return rx_request::create(std::move(chunks));
}

request_type_ptr uring_reactor::prep_device_rx_request(const reactor_config_type& cfg,
                                                       const io_object_type& file,
                                                       uint8_t* dst,
                                                       size_t offset,
                                                       size_t size,
                                                       rmm::cuda_stream_view stream,
                                                       int device_id)
{
  if (size == 0) { return rx_request::create({}); }

  int const fd = file.odirect_handle();
  // align_to_physical aligns the offset down and the end up to IO_BLOCK_SIZE
  // (clamped to the file), giving an O_DIRECT-compliant span.
  auto const phys =
    align_to_physical({static_cast<int64_t>(offset), static_cast<int64_t>(size)}, file.size());
  auto const a_start  = static_cast<size_t>(phys.offset());
  auto const a_end    = a_start + static_cast<size_t>(phys.size());
  size_t alinged_size = phys.size();
  auto manager =
    std::make_shared<request_manager>(size, (alinged_size + cfg.bounce_size - 1) / cfg.bounce_size);

  std::vector<chunk_io_request_type_ptr> chunks;
  for (size_t w = a_start; w < a_end; w += cfg.bounce_size) {
    size_t const read_size = std::min<size_t>(cfg.bounce_size, a_end - w);
    chunks.push_back(make_device_chunk(
      fd, w, read_size, /*host_buf=*/nullptr, offset, size, dst, stream, device_id, manager));
  }
  return rx_request::create(std::move(chunks));
}

request_type_ptr uring_reactor::prep_host_to_device_rx_request(
  const reactor_config_type& cfg,
  const io_object_type& file,
  std::span<io_object_segment> segments,
  uint8_t* dst,
  size_t offset,
  size_t size,
  rmm::cuda_stream_view stream,
  int device_id)
{
  // Device read staged through caller-supplied pinned host buffers.  The
  // provider hands back one chunk-aligned segment per buffer it owns that
  // overlaps the request; those buffers may not cover the whole range.  For a
  // covered chunk we read straight into the provider's buffer; for a gap we
  // either read through an internal bounce slot (use_internal_buffer == true)
  // or skip it entirely.
  if (size == 0 || segments.empty()) { return rx_request::create({}); }

  auto manager =
    std::make_shared<request_manager>(segments.size() * cfg.bounce_size, segments.size());

  int const fd         = file.odirect_handle();
  size_t const req_end = offset + size;

  std::vector<chunk_io_request_type_ptr> chunks;
  size_t const first = (offset / cfg.bounce_size) * cfg.bounce_size;
  for (size_t w = first; w < req_end; w += cfg.bounce_size) {
    uint8_t* host_buf = nullptr;
    bool covered      = false;
    for (auto const& s : segments) {
      if (s.offset == w) {
        host_buf = s.data();
        covered  = true;
        break;
      }
    }
    // Covered: read straight into the provider's buffer.  Gap: read through an
    // internal bounce slot (host_buf == nullptr) or skip it entirely.
    if (!covered) continue;
    chunks.push_back(make_device_chunk(
      fd, w, cfg.bounce_size, host_buf, offset, size, dst, stream, device_id, manager));
  }
  return rx_request::create(std::move(chunks));
}

request_type_ptr uring_reactor::prep_host_rxv_request(const reactor_config_type& cfg,
                                                      const io_object_type& file,
                                                      std::span<io_object_segment> segments)
{
  if (segments.size() == 0) { return rx_request::create(std::vector<chunk_io_request_type_ptr>{}); }

  assert(segments.size() == dst.size() &&
         "create_host_rx_request: ranges and destination spans must be 1:1");

  bool is_device_accessible = segments.front().is_device_accessible();
  auto manager =
    std::make_shared<request_manager>(segments.size() * cfg.bounce_size, segments.size());

  int const fd = is_device_accessible ? file.odirect_handle() : file.buffered_handle();

  std::vector<chunk_io_request_type_ptr> chunks;
  chunks.reserve(segments.size());
  for (auto& s : segments) {
    if (is_device_accessible) {
      assert(s.is_odirect_compatible() &&
             "create_host_rx_request: all segments must be O_DIRECT compatible when "
             "align_for_device is set");
    }

    auto req     = std::make_unique<chunked_rx_request>();
    req->fd      = fd;
    req->chunk   = s;
    req->manager = manager;
    chunks.push_back(std::move(req));
  }
  return rx_request::create(std::move(chunks));
}

void uring_reactor::interrupt() {}

void uring_reactor::shutdown()
{
  if (_worker.joinable()) {
    _stop_source.request_stop();
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

size_t uring_reactor::host_read(const io_object_type& file,
                                size_t offset,
                                size_t size,
                                uint8_t* dst)
{
  if (size == 0) return 0;
  // Loop until either the full requested size is read, EOF (n == 0), or a
  // real error. pread on a regular file should only return short on EOF, but
  // we retry defensively against EINTR and any unexpected short-read paths
  // so callers don't have to.
  size_t total = 0;
  while (total < size) {
    ssize_t n = ::pread(
      file.buffered_handle(), dst + total, size - total, static_cast<off_t>(offset + total));
    if (n < 0) {
      if (errno == EINTR) continue;
      throw std::runtime_error("uring_reactor::host_read pread: " + std::string(strerror(errno)));
    }
    if (n == 0) break;  // EOF
    total += static_cast<size_t>(n);
  }
  return total;
}

void uring_reactor::enqueue(request_type_ptr req)
{
  auto chunks = req->get_all_chunks();
  enqueue_chunks(chunks);
}

void uring_reactor::enqueue_chunks(std::span<chunk_io_request_type_ptr> batch)
{
  bool success = _requests.enqueue_bulk(std::make_move_iterator(batch.data()), batch.size());
  if (!success) {
    throw std::runtime_error("uring_reactor::enqueue_chunks: failed to enqueue bulk requests");
  }
}

void uring_reactor::enqueue_chunk(chunk_io_request_type_ptr request)
{
  bool success = _requests.enqueue(std::move(request));
  if (!success) {
    throw std::runtime_error("uring_reactor::enqueue_chunk: failed to enqueue request");
  }
}

void uring_reactor::worker_loop(std::stop_token stop_token)
{
  static constexpr std::chrono::milliseconds SHUTDOWN_POLL_MS{100};

  std::stop_callback cb(stop_token, [this] {
    spdlog::trace("uring_reactor worker_loop: stop requested");
    _requests.enqueue(nullptr);  // unblock the worker if it's waiting on an empty queue
  });

  using slot_token = slot_pool<NUM_CHUNKS>::token;

  unique_ring ring(2 * NUM_CHUNKS);

  auto blocks = _bounce_storage->get_blocks();
  std::vector<iovec> iovecs;
  iovecs.reserve(blocks.size());
  std::ranges::transform(
    blocks, std::back_inserter(iovecs), [len = _bounce_slot_size](auto* b) mutable {
      return iovec{.iov_base = b, .iov_len = len};
    });

  slot_pool<NUM_CHUNKS> slot_pool;
  std::vector<io_slot> slots;
  slots.reserve(NUM_CHUNKS);
  std::ranges::transform(iovecs, std::back_inserter(slots), [i = 0](auto& b) mutable {
    return io_slot(i++, reinterpret_cast<uint8_t*>(b.iov_base));
  });

  std::array<io_uring_cqe*, NUM_CHUNKS> cqes;
  std::vector<int> incomplete_requests;
  incomplete_requests.reserve(NUM_CHUNKS);
  std::vector<slot_token> copying_slots;
  copying_slots.reserve(NUM_CHUNKS);
  std::unordered_map<int, std::vector<cucascade::cuda::cuda_event>> per_device_copy_events;
  auto n_devices = rmm::get_num_cuda_devices();
  for (int device_id = 0; device_id < n_devices; ++device_id) {
    rmm::cuda_set_device_raii device_guard(rmm::cuda_device_id{device_id});
    auto& events = per_device_copy_events[device_id];
    events.reserve(NUM_CHUNKS);
    std::generate_n(std::back_inserter(events), NUM_CHUNKS, []() {
      return cucascade::cuda::cuda_event{cudaEventDisableTiming};
    });
  }

  int inflight = 0;

  auto poll_copy_completions = [&]() {
    using query_status = cucascade::cuda::cuda_event::query_status;
    copying_slots.erase(std::remove_if(copying_slots.begin(),
                                       copying_slots.end(),
                                       [&](slot_token const& token) {
                                         int si         = token.slot_index();
                                         auto& s        = slots[si];
                                         auto ev_status = s.event->query();
                                         return !(ev_status == query_status::in_progress);
                                       }),
                        copying_slots.end());
  };

  auto drain_and_submit = [&]() {
    int added          = 0;
    bool wait_for_copy = false;
    while (true) {
      auto slot = slot_pool.try_acquire_token();
      if (!slot) {
        if (inflight == 0 && !copying_slots.empty() && !std::exchange(wait_for_copy, true)) {
          SIRIUS_TRY_AND_LOG_EXCEPTION(
            slots[copying_slots.back().slot_index()].event->synchronize(),
            "uring_reactor: failed to synchronize copy event for slot {}",
            copying_slots.back().slot_index());
          poll_copy_completions();
          continue;
        }
        break;
      }

      auto& s                      = slots[slot.slot_index()];
      chunk_io_request_type_ptr dr = nullptr;
      while (dr == nullptr) {
        if (!_requests.try_dequeue(dr) && inflight == 0) { _requests.wait_dequeue(dr); }
        if (dr && dr->manager->has_error()) {
          // If the request is already in error state, skip it.
          dr.reset(nullptr);
          continue;
        }
        break;
      }
      if (dr == nullptr) {
        break;  // queue empty
      }
      cucascade::cuda::cuda_event* cu_event = nullptr;
      if (dr->needs_event_for_synchronization()) {
        cu_event = std::addressof(per_device_copy_events[dr->cpy_req->device_id][s.slot_index]);
      }

      s.on_request(std::move(dr), std::move(slot), cu_event);

      auto* sqe = ring.get_sqe();
      if (!sqe) {
        incomplete_requests.push_back(s.slot_index);
        break;
      }
      s.register_sqe(sqe);
      ++inflight;
      ++added;
    }
    if (added > 0) { ring.submit(added); }
  };

  auto reap_cqes = [&]() {
    unsigned n = ring.peek_cqe_batch(cqes);
    for (auto* cqe : std::span{cqes.data(), n}) {
      uint64_t raw   = io_uring_cqe_get_data64(cqe);
      int si         = static_cast<int>(raw);
      int bytes_read = cqe->res;
      ring.mark_cqe_seen(cqe);
      --inflight;

      auto& s = slots[si];

      if (bytes_read < 0) {
        s.on_error(std::error_code(-bytes_read, std::generic_category()));
        continue;
      }

      s.bytes_read += static_cast<size_t>(bytes_read);
      bool const fully_read = s.bytes_read >= s.req->chunk.size;
      bool const eof        = (bytes_read == 0);

      if (!fully_read && !eof) {
        incomplete_requests.push_back(si);
        continue;
      }

      if (s.req->cpy_req) {
        // todo(amin): should we even handle cuda errors.
        auto err = s.copy_h2d_async();
        if (err != cudaSuccess) {
          s.on_error(err);
          continue;
        }
        if (s.use_internal_buffer) { copying_slots.push_back(s.release_slot()); }
      }
      s.on_complete(s.bytes_read);
    }
  };

  auto resubmit_incomplete_requests = [&]() {
    size_t any_added = 0;
    while (!incomplete_requests.empty()) {
      int si    = incomplete_requests.back();
      auto& s   = slots[si];
      auto* sqe = ring.get_sqe_with_drain();
      if (!sqe) {
        // This should be very unlikely since we reserved enough SQEs for
        // every slot to be re-submitted once, but if it happens we can just
        // wait for the next CQE batch to drain some SQEs and try again then.
        break;
      }
      incomplete_requests.pop_back();
      s.register_sqe(sqe);
      ++inflight;
      ++any_added;
    }
    if (any_added > 0) { ring.submit(any_added); }
  };

  auto clean_up_and_shutdown = [&]() {
    // wait for all in-flight requests to complete so we don't report spurious errors on shutdown
    while (inflight > 0) {
      auto s = ring.wait_for(SHUTDOWN_POLL_MS);
      if (s) {
        spdlog::error("uring_reactor: io_uring_wait_cqe failed during shutdown: {}", strerror(s));
        break;
      }
      reap_cqes();
    }

    // wait for any in-flight copies to complete so we don't introduce illegal accesses when we
    // release the bounce buffers back to the memory resource
    std::for_each(copying_slots.begin(), copying_slots.end(), [&](auto& s) {
      int si     = s.slot_index();
      auto& slot = slots[si];
      if (slot.event) {
        SIRIUS_TRY_AND_LOG_EXCEPTION(slot.event->synchronize(),
                                     "uring_reactor: failed to synchronize copy event for slot {}",
                                     si);
      }
    });

    // Mark all pending requests as canceled so their managers don't wait indefinitely for
    // completion
    chunk_io_request_type_ptr dr = nullptr;
    while (_requests.try_dequeue(dr)) {
      if (dr) {
        dr->manager->report_error(std::make_error_code(std::errc::operation_canceled));
        dr.reset(nullptr);
      } else {
        break;  // queue empty
      }
    }
  };

  ring.register_buffers(iovecs);

  // The main loop: drain the request queue and submit new SQEs, wait for completions and reap
  {
    auto cleanup = absl::MakeCleanup([&]() { clean_up_and_shutdown(); });

    while (!stop_token.stop_requested()) {
      drain_and_submit();

      if (inflight > 0) {
        auto s = ring.wait_for(SHUTDOWN_POLL_MS);
        if (s) {
          spdlog::error("uring_reactor: io_uring_wait_cqe_timeout failed: {}", strerror(s));
          break;
        }
        reap_cqes();
      }

      resubmit_incomplete_requests();

      poll_copy_completions();
    }
  }
}

}  // namespace sirius::io::uring
