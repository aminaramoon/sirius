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

#pragma once

#include "exec/semi_future.hpp"
#include "io/cache/types.hpp"
#include "io/details/slot_pool.hpp"
#include "io/types.hpp"
#include "io/uring/types.hpp"

#include <cuda_runtime.h>

#include <blockingconcurrentqueue.h>
#include <concurrentqueue.h>
#include <cucascade/memory/fixed_size_host_memory_resource.hpp>
#include <liburing.h>

#include <array>
#include <atomic>
#include <memory>
#include <span>
#include <stop_token>
#include <string>
#include <string_view>
#include <thread>

namespace {
inline void cuda_check(cudaError_t e, char const* file, int line)
{
  if (e != cudaSuccess)
    throw std::runtime_error(std::string("CUDA error ") + file + ":" + std::to_string(line) +
                             " – " + cudaGetErrorString(e));
}
}  // namespace

#define CUDA_CHECK(call) cuda_check((call), __FILE__, __LINE__)

namespace sirius::io::uring {

// ---- bounce_slot -----------------------------------------------------------

/**
 * @brief One pinned-memory staging buffer.
 *
 * The buffer is a non-owning pointer into a block owned by the reactor's
 * @c fixed_size_host_memory_resource allocation — the resource frees the
 * memory when the reactor is destroyed.  Used only by managed device reads;
 * host and BYO device reads supply their own destination buffer.
 */
struct bounce_slot {
  void* buf{nullptr};
};

// ---------------------------------------------------------------------------
// local_io_object
// ---------------------------------------------------------------------------

/**
 * @brief Concrete @c sirius_io_object backed by a filesystem path.
 *
 * Passive bag of native handles.  The buffered @c O_RDONLY fd
 * (for @c pread / host_read) and the @c O_DIRECT fd (for reactor-driven
 * device reads) are produced by @c uring_reactor::create_io_object — this
 * class does no I/O of its own.
 */
class local_io_object : public sirius_io_object {
 public:
  local_io_object(std::string path, file_descriptor fd, file_descriptor fd_direct, size_t file_size)
    : _path(std::move(path)),
      _fd(std::move(fd)),
      _fd_direct(std::move(fd_direct)),
      _file_size(file_size)
  {
  }

  [[nodiscard]] const std::string& raw_file_cache_id() const noexcept override { return _path; }
  [[nodiscard]] const std::string& object_path() const noexcept override { return _path; }
  [[nodiscard]] size_t size() const noexcept override { return _file_size; }

  [[nodiscard]] int fd() const noexcept { return _fd.get(); }
  [[nodiscard]] int fd_direct() const noexcept { return _fd_direct.get(); }

  // ---- templated_ioctx / io_object_c requirements -----------------------
  [[nodiscard]] int buffered_handle() const noexcept { return _fd.get(); }
  [[nodiscard]] int odirect_handle() const noexcept { return _fd_direct.get(); }

 private:
  std::string _path;
  file_descriptor _fd;
  file_descriptor _fd_direct;
  size_t _file_size{0};
};

// ---------------------------------------------------------------------------
// uring_reactor
// ---------------------------------------------------------------------------

/**
 * @brief Single-threaded I/O reactor for O_DIRECT device reads.
 *
 * Owns one @c io_uring (O_DIRECT), one worker thread, @c NUM_CHUNKS pinned
 * bounce slots, and an MPSC request queue.  Models the reactor concept
 * consumed by @c templated_ioctx.
 */
class uring_reactor {
 public:
  struct config {
    std::size_t bounce_size{1UL << 20};
    /// When false, every prep path except the BYO-device-buffer read
    /// (prep_device_rx_request) reads through the buffered (page-cache) file
    /// handle instead of the O_DIRECT one.  Defaults to O_DIRECT.
    bool use_odirect{true};
  };

  using native_handle_type        = int;
  using io_object_type            = local_io_object;
  using request_type              = rx_request;
  using request_type_ptr          = std::unique_ptr<rx_request>;
  using chunk_io_request_type     = chunked_rx_request;
  using chunk_io_request_type_ptr = std::unique_ptr<chunked_rx_request>;
  using reactor_config_type       = config;

  /// Bounce slots are allocated from @p mr; their size is taken from
  /// @c mr.get_block_size().  The reactor keeps the @c multiple_blocks_allocation
  /// alive for its lifetime — blocks return to the resource on destruction.
  explicit uring_reactor(cucascade::memory::fixed_size_host_memory_resource& mr,
                         std::string_view tname = "uring_reactor");

  ~uring_reactor();

  uring_reactor(uring_reactor const&)            = delete;
  uring_reactor& operator=(uring_reactor const&) = delete;

  static request_type_ptr prep_host_rx_request(const reactor_config_type& cfg,
                                               const io_object_type& file,
                                               const io_object_segment& segment);

  static request_type_ptr prep_device_rx_request(const reactor_config_type& cfg,
                                                 const io_object_type& file,
                                                 std::byte* dst,
                                                 size_t offset,
                                                 size_t size,
                                                 rmm::cuda_stream_view stream,
                                                 int device_id);

  static request_type_ptr prep_host_to_device_rx_request(const reactor_config_type& cfg,
                                                         const io_object_type& file,
                                                         std::span<io_object_segment> bounce,
                                                         std::byte* dst,
                                                         size_t offset,
                                                         size_t size,
                                                         rmm::cuda_stream_view stream,
                                                         int device_id);

  static request_type_ptr prep_host_rxv_request(const reactor_config_type& cfg,
                                                const io_object_type& file,
                                                std::span<io_object_segment> segments);

  void interrupt();
  void shutdown();

  /// Synchronous buffered host read (pread on @p fd).  Blocks the caller.
  size_t host_read(const io_object_type& file, size_t offset, size_t size, std::byte* dst);

  void enqueue(request_type_ptr req);

  /// Whether @p path can be served by this reactor.  Local-disk only:
  /// returns true iff the path refers to an existing, accessible file.
  [[nodiscard]] static bool supports(std::string_view path);

  /// Open the buffered + O_DIRECT fds for @p path and return them
  /// packaged in a @c local_io_object.  Throws on unsupported paths or
  /// open() failure.
  static std::unique_ptr<io_object_type> create_io_object(std::string path);

  /// fstat the open fd to get the file's current size.
  static size_t size(int native_handle);

  // ---- Backend capabilities --------------------------------------------
  //
  // io_uring + O_DIRECT serves device reads through the reactor's bounce
  // slots, supports batched host reads via @c host_enqueue_bulk, and pairs
  // well with eager prefill since local-disk latencies are low and we want
  // to keep the device fed.
  //
  // supports_device_read / supports_vector_host_read are no longer declared
  // here: they are derived structurally by reactor_traits from the presence of
  // the corresponding create_*_rx_request overloads (both of which this
  // reactor provides).

  static constexpr cache::prefetching_mode preferred_prefetching_mode() noexcept
  {
    return cache::prefetching_mode::immediate;
  }

 private:
  /// Enqueue a whole batch of device read chunks with a single wake
  /// notification.  Preferred over single-op enqueues when a caller
  /// produces several chunks destined for this reactor — amortises the
  /// wait-atomic notify and uses moodycamel's enqueue_bulk path.
  /// The span's elements are moved out; the caller's backing storage
  /// must outlive this call but the contents are left in moved-from state.
  void enqueue_chunks(std::span<chunk_io_request_type_ptr> batch);

  /// Enqueue a whole batch of device read chunks with a single wake
  /// notification.  Preferred over single-op enqueues when a caller
  /// produces several chunks destined for this reactor — amortises the
  /// wait-atomic notify and uses moodycamel's enqueue_bulk path.
  /// The span's elements are moved out; the caller's backing storage
  /// must outlive this call but the contents are left in moved-from state.
  void enqueue_chunk(chunk_io_request_type_ptr request);

 public:
  /// O_DIRECT requires 4 KiB alignment of both file offset and length.
  static cudf::io::text::byte_range_info align_to_physical(cudf::io::text::byte_range_info logical,
                                                           size_t file_size);

 private:
  void worker_loop(const std::stop_token& stop_token);

  // Keeps the bounce-slot blocks alive for the reactor's lifetime.  The
  // multiple_blocks_allocation destructor returns the blocks to the upstream
  // resource when the reactor is destroyed.
  reactor_config_type _config;
  cucascade::memory::fixed_multiple_blocks_allocation _bounce_storage;
  std::size_t _bounce_slot_size;
  std::stop_source _stop_source;
  std::jthread _worker;
  duckdb_moodycamel::BlockingConcurrentQueue<chunk_io_request_type_ptr> _requests;
};

}  // namespace sirius::io::uring
