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

#include <cudf/io/datasource.hpp>
#include <cudf/io/text/byte_range_info.hpp>

#include <cuda_runtime.h>

#include <atomic>
#include <cstddef>
#include <exception>
#include <functional>
#include <memory>
#include <string>

namespace sirius::io {

// ---------------------------------------------------------------------------
// Completion handler
// ---------------------------------------------------------------------------

/// Boost.Asio-style completion handler for async I/O.
/// @param bytes_transferred  Total bytes read on success.
/// @param ep                 Non-null on failure.
using io_completion_handler = std::function<void(size_t bytes_transferred, std::exception_ptr ep)>;

// ---------------------------------------------------------------------------
// IO constants
// ---------------------------------------------------------------------------

static constexpr size_t CHUNK_SIZE    = 1UL << 20;  ///< Bounce-buffer chunk size (1 MiB).
static constexpr size_t NUM_CHUNKS    = 128;        ///< Number of bounce slots per reactor.
static constexpr size_t IO_BLOCK_SIZE = 4096;       ///< O_DIRECT alignment requirement (bytes).

// ---------------------------------------------------------------------------
// sirius_io_object
// ---------------------------------------------------------------------------

/**
 * @brief Abstract per-file handle.  A passive bag of native handles
 * produced by a backend reactor (e.g. file descriptors, CURL easy
 * handles, S3 client state).  Performs no I/O of its own.
 *
 * Inherits from @c std::enable_shared_from_this so the prefetching cache can
 * take a reference to an io_object and safely extend its lifetime via
 * @c shared_from_this() — this enforces at call sites that every io_object
 * passed in is already owned by a @c std::shared_ptr.
 */
class sirius_io_object : public std::enable_shared_from_this<sirius_io_object> {
 public:
  virtual ~sirius_io_object() = default;

  /// Stable identifier used as the prefetching-cache key.  Often equal to
  /// @c object_path() but may differ for backends that need to distinguish
  /// otherwise-equal paths (versioned S3 keys, normalized URLs, …).
  [[nodiscard]] virtual const std::string& raw_file_cache_id() const noexcept = 0;

  /// The path / URL / key the caller used to construct this object.
  [[nodiscard]] virtual const std::string& object_path() const noexcept = 0;

  /// Total size of the underlying object, populated by the reactor at
  /// construction time and stored on the io_object thereafter.
  [[nodiscard]] virtual size_t size() const noexcept = 0;
};

class sirius_io_object_metadata {
 public:
  virtual ~sirius_io_object_metadata() = default;
};

// ---------------------------------------------------------------------------
// request_context
// ---------------------------------------------------------------------------

/**
 * @brief Shared completion state for one logical read call (host or device).
 *
 * A single read may be split into multiple sub-requests. All sub-requests
 * decrement @c pending; the last one resolves the promise.
 */
struct request_context {
  io_completion_handler handler;
  std::atomic<size_t> pending{0};
  size_t total_bytes{0};
  std::atomic<bool> failed{false};
  std::exception_ptr exc;

  void chunk_done()
  {
    if (pending.fetch_sub(1, std::memory_order_acq_rel) == 1) {
      if (failed.load(std::memory_order_relaxed)) {
        handler(0, exc);
      } else {
        handler(total_bytes, nullptr);
      }
    }
  }

  void chunk_failed(std::exception_ptr e)
  {
    bool expected = false;
    if (failed.compare_exchange_strong(expected, true, std::memory_order_relaxed)) {
      exc = std::move(e);
    }
    chunk_done();
  }
};

// ---------------------------------------------------------------------------
// device_read_req / host_read_req
// ---------------------------------------------------------------------------

/**
 * @brief Descriptor for one aligned 1 MiB I/O chunk pushed to a reactor for
 *        a device (GPU) read.  Templated on the backend's native handle type
 *        (e.g. @c int for a POSIX file descriptor).
 */
template <typename Handle>
struct device_read_req {
  Handle handle{};
  size_t file_off{0};
  size_t io_size{0};
  size_t data_off{0};
  size_t data_size{0};
  uint8_t* dst{nullptr};
  cudaStream_t stream{nullptr};
  /// CUDA device index that owns @c dst and @c stream.  The reactor thread
  /// may be running with a different current device, so it must
  /// cudaSetDevice(device_id) before issuing the H2D copy in multi-GPU
  /// deployments.  -1 means "don't switch" (single-GPU fast path).
  int device_id{-1};
  std::shared_ptr<request_context> ctx;
};

/**
 * @brief Descriptor for one buffered host read pushed to a reactor.
 *        Templated on the backend's native handle type.
 */
template <typename Handle>
struct host_read_req {
  Handle handle{};
  size_t offset{0};
  size_t size{0};
  uint8_t* dst{nullptr};
  std::shared_ptr<request_context> ctx;
};

}  // namespace sirius::io
