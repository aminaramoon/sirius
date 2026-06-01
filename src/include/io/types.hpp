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

#include <cstddef>
#include <memory>
#include <string>

namespace sirius::io {

static constexpr size_t IO_BLOCK_SIZE = 4096;  // O_DIRECT page size

/**
 * @brief RAII wrapper for a POSIX file descriptor.
 *
 * Non-copyable, movable. Closes the underlying fd on destruction.
 */
struct file_descriptor {
  int fd{-1};
  file_descriptor() = default;
  explicit file_descriptor(int f) noexcept : fd(f) {}
  ~file_descriptor() noexcept
  {
    if (fd >= 0) ::close(fd);
  }
  file_descriptor(file_descriptor const&)            = delete;
  file_descriptor& operator=(file_descriptor const&) = delete;
  file_descriptor(file_descriptor&& o) noexcept : fd(std::exchange(o.fd, -1)) {}
  file_descriptor& operator=(file_descriptor&& o) noexcept
  {
    if (this != &o) {
      if (fd >= 0) ::close(fd);
      fd = std::exchange(o.fd, -1);
    }
    return *this;
  }
  [[nodiscard]] int get() const noexcept { return fd; }
  [[nodiscard]] int native_handle() const noexcept { return fd; }
  explicit operator bool() const noexcept { return fd >= 0; }
};

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

class io_object_segment {
 public:
  io_object_segment() = default;

  io_object_segment(size_t offset, size_t size) : offset(offset), size(size) {}

  io_object_segment(size_t offset, size_t size, void* data, bool is_device_accessible = false)
    : offset(offset),
      size(size),
      _data(static_cast<uint8_t*>(data)),
      _is_device_accessible(is_device_accessible)

  {
  }

  void set_data(uint8_t* data, bool is_device_accessible = false)
  {
    _data                 = data;
    _is_device_accessible = is_device_accessible;
  }

  [[nodiscard]] uint8_t* data() const noexcept { return _data; }

  [[nodiscard]] bool is_device_accessible() const noexcept { return _is_device_accessible; }

  [[nodiscard]] bool is_buffer_allocated() const noexcept { return _data != nullptr; }

  [[nodiscard]] bool is_odirect_compatible() const noexcept
  {
    return (offset % IO_BLOCK_SIZE == 0) && (size % IO_BLOCK_SIZE == 0) &&
           (_data == nullptr || reinterpret_cast<uintptr_t>(_data) % IO_BLOCK_SIZE == 0);
  }

  size_t offset{0};
  size_t size{0};

 private:
  uint8_t* _data{nullptr};  // null means the destination buffer is not yet allocated; non-null
                            // means the caller has pre-allocated a buffer for this segment, doesn't
                            // mean the buffer is populated with data yet
  bool _is_device_accessible{
    false};  // true if the caller intends to read directly into device memory
};

}  // namespace sirius::io
