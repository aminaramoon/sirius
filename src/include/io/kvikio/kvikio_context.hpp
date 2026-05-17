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

#include "io/io_context.hpp"

#include <cudf/io/datasource.hpp>

#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace sirius::io {

// ---------------------------------------------------------------------------
// kvikio_io_object
// ---------------------------------------------------------------------------

/**
 * @brief @c sirius_io_object that holds a cudf::io::datasource (kvikio-backed
 *        on the default cudf build).
 *
 * Owns the datasource for the file's lifetime; @c kvikio_context's read
 * overrides forward straight to this datasource so we don't have to
 * translate cudf's future-returning API into the push/callback shape that
 * the base class's protected @c _io primitives expect.
 */
class kvikio_io_object final : public sirius_io_object {
 public:
  kvikio_io_object(std::string path, std::shared_ptr<cudf::io::datasource> ds, size_t file_size)
    : _path(std::move(path)), _datasource(std::move(ds)), _file_size(file_size)
  {
  }

  [[nodiscard]] const std::string& raw_file_cache_id() const noexcept final { return _path; }
  [[nodiscard]] const std::string& object_path() const noexcept final { return _path; }
  [[nodiscard]] size_t size() const noexcept final { return _file_size; }

  [[nodiscard]] cudf::io::datasource& datasource() const noexcept { return *_datasource; }

 private:
  std::string _path;
  std::shared_ptr<cudf::io::datasource> _datasource;
  size_t _file_size{0};
};

// ---------------------------------------------------------------------------
// kvikio_context
// ---------------------------------------------------------------------------

/**
 * @brief Fallback @c sirius_ioctx that defers to cudf's default datasource
 *        (kvikio-backed for file paths on a stock cudf build).
 *
 * Why override the public read API directly instead of the protected
 * @c _io primitives?  cudf's async path returns @c std::future<size_t>,
 * but the protected @c host_read_async_io / @c device_read_async_io
 * contract is push/callback (@c io_completion_handler).  Converting one
 * to the other requires either spawning a thread to wait on the future
 * or blocking the caller — both wrong.  Instead we override the public
 * read APIs so the future flows through unchanged; the protected @c _io
 * primitives become unreachable placeholders.
 *
 * Capabilities:
 *   - @c supports_device_read: true (cudf's datasource supports it where the
 *     platform allows, e.g. GDS).
 *   - @c supports_vector_host_read: false — no batched dispatch path.
 *   - @c preferred_prefetching_mode: @c none.
 */
class kvikio_context final : public sirius_ioctx {
 public:
  kvikio_context()           = default;
  ~kvikio_context() override = default;

  void shutdown() override {}

  std::shared_ptr<sirius_io_object> create_io_object(std::string path) override;

  std::unique_ptr<cudf::io::datasource> make_datasource(
    std::shared_ptr<sirius_io_object> io_object) override;

  [[nodiscard]] bool supports(std::string_view path) const override;

  [[nodiscard]] bool supports_device_read() const override { return true; }
  [[nodiscard]] bool supports_vector_host_read() const override { return false; }
  [[nodiscard]] prefetching_mode preferred_prefetching_mode() const override
  {
    return prefetching_mode::none;
  }

  // -- Public read API (override the base virtuals directly) ----------------
  //
  // Each forwards to the cudf datasource held on the io_object.  The cache
  // pointer wired via @c attach_cache is intentionally NOT consulted here:
  // kvikio_context reports @c supports_vector_host_read == false, so the
  // scan_manager won't ever attach a cache to a kvikio_context anyway.

  size_t host_read(sirius_io_object& obj, size_t offset, size_t size, uint8_t* dst) final;

  std::future<size_t> host_read_async(sirius_io_object& obj,
                                      size_t offset,
                                      size_t size,
                                      uint8_t* dst) final;

  size_t device_read(sirius_io_object& obj,
                     size_t offset,
                     size_t size,
                     uint8_t* dst,
                     rmm::cuda_stream_view stream) final;

  std::future<size_t> device_read_async(sirius_io_object& obj,
                                        size_t offset,
                                        size_t size,
                                        uint8_t* dst,
                                        rmm::cuda_stream_view stream) final;

  /// kvikio handles its own alignment internally; pass the logical range
  /// through unchanged.
  cudf::io::text::byte_range_info compute_physical_range(cudf::io::text::byte_range_info logical,
                                                         size_t file_size) const final;

 protected:
  // -- Protected _io primitives -------------------------------------------
  //
  // The base class's default read implementations route through these on a
  // cache miss, but kvikio_context overrides the public read API and never
  // attaches a cache (supports_vector_host_read == false), so these are
  // unreachable from the documented code paths.  They remain pure-virtual
  // on the base, so we provide throwing placeholders to keep the class
  // instantiable; any future caller that bypasses the public API will see
  // a clear failure rather than silent misbehaviour.

  size_t host_read_io(sirius_io_object& obj, size_t offset, size_t size, uint8_t* dst) final;

  void host_read_async_io(sirius_io_object& obj,
                          size_t offset,
                          size_t size,
                          uint8_t* dst,
                          io_completion_handler handler) final;

  size_t device_read_io(sirius_io_object& obj,
                        size_t offset,
                        size_t size,
                        uint8_t* dst,
                        rmm::cuda_stream_view stream) final;

  void device_read_async_io(sirius_io_object& obj,
                            size_t offset,
                            size_t size,
                            uint8_t* dst,
                            rmm::cuda_stream_view stream,
                            io_completion_handler handler) final;

  void host_read_ranges_async_io(sirius_io_object& obj,
                                 std::vector<cudf::io::text::byte_range_info> const& ranges,
                                 std::span<cudf::host_span<std::byte>> dst,
                                 io_completion_handler handler) final;
};

}  // namespace sirius::io
