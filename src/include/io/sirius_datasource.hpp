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
#include "io/prefetching_cache.hpp"

#include <cudf/io/datasource.hpp>
#include <cudf/io/text/byte_range_info.hpp>

#include <span>

namespace sirius::io {

// ---------------------------------------------------------------------------
// sirius_datasource
// ---------------------------------------------------------------------------

/**
 * @brief Concrete @c io_datasource backed by io_uring.
 *
 * Thin delegate: every read method forwards to @c sirius_ioctx, passing the
 * owned @c sirius_io_object by reference.
 *
 * Ownership model: one scan owns one @c sirius_datasource.  The underlying
 * @c sirius_io_object can be shared across multiple datasources (e.g. when
 * the same file is scanned in different pipelines), but the datasource
 * itself stores per-scan state (notably the @c prefetching_handle returned
 * by an @c fadvise call) and is therefore not safe to share.
 */
class sirius_datasource : public cudf::io::datasource {
 public:
  static constexpr size_t NUM_BUFFERS = NUM_CHUNKS;
  static constexpr size_t BUFFER_SIZE = CHUNK_SIZE;

  explicit sirius_datasource(std::shared_ptr<sirius_ioctx> io_ctx,
                             std::shared_ptr<sirius_io_object> io_object);

  ~sirius_datasource() override;

  sirius_datasource(sirius_datasource const&)            = delete;
  sirius_datasource& operator=(sirius_datasource const&) = delete;

  // ---- Context accessors ---------------------------------------------------

  [[nodiscard]] std::shared_ptr<sirius_ioctx> io_ctx() const { return _io_ctx; }

  // ---- cudf::io::datasource overrides ---------------------------------------

  [[nodiscard]] size_t size() const override;

  [[nodiscard]] bool supports_device_read() const override;

  [[nodiscard]] bool supports_vector_host_read() const;

  [[nodiscard]] bool is_device_read_preferred(size_t) const override;

  size_t host_read(size_t offset, size_t size, uint8_t* dst) override;

  std::unique_ptr<datasource::buffer> host_read(size_t offset, size_t size) override;

  std::future<size_t> host_read_async(size_t offset, size_t size, uint8_t* dst) override;

  std::future<std::unique_ptr<datasource::buffer>> host_read_async(size_t offset,
                                                                   size_t size) override;

  std::unique_ptr<datasource::buffer> device_read(size_t offset,
                                                  size_t size,
                                                  rmm::cuda_stream_view stream) override;
  size_t device_read(size_t offset,
                     size_t size,
                     uint8_t* dst,
                     rmm::cuda_stream_view stream) override;

  std::future<size_t> device_read_async(size_t offset,
                                        size_t size,
                                        uint8_t* dst,
                                        rmm::cuda_stream_view stream) override;

  // ---- Advisory IO ---------------------------------------------------------

  /// Hint the IO layer about @p ranges that this scan will (or might) read
  /// soon.
  ///
  /// The behaviour depends on @p site and the io_ctx's
  /// @c preferred_prefetching_mode:
  ///   - @c speculative / @c immediate: only honored when @p site matches
  ///     the ioctx's preferred mode.  Hands @p ranges to the prefetching
  ///     cache and stashes the returned @c prefetching_handle on this
  ///     datasource so a later @c fadvise(disposable) can cancel.
  ///   - @c disposable: always honored.  If a handle is stored (i.e. a
  ///     prior speculative/immediate call enqueued work), cancel it so the
  ///     cache worker drops still-pending entries.
  ///   - @c none: no-op (either the call site asked for nothing, or the
  ///     backend opted out of prefetching).
  ///
  /// Calling @c fadvise with a non-disposable @p site while a handle is
  /// already stored emits a warning: the datasource lifecycle expects one
  /// speculative-or-immediate insert per scan, with a single
  /// @c disposable call at consume time.
  void fadvise(prefetching_mode site,
               std::span<const cudf::io::text::byte_range_info> ranges);

 private:
  std::shared_ptr<sirius_ioctx> _io_ctx;
  std::shared_ptr<sirius_io_object> _io_object;
  /// Handle of the most recent speculative/immediate insert into the
  /// prefetching cache, or empty if none was made.  fadvise(disposable)
  /// uses this to cancel still-pending work.
  prefetching_handle _prefetch_handle;
};

}  // namespace sirius::io
