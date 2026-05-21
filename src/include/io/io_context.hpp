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

#include "io/metadata_store.hpp"
#include "io/prefetch_types.hpp"
#include "io/types.hpp"

#include <rmm/cuda_stream_view.hpp>

#include <cstddef>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace sirius::io {

class prefetching_cache;

// ---------------------------------------------------------------------------
// prefetching_mode
// ---------------------------------------------------------------------------

/**
 * @brief How the prefetching layer should behave on top of a given backend.
 *
 * - @c none: no prefetching.  Either the backend does not support vector host
 *   reads (so the prefetcher cannot batch range requests cheaply) or the
 *   backend explicitly opted out.
 * - @c immediate: prefill the cache ahead of consumer demand.
 * - @c opportunistic: read-ahead on demand — issue extra IO only when triggered by a
 *   consumer read.
 * - @c disposable: prefetching is temporary and can be discarded when no longer needed.
 */
enum class prefetching_mode { none, immediate, opportunistic, disposable };

// ---------------------------------------------------------------------------
// sirius_ioctx
// ---------------------------------------------------------------------------

/**
 * @brief Abstract shared context passed to every datasource.
 *
 * Holds resources that are shared across all datasources (cache, reactor
 * threads, ...). Extend this class to provide a concrete I/O backend.
 */
class sirius_ioctx : public std::enable_shared_from_this<sirius_ioctx> {
  // prefetching_cache's worker_loop is the only caller of the protected
  // host_read_ranges_async_io entry point through a sirius_ioctx* base
  // pointer.  Friending the cache lets that single call site reach in
  // without forcing the vector-read primitive (which most callers never
  // touch) into the public API.
  friend class prefetching_cache;

 public:
  sirius_ioctx();
  virtual ~sirius_ioctx();

  virtual void shutdown() = 0;

  /// Backend-specific factory: open native handles for @p path and
  /// return a populated io_object.  Throws on unsupported / unreachable
  /// paths (callers that want a check-without-open should use
  /// @c supports()).
  virtual std::shared_ptr<sirius_io_object> create_io_object(std::string path) = 0;

  virtual std::unique_ptr<cudf::io::datasource> make_datasource(
    std::shared_ptr<sirius_io_object> io_object) = 0;

  /// Convenience: @c create_io_object + @c make_datasource in one shot.
  std::unique_ptr<cudf::io::datasource> open_datasource(std::string path)
  {
    return make_datasource(create_io_object(std::move(path)));
  }

  /// Whether this backend can serve reads for @p path.  Backends should
  /// validate scheme/protocol support and any backend-specific
  /// preconditions (e.g. file existence for local-disk backends).
  [[nodiscard]] virtual bool supports(std::string_view path) const = 0;

  // -- Backend capabilities ---------------------------------------------------

  /// Whether the backend can stream data directly into device memory
  /// (e.g. via O_DIRECT + GDS).  Used by @c sirius_datasource to answer the
  /// equivalent cudf::io::datasource queries.
  [[nodiscard]] virtual bool supports_device_read() const = 0;

  /// Whether the backend can serve a batch of host reads in a single dispatch
  /// (cf. @c host_read_ranges_async_io).  When false, the prefetching layer
  /// cannot amortise per-request overhead and must fall back to
  /// @c prefetching_mode::none.
  [[nodiscard]] virtual bool supports_vector_host_read() const = 0;

  /// Prefetching strategy the prefetching layer should use against this
  /// backend.  Returns @c prefetching_mode::none whenever
  /// @c supports_vector_host_read() is false; otherwise the backend picks
  /// between eager prefill and on-demand read-ahead based on its IO
  /// characteristics.
  [[nodiscard]] virtual prefetching_mode preferred_prefetching_mode() const = 0;

  /// Build the prefetching cache.  One-shot — calling twice is a no-op
  /// after the first successful build.  The cache holds a raw
  /// back-pointer to this ioctx and stays alive until @ref
  /// shutdown_cache is called (or this ioctx is destroyed).  @p pool
  /// is non-owning; the caller (typically @c scan_manager) guarantees
  /// it outlives the cache.
  ///
  /// The cache constructs itself in an "armed" or "unarmed" state
  /// depending on @p pool and @c supports_vector_host_read(); the
  /// ioctx is unaware of that distinction — it simply forwards lookups
  /// through @c cache().
  void initialize_cache(buffer_pool* pool, size_t inflight_budget_chunks);

  /// Tear down the cache (drains background workers and any in-flight
  /// IO via @c admission_control).  Idempotent.  The owner (scan
  /// manager) calls this BEFORE releasing the @c buffer_pool the cache
  /// was constructed with — otherwise workers may issue final IO
  /// against a destroyed pool.
  void shutdown_cache() noexcept;

  /// Every concrete derived class MUST call this as the very first
  /// statement in its destructor.  It drains the cache (so its workers
  /// stop issuing IO) while the derived object's reactors / handles
  /// are still alive.  Without this, the cache's defensive shutdown
  /// in @c ~sirius_ioctx would run AFTER the derived part of the
  /// object has been destroyed, and worker callbacks would reach
  /// already-destroyed reactors.
  ///
  /// Idempotent — calling @c shutdown_cache directly before this is
  /// fine.  Cheap when no cache was ever initialised.
  void pre_destroy() noexcept { shutdown_cache(); }

  [[nodiscard]] prefetching_cache* cache() noexcept { return _cache.get(); }

  /// True iff @c host_read / @c device_read should consult the cache
  /// before falling through to the backend.  Computed live so it tracks
  /// @ref initialize_cache / @ref shutdown_cache transitions.
  [[nodiscard]] bool uses_prefetching_cache() const noexcept
  {
    return _cache != nullptr && supports_vector_host_read();
  }

  /// Per-file metadata cache that lives independently of the prefetching
  /// cache.  Always available — callers that have parsed file metadata
  /// (e.g. a parquet footer) park it here so a later scan of the same
  /// path can skip the parse without depending on whether the
  /// prefetching machinery has been wired up.
  [[nodiscard]] metadata_store& metadata() noexcept { return _metadata_store; }
  [[nodiscard]] metadata_store const& metadata() const noexcept { return _metadata_store; }

  // -- Read API ---------------------------------------------------------------

  virtual size_t host_read(sirius_io_object& obj, size_t offset, size_t size, uint8_t* dst);

  virtual std::future<size_t> host_read_async(sirius_io_object& obj,
                                              size_t offset,
                                              size_t size,
                                              uint8_t* dst);

  virtual size_t device_read(
    sirius_io_object& obj, size_t offset, size_t size, uint8_t* dst, rmm::cuda_stream_view stream);

  virtual std::future<size_t> device_read_async(
    sirius_io_object& obj, size_t offset, size_t size, uint8_t* dst, rmm::cuda_stream_view stream);

  /// Issue a device read reusing the pre-allocated bounce buffers in @p buffer.
  /// @p buffer must be in the @c loading state (caller already stole it from
  /// the prefetching cache via @c prefetching_cache::read()).  On completion
  /// the handler fires and the entry transitions to @c cached or @c empty.
  /// Default implementation throws; only meaningful for backends that support
  /// vector host reads (i.e. when @c supports_vector_host_read() is true).
  virtual void device_read_async_io_using(sirius_io_object& obj,
                                          size_t offset,
                                          size_t size,
                                          uint8_t* dst,
                                          rmm::cuda_stream_view stream,
                                          cached_host_buffer buffer,
                                          io_completion_handler handler);

  // -- Physical range alignment ------------------------------------------------

  virtual cudf::io::text::byte_range_info compute_physical_range(
    cudf::io::text::byte_range_info logical, size_t file_size) const = 0;

  ///

 protected:
  virtual size_t host_read_io(sirius_io_object& obj, size_t offset, size_t size, uint8_t* dst) = 0;

  virtual void host_read_async_io(sirius_io_object& obj,
                                  size_t offset,
                                  size_t size,
                                  uint8_t* dst,
                                  io_completion_handler handler) = 0;

  virtual size_t device_read_io(sirius_io_object& obj,
                                size_t offset,
                                size_t size,
                                uint8_t* dst,
                                rmm::cuda_stream_view stream) = 0;

  virtual void device_read_async_io(sirius_io_object& obj,
                                    size_t offset,
                                    size_t size,
                                    uint8_t* dst,
                                    rmm::cuda_stream_view stream,
                                    io_completion_handler handler) = 0;

  virtual void host_read_ranges_async_io(sirius_io_object& obj,
                                         std::vector<cudf::io::text::byte_range_info> const& ranges,
                                         std::span<cudf::host_span<std::byte>> dst,
                                         io_completion_handler handler) = 0;

 protected:
  /// Owned by this ioctx.  Built by @ref initialize_cache, destroyed
  /// by @ref shutdown_cache (or the ioctx destructor as a safety net,
  /// though callers are expected to drive the lifecycle explicitly so
  /// reactors stay alive while workers drain).
  std::unique_ptr<prefetching_cache> _cache;

  /// Independent of the prefetching machinery — exposed via @c metadata().
  metadata_store _metadata_store;
};

}  // namespace sirius::io
