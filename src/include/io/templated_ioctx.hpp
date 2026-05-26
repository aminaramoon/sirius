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
#include "io/io_context.hpp"
#include "io/prefetching_cache.hpp"
#include "io/sirius_datasource.hpp"

#include <rmm/device_buffer.hpp>

#include <algorithm>
#include <atomic>
#include <cassert>
#include <concepts>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string_view>
#include <utility>
#include <vector>

namespace sirius::io {

namespace detail {

/// Wrap cudaGetDevice with an exception on failure.  Without the check a
/// failure leaves the out-param at -1, the reactor's `if (device_id >= 0)
/// cudaSetDevice` skips, and on multi-GPU the H2D copy lands on whichever
/// device the reactor thread happens to have current — wrong-device data
/// corruption or a downstream cudaMemcpyAsync error on a different code
/// path.  Throwing here surfaces the original error at the dispatch site.
[[nodiscard]] inline int current_cuda_device()
{
  int id          = -1;
  cudaError_t err = cudaGetDevice(&id);
  if (err != cudaSuccess) {
    cudaGetLastError();
    throw std::runtime_error(std::string("templated_ioctx: cudaGetDevice failed: ") +
                             cudaGetErrorString(err));
  }
  return id;
}

}  // namespace detail

// ---------------------------------------------------------------------------
// Concept: io_object_c
// ---------------------------------------------------------------------------
//
// A concrete io_object for use with templated_ioctx<Reactor> must:
//   - derive from sirius_io_object
//   - expose device_handle() / host_handle() returning Handle

template <class O, class Handle>
concept io_object_c = std::derived_from<O, sirius_io_object> && requires(O o) {
  { o.device_handle() } -> std::convertible_to<Handle>;
  { o.host_handle() } -> std::convertible_to<Handle>;
};

// ---------------------------------------------------------------------------
// Concept: io_reactor_c
// ---------------------------------------------------------------------------

template <class R>
concept io_reactor_c = requires(R r,
                                std::span<typename R::device_read_req_type> dbatch,
                                std::span<typename R::host_read_req_type> hbatch,
                                typename R::host_read_req_type hreq,
                                typename R::native_handle_type handle,
                                size_t offset,
                                size_t size,
                                uint8_t* dst,
                                cudf::io::text::byte_range_info logical,
                                size_t file_size,
                                std::string_view path,
                                std::string path_str) {
  typename R::native_handle_type;
  typename R::io_object_type;
  typename R::device_read_req_type;
  typename R::host_read_req_type;

  requires io_object_c<typename R::io_object_type, typename R::native_handle_type>;

  { r.enqueue_bulk(dbatch) };
  { r.host_read(handle, offset, size, dst) } -> std::same_as<size_t>;
  { r.host_read_async(std::move(hreq)) };
  { r.host_enqueue_bulk(hbatch) };
  { r.shutdown() };
  { R::align_to_physical(logical, file_size) } -> std::same_as<cudf::io::text::byte_range_info>;
  { R::supports(path) } -> std::same_as<bool>;
  {
    R::create_io_object(std::move(path_str))
  } -> std::same_as<std::unique_ptr<typename R::io_object_type>>;
  { R::size(handle) } -> std::same_as<size_t>;

  // -- capabilities --------------------------------------------------------
  { R::supports_device_read() } -> std::same_as<bool>;
  { R::supports_vector_host_read() } -> std::same_as<bool>;
  { R::preferred_prefetching_mode() } -> std::same_as<prefetching_mode>;
};

// ---------------------------------------------------------------------------
// templated_ioctx
// ---------------------------------------------------------------------------
//
// Generic ioctx implementation parameterised on a backend Reactor.  Owns a
// pool of reactors, the cache (via sirius_ioctx base), and implements all of
// the sirius_ioctx read API using only the operations named by io_reactor_c
// and io_object_c.  Backends plug in by providing a Reactor + io_object type
// pair that satisfies those concepts.

template <io_reactor_c Reactor>
class templated_ioctx : public sirius_ioctx {
 public:
  using reactor_type         = Reactor;
  using native_handle_type   = typename Reactor::native_handle_type;
  using io_object_type       = typename Reactor::io_object_type;
  using device_read_req_type = typename Reactor::device_read_req_type;
  using host_read_req_type   = typename Reactor::host_read_req_type;

  /// Construct with a pre-built vector of reactors (most flexible).
  explicit templated_ioctx(std::vector<std::unique_ptr<Reactor>> reactors)
    : _reactors(std::move(reactors))
  {
  }

  /// Construct by invoking @p factory @p n_reactors times to build the pool.
  template <class Factory>
    requires std::invocable<Factory&> &&
             std::convertible_to<std::invoke_result_t<Factory&>, std::unique_ptr<Reactor>>
  templated_ioctx(size_t n_reactors, Factory factory)
  {
    _reactors.reserve(n_reactors);
    for (size_t i = 0; i < n_reactors; ++i)
      _reactors.emplace_back(factory());
  }

  ~templated_ioctx() override
  {
    // pre_destroy() drains the cache (if any) so its workers stop
    // issuing IO BEFORE we tear down the reactors below.  Must be the
    // first statement in every derived dtor — see sirius_ioctx for
    // the full contract.
    this->pre_destroy();
    shutdown();
  }

  void shutdown() override
  {
    for (auto& r : _reactors) {
      r->shutdown();
    }
  }

  std::unique_ptr<cudf::io::datasource> make_datasource(
    std::shared_ptr<sirius_io_object> io_object) override
  {
    return std::make_unique<sirius_datasource>(shared_from_this(), std::move(io_object));
  }

  std::shared_ptr<sirius_io_object> create_io_object(std::string path) override
  {
    return std::shared_ptr<sirius_io_object>(Reactor::create_io_object(std::move(path)));
  }

  [[nodiscard]] bool supports(std::string_view path) const override
  {
    return Reactor::supports(path);
  }

  // -- Capabilities (delegate to the reactor) -------------------------------

  [[nodiscard]] bool supports_device_read() const override
  {
    return Reactor::supports_device_read();
  }

  [[nodiscard]] bool supports_vector_host_read() const override
  {
    return Reactor::supports_vector_host_read();
  }

  [[nodiscard]] prefetching_mode preferred_prefetching_mode() const override
  {
    // Vector host read is the cheap dispatch path the prefetcher relies on.
    // Without it, force prefetching off regardless of the reactor's preference.
    if (!Reactor::supports_vector_host_read()) return prefetching_mode::none;
    return Reactor::preferred_prefetching_mode();
  }

  Reactor& next_reactor()
  {
    size_t idx = _next.fetch_add(1, std::memory_order_relaxed) % _reactors.size();
    return *_reactors.at(idx);
  }

  // -- Host reads (generic: delegate to io_object_type + std::async) --------

  size_t host_read_io(sirius_io_object& obj, size_t offset, size_t size, uint8_t* dst) override
  {
    auto& tobj = as_typed(obj);
    size       = std::min(size, tobj.size() > offset ? tobj.size() - offset : size_t{0});
    if (size == 0) return 0;
    return next_reactor().host_read(tobj.host_handle(), offset, size, dst);
  }

  exec::semi_future<size_t> host_read_async_io(sirius_io_object& obj,
                                               size_t offset,
                                               size_t size,
                                               uint8_t* dst) override
  {
    auto& tobj = as_typed(obj);
    size       = std::min(size, tobj.size() > offset ? tobj.size() - offset : size_t{0});

    // size==0 case is folded into create(): it returns a null ctx and
    // a ready future with 0 bytes.
    auto [ctx, fut] = request_context::create(size == 0 ? 0 : 1, size);
    if (!ctx) return std::move(fut);

    host_read_req_type req;
    req.handle = tobj.host_handle();
    req.offset = offset;
    req.size   = size;
    req.dst    = dst;
    req.ctx    = std::move(ctx);
    next_reactor().host_read_async(std::move(req));
    return std::move(fut);
  }

  // -- Device reads (generic chunking; reactor-backed) ----------------------

  size_t device_read_io(sirius_io_object& obj,
                        size_t offset,
                        size_t size,
                        uint8_t* dst,
                        rmm::cuda_stream_view stream) override
  {
    return std::move(enqueue_device_read(as_typed(obj), offset, size, dst, stream.value())).get();
  }

  exec::semi_future<size_t> device_read_async_io(sirius_io_object& obj,
                                                 size_t offset,
                                                 size_t size,
                                                 uint8_t* dst,
                                                 rmm::cuda_stream_view stream) override
  {
    return enqueue_device_read(as_typed(obj), offset, size, dst, stream.value());
  }

  // -- Device read using pre-allocated bounce buffers -----------------------

  /// Issue an async device read for the range [offset, offset+size) into
  /// @p dst, reusing the pinned host chunks already allocated in
  /// @p buffer.  @p buffer MUST be in the @c loading state (the caller
  /// already flipped it from @c allocated via
  /// @c prefetching_cache::read()).
  ///
  /// On IO completion the reactor fires @p handler.  The wrapped
  /// completion callback also calls @p buffer.mark_cached() on success or
  /// @p buffer.mark_load_failed() on failure, which transitions the entry
  /// to its terminal state and wakes any parked readers.
  ///
  /// @p offset must be **file-absolute** (same coordinate space as the
  /// entry's @c physical_range.offset()); see
  /// @c cached_host_buffer::prepare_device_requests for the full contract.
  exec::semi_future<size_t> device_read_async_io_using(sirius_io_object& obj,
                                                       size_t offset,
                                                       size_t size,
                                                       uint8_t* dst,
                                                       rmm::cuda_stream_view stream,
                                                       cached_host_buffer buffer) override
  {
    auto& tobj = as_typed(obj);

    // Capture the caller's device before we move `buffer` into shared
    // ownership: a throw here unwinds the stack with `buffer` still
    // local, so its dtor's mark_load_failed deallocates the entry's
    // chunks and resets state instead of stranding them.
    int const device_id = detail::current_cuda_device();

    auto reqs = buffer.template prepare_device_requests<native_handle_type>(
      tobj.device_handle(), offset, size, dst, stream.value(), device_id);

    if (reqs.empty()) {
      buffer.mark_load_failed();
      return exec::make_semi_future_with([]() -> size_t {
        throw std::runtime_error("device_read_async_io_using: cache entry has no chunks");
      });
    }

    // shared_ptr keeps the buffer alive across the deferred completion stage;
    // cached_host_buffer itself is move-only.  Capture stream.value() so the
    // success path can defer the read-pin release via cudaStreamAddCallback —
    // this keeps the entry pinned across the H2D copy that the reactor
    // launches against the entry's chunks, preventing the evictor from
    // recycling pool buffers while the copy is still in flight.
    auto shared_buf = std::make_shared<cached_host_buffer>(std::move(buffer));
    auto raw_stream = stream.value();

    auto [ctx, fut] = request_context::create(reqs.size(), size);
    assert(ctx);  // n_chunks == reqs.size() > 0, so create() never returns null here

    for (auto& r : reqs)
      r.ctx = ctx;

    auto const m = _reactors.size();
    size_t start = _next.fetch_add(1, std::memory_order_relaxed) % m;
    size_t base  = reqs.size() / m;
    size_t rem   = reqs.size() % m;
    size_t off   = 0;
    for (size_t k = 0; k < m; ++k) {
      size_t group_size = base + (k < rem ? 1 : 0);
      if (group_size == 0) continue;
      size_t reactor_idx = (start + k) % m;
      _reactors[reactor_idx]->enqueue_bulk(
        std::span<device_read_req_type>(reqs.data() + off, group_size));
      off += group_size;
    }

    // Chain the entry-state bookkeeping onto the future.  The deferred
    // stage runs on whichever thread fires the upstream completion (the
    // reactor thread in push mode, or the caller's thread in pull mode),
    // so mark_cached / mark_load_failed lands before the consumer
    // observes the result.
    return std::move(fut).defer(
      [buf = std::move(shared_buf), raw_stream](exec::try_t<size_t>&& t) -> exec::try_t<size_t> {
        if (t.has_exception()) {
          buf->mark_load_failed();
        } else {
          buf->mark_cached_with_stream(raw_stream);
        }
        return std::move(t);
      });
  }

  // -- Batch host reads (generic: dispatch to reactor host_read_async) ------

  exec::semi_future<size_t> host_read_ranges_async_io(
    sirius_io_object& obj,
    std::vector<cudf::io::text::byte_range_info> const& ranges,
    std::span<cudf::host_span<std::byte>> dst) override
  {
    auto& tobj     = as_typed(obj);
    auto file_size = tobj.size();

    if (ranges.empty()) { return exec::make_semi_future<size_t>(size_t{0}); }

    // Build every valid request up-front; we need the exact count before
    // creating the ctx so that pending starts at the right value, and we
    // want the full vector in hand to split across reactors.
    std::vector<host_read_req_type> reqs;
    reqs.reserve(ranges.size());
    size_t total = 0;
    for (size_t i = 0; i < ranges.size(); ++i) {
      auto off  = static_cast<size_t>(ranges[i].offset());
      size_t sz = std::min(static_cast<size_t>(ranges[i].size()),
                           file_size > off ? file_size - off : size_t{0});
      if (sz == 0 || sz > dst[i].size()) continue;
      host_read_req_type req;
      req.handle = tobj.host_handle();
      req.offset = off;
      req.size   = sz;
      req.dst    = reinterpret_cast<uint8_t*>(dst[i].data());
      reqs.push_back(std::move(req));
      total += sz;
    }
    // reqs.empty() case is folded into create(): it returns a null ctx
    // and a ready future with 0 bytes.  Early-return retained as a fast
    // path that skips the per-reactor split loop below.
    auto [ctx, fut] = request_context::create(reqs.size(), total);
    if (!ctx) return std::move(fut);
    for (auto& r : reqs)
      r.ctx = ctx;

    // Split evenly across reactors with a rotating start.  Each reactor
    // gets a contiguous slice of @c reqs as a span — no per-reactor vector
    // allocation, no per-request moves.  host_enqueue_bulk moves elements
    // out of the span; @c reqs remains alive until end of scope.
    auto const m = _reactors.size();
    size_t start = _next.fetch_add(1, std::memory_order_relaxed) % m;
    size_t base  = reqs.size() / m;
    size_t rem   = reqs.size() % m;
    size_t off   = 0;
    for (size_t k = 0; k < m; ++k) {
      size_t group_size = base + (k < rem ? 1 : 0);
      if (group_size == 0) continue;
      size_t reactor_idx = (start + k) % m;
      _reactors[reactor_idx]->host_enqueue_bulk(
        std::span<host_read_req_type>(reqs.data() + off, group_size));
      off += group_size;
    }
    return std::move(fut);
  }

  cudf::io::text::byte_range_info compute_physical_range(cudf::io::text::byte_range_info logical,
                                                         size_t file_size) const override
  {
    return Reactor::align_to_physical(logical, file_size);
  }

 protected:
  std::vector<std::unique_ptr<Reactor>> _reactors;
  std::atomic<size_t> _next{0};

 private:
  static io_object_type& as_typed(sirius_io_object& obj)
  {
    // make_datasource() only accepts sirius_io_object subclasses; backends
    // are expected to hand in an io_object_type instance.  A misuse here is
    // a programmer error, not runtime user input.
    return static_cast<io_object_type&>(obj);
  }

  exec::semi_future<size_t> enqueue_device_read(
    io_object_type& obj, size_t offset, size_t size, uint8_t* dst, cudaStream_t stream)
  {
    auto file_size = obj.size();
    if (size == 0 || offset >= file_size) {
      // Nothing to read — return a ready future with 0 bytes.
      return exec::make_semi_future<size_t>(size_t{0});
    }
    size = std::min(size, file_size - offset);

    auto phys = Reactor::align_to_physical(
      {static_cast<int64_t>(offset), static_cast<int64_t>(size)}, file_size);
    size_t a_start = static_cast<size_t>(phys.offset());
    size_t a_end   = a_start + static_cast<size_t>(phys.size());
    size_t prefix  = offset - a_start;

    size_t n_chunks = (a_end - a_start + CHUNK_SIZE - 1) / CHUNK_SIZE;

    // Capture the caller's CUDA device BEFORE creating the request_context
    // so that a failure throws cleanly without leaving a ctx whose
    // destructor safety net would fire a less specific error.  In
    // multi-GPU usage a single reactor thread serves streams bound to
    // different devices, and silently defaulting to "don't switch"
    // lands the H2D on the wrong device.
    int const device_id = detail::current_cuda_device();

    auto [ctx, fut] = request_context::create(n_chunks, size);
    if (!ctx) return std::move(fut);

    // Build the chunks into one flat vector, then hand each reactor a
    // contiguous span slice.  Collapses N wake-notifies to at most M
    // (reactor count) and avoids M per-reactor vector allocations.
    std::vector<device_read_req_type> reqs;
    reqs.reserve(n_chunks);

    size_t produced = 0;
    for (size_t cur = a_start; cur < a_end; cur += CHUNK_SIZE) {
      device_read_req_type req;
      req.handle    = obj.device_handle();
      req.file_off  = cur;
      req.io_size   = std::min(CHUNK_SIZE, a_end - cur);
      req.data_off  = (cur == a_start) ? prefix : 0;
      req.data_size = std::min(req.io_size - req.data_off, size - produced);
      req.dst       = dst + produced;
      req.stream    = stream;
      req.device_id = device_id;
      req.ctx       = ctx;
      produced += req.data_size;
      reqs.push_back(std::move(req));
    }

    auto const m = _reactors.size();
    size_t start = _next.fetch_add(1, std::memory_order_relaxed) % m;
    size_t base  = reqs.size() / m;
    size_t rem   = reqs.size() % m;
    size_t off   = 0;
    for (size_t k = 0; k < m; ++k) {
      size_t group_size = base + (k < rem ? 1 : 0);
      if (group_size == 0) continue;
      size_t reactor_idx = (start + k) % m;
      _reactors[reactor_idx]->enqueue_bulk(
        std::span<device_read_req_type>(reqs.data() + off, group_size));
      off += group_size;
    }
    return std::move(fut);
  }
};

}  // namespace sirius::io
