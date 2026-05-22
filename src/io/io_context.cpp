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

#include "io/io_context.hpp"

#include "driver_types.h"
#include "io/prefetching_cache.hpp"

#include <rmm/device_buffer.hpp>

#include <cuda_runtime.h>

#include <cucascade/cuda/event.hpp>

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <exception>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace sirius::io {

sirius_ioctx::sirius_ioctx() = default;
sirius_ioctx::~sirius_ioctx()
{
  // Derived destructors are required to call @c pre_destroy() as
  // their first statement, which drains the cache while reactors are
  // still alive.  By the time we get here, _cache must be empty;
  // anything else is a contract violation that would race the
  // cache's worker against partially-destroyed derived state.
  assert(!_cache && "derived ioctx forgot to call pre_destroy() in its destructor");
}

void sirius_ioctx::initialize_cache(buffer_pool* pool, size_t inflight_budget_chunks)
{
  // One-shot.  Repeated calls are silent no-ops so callers can be
  // robust to multiple wiring sites.
  if (_cache) return;
  _cache = std::make_unique<prefetching_cache>(pool, this, inflight_budget_chunks);
}

void sirius_ioctx::shutdown_cache() noexcept { _cache.reset(); }

void sirius_ioctx::device_read_async_io_using(sirius_io_object&,
                                              size_t,
                                              size_t,
                                              uint8_t*,
                                              rmm::cuda_stream_view,
                                              cached_host_buffer buffer,
                                              io_completion_handler)
{
  throw std::runtime_error(
    "sirius_ioctx: device_read_async_io_using not supported by this backend");
}

namespace {

std::future<size_t> copy_pinned_slices_to_device(
  std::vector<cudf::io::datasource::non_owning_buffer> const& slices,
  uint8_t* dst,
  rmm::cuda_stream_view stream)
{
  // Skip empty slices without touching CUDA.
  size_t n_nonempty =
    std::count_if(slices.begin(), slices.end(), [](auto const& s) { return s.size() > 0; });

  if (n_nonempty == 0) return std::async(std::launch::deferred, []() { return size_t{0}; });

  cucascade::cuda::cuda_event copy_done_event;

  // Fast path: one slice (common after pinned_view::slice coalescing when
  // chunks are contiguous in slab memory). Plain cudaMemcpyAsync avoids the
  // batch-API per-call overhead.
  if (n_nonempty == 1) {
    size_t copied = 0;
    for (auto const& s : slices) {
      if (s.size() == 0) continue;
      auto err =
        cudaMemcpyAsync(dst + copied, s.data(), s.size(), cudaMemcpyHostToDevice, stream.value());
      if (err != cudaSuccess) {
        throw std::runtime_error(std::string("sirius_ioctx: cudaMemcpyAsync failed: ") +
                                 cudaGetErrorString(err));
      }
      copied += s.size();
    }
    copy_done_event.record(stream);
    return std::async(std::launch::deferred, [e = std::move(copy_done_event), copied]() mutable {
      e.synchronize();
      return copied;
    });
  }

  // Build the batch descriptors. Copies within a batch are unordered with
  // respect to each other but the whole batch is stream-ordered; all copies
  // have disjoint destination ranges.
  std::vector<void*> dsts;
  std::vector<void const*> srcs;
  std::vector<size_t> sizes;
  dsts.reserve(n_nonempty);
  srcs.reserve(n_nonempty);
  sizes.reserve(n_nonempty);

  size_t copied = 0;
  for (auto const& s : slices) {
    auto n = s.size();
    if (n == 0) continue;
    dsts.push_back(dst + copied);
    srcs.push_back(s.data());
    sizes.push_back(n);
    copied += n;
  }

#if CUDART_VERSION < 12080
  // No batch API: fall back to sequential cudaMemcpyAsync on the caller's stream.
  size_t total_copied = 0;
  for (size_t i = 0; i < dsts.size(); ++i) {
    auto err = cudaMemcpyAsync(dsts[i], srcs[i], sizes[i], cudaMemcpyHostToDevice, stream.value());
    if (err != cudaSuccess) {
      throw std::runtime_error(std::string("sirius_ioctx: cudaMemcpyAsync failed at idx ") +
                               std::to_string(i) + ": " + cudaGetErrorString(err));
    }
    total_copied += sizes[i];
  }
  copy_done_event.record(stream);
  return std::async(std::launch::deferred,
                    [e = std::move(copy_done_event), total_copied]() mutable {
                      e.synchronize();
                      return total_copied;
                    });
#else
  // Batch API available. It does not support legacy/null streams, so in
  // that case redirect to cudaStreamPerThread.  Once the batch completes
  // we insert a GPU-side cudaStreamWaitEvent edge from per-thread back
  // to the legacy stream so any work the caller queues on the legacy
  // stream after we return is correctly ordered behind our copy.
  // Without that edge the legacy stream and per-thread stream are
  // unordered and a caller reading `dst` on the legacy stream races
  // with our batch.
  bool const stream_is_legacy = (stream.value() == nullptr) || (stream.value() == cudaStreamLegacy);
  cudaStream_t stream_to_use  = stream_is_legacy ? cudaStreamPerThread : stream.value();

  cudaMemcpyAttributes attrs{};
  attrs.srcAccessOrder  = cudaMemcpySrcAccessOrderStream;
  attrs.srcLocHint.type = cudaMemLocationTypeHost;
  attrs.dstLocHint.type = cudaMemLocationTypeDevice;
  attrs.flags           = 0;

  size_t attrs_idx = 0;  // attrs applies to all copies -> single entry at index 0
  size_t fail_idx  = 0;  // out-param: driver writes failing copy index on error

#if CUDART_VERSION < 13000
  auto err = cudaMemcpyBatchAsync(dsts.data(),
                                  srcs.data(),
                                  sizes.data(),
                                  n_nonempty,
                                  &attrs,
                                  &attrs_idx,
                                  1,
                                  &fail_idx,
                                  stream_to_use);
#else
  auto err = cudaMemcpyBatchAsync(
    dsts.data(), srcs.data(), sizes.data(), n_nonempty, &attrs, &attrs_idx, 1, stream_to_use);
#endif
  if (err != cudaSuccess) {
    cudaGetLastError();
    throw std::runtime_error(std::string("sirius_ioctx: cudaMemcpyBatchAsync failed at idx ") +
                             std::to_string(fail_idx) + ": " + cudaGetErrorString(err));
  }
  copy_done_event.record(stream_to_use);
  if (stream_is_legacy) {
    // Add a GPU-side sync edge from per-thread back to the caller's
    // legacy stream so caller work submitted on the legacy stream after
    // device_read returns observes our copy.  Host cost: O(10ns).
    cudaError_t wait_err = cudaStreamWaitEvent(stream.value(), copy_done_event.get(), 0);
    if (wait_err != cudaSuccess) {
      cudaGetLastError();
      throw std::runtime_error(std::string("sirius_ioctx: cudaStreamWaitEvent failed: ") +
                               cudaGetErrorString(wait_err));
    }
  }
  return std::async(std::launch::deferred, [e = std::move(copy_done_event), copied]() mutable {
    e.synchronize();
    return copied;
  });
#endif
}
}  // namespace

size_t sirius_ioctx::host_read(sirius_io_object& obj, size_t offset, size_t size, uint8_t* dst)
{
  // uses_prefetching_cache() gates the lookup on both "a cache exists"
  // and "this backend can serve the vector host reads the prefetcher
  // needs" — backends like the kvikio fallback never carry a cache for
  // reads, so the map lookup is skipped entirely.
  if (uses_prefetching_cache()) {
    if (auto view = _cache->read(obj, offset, size); view) {
      auto slices   = view.slice(offset, size);
      size_t copied = 0;
      for (auto const& s : slices) {
        std::memcpy(dst + copied, s.data(), s.size());
        copied += s.size();
      }
      return copied;
    }
  }
  return host_read_io(obj, offset, size, dst);
}

std::future<size_t> sirius_ioctx::host_read_async(sirius_io_object& obj,
                                                  size_t offset,
                                                  size_t size,
                                                  uint8_t* dst)
{
  if (uses_prefetching_cache()) {
    if (auto view = _cache->read(obj, offset, size); view) {
      auto slices = view.slice(offset, size);
      try {
        size_t copied = 0;
        for (auto const& s : slices) {
          std::memcpy(dst + copied, s.data(), s.size());
          copied += s.size();
        }
        return std::async(std::launch::deferred, [copied]() { return copied; });
      } catch (...) {
        return std::async(std::launch::deferred, [e = std::current_exception()]() -> size_t {
          std::rethrow_exception(e);
        });
      }
    }
  }
  auto promise = std::make_shared<std::promise<size_t>>();
  host_read_async_io(
    obj, offset, size, dst, [promise](size_t bytes_transferred, std::exception_ptr ep) {
      if (ep) {
        promise->set_exception(std::move(ep));
      } else {
        promise->set_value(bytes_transferred);
      }
    });
  return promise->get_future();
}

size_t sirius_ioctx::device_read(
  sirius_io_object& obj, size_t offset, size_t size, uint8_t* dst, rmm::cuda_stream_view stream)
{
  if (uses_prefetching_cache()) {
    cached_host_buffer chb;
    if (auto view = _cache->read(obj, offset, size, stream.value(), &chb); view) {
      auto slices = view.slice(offset, size);
      auto copied = copy_pinned_slices_to_device(slices, dst, stream);
      return copied.get();
    }
    if (chb) {
      // Allocated-steal path: chunks are pre-assigned, dispatch
      // file → bounce → device directly through the new ioctx API.  Route
      // any synchronous throw from device_read_async_io_using through the
      // promise so fut.get() surfaces the original exception instead of
      // std::future_error(broken_promise).
      std::promise<size_t> p;
      auto fut = p.get_future();
      try {
        device_read_async_io_using(obj,
                                   offset,
                                   size,
                                   dst,
                                   stream,
                                   std::move(chb),
                                   [&p](size_t bytes_transferred, std::exception_ptr ep) {
                                     if (ep) {
                                       p.set_exception(std::move(ep));
                                     } else {
                                       p.set_value(bytes_transferred);
                                     }
                                   });
      } catch (...) {
        p.set_exception(std::current_exception());
      }
      return fut.get();
    }
  }
  return device_read_io(obj, offset, size, dst, stream);
}

std::future<size_t> sirius_ioctx::device_read_async(
  sirius_io_object& obj, size_t offset, size_t size, uint8_t* dst, rmm::cuda_stream_view stream)
{
  if (uses_prefetching_cache()) {
    cached_host_buffer chb;
    if (auto view = _cache->read(obj, offset, size, stream.value(), &chb); view) {
      auto slices = view.slice(offset, size);
      try {
        return copy_pinned_slices_to_device(slices, dst, stream);
      } catch (...) {
        return std::async(std::launch::deferred, [e = std::current_exception()]() -> size_t {
          std::rethrow_exception(e);
        });
      }
    }
    if (chb) {
      // Allocated-steal path: same as the sync flow above, but expose the
      // future to the caller instead of blocking on it.  Synchronous throws
      // are routed through the promise so callers always observe failure
      // via the returned future rather than as a propagated exception.
      auto promise = std::make_shared<std::promise<size_t>>();
      try {
        device_read_async_io_using(obj,
                                   offset,
                                   size,
                                   dst,
                                   stream,
                                   std::move(chb),
                                   [promise](size_t bytes_transferred, std::exception_ptr ep) {
                                     if (ep) {
                                       promise->set_exception(std::move(ep));
                                     } else {
                                       promise->set_value(bytes_transferred);
                                     }
                                   });
      } catch (...) {
        promise->set_exception(std::current_exception());
      }
      return promise->get_future();
    }
  }
  // Same try/catch shape as the allocated-steal path above: a synchronous
  // throw from device_read_async_io (e.g. cudaGetDevice failure inside
  // enqueue_device_read) must be delivered via the returned future, not
  // propagated to the caller — otherwise the future has no setter and
  // callers see broken_promise instead of the real error.
  auto promise = std::make_shared<std::promise<size_t>>();
  try {
    device_read_async_io(
      obj, offset, size, dst, stream, [promise](size_t bytes_transferred, std::exception_ptr ep) {
        if (ep) {
          promise->set_exception(std::move(ep));
        } else {
          promise->set_value(bytes_transferred);
        }
      });
  } catch (...) {
    promise->set_exception(std::current_exception());
  }
  return promise->get_future();
}

}  // namespace sirius::io
