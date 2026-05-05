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

#include "io/prefetching_cache.hpp"

#include <rmm/device_buffer.hpp>

#include <cuda_runtime.h>

#include <exception>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace sirius::io {

sirius_ioctx::sirius_ioctx()  = default;
sirius_ioctx::~sirius_ioctx() = default;

void sirius_ioctx::initialize_cache(buffer_pool& pool, size_t inflight_budget_chunks)
{
  _cache = std::make_unique<prefetching_cache>(pool, this, inflight_budget_chunks);
}

namespace {

// Copy each pinned-host slice to the device buffer on @p stream.
// Returns the total bytes issued (== sum of slice sizes).
size_t copy_pinned_slices_to_device(
  std::vector<cudf::io::datasource::non_owning_buffer> const& slices,
  uint8_t* dst,
  rmm::cuda_stream_view stream)
{
  // Skip empty slices without touching CUDA.
  size_t n_nonempty = 0;
  for (auto const& s : slices)
    if (s.size() > 0) ++n_nonempty;

  if (n_nonempty == 0) return 0;

  // Fast path: one slice (common after pinned_view::slice coalescing when
  // chunks are contiguous in slab memory).  Plain cudaMemcpyAsync avoids the
  // batch-API per-call overhead.
  if (n_nonempty == 1) {
    size_t copied = 0;
    for (auto const& s : slices) {
      if (s.size() == 0) continue;
      auto err =
        cudaMemcpyAsync(dst + copied, s.data(), s.size(), cudaMemcpyHostToDevice, stream.value());
      if (err != cudaSuccess)
        throw std::runtime_error(std::string("sirius_ioctx: cudaMemcpyAsync failed: ") +
                                 cudaGetErrorString(err));
      copied += s.size();
    }
    return copied;
  }

  // Batch path: hand all non-contiguous slices to the driver in one call.
  // Copies within a batch are unordered with respect to each other but the
  // whole batch is stream-ordered; all copies have disjoint
  // destination ranges.
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

#if CUDA_VERSION >= 12080
  cudaMemcpyAttributes attrs{};
  attrs.srcAccessOrder  = cudaMemcpySrcAccessOrderStream;
  attrs.srcLocHint.type = cudaMemLocationTypeHost;
  attrs.dstLocHint.type = cudaMemLocationTypeDevice;
  attrs.flags           = 0;
  size_t attrs_idx      = 0;
  size_t fail_idx       = 0;

  auto err = cudaMemcpyBatchAsync(
    dsts.data(), srcs.data(), sizes.data(), n_nonempty, &attrs, &attrs_idx, 1, stream.value());
  if (err != cudaSuccess)
    throw std::runtime_error(std::string("sirius_ioctx: cudaMemcpyBatchAsync failed at idx ") +
                             std::to_string(fail_idx) + ": " + cudaGetErrorString(err));
#else
  for (size_t i = 0; i < n_nonempty; ++i) {
    auto err = cudaMemcpyAsync(dsts[i], srcs[i], sizes[i], cudaMemcpyHostToDevice, stream.value());
    if (err != cudaSuccess)
      throw std::runtime_error(std::string("sirius_ioctx: cudaMemcpyAsync failed at idx ") +
                               std::to_string(i) + ": " + cudaGetErrorString(err));
  }
#endif
  return copied;
}

}  // namespace

size_t sirius_ioctx::host_read(sirius_io_object& obj, size_t offset, size_t size, uint8_t* dst)
{
  if (_cache) {
    if (auto view = _cache->read(obj, offset, size, 0); view) {
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

void sirius_ioctx::host_read_async(
  sirius_io_object& obj, size_t offset, size_t size, uint8_t* dst, io_completion_handler handler)
{
  if (_cache) {
    if (auto view = _cache->read(obj, offset, size, 0); view) {
      auto slices = view.slice(offset, size);
      try {
        size_t copied = 0;
        for (auto const& s : slices) {
          std::memcpy(dst + copied, s.data(), s.size());
          copied += s.size();
        }
        handler(copied, nullptr);
      } catch (...) {
        handler(0, std::current_exception());
      }
      return;
    }
  }
  host_read_async_io(obj, offset, size, dst, std::move(handler));
}

size_t sirius_ioctx::device_read(
  sirius_io_object& obj, size_t offset, size_t size, uint8_t* dst, rmm::cuda_stream_view stream)
{
  if (_cache) {
    if (auto view = _cache->read(obj, offset, size, stream.value()); view) {
      auto slices = view.slice(offset, size);
      return copy_pinned_slices_to_device(slices, dst, stream);
    }
  }
  return device_read_io(obj, offset, size, dst, stream);
}

void sirius_ioctx::device_read_async(sirius_io_object& obj,
                                     size_t offset,
                                     size_t size,
                                     uint8_t* dst,
                                     rmm::cuda_stream_view stream,
                                     io_completion_handler handler)
{
  if (_cache) {
    if (auto view = _cache->read(obj, offset, size, stream.value()); view) {
      auto slices = view.slice(offset, size);
      try {
        auto copied = copy_pinned_slices_to_device(slices, dst, stream);
        handler(copied, nullptr);
      } catch (...) {
        handler(0, std::current_exception());
      }
      return;
    }
  }
  device_read_async_io(obj, offset, size, dst, stream, std::move(handler));
}

}  // namespace sirius::io
