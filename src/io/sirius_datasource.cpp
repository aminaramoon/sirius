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

#include "io/sirius_datasource.hpp"

#include "io/prefetching_cache.hpp"

#include <rmm/device_buffer.hpp>

#include <fcntl.h>
#include <spdlog/spdlog.h>
#include <sys/stat.h>

#include <algorithm>
#include <future>
#include <memory>
#include <utility>
#include <vector>

namespace sirius::io {

sirius_datasource::sirius_datasource(std::shared_ptr<sirius_ioctx> io_ctx,
                                     std::shared_ptr<sirius_io_object> io_object)
  : _io_ctx(std::move(io_ctx)), _io_object(std::move(io_object))
{
}

sirius_datasource::~sirius_datasource() {}

size_t sirius_datasource::size() const { return _io_object->size(); }

bool sirius_datasource::supports_device_read() const { return _io_ctx->supports_device_read(); }

bool sirius_datasource::supports_vector_host_read() const
{
  return _io_ctx->supports_vector_host_read();
}

bool sirius_datasource::is_device_read_preferred(size_t) const
{
  return _io_ctx->supports_device_read();
}

size_t sirius_datasource::host_read(size_t offset, size_t size, uint8_t* dst)
{
  return _io_ctx->host_read(*_io_object, offset, size, dst);
}

std::unique_ptr<cudf::io::datasource::buffer> sirius_datasource::host_read(size_t offset,
                                                                           size_t size)
{
  std::vector<uint8_t> buf(size);
  _io_ctx->host_read(*_io_object, offset, size, buf.data());
  return cudf::io::datasource::buffer::create(std::move(buf));
}

std::future<size_t> sirius_datasource::host_read_async(size_t offset, size_t size, uint8_t* dst)
{
  return _io_ctx->host_read_async(*_io_object, offset, size, dst);
}

std::future<std::unique_ptr<cudf::io::datasource::buffer>> sirius_datasource::host_read_async(
  size_t offset, size_t size)
{
  auto file_size = _io_object->size();
  size           = std::min(size, file_size > offset ? file_size - offset : size_t{0});
  auto buf       = std::make_shared<std::vector<uint8_t>>(size);
  auto inner_fut = std::make_shared<std::future<size_t>>(
    _io_ctx->host_read_async(*_io_object, offset, size, buf->data()));
  return std::async(std::launch::deferred, [buf, inner_fut]() mutable {
    auto n = inner_fut->get();
    buf->resize(n);
    return datasource::buffer::create(std::move(*buf));
  });
}

std::unique_ptr<cudf::io::datasource::buffer> sirius_datasource::device_read(
  size_t offset, size_t size, rmm::cuda_stream_view stream)
{
  rmm::device_buffer buf(size, stream);
  size_t n =
    _io_ctx->device_read(*_io_object, offset, size, static_cast<uint8_t*>(buf.data()), stream);
  buf.resize(n, stream);
  return cudf::io::datasource::buffer::create(std::move(buf));
}

size_t sirius_datasource::device_read(size_t offset,
                                      size_t size,
                                      uint8_t* dst,
                                      rmm::cuda_stream_view stream)
{
  return _io_ctx->device_read(*_io_object, offset, size, dst, stream);
}

std::future<size_t> sirius_datasource::device_read_async(size_t offset,
                                                         size_t size,
                                                         uint8_t* dst,
                                                         rmm::cuda_stream_view stream)
{
  return _io_ctx->device_read_async(*_io_object, offset, size, dst, stream);
}

void sirius_datasource::fadvise(prefetching_mode site,
                                std::span<const cudf::io::text::byte_range_info> ranges)
{
  // Disposable is always honored, regardless of the backend's preferred
  // mode.  Cancel any handle a prior speculative/immediate call left on
  // this datasource; if there's no handle, the cache worker has already
  // taken (or never had) the request — nothing to do.
  if (site == prefetching_mode::disposable) {
    _prefetch_handle.cancel();
    _prefetch_handle = {};
    return;
  }

  // Speculative / immediate: only honored when the backend asked for this
  // particular call site.  none falls through to no-op (the caller blindly
  // calls fadvise at every tier and the backend's preference is what
  // decides where the work actually lands).
  auto const preferred = _io_ctx->preferred_prefetching_mode();
  if (preferred == prefetching_mode::none || site != preferred) { return; }

  // The contract is "one scan, one datasource": a second
  // speculative/immediate fadvise on a datasource that already carries a
  // handle is a caller bug.  Warn loudly; cancel the stale handle so the
  // worker drops the old request and we don't leak both into the cache.
  if (_prefetch_handle) {
    spdlog::warn(
      "sirius_datasource::fadvise: a prefetching_handle was already stored on "
      "this datasource (path={}); cancelling the stale request.  Each scan "
      "should own a unique datasource.",
      _io_object->object_path());
    _prefetch_handle.cancel();
    _prefetch_handle = {};
  }

  auto* cache = _io_ctx->cache();
  if (cache == nullptr) { return; }

  // Hand the ranges to the cache.  insert() returns an empty handle when
  // it didn't enqueue any new work (dormant cache, every range coalesced
  // with an existing entry); we only stash a real handle.
  std::vector<cudf::io::text::byte_range_info> owned_ranges(ranges.begin(), ranges.end());
  auto handle = cache->insert(*_io_object, owned_ranges);
  if (handle) { _prefetch_handle = std::move(handle); }
}

}  // namespace sirius::io
