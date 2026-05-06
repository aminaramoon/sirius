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
  spdlog::debug("sirius_datasource[{}]: opened size={}B cache={}",
                _io_object->object_path(),
                _io_object->size(),
                _io_ctx->cache() != nullptr ? "on" : "off");
}

sirius_datasource::~sirius_datasource()
{
  // Cache summary is process-wide, so logging it per-datasource over-reports;
  // do it only when the cache is on, gated to debug.
  if (_io_ctx && _io_ctx->cache() != nullptr) {
    spdlog::debug("sirius_datasource[{}]: closing — {}",
                  _io_object->object_path(),
                  _io_ctx->cache()->summary());
  } else {
    spdlog::debug("sirius_datasource[{}]: closing (no cache)", _io_object->object_path());
  }
}

size_t sirius_datasource::size() const { return _io_object->size(); }

bool sirius_datasource::supports_device_read() const { return true; }

bool sirius_datasource::is_device_read_preferred(size_t) const { return true; }

size_t sirius_datasource::host_read(size_t offset, size_t size, uint8_t* dst)
{
  spdlog::trace(
    "sirius_datasource[{}]: host_read offset={} size={}", _io_object->object_path(), offset, size);
  return _io_ctx->host_read(*_io_object, offset, size, dst);
}

std::unique_ptr<cudf::io::datasource::buffer> sirius_datasource::host_read(size_t offset,
                                                                           size_t size)
{
  spdlog::trace("sirius_datasource[{}]: host_read(buf) offset={} size={}",
                _io_object->object_path(),
                offset,
                size);
  std::vector<uint8_t> buf(size);
  _io_ctx->host_read(*_io_object, offset, size, buf.data());
  return cudf::io::datasource::buffer::create(std::move(buf));
}

std::future<size_t> sirius_datasource::host_read_async(size_t offset, size_t size, uint8_t* dst)
{
  spdlog::trace("sirius_datasource[{}]: host_read_async offset={} size={}",
                _io_object->object_path(),
                offset,
                size);
  return _io_ctx->host_read_async(*_io_object, offset, size, dst);
}

std::future<std::unique_ptr<cudf::io::datasource::buffer>> sirius_datasource::host_read_async(
  size_t offset, size_t size)
{
  auto file_size = _io_object->size();
  size           = std::min(size, file_size > offset ? file_size - offset : size_t{0});
  spdlog::trace("sirius_datasource[{}]: host_read_async(buf) offset={} size={}",
                _io_object->object_path(),
                offset,
                size);
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
  spdlog::trace("sirius_datasource[{}]: device_read(buf) offset={} size={}",
                _io_object->object_path(),
                offset,
                size);
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
  spdlog::trace("sirius_datasource[{}]: device_read offset={} size={}",
                _io_object->object_path(),
                offset,
                size);
  return _io_ctx->device_read(*_io_object, offset, size, dst, stream);
}

std::future<size_t> sirius_datasource::device_read_async(size_t offset,
                                                         size_t size,
                                                         uint8_t* dst,
                                                         rmm::cuda_stream_view stream)
{
  spdlog::trace("sirius_datasource[{}]: device_read_async offset={} size={}",
                _io_object->object_path(),
                offset,
                size);
  return _io_ctx->device_read_async(*_io_object, offset, size, dst, stream);
}

}  // namespace sirius::io
