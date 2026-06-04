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

#include "io/kvikio/kvikio_context.hpp"

#include "io/sirius_datasource.hpp"

#include <cstdint>
#include <stdexcept>
#include <utility>

namespace sirius::io {

namespace {

const kvikio_io_object& as_kvikio(const sirius_io_object& obj)
{
  // Concrete type is enforced by create_io_object below; a mismatch is a
  // programmer error (e.g. mixing io_objects across backends), not user
  // input, so a static_cast is appropriate.
  return static_cast<const kvikio_io_object&>(obj);
}

}  // namespace

std::shared_ptr<sirius_io_object> kvikio_context::create_io_object(std::string path)
{
  // cudf::io::datasource::create returns a unique_ptr; promote to shared_ptr
  // so the kvikio_io_object can expose access without transferring
  // ownership (the io_object outlives any single sirius_datasource we hand
  // back from make_datasource).
  std::shared_ptr<cudf::io::datasource> ds = cudf::io::datasource::create(path);
  auto const file_size                     = ds->size();
  return std::make_shared<kvikio_io_object>(std::move(path), std::move(ds), file_size);
}

std::unique_ptr<cudf::io::datasource> kvikio_context::make_datasource(
  std::shared_ptr<sirius_io_object> io_object)
{
  // Return a sirius_datasource so cudf sees a single datasource type across
  // backends.  Its read methods call back into this ioctx via the public
  // host_read / device_read overrides above.
  return std::make_unique<sirius_datasource>(shared_from_this(), std::move(io_object));
}

bool kvikio_context::supports(std::string_view /*path*/) const noexcept
{
  // cudf::io::datasource::create handles file paths, URIs, and registered
  // protocol handlers; the actual feasibility check happens at
  // create_io_object time, where opening the file may throw.
  return true;
}

// -- Public read API ---------------------------------------------------------

size_t kvikio_context::host_read_sync(const sirius_io_object& obj,
                                      size_t offset,
                                      size_t size,
                                      std::byte* dst)
{
  return as_kvikio(obj).datasource().host_read(offset, size, reinterpret_cast<uint8_t*>(dst));
}

exec::semi_future<size_t> kvikio_context::host_read_async(const sirius_io_object& obj,
                                                          size_t offset,
                                                          size_t size,
                                                          std::byte* dst) noexcept
{
  // cudf's datasource returns a std::future; the base contract is an
  // exec::semi_future.  make_semi_future_with materialises the result (and
  // captures any exception) into a ready semi_future.
  return exec::make_semi_future_with([&] {
    return as_kvikio(obj)
      .datasource()
      .host_read_async(offset, size, reinterpret_cast<uint8_t*>(dst))
      .get();
  });
}

exec::semi_future<size_t> kvikio_context::device_read_async(const sirius_io_object& obj,
                                                            size_t offset,
                                                            size_t size,
                                                            std::byte* dst,
                                                            rmm::cuda_stream_view stream) noexcept
{
  return exec::make_semi_future_with([&] {
    return as_kvikio(obj)
      .datasource()
      .device_read_async(offset, size, reinterpret_cast<uint8_t*>(dst), stream)
      .get();
  });
}

cudf::io::text::byte_range_info kvikio_context::compute_physical_range(
  cudf::io::text::byte_range_info logical, size_t /*file_size*/) const noexcept
{
  return logical;
}

// -- Protected placeholders --------------------------------------------------
//
// These are unreachable on the documented code paths: kvikio_context
// overrides the public host_read/device_read entries that the base class's
// _io primitives feed, and never attaches a prefetching_cache (which is
// the only caller of host_read_ranges_async_io).  A throw here surfaces
// any future code path that bypasses the public API.

namespace {
[[noreturn]] void unreachable_io_primitive(const char* name)
{
  throw std::logic_error(std::string("kvikio_context::") + name +
                         " is unreachable — the public read API is overridden directly");
}
}  // namespace

size_t kvikio_context::host_read_io(const sirius_io_object& /*obj*/,
                                    size_t /*offset*/,
                                    size_t /*size*/,
                                    std::byte* /*dst*/)
{
  unreachable_io_primitive("host_read_io");
}

exec::semi_future<size_t> kvikio_context::host_read_async_io(const sirius_io_object& /*obj*/,
                                                             size_t /*offset*/,
                                                             size_t /*size*/,
                                                             std::byte* /*dst*/) noexcept
{
  unreachable_io_primitive("host_read_async_io");
}

exec::semi_future<size_t> kvikio_context::device_read_async_io(
  const sirius_io_object& obj,
  size_t offset,
  size_t size,
  std::byte* dst,
  rmm::cuda_stream_view stream) noexcept
{
  unreachable_io_primitive("device_read_async_io");
}

exec::semi_future<size_t> kvikio_context::host_to_device_read_async_io(
  const sirius_io_object& obj,
  std::span<io_object_segment> slices,
  size_t offset,
  size_t size,
  std::byte* device_dst,
  rmm::cuda_stream_view stream) noexcept
{
  unreachable_io_primitive("host_to_device_read_async_io");
}

exec::semi_future<size_t> kvikio_context::host_read_ranges_async_io(
  const sirius_io_object& obj, std::span<io_object_segment> segments) noexcept
{
  unreachable_io_primitive("host_read_ranges_async_io");
}

}  // namespace sirius::io
