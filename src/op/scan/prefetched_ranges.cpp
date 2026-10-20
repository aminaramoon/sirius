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

#include "op/scan/prefetched_ranges.hpp"

#include <rmm/cuda_device.hpp>

#include <cuda_runtime_api.h>

#include <optional>
#include <stdexcept>
#include <utility>

namespace sirius::op::scan {

//===----------------------------------------------------------------------===//
// device_stream
//===----------------------------------------------------------------------===//

std::shared_ptr<device_stream> device_stream::create(int device)
{
  rmm::cuda_set_device_raii bind{rmm::cuda_device_id{device}};
  auto owner    = std::shared_ptr<device_stream>(new device_stream);
  owner->device = device;
  owner->stream = std::make_unique<rmm::cuda_stream>(rmm::cuda_stream::flags::non_blocking);
  return owner;
}

namespace {

/// Binds @p device into @p bind (left empty when the switch fails).  Destructors use it: a failed
/// device switch must not terminate the process, and the frees that follow are still attempted.
void try_bind_device(std::optional<rmm::cuda_set_device_raii>& bind, int device) noexcept
{
  try {
    bind.emplace(rmm::cuda_device_id{device});
  } catch (...) {
    bind.reset();
  }
}

}  // namespace

device_stream::~device_stream()
{
  if (!stream) { return; }
  // A stream belongs to the device it was created on; destroy it there, after the work still
  // queued on it (stream-ordered frees of the buffers it allocated) has finished.
  std::optional<rmm::cuda_set_device_raii> bind;
  try_bind_device(bind, device);
  stream->synchronize_no_throw();
  stream.reset();
}

//===----------------------------------------------------------------------===//
// prefetched_ranges
//===----------------------------------------------------------------------===//

prefetched_ranges::prefetched_ranges(int device_id,
                                     std::shared_ptr<device_stream const> stream_owner,
                                     std::vector<device_ranges> ranges,
                                     std::size_t total_bytes,
                                     release_hook on_release) noexcept
  : _device_id(device_id),
    _stream_owner(std::move(stream_owner)),
    _total_bytes(total_bytes),
    _on_release(std::move(on_release)),
    _ranges(std::move(ranges))
{
}

prefetched_ranges::~prefetched_ranges()
{
  if (!_released.load(std::memory_order_acquire)) {
    auto const reason = _wrong_device.load(std::memory_order_acquire) ? release_reason::wrong_device
                        : current() == state::failed                  ? release_reason::failed
                                                                      : release_reason::unconsumed;
    fire_release(reason);
  }
  if (_ranges.empty()) { return; }
  // Frees are stream-ordered on each buffer's stream -- the allocation stream, still alive
  // because _stream_owner is destroyed after this body -- and belong to the buffers' device.
  std::optional<rmm::cuda_set_device_raii> bind;
  try_bind_device(bind, _device_id);
  _ranges.clear();
}

bool prefetched_ranges::settle(state next, std::exception_ptr error, bool abandoned) noexcept
{
  {
    std::lock_guard lock(_mutex);
    if (_state != state::pending) { return false; }
    if (abandoned) {
      error = std::make_exception_ptr(
        std::runtime_error("device prefetch abandoned: consumer overtook the readahead"));
    }
    _state     = next;
    _failure   = std::move(error);
    _abandoned = abandoned;
  }
  _cv.notify_all();
  return true;
}

bool prefetched_ranges::set_ready() noexcept { return settle(state::ready, nullptr, false); }

bool prefetched_ranges::set_failed(std::exception_ptr error) noexcept
{
  return settle(state::failed, std::move(error), false);
}

bool prefetched_ranges::abandon() noexcept { return settle(state::failed, nullptr, true); }

prefetched_ranges::state prefetched_ranges::wait_for(std::chrono::milliseconds timeout) const
{
  auto const start = std::chrono::steady_clock::now();
  state seen{state::pending};
  {
    std::unique_lock lock(_mutex);
    _cv.wait_for(lock, timeout, [this] { return _state != state::pending; });
    seen = _state;
  }
  auto const waited = std::chrono::steady_clock::now() - start;
  _wait_ns.fetch_add(static_cast<std::uint64_t>(
                       std::chrono::duration_cast<std::chrono::nanoseconds>(waited).count()),
                     std::memory_order_relaxed);
  return seen;
}

prefetched_ranges::state prefetched_ranges::current() const noexcept
{
  std::lock_guard lock(_mutex);
  return _state;
}

std::exception_ptr prefetched_ranges::failure() const noexcept
{
  std::lock_guard lock(_mutex);
  return _failure;
}

bool prefetched_ranges::abandoned() const noexcept
{
  std::lock_guard lock(_mutex);
  return _abandoned;
}

void prefetched_ranges::flag_wrong_device() noexcept
{
  _wrong_device.store(true, std::memory_order_release);
}

std::optional<std::vector<prefetched_ranges::device_ranges>> prefetched_ranges::take(
  rmm::cuda_stream_view consumer_stream)
{
  // The returned buffers are freed on the caller's current device: refuse any other.
  int current_device = -1;
  if (cudaGetDevice(&current_device) != cudaSuccess || current_device != _device_id) {
    flag_wrong_device();
    return std::nullopt;
  }
  std::vector<device_ranges> out;
  {
    std::lock_guard lock(_mutex);
    if (_state != state::ready || _taken || _wrong_device.load(std::memory_order_acquire)) {
      return std::nullopt;
    }
    _taken = true;
    // The consumer frees these after its decode: rebinding first makes that free trail the
    // decode kernels on the consumer's stream instead of racing them on the allocation stream.
    for (auto& entry : _ranges) {
      for (auto& buffer : entry.buffers) {
        buffer.set_stream(consumer_stream);
      }
    }
    out = std::move(_ranges);
    _ranges.clear();
  }
  fire_release(release_reason::consumed);
  return out;
}

bool prefetched_ranges::taken() const noexcept
{
  std::lock_guard lock(_mutex);
  return _taken;
}

std::size_t prefetched_ranges::entry_count() const noexcept
{
  std::lock_guard lock(_mutex);
  return _ranges.size();
}

void prefetched_ranges::fire_release(release_reason reason) noexcept
{
  if (_released.exchange(true, std::memory_order_acq_rel)) { return; }
  if (!_on_release) { return; }
  _on_release(release_event{
    .reason = reason, .bytes = _total_bytes, .wait_ns = _wait_ns.load(std::memory_order_relaxed)});
  // Drop the hook's captures (the per-device window) now rather than with this object.
  _on_release = nullptr;
}

}  // namespace sirius::op::scan
