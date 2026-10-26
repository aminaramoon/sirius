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

#include "scan_manager/device_prefetch.hpp"

#include "log/logging.hpp"
#include "op/scan/gpu_ingestible_types.hpp"
#include "op/scan/prefetched_ranges.hpp"

#include <rmm/aligned.hpp>
#include <rmm/cuda_device.hpp>
#include <rmm/device_buffer.hpp>
#include <rmm/error.hpp>

#include <absl/cleanup/cleanup.h>
#include <cucascade/cudf/datasource.hpp>
#include <cucascade/exec/semi_future.hpp>
#include <cucascade/exec/try.hpp>
#include <cucascade/io/cache/fs_cache.hpp>
#include <cucascade/io/io_context.hpp>
#include <cucascade/io/types.hpp>
#include <cucascade/memory/memory_reservation.hpp>
#include <cucascade/memory/memory_space.hpp>
#include <cucascade/memory/reservation_aware_resource_adaptor.hpp>

#include <algorithm>
#include <cassert>
#include <mutex>
#include <new>
#include <stdexcept>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

namespace sirius::scan_manager {

namespace {

using op::scan::prefetched_ranges;
using op::scan::scan_info;
using read_result   = cucascade::exec::try_t<std::size_t>;
using read_callback = cucascade::exec::invocable<void(read_result&&)>;
using done_callback = sirius::exec::invocable<void(device_prefetch_result) noexcept>;
// Installing a callback of the future's own stored type is a move and a compare-exchange; any
// other type is wrapped in a heap-allocated adapter, whose allocation could fail after the read
// was submitted.
static_assert(
  std::is_same_v<read_callback, cucascade::exec::semi_future<std::size_t>::callback_type>);

/// An exception_ptr holding a std::runtime_error, or whatever constructing it threw.
std::exception_ptr runtime_error_ptr(char const* what) noexcept
{
  try {
    return std::make_exception_ptr(std::runtime_error(what));
  } catch (...) {
    return std::current_exception();
  }
}

/// Blocks until @p f settles and returns its result; never throws.  @p f must be valid.
read_result settle_inline(cucascade::exec::semi_future<std::size_t>&& f) noexcept
{
  try {
    return std::move(f).get_try();
  } catch (...) {
    return read_result{std::current_exception()};
  }
}

/// Fan-in over one attempt's per-hint reads.  Settles the prefetch and reports once every hint
/// and the issuer's guard have arrived.  Shared by every read callback, so it outlives issue();
/// everything here may run on any thread (an IO completion, a pipeline thread draining the
/// completion poll, the worker inline), so it is noexcept and blocks only on its own mutex.
struct completion {
  completion(std::vector<std::size_t> expected, std::size_t logical, done_callback done)
    : _remaining(expected.size() + 1),
      _expected_bytes(std::move(expected)),
      _bytes_logical(logical),
      _on_done(std::move(done))
  {
  }

  /// Hint @p k settled with @p result.  A short count is a failure: the backend clamps a read
  /// at end of file and still succeeds with the bytes it found.
  void arrive(std::size_t k, read_result&& result) noexcept
  {
    if (result.has_exception()) {
      record(std::move(result).exception());
    } else if (!result.has_value()) {
      record_failure("device read settled without a result");
    } else if (result.value() != _expected_bytes[k]) {
      try {
        record_failure("short device read: expected " + std::to_string(_expected_bytes[k]) +
                       " bytes, got " + std::to_string(result.value()));
      } catch (...) {
        record(std::current_exception());
      }
    }
    settle();
  }

  /// Drops the issuer's guard once every hint was submitted (or settled).
  void release_guard() noexcept { settle(); }

  std::shared_ptr<prefetched_ranges> keep_alive;  ///< until the last read settled

 private:
  void record(std::exception_ptr error) noexcept
  {
    std::lock_guard lock(_mutex);
    if (!_first_error) { _first_error = std::move(error); }
  }

  void record_failure(std::string const& what) noexcept
  {
    try {
      record(std::make_exception_ptr(std::runtime_error(what)));
    } catch (...) {
      record(std::current_exception());
    }
  }

  void settle() noexcept
  {
    if (_remaining.fetch_sub(1, std::memory_order_acq_rel) != 1) { return; }
    std::exception_ptr error;
    {
      std::lock_guard lock(_mutex);
      error = _first_error;
    }
    auto ranges = std::move(keep_alive);
    if (ranges) {
      if (error) {
        std::ignore = ranges->set_failed(error);
      } else {
        std::ignore = ranges->set_ready();
      }
    }
    // Before on_done: once it ran, the fan-in no longer keeps the buffers alive.
    ranges.reset();
    if (_on_done) {
      _on_done(device_prefetch_result{.bytes = _bytes_logical, .ok = !error, .error = error});
      _on_done = nullptr;
    }
  }

  std::atomic<std::size_t> _remaining;
  std::vector<std::size_t> _expected_bytes;
  std::size_t _bytes_logical;
  done_callback _on_done;
  std::mutex _mutex;
  std::exception_ptr _first_error;
};

/// Abandons an installed attempt on every exit before its reads were submitted, so no consumer
/// can wait on a prefetch nobody will settle.
class slot_guard {
 public:
  slot_guard(scan_info& split, prefetched_ranges& ranges) noexcept : _split(split), _ranges(ranges)
  {
  }
  ~slot_guard()
  {
    if (_armed) { release(); }
  }
  slot_guard(slot_guard const&)            = delete;
  slot_guard& operator=(slot_guard const&) = delete;

  void disarm() noexcept { _armed = false; }

 private:
  void release() noexcept
  {
    // No ticket bookkeeping here: the readahead published the split's scan_ticket before this
    // attempt and settles it when the attempt does not issue (a consumer that already reported
    // `reading` has joined that ticket, which then transfers to its read), so nothing has to be
    // undone for the consumer's sake.
    std::ignore = _ranges.abandon();
    _split.discard_prefetched_ranges();
  }

  scan_info& _split;
  prefetched_ranges& _ranges;
  bool _armed{true};
};

/// The release hook of one attempt: decrements exactly what the issuer added to @p window when it
/// installed the attempt, and books where the buffers went.  Atomics only: it runs on whichever
/// thread drops the attempt's last reference.
prefetched_ranges::release_hook make_release_hook(std::shared_ptr<device_window> window,
                                                  std::size_t bytes_reserved)
{
  return [w = std::move(window), bytes_reserved](prefetched_ranges::release_event e) noexcept {
    w->parked_splits.fetch_sub(1, std::memory_order_relaxed);
    w->parked_bytes.fetch_sub(bytes_reserved, std::memory_order_relaxed);
    switch (e.reason) {
      case prefetched_ranges::release_reason::consumed:
        w->consumed.fetch_add(1, std::memory_order_relaxed);
        w->consumed_bytes.fetch_add(e.bytes, std::memory_order_relaxed);
        break;
      case prefetched_ranges::release_reason::unconsumed:
        w->unconsumed.fetch_add(1, std::memory_order_relaxed);
        break;
      case prefetched_ranges::release_reason::failed:
        w->failed.fetch_add(1, std::memory_order_relaxed);
        break;
      case prefetched_ranges::release_reason::wrong_device:
        w->wrong_device.fetch_add(1, std::memory_order_relaxed);
        break;
    }
    w->wait_ns.fetch_add(e.wait_ns, std::memory_order_relaxed);
  };
}

/// Where a split's IO goes, for log lines.
std::string path_of(scan_info::fadvise_entry const& hint)
{
  return hint.datasource ? hint.datasource->get_io_object().object_path() : std::string{"<none>"};
}

}  // namespace

device_prefetch_issuer::device_prefetch_issuer(
  cucascade::memory::memory_space& space,
  std::shared_ptr<op::scan::device_stream const> stream,
  std::shared_ptr<device_window> window,
  double fraction,
  std::size_t max_parked_splits) noexcept
  : _space(space),
    _stream(std::move(stream)),
    _window(std::move(window)),
    _fraction(fraction >= 0.0 ? std::min(fraction, 1.0) : 0.0),  // NaN -> 0
    _max_parked_splits(max_parked_splits)
{
  assert(fraction >= 0.0 && fraction <= 1.0);
}

int device_prefetch_issuer::device_id() const noexcept { return _space.get_device_id(); }

device_prefetch_issue device_prefetch_issuer::issue(std::shared_ptr<scan_info> const& split,
                                                    done_callback on_done)
{
  // try_issue() submits under its own catch: whatever reaches here was thrown before submission.
  try {
    return try_issue(split, std::move(on_done));
  } catch (std::exception const& e) {
    SIRIUS_LOG_WARN("[readahead] device prefetch on GPU {} not issued: {}", device_id(), e.what());
  } catch (...) {
    SIRIUS_LOG_WARN("[readahead] device prefetch on GPU {} not issued: unknown exception",
                    device_id());
  }
  return device_prefetch_issue::error;
}

device_prefetch_issue device_prefetch_issuer::try_issue(std::shared_ptr<scan_info> const& split,
                                                        done_callback on_done)
{
  // 1. What the split would read: per hint the logical bytes, and the reservation, which is what
  //    the allocator will track (every buffer rounded up to the allocation alignment).
  if (!split || !split->supports_device_prefetch()) { return device_prefetch_issue::unsupported; }
  auto const hints = split->fadvise_hints();
  if (hints.empty()) { return device_prefetch_issue::no_ranges; }
  std::vector<std::size_t> expected_bytes;
  std::size_t bytes_logical  = 0;
  std::size_t bytes_reserved = 0;
  try {
    expected_bytes.reserve(hints.size());
  } catch (std::bad_alloc const&) {
    return device_prefetch_issue::error;
  }
  for (auto const& hint : hints) {
    if (!hint.datasource) { return device_prefetch_issue::unsupported; }
    std::size_t hint_bytes = 0;
    for (auto const& range : hint.ranges) {
      auto const size = static_cast<std::size_t>(range.size());
      hint_bytes += size;
      bytes_reserved += rmm::align_up(size, rmm::CUDA_ALLOCATION_ALIGNMENT);
    }
    expected_bytes.push_back(hint_bytes);
    bytes_logical += hint_bytes;
  }
  if (bytes_logical == 0) { return device_prefetch_issue::no_ranges; }

  // 2. The parked window holds at most half the admission budget, so a split above that would
  //    never be admitted, not even into an empty window: refuse, no back-off.
  auto const limit      = _space.get_max_memory();
  auto const budget     = _fraction * static_cast<double>(limit);
  auto const window_cap = budget / 2;
  if (static_cast<double>(bytes_reserved) > window_cap) { return device_prefetch_issue::too_large; }

  // 3. The parked window.  Relaxed loads: a racing release only makes the check conservative.
  if (_window->parked_splits.load(std::memory_order_relaxed) >= _max_parked_splits ||
      static_cast<double>(_window->parked_bytes.load(std::memory_order_relaxed) + bytes_reserved) >
        window_cap) {
    return device_prefetch_issue::window_full;
  }

  // 4. The consumer got here first: its own read is already under way.
  if (split->has_fallen_behind()) { return device_prefetch_issue::fell_behind; }

  // 5. Allocation, reads and every free below belong to this space's GPU.  Declared first so it
  //    outlives every buffer and the objects that may hold the last reference to them.
  rmm::cuda_set_device_raii device_guard{rmm::cuda_device_id{_space.get_device_id()}};
  auto const stream = _stream->view();

  // 6. Reserve first, so concurrent reservations are seen, then keep the usage ceiling: after
  //    this split, at least (1 - fraction) x limit must remain available.
  auto reservation = _space.make_reservation_or_null(bytes_reserved);
  if (!reservation) { return device_prefetch_issue::memory; }
  if (_space.get_available_memory() < static_cast<std::size_t>((1.0 - _fraction) * limit)) {
    reservation.reset();
    return device_prefetch_issue::memory;
  }

  // 7-8. Allocate one buffer per hint range under the reservation, attached to this thread only
  //      for the allocations; the allocated bytes stay counted until their consumer frees them.
  std::vector<prefetched_ranges::device_ranges> entries;
  {
    auto* allocator = reservation->get_memory_resource_of<cucascade::memory::Tier::GPU>();
    if (allocator == nullptr ||
        !allocator->attach_reservation_to_tracker(stream.value(), std::move(reservation))) {
      SIRIUS_LOG_WARN(
        "[readahead] could not attach the device prefetch reservation on GPU {}; skipping",
        _space.get_device_id());
      return device_prefetch_issue::error;  // not a memory shortage: no back-off
    }
    absl::Cleanup detach = [allocator, s = stream.value()] {
      allocator->reset_stream_reservation(s);
    };
    try {
      entries.resize(hints.size());
      for (std::size_t k = 0; k < hints.size(); ++k) {
        auto& buffers = entries[k].buffers;
        buffers.reserve(hints[k].ranges.size());
        for (auto const& range : hints[k].ranges) {
          buffers.emplace_back(
            static_cast<std::size_t>(range.size()), stream, _space.get_default_allocator());
        }
      }
    } catch (rmm::out_of_memory const&) {  // cucascade_out_of_memory among them
      return device_prefetch_issue::memory;
    } catch (std::bad_alloc const&) {
      return device_prefetch_issue::memory;
    }
  }

  // 9. Build everything that can throw BEFORE the slot is installed: once a consumer can see the
  //    prefetch, the only way out without issuing is the slot guard's abandon.
  std::shared_ptr<completion> fan_in;
  std::vector<std::vector<cucascade::io::slice>> slices;
  std::vector<read_callback> callbacks;
  std::shared_ptr<prefetched_ranges> ranges;
  try {
    for (auto& entry : entries) {
      entry.spans.reserve(entry.buffers.size());
      for (auto const& buffer : entry.buffers) {
        entry.spans.emplace_back(static_cast<std::uint8_t const*>(buffer.data()), buffer.size());
      }
    }
    // (a) the fan-in
    fan_in =
      std::make_shared<completion>(std::move(expected_bytes), bytes_logical, std::move(on_done));
    // (b) one vectored read per hint; zero-size ranges read nothing and get no slice
    slices.resize(hints.size());
    for (std::size_t k = 0; k < hints.size(); ++k) {
      auto const& ranges_k = hints[k].ranges;
      slices[k].reserve(ranges_k.size());
      for (std::size_t j = 0; j < ranges_k.size(); ++j) {
        if (ranges_k[j].size() == 0) { continue; }
        slices[k].emplace_back(static_cast<std::size_t>(ranges_k[j].offset()),
                               static_cast<std::size_t>(ranges_k[j].size()),
                               static_cast<std::uint8_t*>(entries[k].buffers[j].data()));
      }
    }
    // (c) the read callbacks, already of the exact type install_callback stores, so installing
    //     one is a move
    callbacks.reserve(hints.size());
    for (std::size_t k = 0; k < hints.size(); ++k) {
      callbacks.emplace_back([c = fan_in, k](read_result&& r) mutable noexcept {
        c->arrive(k, std::move(r));
        c.reset();
      });
    }
    // (d) the release hook
    auto hook = make_release_hook(_window, bytes_reserved);
    // (e) + (f): the window counts the attempt, and the object whose hook takes the count back
    //     exists right after; only its allocation can fail in between.
    _window->parked_splits.fetch_add(1, std::memory_order_relaxed);
    _window->parked_bytes.fetch_add(bytes_reserved, std::memory_order_relaxed);
    try {
      ranges = std::make_shared<prefetched_ranges>(
        _space.get_device_id(), _stream, std::move(entries), bytes_logical, std::move(hook));
    } catch (...) {
      _window->parked_splits.fetch_sub(1, std::memory_order_relaxed);
      _window->parked_bytes.fetch_sub(bytes_reserved, std::memory_order_relaxed);
      throw;
    }
    // (g) the fan-in keeps the buffers alive until the last read settled
    fan_in->keep_alive = ranges;
  } catch (std::exception const& e) {
    SIRIUS_LOG_WARN(
      "[readahead] device prefetch for '{}' not issued: {}", path_of(hints[0]), e.what());
    return device_prefetch_issue::error;
  } catch (...) {
    SIRIUS_LOG_WARN("[readahead] device prefetch for '{}' not issued: unknown exception",
                    path_of(hints[0]));
    return device_prefetch_issue::error;
  }

  // 10. Install.  The consumer's `reading` report takes no cold-read ticket of its own: it joins
  //     the scan_ticket the readahead published before this attempt.  From here every non-issued
  //     exit goes through the guard.
  if (!split->set_prefetched_ranges(ranges)) {
    SIRIUS_LOG_WARN("[readahead] device prefetch for '{}' not issued: slot already occupied",
                    path_of(hints[0]));
    return device_prefetch_issue::error;
  }
  slot_guard guard{*split, *ranges};

  // 11. The consumer may have overtaken us while we allocated: abandon (a consumer already
  //     waiting wakes at once) rather than race its read.
  if (split->has_fallen_behind()) { return device_prefetch_issue::fell_behind; }

  // 12. Submit one low-priority vectored device read per hint.  A submitted read is never
  //     arrived early: its buffers must outlive it.
  //
  //     Invariant: nothing below throws, every hint arrives exactly once, and the call ends in
  //     `issued`.  Any other exit after a submission would release the readahead ticket twice
  //     (the caller's and on_done's) or leave the fan-in short of an arrival and leak the buffers.
  for (std::size_t k = 0; k < hints.size(); ++k) {
    if (slices[k].empty()) {
      fan_in->arrive(k, read_result{std::size_t{0}});
      continue;
    }
    auto const& ds = *hints[k].datasource;
    cucascade::exec::semi_future<std::size_t> f;
    try {
      auto io_ctx = ds.io_ctx();
      // Read once: shutdown_cache() may reset it between a check and the call.
      auto* cache = io_ctx->can_use_fs_cache() ? io_ctx->cache() : nullptr;
      if (cache != nullptr) {
        f = cache->device_read_ranges_async(
          ds.get_io_object(), slices[k], stream.value(), nullptr, cucascade::io::io_priority::low);
      } else {
        f = io_ctx->device_readv_async_io(
          ds.get_io_object(), slices[k], stream.value(), cucascade::io::io_priority::low);
      }
    } catch (...) {
      // Both reads report failures through the future once submitted: a throw means nothing was.
      fan_in->arrive(k, read_result{std::current_exception()});
      continue;
    }
    // A read that handed back no future has nothing to install on and nothing anyone could wait
    // for: it is a failed hint, never an installed one.
    if (!f.valid()) {
      fan_in->arrive(k, read_result{runtime_error_ptr("device read returned no future")});
      continue;
    }
    try {
      std::move(f).install_callback(std::move(callbacks[k]));
    } catch (...) {
      // The IO is submitted.  install_callback resets the future's state only after the callback
      // is stored, so a still-valid future means the callback is NOT installed: wait for the read
      // here (the single blocking path, reachable only through an allocation failure) and arrive
      // with its real result.  A future that became invalid means the callback IS installed and
      // arrives by itself.
      try {
        SIRIUS_LOG_ERROR(
          "[readahead] install_callback failed for hint {} on GPU {}; waiting for the read inline",
          k,
          _space.get_device_id());
      } catch (...) {
      }
      // The && overload leaves the state in place when install fails, so f is still usable.
      if (f.valid()) {  // NOLINT(bugprone-use-after-move)
        fan_in->arrive(k, settle_inline(std::move(f)));
      }
    }
  }

  // 13. Issued: the fan-in owns the outcome from here.
  guard.disarm();
  fan_in->release_guard();
  return device_prefetch_issue::issued;
}

}  // namespace sirius::scan_manager
