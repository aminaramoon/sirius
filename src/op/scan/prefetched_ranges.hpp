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

#include "exec/invocable.hpp"

#include <cudf/utilities/span.hpp>

#include <rmm/cuda_stream.hpp>
#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_buffer.hpp>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <vector>

namespace sirius::op::scan {

/// A dedicated non-blocking CUDA stream on one GPU, shared by reference between the readahead
/// tier that allocates device prefetches on it and every prefetch allocated on it, so the stream
/// outlives in-flight IO and untaken buffers whichever side lets go first.
struct device_stream {
  int device{-1};
  std::unique_ptr<rmm::cuda_stream> stream;  ///< destroyed with @ref device bound

  /// Creates a non-blocking stream on @p device (bound for the call).
  [[nodiscard]] static std::shared_ptr<device_stream> create(int device);

  [[nodiscard]] rmm::cuda_stream_view view() const noexcept { return stream->view(); }

  ~device_stream();
  device_stream(device_stream const&)            = delete;
  device_stream& operator=(device_stream const&) = delete;

 private:
  device_stream() = default;
};

/// Device-resident copies of one split's hint ranges, read ahead of demand by the readahead's
/// device tier.  Entry k mirrors scan_info::fadvise_hints()[k]; buffer j of entry k holds range j
/// of that hint (the column-chunk order the hybrid scan reader wants).  Shared between the split's
/// slot and the IO completion that fills it.
class prefetched_ranges {
 public:
  enum class state : int { pending = 0, ready = 1, failed = 2 };

  /// Why the buffers left this object -- reported once to the release hook.
  enum class release_reason : std::uint8_t { consumed, unconsumed, failed, wrong_device };

  struct release_event {
    release_reason reason;
    std::size_t bytes;      ///< logical bytes (total_bytes())
    std::uint64_t wait_ns;  ///< time consumers spent in wait_for() on this object
  };

  /// Must be noexcept and touch only atomics: it may run on any thread -- an IO completion, the
  /// evictor, a pipeline thread, the readahead worker -- and must own what it touches
  /// (shared_ptr), because it can outlive whoever installed it.
  using release_hook = sirius::exec::invocable<void(release_event) noexcept>;

  struct device_ranges {
    std::vector<rmm::device_buffer> buffers;                   ///< one per hint range, hint order
    std::vector<cudf::device_span<std::uint8_t const>> spans;  ///< parallel to buffers
  };

  /// @p ranges were allocated on @p stream_owner->view() (device @p device_id).  @p on_release
  /// fires exactly once: from take() (consumed) or from the destructor (any other reason).
  prefetched_ranges(int device_id,
                    std::shared_ptr<device_stream const> stream_owner,
                    std::vector<device_ranges> ranges,
                    std::size_t total_bytes,
                    release_hook on_release) noexcept;

  /// Fires the hook if take() did not, then frees whatever take() did not move out
  /// (stream-ordered on the allocation stream, device bound).  Must not run while an IO may still
  /// write the buffers: the producer keeps a shared_ptr until its last read settled.
  ~prefetched_ranges();

  prefetched_ranges(prefetched_ranges const&)            = delete;
  prefetched_ranges& operator=(prefetched_ranges const&) = delete;

  // ---- producer side (readahead worker / IO completion thread; noexcept, host-only) ----
  //
  // Each transition leaves pending at most once and returns whether this call made it.  The
  // caller must hold a reference (shared_ptr) for the duration of the call: a waiter woken by the
  // transition may drop its own reference right away.  The producer and the waiter both do.

  /// pending -> ready; idempotent.  Call only once every write into the buffers is
  /// host-observed complete (or the allocation stream was synchronized after them): a consumer
  /// reads the bytes on its own stream right after ready, and take() only rebinds the
  /// deallocation stream -- it does not order the consumer's stream after the allocation stream.
  bool set_ready() noexcept;
  /// pending -> failed; keeps the first error.
  bool set_failed(std::exception_ptr error) noexcept;
  /// pending -> failed with "device prefetch abandoned: consumer overtook the readahead" and the
  /// abandoned flag, so a consumer already waiting wakes at once and reads for itself.
  bool abandon() noexcept;

  // ---- consumer side (pipeline thread) ----

  /// Blocks until the state leaves pending or @p timeout passes; returns the state seen and
  /// accumulates the time spent for the release event.  The caller must hold a reference
  /// (shared_ptr) while it waits.
  [[nodiscard]] state wait_for(std::chrono::milliseconds timeout) const;
  [[nodiscard]] state current() const noexcept;
  [[nodiscard]] std::exception_ptr failure() const noexcept;
  [[nodiscard]] bool abandoned() const noexcept;

  /// The consumer decodes on another GPU than the buffers live on: makes the destructor report
  /// wrong_device.  take() refuses after this.
  void flag_wrong_device() noexcept;

  /// Move the buffers out for a decode on @p consumer_stream after rebinding every buffer's
  /// deallocation stream to it.  Only when ready, exactly once, and not flagged wrong_device
  /// (otherwise nullopt).  Fires the hook with reason consumed.  The returned spans stay valid
  /// while the returned buffers live.
  ///
  /// Device contract: the calling thread's current device must be device_id() both here and when
  /// the returned buffers are destroyed -- rmm::device_buffer's destructor frees on the current
  /// device without binding its own.  Check device_id() first and flag_wrong_device() on a
  /// mismatch; as a guard, a call made on another current device flags wrong_device itself and
  /// returns nullopt.
  [[nodiscard]] std::optional<std::vector<device_ranges>> take(
    rmm::cuda_stream_view consumer_stream);
  /// The entries take() would move out, without taking them (the hook does not fire): empty
  /// unless ready, untaken and not flagged wrong_device.  For the single consumer, which may
  /// check them before committing to take(); valid until take() or destruction.
  [[nodiscard]] std::span<device_ranges const> peek() const noexcept;
  [[nodiscard]] bool taken() const noexcept;

  [[nodiscard]] std::size_t total_bytes() const noexcept { return _total_bytes; }
  [[nodiscard]] int device_id() const noexcept { return _device_id; }
  /// Entries still held: the hint count until take() moves them out, 0 after.
  [[nodiscard]] std::size_t entry_count() const noexcept;

 private:
  /// Exactly once (the _released exchange decides who reports).
  void fire_release(release_reason reason) noexcept;
  /// Leaves pending for @p next under the mutex; false when the state had already left it.
  /// With @p abandoned the abandon error is built here, only when the transition happens.
  bool settle(state next, std::exception_ptr error, bool abandoned) noexcept;

  int _device_id;
  /// Declared BEFORE _ranges, so it is destroyed after them: an untaken buffer's stream-ordered
  /// free is enqueued on a stream that still exists.
  std::shared_ptr<device_stream const> _stream_owner;
  std::size_t _total_bytes;
  release_hook _on_release;
  std::atomic<bool> _released{false};
  std::atomic<bool> _wrong_device{false};
  mutable std::atomic<std::uint64_t> _wait_ns{0};
  mutable std::mutex _mutex;
  mutable std::condition_variable _cv;
  state _state{state::pending};
  bool _abandoned{false};
  bool _taken{false};
  std::exception_ptr _failure;
  std::vector<device_ranges> _ranges;  ///< emptied by take(); declared LAST
};

}  // namespace sirius::op::scan
