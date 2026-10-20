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
#include "op/scan/prefetched_ranges.hpp"

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <memory>

namespace cucascade::memory {
class memory_space;
}  // namespace cucascade::memory

namespace sirius::op::scan {
class scan_info;
}  // namespace sirius::op::scan

namespace sirius::scan_manager {

/// Landed-but-unconsumed device prefetches on one GPU, and what became of them.  A shared_ptr of
/// atomics: the release hooks of prefetched_ranges capture it, so it is written from any thread
/// and must outlive the readahead manager that created it.
///
/// A prefetch counts as parked from the moment the issuer installs it until its release hook
/// fires (at take() or at destruction).  On destruction the hook fires before the buffers are
/// freed, so for that short span the window already admits new bytes that the reservation
/// accounting does not yet see released -- a brief over-admission bounded by one split.
struct device_window {
  std::atomic<std::size_t> parked_splits{0};
  std::atomic<std::size_t> parked_bytes{0};  ///< reserved (256-aligned) bytes parked
  std::atomic<std::uint64_t> consumed{0};    ///< take() happened
  std::atomic<std::uint64_t> consumed_bytes{0};
  std::atomic<std::uint64_t> unconsumed{0};  ///< freed without a take (split died, mismatch, ...)
  std::atomic<std::uint64_t> failed{0};      ///< freed after failed/abandoned
  std::atomic<std::uint64_t> wrong_device{0};
  std::atomic<std::uint64_t> wait_ns{0};  ///< consumers' time in wait_for()
};

enum class device_prefetch_issue : std::uint8_t {
  issued,       ///< reads are in flight; on_done fires exactly once (possibly already did)
  unsupported,  ///< the split cannot consume device copies
  no_ranges,    ///< nothing to read (no hints / zero bytes)
  fell_behind,  ///< the consumer reached the split first (before or right after install)
  too_large,    ///< bytes_reserved > fraction x limit; no back-off
  window_full,  ///< the per-device parked window is full; no back-off
  memory,       ///< reservation refused, floor violated, or allocation OOM -> caller backs off
  error,        ///< a host-side failure before anything was submitted (logged); no back-off
};

struct device_prefetch_result {
  std::size_t bytes{0};      ///< logical bytes requested (== prefetched_ranges::total_bytes)
  bool ok{false};            ///< every read settled with the expected byte count
  std::exception_ptr error;  ///< first error when !ok
};

/// Reads one split's hint ranges straight into device buffers on one GPU under the readahead's
/// memory policy.  One per GPU, driven by the readahead worker thread; never blocks on memory or
/// IO (the one exception: a read whose completion callback could not be installed is awaited
/// inline, so its buffers are never freed under it).
///
/// Admission, with bytes_reserved = the split's ranges each rounded up to the allocation
/// alignment: refused as too_large when bytes_reserved exceeds fraction x the space's reservation
/// limit; as window_full when the parked window already holds max_parked_splits splits or would
/// exceed half that budget; as memory when the reservation is refused or would leave less than
/// (1 - fraction) x limit available.
class device_prefetch_issuer {
 public:
  device_prefetch_issuer(cucascade::memory::memory_space& space,
                         std::shared_ptr<op::scan::device_stream const> stream,
                         std::shared_ptr<device_window> window,
                         double fraction,
                         std::size_t max_parked_splits) noexcept;

  /// Attempt phase 1 for @p split.  On `issued` the split carries a pending prefetched_ranges, its
  /// prefetch_state is `prefetched`, the window counts it, and @p on_done runs exactly once (on an
  /// IO thread, or inline before this returns).  On every other value nothing stays installed on
  /// the split (an installed slot was abandoned, so no consumer can wait on it) and @p on_done is
  /// never called.
  [[nodiscard]] device_prefetch_issue issue(
    std::shared_ptr<op::scan::scan_info> const& split,
    sirius::exec::invocable<void(device_prefetch_result) noexcept> on_done);

  [[nodiscard]] int device_id() const noexcept;
  [[nodiscard]] device_window const& window() const noexcept { return *_window; }

 private:
  cucascade::memory::memory_space& _space;
  std::shared_ptr<op::scan::device_stream const> _stream;
  std::shared_ptr<device_window> _window;
  double _fraction;
  std::size_t _max_parked_splits;
};

}  // namespace sirius::scan_manager
