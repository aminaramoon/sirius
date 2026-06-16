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

#include <condition_variable>
#include <cstddef>
#include <functional>
#include <mutex>

namespace sirius::io {

// ---------------------------------------------------------------------------
// completion_controller — unbounded slot tracking with completion callback
// ---------------------------------------------------------------------------
//
// Like admission_control, hands out RAII `slot`s and releases them on slot
// destruction — but the number of outstanding slots is unbounded, so
// acquire() never blocks.  Its purpose is to track in-flight work so callers
// can either:
//   - block in wait_for_idle() until every slot has been released, or
//   - register an on_completion callback that fires when the count of
//     outstanding slots drops to zero.
//
// The on_completion callback is invoked each time the outstanding-slot count
// transitions to zero while a callback is set.  If a callback is installed
// when no slots are outstanding, it fires immediately.  Callbacks are always
// invoked without holding the internal mutex, so it is safe for them to
// acquire new slots or call back into this controller.

class completion_controller {
 public:
  using completion_fn = std::function<void()>;

  class slot {
   public:
    slot() = default;
    ~slot();

    slot(slot&& o) noexcept;
    slot& operator=(slot&& o) noexcept;

    slot(slot const&)            = delete;
    slot& operator=(slot const&) = delete;

    /// True if this slot holds a live reservation.
    explicit operator bool() const noexcept { return _ctrl != nullptr; }

   private:
    friend class completion_controller;
    explicit slot(completion_controller* ctrl) noexcept : _ctrl(ctrl) {}

    completion_controller* _ctrl{nullptr};
  };

  completion_controller()  = default;
  ~completion_controller() = default;

  completion_controller(completion_controller const&)            = delete;
  completion_controller& operator=(completion_controller const&) = delete;

  /// Hand out a slot, incrementing the outstanding count.  Never blocks.
  [[nodiscard]] slot acquire();

  /// Block until every slot handed out by @c acquire has been destroyed.
  void wait_for_idle();

  /// Install (or replace) the callback invoked when the outstanding-slot
  /// count reaches zero.  If no slots are currently outstanding, the callback
  /// fires immediately on the calling thread.  Pass an empty function to
  /// clear the callback.
  void on_completion(completion_fn fn);

  /// Number of slots currently outstanding.
  [[nodiscard]] size_t active() const noexcept;

 private:
  void release() noexcept;

  size_t _active_slots{0};
  completion_fn _on_completion;
  mutable std::mutex _mtx;
  std::condition_variable _cv;
};

}  // namespace sirius::io
