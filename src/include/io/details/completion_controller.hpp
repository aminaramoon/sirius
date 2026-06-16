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
#include <mutex>
#include <stop_token>

namespace sirius::io {

// ---------------------------------------------------------------------------
// completion_controller — unbounded slot tracking with one-shot completion
// ---------------------------------------------------------------------------
//
// Like admission_control, hands out RAII `slot`s and releases them on slot
// destruction — but the number of outstanding slots is unbounded, so
// acquire() never blocks.  Its purpose is to track in-flight work so callers
// can detect when every issued slot has been released ("drained").
//
// Completion is modelled as a std::stop_source: the first time the
// outstanding-slot count transitions to zero, request_stop() is called on the
// internal stop_source.  Subscribers observe completion by attaching their own
// std::stop_callback to completion_token():
//
//     std::stop_callback cb{controller.completion_token(),
//                           [] { /* all work drained */ }};
//
// std::stop_callback handles the registration race for free: if the token is
// already stopped when the callback is constructed, it fires immediately on
// the constructing thread.  request_stop() is always invoked WITHOUT holding
// the internal mutex, so callbacks may safely re-enter this controller.
//
// Completion is ONE-SHOT: once drained, the stop_token stays stopped.
// Acquiring further slots after that point is harmless (request_stop() is
// idempotent) but will not re-fire the callbacks.
//
// Spurious-completion guard: if a producer issues slots in a loop, the count
// can briefly hit zero between the first release and the next acquire, firing
// completion early.  The producer should hold a "primer" slot for the duration
// of production and release it only after all work has been enqueued.

class completion_controller {
 public:
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

  /// Token that becomes stopped the first time all outstanding slots drain.
  /// Attach a std::stop_callback to it to be notified of completion.
  [[nodiscard]] std::stop_token completion_token() const noexcept
  {
    return _completion.get_token();
  }

  /// True once completion has been signalled (all slots have drained at least
  /// once).
  [[nodiscard]] bool completed() const noexcept { return _completion.stop_requested(); }

  /// Number of slots currently outstanding.
  [[nodiscard]] size_t active() const noexcept;

 private:
  void release() noexcept;

  size_t _active_slots{0};
  mutable std::mutex _mtx;
  std::condition_variable _cv;
  std::stop_source _completion;
};

}  // namespace sirius::io
