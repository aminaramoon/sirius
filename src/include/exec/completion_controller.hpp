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

namespace sirius::exec {

// ---------------------------------------------------------------------------
// completion_controller — unbounded slot tracking with one-shot completion
// ---------------------------------------------------------------------------
//
// Like admission_control, hands out RAII `slot`s and releases them on slot
// destruction — but the number of outstanding slots is unbounded, so
// acquire() never blocks.  Its purpose is to track in-flight work so callers
// can detect when every issued slot has been released ("drained").
//
// Completion is modelled as a std::stop_source.  Subscribers observe it by
// attaching their own std::stop_callback to completion_token():
//
//     std::stop_callback cb{controller.completion_token(),
//                           [] { /* all work drained */ }};
//
// std::stop_callback handles the registration race for free: if the token is
// already stopped when the callback is constructed, it fires immediately on
// the constructing thread.
//
// Completion fires exactly once and ONLY after the producer signals it is done
// issuing work via close().  Concretely, request_stop() is called when EITHER:
//   - close() is invoked while no slots are outstanding, or
//   - the last outstanding slot drains after close() has been invoked.
// Gating on close() avoids spurious early completion: a producer issuing slots
// in a loop can transiently hit zero outstanding slots between releases, but
// that does not fire completion until it has called close().
//
// As a backstop, the destructor also requests stop, so the token never leaks
// un-stopped.  request_stop() is idempotent, so the callbacks fire at most
// once regardless of which path triggers it.  It is always invoked WITHOUT
// holding the internal mutex, so callbacks may safely re-enter this controller.
//
// Acquiring further slots after completion is harmless but will not re-fire the
// callbacks.

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

  completion_controller() = default;

  ~completion_controller();

  completion_controller(completion_controller const&)            = delete;
  completion_controller& operator=(completion_controller const&) = delete;

  /// Hand out a slot, incrementing the outstanding count.  Never blocks.
  [[nodiscard]] slot acquire();

  /// Signal that no more work will be issued.  Completion fires now if no
  /// slots are outstanding, otherwise when the last one drains.  Idempotent.
  void close();

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
  bool _closing{false};
  mutable std::mutex _mtx;
  std::condition_variable _cv;
  std::stop_source _completion;
};

}  // namespace sirius::exec
