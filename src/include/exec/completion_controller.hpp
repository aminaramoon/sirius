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
#include <memory>
#include <mutex>
#include <stop_token>

namespace sirius::exec {

// ---------------------------------------------------------------------------
// completion_token — opaque RAII handle owning a completion subscription
// ---------------------------------------------------------------------------
//
// Returned by completion_controller::on_completion().  It type-erases the
// registered callback and owns its lifetime: destroying the token deregisters
// the callback.  It has no public API — it exists only to be moved around
// (behind the returned unique_ptr) and eventually destroyed.  Internally it
// wraps a std::stop_callback, which gives the one-shot firing and the
// register-after-already-fired behaviour for free.

class completion_token {
 public:
  completion_token(completion_token const&)            = delete;
  completion_token& operator=(completion_token const&) = delete;
  ~completion_token()                                  = default;

 private:
  friend class completion_controller;
  completion_token(std::stop_token tok, std::function<void()> fn)
    : _cb(std::move(tok), std::move(fn))
  {
  }

  std::stop_callback<std::function<void()>> _cb;
};

// ---------------------------------------------------------------------------
// completion_controller — unbounded slot tracking with one-shot completion
// ---------------------------------------------------------------------------
//
// Like admission_control, hands out RAII `slot`s and releases them on slot
// destruction — but the number of outstanding slots is unbounded, so
// acquire() never blocks.  Its purpose is to track in-flight work so callers
// can detect when every issued slot has been released ("drained").
//
// Subscribers register a callback via on_completion(), which returns an owning
// completion_token handle:
//
//     auto tok = controller.on_completion([] { /* all work drained */ });
//     // ... keep `tok` alive for as long as the callback should stay armed;
//     // destroying it deregisters the callback.
//
// If completion has already fired when on_completion() is called, the callback
// runs immediately on the calling thread before on_completion() returns.
//
// Completion fires exactly once and ONLY after the producer signals it is done
// issuing work via close().  Concretely, completion is triggered when EITHER:
//   - close() is invoked while no slots are outstanding, or
//   - the last outstanding slot drains after close() has been invoked.
// Gating on close() avoids spurious early completion: a producer issuing slots
// in a loop can transiently hit zero outstanding slots between releases, but
// that does not fire completion until it has called close().
//
// As a backstop, the destructor also signals completion, so registered
// callbacks never leak un-fired.  The trigger is idempotent, so callbacks fire
// at most once regardless of which path triggers it.  It is always invoked
// WITHOUT holding the internal mutex, so callbacks may safely re-enter this
// controller.
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

  /// Register @p fn to run exactly once when completion fires (see class
  /// docs).  The returned handle owns the subscription: keep it alive for as
  /// long as the callback should stay armed; destroying it deregisters @p fn.
  /// If completion has already fired, @p fn runs immediately on the calling
  /// thread before this returns.
  [[nodiscard]] std::unique_ptr<completion_token> on_completion(std::function<void()> fn);

  /// True once completion has been signalled (all slots have drained at least
  /// once after close()).
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
