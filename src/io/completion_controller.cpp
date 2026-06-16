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

#include "io/details/completion_controller.hpp"

#include <utility>

namespace sirius::io {

completion_controller::slot completion_controller::acquire()
{
  std::lock_guard lk(_mtx);
  ++_active_slots;
  return slot{this};
}

void completion_controller::release() noexcept
{
  completion_fn fire;
  {
    std::lock_guard lk(_mtx);
    --_active_slots;
    if (_active_slots == 0) {
      // Copy (not move) the callback so it survives repeated drains.
      fire = _on_completion;
    }
  }
  // notify_all (rather than notify_one): every wait_for_idle() waiter parks on
  // this same _cv and all need to observe _active_slots dropping to zero.
  _cv.notify_all();
  // Invoke the completion callback outside the lock so it may re-enter this
  // controller (e.g. acquire new slots) without deadlocking.
  if (fire) fire();
}

void completion_controller::wait_for_idle()
{
  std::unique_lock lk(_mtx);
  _cv.wait(lk, [&] { return _active_slots == 0; });
}

void completion_controller::on_completion(completion_fn fn)
{
  completion_fn fire;
  {
    std::lock_guard lk(_mtx);
    _on_completion = std::move(fn);
    if (_on_completion && _active_slots == 0) fire = _on_completion;
  }
  // Fire outside the lock, mirroring release()'s reentrancy guarantee.
  if (fire) fire();
}

size_t completion_controller::active() const noexcept
{
  std::lock_guard lk(_mtx);
  return _active_slots;
}

completion_controller::slot::~slot()
{
  if (_ctrl) _ctrl->release();
}

completion_controller::slot::slot(slot&& o) noexcept : _ctrl(o._ctrl) { o._ctrl = nullptr; }

completion_controller::slot& completion_controller::slot::operator=(slot&& o) noexcept
{
  if (this != &o) {
    if (_ctrl) _ctrl->release();
    _ctrl   = o._ctrl;
    o._ctrl = nullptr;
  }
  return *this;
}

}  // namespace sirius::io
