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

namespace sirius::io {

completion_controller::slot completion_controller::acquire()
{
  std::lock_guard lk(_mtx);
  ++_active_slots;
  return slot{this};
}

void completion_controller::release() noexcept
{
  bool drained;
  {
    std::lock_guard lk(_mtx);
    --_active_slots;
    drained = (_active_slots == 0);
  }
  // notify_all (rather than notify_one): every wait_for_idle() waiter parks on
  // this same _cv and all need to observe _active_slots dropping to zero.
  _cv.notify_all();
  // Signal completion outside the lock: request_stop() invokes any registered
  // std::stop_callbacks synchronously on this thread, and they may re-enter
  // this controller.  request_stop() is idempotent and thread-safe, so a
  // racing acquire/release that re-reaches zero is harmless.
  if (drained) _completion.request_stop();
}

void completion_controller::wait_for_idle()
{
  std::unique_lock lk(_mtx);
  _cv.wait(lk, [&] { return _active_slots == 0; });
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
