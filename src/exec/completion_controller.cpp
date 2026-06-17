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

#include <exec/completion_controller.hpp>

namespace sirius::exec {

completion_controller::~completion_controller()
{
  // Backstop for the case where the producer never called close(): make sure
  // the token is stopped so subscribers are not left waiting forever.
  // Idempotent — a no-op if completion already fired.
  _completion.request_stop();
}

completion_controller::slot completion_controller::acquire()
{
  std::lock_guard lk(_mtx);
  ++_active_slots;
  return slot{this};
}

std::unique_ptr<completion_token> completion_controller::on_completion(std::function<void()> fn)
{
  // completion_controller is a friend of completion_token, so it can reach the
  // private constructor directly; std::make_unique cannot, hence the explicit
  // new.  The std::stop_callback inside the token fires immediately on this
  // thread if completion has already been signalled.
  return std::unique_ptr<completion_token>(
    new completion_token(_completion.get_token(), std::move(fn)));
}

void completion_controller::close()
{
  bool fire;
  {
    std::lock_guard lk(_mtx);
    _closing = true;
    // Fire now only if there is nothing left to drain; otherwise release()
    // fires when the last outstanding slot goes away.
    fire = (_active_slots == 0);
  }
  if (fire) _completion.request_stop();
}

void completion_controller::release() noexcept
{
  bool fire;
  {
    std::lock_guard lk(_mtx);
    --_active_slots;
    // Only complete once the producer has closed; transient zeros before
    // close() must not fire the callbacks.
    fire = _closing && (_active_slots == 0);
  }
  // notify_all (rather than notify_one): every wait_for_idle() waiter parks on
  // this same _cv and all need to observe _active_slots dropping to zero.
  _cv.notify_all();
  // Signal completion outside the lock: request_stop() invokes any registered
  // std::stop_callbacks synchronously on this thread, and they may re-enter
  // this controller.  request_stop() is idempotent, so it fires the callbacks
  // at most once across close()/release()/the destructor.
  if (fire) _completion.request_stop();
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

}  // namespace sirius::exec
