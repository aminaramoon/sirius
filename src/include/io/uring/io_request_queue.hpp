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

// Unified MPSC request queue for uring_reactor.
//
// Wraps two moodycamel ConcurrentQueues (device and host requests) behind a
// single interface: try_dequeue_*, empty(), notify(), and wait().  Producers
// call the try_enqueue_* methods, which bump the sequence counter on success;
// the worker thread dequeues directly via try_dequeue_* and uses wait() /
// notify() for parking.

#pragma once

#include "io/types.hpp"

#include <concurrentqueue.h>

#include <atomic>
#include <cstdint>

namespace sirius::io {

template <typename NativeHandle>
class io_request_queue {
 public:
  using device_req = device_read_req<NativeHandle>;
  using host_req   = host_read_req<NativeHandle>;

  // --- Producer API -------------------------------------------------------

  template <typename It>
  bool try_enqueue_device_bulk(std::move_iterator<It> first, size_t n)
  {
    if (!_device.enqueue_bulk(first, n)) return false;
    notify();
    return true;
  }

  bool try_enqueue_host(host_req r)
  {
    if (!_host.enqueue(std::move(r))) return false;
    notify();
    return true;
  }

  template <typename It>
  bool try_enqueue_host_bulk(std::move_iterator<It> first, size_t n)
  {
    if (!_host.enqueue_bulk(first, n)) return false;
    notify();
    return true;
  }

  // --- Consumer API -------------------------------------------------------

  bool try_dequeue_device(device_req& out) { return _device.try_dequeue(out); }
  bool try_dequeue_host(host_req& out) { return _host.try_dequeue(out); }

  [[nodiscard]] bool empty() const noexcept
  {
    return _device.size_approx() == 0 && _host.size_approx() == 0;
  }

  // --- Synchronization ----------------------------------------------------

  [[nodiscard]] uint64_t current_seq() const noexcept
  {
    return _seq.load(std::memory_order_acquire);
  }

  // Park until the sequence changes from `seq`.  Returns immediately if it
  // already changed — no lost-wakeup.
  void wait(uint64_t seq) noexcept { _seq.wait(seq, std::memory_order_relaxed); }

  // Bump the sequence and wake one waiter.  Called by producers after a
  // successful enqueue and by cuda_copy_cb after a H2D copy completes.
  void notify() noexcept
  {
    _seq.fetch_add(1, std::memory_order_release);
    _seq.notify_one();
  }

 private:
  duckdb_moodycamel::ConcurrentQueue<device_req> _device;
  duckdb_moodycamel::ConcurrentQueue<host_req> _host;
  std::atomic<uint64_t> _seq{0};
};

}  // namespace sirius::io
