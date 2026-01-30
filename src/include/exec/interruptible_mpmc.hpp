/*
 * Copyright 2025, Sirius Contributors.
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

#include <blockingconcurrentqueue.h>

#include <atomic>
#include <concepts>
#include <optional>

namespace sirius::exec {

template <typename T>
class interruptible_mpmc {
 public:
  using value_type = T;

 private:
  // The underlying high-performance queue
  duckdb_moodycamel::BlockingConcurrentQueue<value_type> queue;

  // Atomic flag to manage the shutdown state
  std::atomic<bool> _is_active{true};

 public:
  interruptible_mpmc() = default;
  // Delete copy/move to prevent unsafe duplication of the internal queue
  interruptible_mpmc(const interruptible_mpmc&)            = delete;
  interruptible_mpmc& operator=(const interruptible_mpmc&) = delete;

  [[nodiscard]] bool is_open() const noexcept { return _is_active.load(std::memory_order_relaxed); }

  /**
   * \brief Pushes an item into the queue (move version).
   * \note Only available if value_type is move constructible.
   * \return Returns false if the queue has been stopped/interrupted.
   */
  [[nodiscard]] bool push(value_type item)
    requires std::move_constructible<value_type>
  {
    if (!_is_active.load(std::memory_order_relaxed)) { return false; }
    queue.enqueue(std::move(item));
    return true;
  }

  /**
   * \brief Pushes an item into the queue (copy version).
   * \note Only available if value_type is copy constructible.
   * \return Returns false if the queue has been stopped/interrupted.
   */
  [[nodiscard]] bool push(const value_type& item)
    requires std::copy_constructible<value_type>
  {
    if (!_is_active.load(std::memory_order_relaxed)) { return false; }
    queue.enqueue(item);
    return true;
  }

  /**
   * \brief Constructs an item in-place and pushes it into the queue.
   * \note Only available if value_type is constructible from Args.
   * \return Returns false if the queue has been stopped/interrupted.
   */
  template <typename... Args>
    requires std::constructible_from<value_type, Args...>
  [[nodiscard]] bool emplace(Args&&... args)
  {
    if (!_is_active.load(std::memory_order_relaxed)) { return false; }
    queue.enqueue(value_type(std::forward<Args>(args)...));
    return true;
  }

  /**
   * \brief Blocks waiting for an item.
   * \note Only available if value_type is default initializable and move constructible.
   * \return Returns std::nullopt if the queue is interrupted (shutdown).
   */
  std::optional<value_type> pop()
    requires std::default_initializable<value_type> && std::move_constructible<value_type>
  {
    value_type item;
    while (_is_active.load(std::memory_order_relaxed)) {
      if (queue.wait_dequeue_timed(item, 10000)) { return std::move(item); }
    }
    return std::nullopt;
  }

  /**
   * \brief Attempts to pop without blocking.
   * \note Only available if value_type is default initializable and move constructible.
   * \return Returns std::nullopt if the queue is empty.
   */
  std::optional<value_type> try_pop()
    requires std::default_initializable<value_type> && std::move_constructible<value_type>
  {
    value_type item;
    if (queue.try_dequeue(item)) { return std::move(item); }
    return std::nullopt;
  }

  /**
   * Interrupts the queue.
   * \brief Sets the active flag to false.
   * Consumer threads will see this flag on their next loop cycle (max 10ms delay).
   */
  void interrupt() noexcept
  {
    _is_active.store(false);
  }

  /**
   * \brief Resets the queue state to active (useful for restarting workers).
   */
  void reset() { _is_active.store(true, std::memory_order_relaxed); }
};

}  // namespace sirius::exec
