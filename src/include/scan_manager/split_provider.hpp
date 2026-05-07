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

#include "scan_manager/split_connector.hpp"

#include <atomic>
#include <exception>
#include <memory>
#include <utility>
#include <vector>

namespace sirius::op {
class operator_data;
}  // namespace sirius::op

namespace sirius::scan_manager {

/**
 * @brief Abstract producer of splits for a scan operator.
 *
 * Concrete providers implement a single virtual method, @ref create_split,
 * which processes the next unit of metadata and returns the resulting splits
 * (or an empty vector when production is complete). Implementations must be
 * thread-safe — typically via an atomic counter that claims the next work
 * unit — because @ref run() invokes @ref create_split concurrently from
 * multiple worker tasks.
 *
 * The non-virtual @ref run() helper enqueues a self-fanning chain of tasks on
 * the supplied scheduler. Each task calls @ref create_split once: if it got a
 * non-empty result, it enqueues a sibling on the scheduler before pushing its
 * splits to the connector, so consecutive batches overlap. Actual concurrency
 * is whatever the scheduler runs in parallel — the provider does not pick a
 * worker count. The connector is closed in the shared @c worker_state's
 * destructor, which fires when the last task in the chain releases its
 * @c shared_ptr; any captured exception (first-writer-wins via atomic CAS) is
 * forwarded so consumers see the failure through
 * @ref split_connector::get_next_split.
 */
class split_provider {
 public:
  virtual ~split_provider() = default;

  /**
   * @brief Drive @ref create_split() to completion, dispatching work onto
   *        @p scheduler and pushing results into @p connector.
   *
   * Enqueues one task on @p scheduler and returns immediately. Each task
   * calls @ref create_split once; if it produced splits it enqueues a sibling
   * (so the next batch can run in parallel) and then pushes its splits to
   * @p connector. When the last task in the chain releases its reference to
   * the shared coordination state, the connector is closed; any exception
   * captured by a worker is passed to @c connector.close() so consumers see
   * the failure when @c split_connector::get_next_split is drained.
   *
   * @tparam Scheduler Anything with a @c enqueue(callable) method that runs
   *                   the callable asynchronously. @c static_thread_pool and
   *                   @c scoped_dispatcher both satisfy this shape.
   */
  template <typename Scheduler>
  void run(Scheduler& scheduler, split_connector& connector);

 protected:
  /**
   * @brief Process the next unit of metadata and return its splits.
   *
   * Called concurrently by @ref run()'s workers. Implementations claim the
   * next work unit with an atomic counter (no internal mutex needed) and run
   * the heavy work on the calling thread. Return an empty vector when there
   * is no more work.
   */
  virtual std::vector<std::unique_ptr<op::operator_data>> create_split() = 0;

  /**
   * @brief Push a split into a connector.
   *
   * The only entry point for enqueueing splits: @ref split_connector::push_split
   * is private and reaches in only via the @c friend relationship between
   * @ref split_connector and this class. Because the helper is a protected
   * static, only @ref split_provider and its subclasses can call it, so
   * unrelated code cannot bypass the provider abstraction.
   *
   * Defined out-of-line because the unique_ptr's deleter needs the complete
   * @c op::operator_data type, which we keep forward-declared in this header.
   */
  static void push_to_connector(split_connector& connector,
                                std::unique_ptr<op::operator_data> split);

 private:
  /// RAII coordination shared across the task chain. Destructor closes the
  /// connector with the captured exception (if any) once the last task drops
  /// its ref. Workers race to record the first error via CAS, so no mutex is
  /// needed and the destructor reads `error_ptr` after all writers have gone.
  struct worker_state {
    split_connector& connector;
    std::atomic<bool> error_set{false};
    std::exception_ptr error_ptr;

    static std::shared_ptr<worker_state> create(split_connector& connector)
    {
      return std::shared_ptr<worker_state>(new worker_state(connector));
    }

    void set_error(std::exception_ptr eptr)
    {
      bool expected = false;
      if (error_set.compare_exchange_strong(expected, true)) { error_ptr = std::move(eptr); }
    }

    ~worker_state()
    {
      // close()-side exceptions are swallowed because destructors must not
      // propagate; close() is best-effort wakeup.
      try {
        connector.close(error_ptr);
      } catch (...) {
      }
    }

   private:
    explicit worker_state(split_connector& c) : connector(c) {}
  };

  template <typename Scheduler>
  void schedule_worker(Scheduler& scheduler, std::shared_ptr<worker_state> state);
};

template <typename Scheduler>
void split_provider::run(Scheduler& scheduler, split_connector& connector)
{
  schedule_worker(scheduler, worker_state::create(connector));
}

template <typename Scheduler>
void split_provider::schedule_worker(Scheduler& scheduler, std::shared_ptr<worker_state> state)
{
  scheduler.enqueue([this, state = std::move(state), &scheduler]() {
    try {
      auto splits = create_split();
      if (splits.empty()) { return; }
      // Spawn a sibling so the next batch can run while we push ours.
      schedule_worker(scheduler, state);
      for (auto& split : splits) {
        push_to_connector(state->connector, std::move(split));
      }
    } catch (...) {
      state->set_error(std::current_exception());
    }
  });
}

}  // namespace sirius::scan_manager
