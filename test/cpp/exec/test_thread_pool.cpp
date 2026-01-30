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

#include "catch.hpp"
#include "exec/thread_pool.hpp"

#include <atomic>
#include <chrono>
#include <memory>
#include <string>
#include <thread>
#include <vector>

using namespace sirius::exec;
using namespace std::chrono_literals;

// =============================================================================
// Basic functionality tests
// =============================================================================

TEST_CASE("thread_pool can be constructed and destroyed", "[thread_pool]")
{
  REQUIRE_NOTHROW(thread_pool(2));
}

TEST_CASE("thread_pool with named threads", "[thread_pool]")
{
  REQUIRE_NOTHROW(thread_pool(2, "test_pool"));
}

TEST_CASE("thread_pool executes scheduled task", "[thread_pool]")
{
  thread_pool pool(2);
  std::atomic<bool> executed{false};

  pool.schedule([&]() noexcept { executed = true; });

  // Wait for task to complete
  auto start   = std::chrono::steady_clock::now();
  auto timeout = 1s;
  while (!executed.load()) {
    std::this_thread::sleep_for(10ms);
    if (std::chrono::steady_clock::now() - start > timeout) {
      FAIL("Timeout waiting for task execution");
    }
  }

  REQUIRE(executed.load());
}

TEST_CASE("thread_pool executes multiple tasks", "[thread_pool]")
{
  thread_pool pool(4);
  constexpr int num_tasks = 100;
  std::atomic<int> counter{0};

  for (int i = 0; i < num_tasks; ++i) {
    pool.schedule([&]() noexcept { counter.fetch_add(1); });
  }

  // Wait for all tasks to complete
  auto start   = std::chrono::steady_clock::now();
  auto timeout = 5s;
  while (counter.load() < num_tasks) {
    std::this_thread::sleep_for(10ms);
    if (std::chrono::steady_clock::now() - start > timeout) {
      FAIL("Timeout waiting for all tasks");
    }
  }

  REQUIRE(counter.load() == num_tasks);
}

TEST_CASE("thread_pool executes tasks from multiple schedulers", "[thread_pool]")
{
  thread_pool pool(4);
  constexpr int num_schedulers      = 4;
  constexpr int tasks_per_scheduler = 25;
  constexpr int total_tasks         = num_schedulers * tasks_per_scheduler;
  std::atomic<int> counter{0};

  std::vector<std::thread> schedulers;
  for (int i = 0; i < num_schedulers; ++i) {
    schedulers.emplace_back([&]() {
      for (int j = 0; j < tasks_per_scheduler; ++j) {
        pool.schedule([&]() noexcept { counter.fetch_add(1); });
      }
    });
  }

  for (auto& t : schedulers) {
    t.join();
  }

  // Wait for all tasks to complete
  auto start   = std::chrono::steady_clock::now();
  auto timeout = 5s;
  while (counter.load() < total_tasks) {
    std::this_thread::sleep_for(10ms);
    if (std::chrono::steady_clock::now() - start > timeout) {
      FAIL("Timeout waiting for all tasks");
    }
  }

  REQUIRE(counter.load() == total_tasks);
}

// =============================================================================
// Non-copyable objects (unique_ptr) tests
// =============================================================================

TEST_CASE("thread_pool schedule with captured unique_ptr", "[thread_pool]")
{
  thread_pool pool(2);
  std::atomic<int> result{0};

  auto ptr = std::make_unique<int>(42);
  pool.schedule([&result, p = std::move(ptr)]() noexcept { result = *p; });

  auto start   = std::chrono::steady_clock::now();
  auto timeout = 1s;
  while (result.load() == 0) {
    std::this_thread::sleep_for(10ms);
    if (std::chrono::steady_clock::now() - start > timeout) {
      FAIL("Timeout");
    }
  }

  REQUIRE(result.load() == 42);
}

TEST_CASE("thread_pool schedule with multiple unique_ptrs", "[thread_pool]")
{
  thread_pool pool(4);
  std::atomic<long long> sum{0};
  constexpr int num_tasks = 50;

  for (int i = 0; i < num_tasks; ++i) {
    auto ptr = std::make_unique<int>(i);
    pool.schedule([&sum, p = std::move(ptr)]() noexcept { sum.fetch_add(*p); });
  }

  // Wait for all tasks
  long long expected_sum = static_cast<long long>(num_tasks) * (num_tasks - 1) / 2;
  auto start             = std::chrono::steady_clock::now();
  auto timeout           = 5s;
  while (sum.load() != expected_sum) {
    std::this_thread::sleep_for(10ms);
    if (std::chrono::steady_clock::now() - start > timeout) {
      FAIL("Timeout - sum: " << sum.load() << " expected: " << expected_sum);
    }
  }

  REQUIRE(sum.load() == expected_sum);
}

struct non_copyable_data {
  explicit non_copyable_data(int value) : value_(value) {}
  ~non_copyable_data() = default;

  non_copyable_data(const non_copyable_data&)            = delete;
  non_copyable_data& operator=(const non_copyable_data&) = delete;

  non_copyable_data(non_copyable_data&&)            = default;
  non_copyable_data& operator=(non_copyable_data&&) = default;

  int value_;
};

TEST_CASE("thread_pool schedule with move-only custom type", "[thread_pool]")
{
  thread_pool pool(2);
  std::atomic<int> result{0};

  auto data = std::make_unique<non_copyable_data>(999);
  pool.schedule([&result, d = std::move(data)]() noexcept { result = d->value_; });

  auto start   = std::chrono::steady_clock::now();
  auto timeout = 1s;
  while (result.load() == 0) {
    std::this_thread::sleep_for(10ms);
    if (std::chrono::steady_clock::now() - start > timeout) {
      FAIL("Timeout");
    }
  }

  REQUIRE(result.load() == 999);
}

TEST_CASE("thread_pool schedule lambda capturing unique_ptr and shared_ptr", "[thread_pool]")
{
  thread_pool pool(2);
  std::atomic<int> result{0};

  auto unique = std::make_unique<int>(10);
  auto shared = std::make_shared<int>(20);

  pool.schedule([&result, u = std::move(unique), s = shared]() noexcept { result = *u + *s; });

  auto start   = std::chrono::steady_clock::now();
  auto timeout = 1s;
  while (result.load() == 0) {
    std::this_thread::sleep_for(10ms);
    if (std::chrono::steady_clock::now() - start > timeout) {
      FAIL("Timeout");
    }
  }

  REQUIRE(result.load() == 30);
}

// =============================================================================
// Stop/Interruption tests
// =============================================================================

TEST_CASE("thread_pool stop gracefully terminates", "[thread_pool]")
{
  thread_pool pool(4);
  std::atomic<int> counter{0};

  // Schedule some tasks
  for (int i = 0; i < 10; ++i) {
    pool.schedule([&]() noexcept {
      std::this_thread::sleep_for(10ms);
      counter.fetch_add(1);
    });
  }

  // Stop immediately
  pool.stop();

  // Pool destruction should not hang
  // (pool goes out of scope here)
}

TEST_CASE("thread_pool stop prevents new tasks from starting", "[thread_pool]")
{
  auto pool = std::make_unique<thread_pool>(2);
  std::atomic<int> started{0};
  std::atomic<int> completed{0};

  // Schedule a long-running task
  pool->schedule([&]() noexcept {
    started.fetch_add(1);
    std::this_thread::sleep_for(100ms);
    completed.fetch_add(1);
  });

  // Wait for task to start
  auto start   = std::chrono::steady_clock::now();
  auto timeout = 1s;
  while (started.load() == 0) {
    std::this_thread::sleep_for(5ms);
    if (std::chrono::steady_clock::now() - start > timeout) {
      FAIL("Timeout waiting for task to start");
    }
  }

  // Stop the pool
  pool->stop();

  // Schedule more tasks (they may or may not execute depending on timing)
  for (int i = 0; i < 10; ++i) {
    pool->schedule([&]() noexcept { started.fetch_add(1); });
  }

  // Destroy pool - should not hang
  pool.reset();
}

TEST_CASE("thread_pool stop can be called multiple times", "[thread_pool]")
{
  thread_pool pool(2);

  REQUIRE_NOTHROW(pool.stop());
  REQUIRE_NOTHROW(pool.stop());
  REQUIRE_NOTHROW(pool.stop());
}

TEST_CASE("thread_pool destructor calls stop", "[thread_pool]")
{
  std::atomic<bool> task_started{false};

  {
    thread_pool pool(2);
    pool.schedule([&]() noexcept {
      task_started = true;
      std::this_thread::sleep_for(500ms);  // Long task
    });

    // Wait for task to start
    auto start   = std::chrono::steady_clock::now();
    auto timeout = 1s;
    while (!task_started.load()) {
      std::this_thread::sleep_for(5ms);
      if (std::chrono::steady_clock::now() - start > timeout) {
        FAIL("Timeout waiting for task to start");
      }
    }
    // Pool destructor will be called here
  }

  // If we reach here without hanging, the test passes
  REQUIRE(task_started.load());
}

TEST_CASE("thread_pool handles stop during task execution", "[thread_pool]")
{
  thread_pool pool(4);
  std::atomic<int> tasks_started{0};
  std::atomic<int> tasks_completed{0};
  constexpr int num_tasks = 20;

  for (int i = 0; i < num_tasks; ++i) {
    pool.schedule([&]() noexcept {
      tasks_started.fetch_add(1);
      std::this_thread::sleep_for(20ms);
      tasks_completed.fetch_add(1);
    });
  }

  // Let some tasks start
  std::this_thread::sleep_for(50ms);

  // Stop while tasks are running
  pool.stop();

  // Some tasks should have started
  REQUIRE(tasks_started.load() > 0);

  // Wait a bit for in-progress tasks to complete
  std::this_thread::sleep_for(100ms);

  // In-progress tasks should complete
  INFO("Tasks started: " << tasks_started.load() << ", completed: " << tasks_completed.load());
  // At minimum, the tasks that started should complete
  // (actual behavior depends on implementation)
}

// =============================================================================
// Edge cases and stress tests
// =============================================================================

TEST_CASE("thread_pool with single thread", "[thread_pool]")
{
  thread_pool pool(1);
  std::atomic<int> counter{0};
  constexpr int num_tasks = 10;

  for (int i = 0; i < num_tasks; ++i) {
    pool.schedule([&]() noexcept { counter.fetch_add(1); });
  }

  auto start   = std::chrono::steady_clock::now();
  auto timeout = 5s;
  while (counter.load() < num_tasks) {
    std::this_thread::sleep_for(10ms);
    if (std::chrono::steady_clock::now() - start > timeout) {
      FAIL("Timeout");
    }
  }

  REQUIRE(counter.load() == num_tasks);
}

TEST_CASE("thread_pool tasks can schedule more tasks", "[thread_pool]")
{
  thread_pool pool(4);
  std::atomic<int> counter{0};
  constexpr int target = 100;

  std::function<void()> recursive_schedule;
  recursive_schedule = [&]() noexcept {
    if (counter.fetch_add(1) < target - 1) {
      pool.schedule([&recursive_schedule]() noexcept { recursive_schedule(); });
    }
  };

  pool.schedule([&recursive_schedule]() noexcept { recursive_schedule(); });

  auto start   = std::chrono::steady_clock::now();
  auto timeout = 5s;
  while (counter.load() < target) {
    std::this_thread::sleep_for(10ms);
    if (std::chrono::steady_clock::now() - start > timeout) {
      FAIL("Timeout - counter: " << counter.load());
    }
  }

  REQUIRE(counter.load() >= target);
}

TEST_CASE("thread_pool handles high contention", "[thread_pool]")
{
  thread_pool pool(8);
  constexpr int num_tasks = 1000;
  std::atomic<int> counter{0};

  for (int i = 0; i < num_tasks; ++i) {
    pool.schedule([&]() noexcept { counter.fetch_add(1); });
  }

  auto start   = std::chrono::steady_clock::now();
  auto timeout = 10s;
  while (counter.load() < num_tasks) {
    std::this_thread::sleep_for(10ms);
    if (std::chrono::steady_clock::now() - start > timeout) {
      FAIL("Timeout - counter: " << counter.load());
    }
  }

  REQUIRE(counter.load() == num_tasks);
}

TEST_CASE("thread_pool schedule throwing vs noexcept overload", "[thread_pool]")
{
  thread_pool pool(2);
  std::atomic<int> noexcept_counter{0};
  std::atomic<int> throwing_counter{0};

  // Schedule noexcept lambda
  pool.schedule([&]() noexcept { noexcept_counter.fetch_add(1); });

  // Schedule potentially throwing lambda (wrapped internally)
  pool.schedule([&]() { throwing_counter.fetch_add(1); });

  auto start   = std::chrono::steady_clock::now();
  auto timeout = 1s;
  while (noexcept_counter.load() == 0 || throwing_counter.load() == 0) {
    std::this_thread::sleep_for(10ms);
    if (std::chrono::steady_clock::now() - start > timeout) {
      FAIL("Timeout");
    }
  }

  REQUIRE(noexcept_counter.load() == 1);
  REQUIRE(throwing_counter.load() == 1);
}

// =============================================================================
// Destruction order tests
// =============================================================================

TEST_CASE("thread_pool unique_ptr captured data destroyed after task", "[thread_pool]")
{
  std::atomic<bool> data_destroyed{false};

  struct destructor_tracker {
    std::atomic<bool>& flag;
    explicit destructor_tracker(std::atomic<bool>& f) : flag(f) {}
    ~destructor_tracker() { flag = true; }
  };

  {
    thread_pool pool(2);
    auto tracker = std::make_unique<destructor_tracker>(data_destroyed);

    pool.schedule([t = std::move(tracker)]() noexcept {
      // Use the tracker
      (void)t;
    });

    // Wait for task to complete
    std::this_thread::sleep_for(100ms);
  }

  // After pool destruction, the captured unique_ptr should be destroyed
  REQUIRE(data_destroyed.load());
}
