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
#include "exec/interruptible_mpmc.hpp"

#include <atomic>
#include <chrono>
#include <concepts>
#include <memory>
#include <thread>
#include <type_traits>
#include <vector>

using namespace sirius::exec;
using namespace std::chrono_literals;

// =============================================================================
// Compile-time concept verification
// =============================================================================

// Helper to check if a method exists via concepts
template <typename Q>
concept has_copy_push = requires(Q& q, const typename Q::value_type& v) {
  { q.push(v) } -> std::same_as<bool>;
};

template <typename Q>
concept has_move_push = requires(Q& q, typename Q::value_type&& v) {
  { q.push(std::move(v)) } -> std::same_as<bool>;
};

template <typename Q>
concept has_pop = requires(Q& q) {
  { q.pop() } -> std::same_as<std::optional<typename Q::value_type>>;
};

template <typename Q>
concept has_try_pop = requires(Q& q) {
  { q.try_pop() } -> std::same_as<std::optional<typename Q::value_type>>;
};

template <typename Q>
concept has_interrupt = requires(Q& q) {
  { q.interrupt() } -> std::same_as<void>;
};

// Copyable and movable type
struct copyable_movable {
  int value{0};
  copyable_movable() = default;
  copyable_movable(int v) : value(v) {}
  copyable_movable(const copyable_movable&)            = default;
  copyable_movable& operator=(const copyable_movable&) = default;
  copyable_movable(copyable_movable&&)                 = default;
  copyable_movable& operator=(copyable_movable&&)      = default;
};

// Move-only type (non-copyable)
struct move_only {
  int value{0};
  move_only() = default;
  move_only(int v) : value(v) {}
  move_only(const move_only&)            = delete;
  move_only& operator=(const move_only&) = delete;
  move_only(move_only&&)                 = default;
  move_only& operator=(move_only&&)      = default;
};

// Static assertions for copyable_movable (all methods should be available)
static_assert(has_copy_push<interruptible_mpmc<copyable_movable>>,
              "copy push should be available for copyable types");
static_assert(has_move_push<interruptible_mpmc<copyable_movable>>,
              "move push should be available for movable types");
static_assert(has_pop<interruptible_mpmc<copyable_movable>>,
              "pop should be available for default_initializable + move_constructible types");
static_assert(has_try_pop<interruptible_mpmc<copyable_movable>>,
              "try_pop should be available for default_initializable + move_constructible types");
static_assert(has_interrupt<interruptible_mpmc<copyable_movable>>,
              "interrupt should be available for default_initializable + move_constructible types");

// Static assertions for move_only (copy push should NOT be available)
static_assert(!has_copy_push<interruptible_mpmc<move_only>>,
              "copy push should NOT be available for non-copyable types");
static_assert(has_move_push<interruptible_mpmc<move_only>>,
              "move push should be available for movable types");
static_assert(has_pop<interruptible_mpmc<move_only>>,
              "pop should be available for default_initializable + move_constructible types");
static_assert(has_try_pop<interruptible_mpmc<move_only>>,
              "try_pop should be available for default_initializable + move_constructible types");
static_assert(has_interrupt<interruptible_mpmc<move_only>>,
              "interrupt should be available for default_initializable + move_constructible types");

// Static assertions for unique_ptr (move-only, default_initializable)
static_assert(!has_copy_push<interruptible_mpmc<std::unique_ptr<int>>>,
              "copy push should NOT be available for unique_ptr");
static_assert(has_move_push<interruptible_mpmc<std::unique_ptr<int>>>,
              "move push should be available for unique_ptr");
static_assert(has_pop<interruptible_mpmc<std::unique_ptr<int>>>,
              "pop should be available for unique_ptr");
static_assert(has_try_pop<interruptible_mpmc<std::unique_ptr<int>>>,
              "try_pop should be available for unique_ptr");
static_assert(has_interrupt<interruptible_mpmc<std::unique_ptr<int>>>,
              "interrupt should be available for unique_ptr");

// =============================================================================
// Basic functionality tests
// =============================================================================

TEST_CASE("interruptible_mpmc basic push and pop", "[interruptible_mpmc]")
{
  interruptible_mpmc<int> queue;

  REQUIRE(queue.is_open());
  REQUIRE(queue.push(42));

  auto result = queue.try_pop();
  REQUIRE(result.has_value());
  REQUIRE(*result == 42);
}

TEST_CASE("interruptible_mpmc push with move semantics", "[interruptible_mpmc]")
{
  interruptible_mpmc<std::string> queue;

  std::string value = "hello world";
  REQUIRE(queue.push(std::move(value)));

  auto result = queue.try_pop();
  REQUIRE(result.has_value());
  REQUIRE(*result == "hello world");
}

TEST_CASE("interruptible_mpmc push with const reference", "[interruptible_mpmc]")
{
  interruptible_mpmc<int> queue;

  const int value = 123;
  REQUIRE(queue.push(value));

  auto result = queue.try_pop();
  REQUIRE(result.has_value());
  REQUIRE(*result == 123);
}

TEST_CASE("interruptible_mpmc try_pop returns nullopt on empty queue", "[interruptible_mpmc]")
{
  interruptible_mpmc<int> queue;

  auto result = queue.try_pop();
  REQUIRE_FALSE(result.has_value());
}

TEST_CASE("interruptible_mpmc multiple items FIFO order", "[interruptible_mpmc]")
{
  interruptible_mpmc<int> queue;

  for (int i = 0; i < 10; ++i) {
    REQUIRE(queue.push(i));
  }

  for (int i = 0; i < 10; ++i) {
    auto result = queue.try_pop();
    REQUIRE(result.has_value());
    REQUIRE(*result == i);
  }
}

// =============================================================================
// Non-copyable objects (unique_ptr) tests
// =============================================================================

TEST_CASE("interruptible_mpmc works with unique_ptr (move-only types)", "[interruptible_mpmc]")
{
  interruptible_mpmc<std::unique_ptr<int>> queue;

  auto ptr = std::make_unique<int>(42);
  REQUIRE(queue.push(std::move(ptr)));

  auto result = queue.try_pop();
  REQUIRE(result.has_value());
  REQUIRE(*result != nullptr);
  REQUIRE(**result == 42);
}

TEST_CASE("interruptible_mpmc multiple unique_ptrs", "[interruptible_mpmc]")
{
  interruptible_mpmc<std::unique_ptr<std::string>> queue;

  std::vector<std::string> expected = {"alpha", "beta", "gamma", "delta"};

  for (const auto& s : expected) {
    REQUIRE(queue.push(std::make_unique<std::string>(s)));
  }

  for (const auto& s : expected) {
    auto result = queue.try_pop();
    REQUIRE(result.has_value());
    REQUIRE(*result != nullptr);
    REQUIRE(**result == s);
  }
}

struct non_copyable_resource {
  explicit non_copyable_resource(int id) : id_(id) {}
  ~non_copyable_resource() = default;

  non_copyable_resource(const non_copyable_resource&)            = delete;
  non_copyable_resource& operator=(const non_copyable_resource&) = delete;

  non_copyable_resource(non_copyable_resource&&)            = default;
  non_copyable_resource& operator=(non_copyable_resource&&) = default;

  int id_;
};

TEST_CASE("interruptible_mpmc works with non-copyable custom type", "[interruptible_mpmc]")
{
  interruptible_mpmc<std::unique_ptr<non_copyable_resource>> queue;

  REQUIRE(queue.push(std::make_unique<non_copyable_resource>(999)));

  auto result = queue.try_pop();
  REQUIRE(result.has_value());
  REQUIRE(*result != nullptr);
  REQUIRE((*result)->id_ == 999);
}

// =============================================================================
// Emplace tests
// =============================================================================

struct emplace_test_struct {
  int a {0};
  std::string b {""};
  double c {0.0};

  emplace_test_struct() = default;

  emplace_test_struct(int a_, std::string b_, double c_) : a(a_), b(std::move(b_)), c(c_) {}
};

TEST_CASE("interruptible_mpmc emplace constructs in-place", "[interruptible_mpmc]")
{
  interruptible_mpmc<emplace_test_struct> queue;

  REQUIRE(queue.emplace(42, "hello", 3.14));

  auto result = queue.try_pop();
  REQUIRE(result.has_value());
  REQUIRE(result->a == 42);
  REQUIRE(result->b == "hello");
  REQUIRE(result->c == 3.14);
}

TEST_CASE("interruptible_mpmc emplace with string", "[interruptible_mpmc]")
{
  interruptible_mpmc<std::string> queue;

  REQUIRE(queue.emplace("constructed in place"));

  auto result = queue.try_pop();
  REQUIRE(result.has_value());
  REQUIRE(*result == "constructed in place");
}

TEST_CASE("interruptible_mpmc emplace with pair", "[interruptible_mpmc]")
{
  interruptible_mpmc<std::pair<int, std::string>> queue;

  REQUIRE(queue.emplace(123, "value"));

  auto result = queue.try_pop();
  REQUIRE(result.has_value());
  REQUIRE(result->first == 123);
  REQUIRE(result->second == "value");
}

TEST_CASE("interruptible_mpmc emplace fails after interrupt", "[interruptible_mpmc]")
{
  interruptible_mpmc<std::string> queue;

  queue.interrupt();
  REQUIRE_FALSE(queue.emplace("should fail"));
}

TEST_CASE("interruptible_mpmc emplace multiple items", "[interruptible_mpmc]")
{
  interruptible_mpmc<emplace_test_struct> queue;

  for (int i = 0; i < 10; ++i) {
    REQUIRE(queue.emplace(i, "item_" + std::to_string(i), i * 1.5));
  }

  for (int i = 0; i < 10; ++i) {
    auto result = queue.try_pop();
    REQUIRE(result.has_value());
    REQUIRE(result->a == i);
    REQUIRE(result->b == "item_" + std::to_string(i));
    REQUIRE(result->c == i * 1.5);
  }
}

// =============================================================================
// Interruption tests
// =============================================================================

TEST_CASE("interruptible_mpmc interrupt closes queue", "[interruptible_mpmc]")
{
  interruptible_mpmc<int> queue;

  REQUIRE(queue.is_open());
  queue.interrupt();
  REQUIRE_FALSE(queue.is_open());
}

TEST_CASE("interruptible_mpmc push fails after interrupt", "[interruptible_mpmc]")
{
  interruptible_mpmc<int> queue;

  queue.interrupt();
  REQUIRE_FALSE(queue.push(42));
}

TEST_CASE("interruptible_mpmc interrupt clears existing items", "[interruptible_mpmc]")
{
  interruptible_mpmc<int> queue;

  // Push some items
  for (int i = 0; i < 5; ++i) {
    REQUIRE(queue.push(i));
  }

  // Interrupt clears the queue
  queue.interrupt();

  // Queue should be empty
  auto result = queue.try_pop();
  REQUIRE_FALSE(result.has_value());
}

TEST_CASE("interruptible_mpmc blocking pop returns nullopt after interrupt", "[interruptible_mpmc]")
{
  interruptible_mpmc<int> queue;
  std::atomic<bool> pop_returned{false};
  std::optional<int> pop_result;

  std::thread consumer([&]() {
    pop_result   = queue.pop();
    pop_returned = true;
  });

  // Give the consumer time to block
  std::this_thread::sleep_for(50ms);
  REQUIRE_FALSE(pop_returned.load());

  // Interrupt should unblock the consumer
  queue.interrupt();

  // Wait for consumer to return
  auto start   = std::chrono::steady_clock::now();
  auto timeout = 1s;
  while (!pop_returned.load()) {
    std::this_thread::sleep_for(10ms);
    if (std::chrono::steady_clock::now() - start > timeout) {
      consumer.detach();
      FAIL("Timeout waiting for pop to return after interrupt");
    }
  }

  consumer.join();
  REQUIRE_FALSE(pop_result.has_value());
}

TEST_CASE("interruptible_mpmc interrupt wakes up multiple blocked consumers", "[interruptible_mpmc]")
{
  interruptible_mpmc<int> queue;
  constexpr int num_consumers = 4;
  std::atomic<int> consumers_returned{0};

  std::vector<std::thread> consumers;
  for (int i = 0; i < num_consumers; ++i) {
    consumers.emplace_back([&]() {
      auto result = queue.pop();
      REQUIRE_FALSE(result.has_value());
      consumers_returned.fetch_add(1);
    });
  }

  // Give consumers time to block
  std::this_thread::sleep_for(50ms);
  REQUIRE(consumers_returned.load() == 0);

  // Interrupt should wake all consumers
  queue.interrupt();

  // Wait for all consumers to return
  auto start   = std::chrono::steady_clock::now();
  auto timeout = 2s;
  while (consumers_returned.load() < num_consumers) {
    std::this_thread::sleep_for(20ms);
    if (std::chrono::steady_clock::now() - start > timeout) {
      for (auto& c : consumers) {
        if (c.joinable()) c.detach();
      }
      FAIL("Timeout waiting for all consumers to return");
    }
  }

  for (auto& c : consumers) {
    c.join();
  }
  REQUIRE(consumers_returned.load() == num_consumers);
}

TEST_CASE("interruptible_mpmc reset after interrupt re-enables queue", "[interruptible_mpmc]")
{
  interruptible_mpmc<int> queue;

  queue.interrupt();
  REQUIRE_FALSE(queue.is_open());
  REQUIRE_FALSE(queue.push(1));

  queue.reset();
  REQUIRE(queue.is_open());
  REQUIRE(queue.push(42));

  auto result = queue.try_pop();
  REQUIRE(result.has_value());
  REQUIRE(*result == 42);
}

// =============================================================================
// Multi-threaded tests
// =============================================================================

TEST_CASE("interruptible_mpmc concurrent producers and consumers", "[interruptible_mpmc]")
{
  interruptible_mpmc<int> queue;
  constexpr int num_producers       = 4;
  constexpr int num_consumers       = 4;
  constexpr int items_per_producer  = 100;
  constexpr int total_items         = num_producers * items_per_producer;

  std::atomic<int> produced_count{0};
  std::atomic<int> consumed_count{0};

  std::vector<std::thread> producers;
  std::vector<std::thread> consumers;

  // Start consumers
  for (int i = 0; i < num_consumers; ++i) {
    consumers.emplace_back([&]() {
      while (true) {
        auto result = queue.try_pop();
        if (result.has_value()) {
          consumed_count.fetch_add(1);
        } else if (!queue.is_open()) {
          break;
        } else {
          std::this_thread::yield();
        }
      }
    });
  }

  // Start producers
  for (int i = 0; i < num_producers; ++i) {
    producers.emplace_back([&, producer_id = i]() {
      for (int j = 0; j < items_per_producer; ++j) {
        int value = producer_id * items_per_producer + j;
        while (!queue.push(value)) {
          if (!queue.is_open()) return;
          std::this_thread::yield();
        }
        produced_count.fetch_add(1);
      }
    });
  }

  // Wait for all producers to finish
  for (auto& p : producers) {
    p.join();
  }

  // Wait for all items to be consumed
  auto start   = std::chrono::steady_clock::now();
  auto timeout = 5s;
  while (consumed_count.load() < total_items) {
    std::this_thread::sleep_for(10ms);
    if (std::chrono::steady_clock::now() - start > timeout) {
      queue.interrupt();
      for (auto& c : consumers) {
        if (c.joinable()) c.detach();
      }
      FAIL("Timeout waiting for consumption");
    }
  }

  // Stop consumers
  queue.interrupt();
  for (auto& c : consumers) {
    c.join();
  }

  REQUIRE(produced_count.load() == total_items);
  REQUIRE(consumed_count.load() == total_items);
}

TEST_CASE("interruptible_mpmc concurrent push with unique_ptrs", "[interruptible_mpmc]")
{
  interruptible_mpmc<std::unique_ptr<int>> queue;
  constexpr int num_producers      = 4;
  constexpr int items_per_producer = 50;
  constexpr int total_items        = num_producers * items_per_producer;

  std::atomic<int> produced_count{0};
  std::atomic<int> consumed_count{0};
  std::atomic<long long> sum{0};

  std::vector<std::thread> producers;
  std::thread consumer([&]() {
    while (true) {
      auto result = queue.try_pop();
      if (result.has_value() && *result) {
        sum.fetch_add(**result);
        consumed_count.fetch_add(1);
      } else if (!queue.is_open() && consumed_count.load() >= total_items) {
        break;
      } else {
        std::this_thread::yield();
      }
    }
  });

  for (int i = 0; i < num_producers; ++i) {
    producers.emplace_back([&, producer_id = i]() {
      for (int j = 0; j < items_per_producer; ++j) {
        int value = producer_id * items_per_producer + j;
        while (!queue.push(std::make_unique<int>(value))) {
          if (!queue.is_open()) return;
          std::this_thread::yield();
        }
        produced_count.fetch_add(1);
      }
    });
  }

  for (auto& p : producers) {
    p.join();
  }

  // Wait for consumption
  auto start   = std::chrono::steady_clock::now();
  auto timeout = 5s;
  while (consumed_count.load() < total_items) {
    std::this_thread::sleep_for(10ms);
    if (std::chrono::steady_clock::now() - start > timeout) {
      queue.interrupt();
      if (consumer.joinable()) consumer.detach();
      FAIL("Timeout");
    }
  }

  queue.interrupt();
  consumer.join();

  // Verify sum: 0 + 1 + 2 + ... + (total_items-1) = total_items*(total_items-1)/2
  long long expected_sum = static_cast<long long>(total_items) * (total_items - 1) / 2;
  REQUIRE(sum.load() == expected_sum);
}
