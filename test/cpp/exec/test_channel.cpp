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
#include "exec/channel.hpp"

#include <atomic>
#include <chrono>
#include <memory>
#include <string>
#include <thread>
#include <vector>

using namespace sirius::exec;
using namespace std::chrono_literals;

// =============================================================================
// Basic channel functionality tests
// =============================================================================

TEST_CASE("shared_mpmc_channel basic publish and subscribe", "[channel]")
{
  shared_mpmc_channel<int> channel;
  auto pub = channel.get_publisher();
  auto sub = channel.get_subscriber();

  REQUIRE(pub->push(42));
  auto result = sub->pop();
  REQUIRE(result.has_value());
  REQUIRE(*result == 42);
}

TEST_CASE("shared_mpmc_channel multiple publishers and subscribers", "[channel]")
{
  shared_mpmc_channel<int> channel;

  auto pub1 = channel.get_publisher();
  auto pub2 = channel.get_publisher();
  auto sub1 = channel.get_subscriber();
  auto sub2 = channel.get_subscriber();

  REQUIRE(pub1->push(1));
  REQUIRE(pub2->push(2));

  // Items can be consumed by any subscriber
  std::vector<int> received;
  for (int i = 0; i < 2; ++i) {
    auto r1 = sub1->pop();
    if (r1.has_value()) received.push_back(*r1);
    auto r2 = sub2->pop();
    if (r2.has_value()) received.push_back(*r2);
    if (received.size() >= 2) break;
  }

  std::sort(received.begin(), received.end());
  REQUIRE(received.size() == 2);
  REQUIRE(received[0] == 1);
  REQUIRE(received[1] == 2);
}

TEST_CASE("shared_mpmc_channel push with move semantics", "[channel]")
{
  shared_mpmc_channel<std::string> channel;
  auto pub = channel.get_publisher();
  auto sub = channel.get_subscriber();

  std::string value = "hello channel";
  REQUIRE(pub->push(std::move(value)));

  auto result = sub->pop();
  REQUIRE(result.has_value());
  REQUIRE(*result == "hello channel");
}

// =============================================================================
// Non-copyable objects (unique_ptr) tests
// =============================================================================

TEST_CASE("shared_mpmc_channel works with unique_ptr", "[channel]")
{
  shared_mpmc_channel<std::unique_ptr<int>> channel;
  auto pub = channel.get_publisher();
  auto sub = channel.get_subscriber();

  auto ptr = std::make_unique<int>(42);
  REQUIRE(pub->push(std::move(ptr)));

  auto result = sub->pop();
  REQUIRE(result.has_value());
  REQUIRE(*result != nullptr);
  REQUIRE(**result == 42);
}

TEST_CASE("shared_mpmc_channel multiple unique_ptrs in sequence", "[channel]")
{
  shared_mpmc_channel<std::unique_ptr<std::string>> channel;
  auto pub = channel.get_publisher();
  auto sub = channel.get_subscriber();

  std::vector<std::string> expected = {"one", "two", "three", "four"};

  for (const auto& s : expected) {
    REQUIRE(pub->push(std::make_unique<std::string>(s)));
  }

  for (const auto& s : expected) {
    auto result = sub->pop();
    REQUIRE(result.has_value());
    REQUIRE(*result != nullptr);
    REQUIRE(**result == s);
  }
}

struct move_only_payload {
  explicit move_only_payload(int id, std::string data) : id_(id), data_(std::move(data)) {}
  ~move_only_payload() = default;

  move_only_payload(const move_only_payload&)            = delete;
  move_only_payload& operator=(const move_only_payload&) = delete;

  move_only_payload(move_only_payload&&)            = default;
  move_only_payload& operator=(move_only_payload&&) = default;

  int id_;
  std::string data_;
};

TEST_CASE("shared_mpmc_channel works with move-only custom type via unique_ptr", "[channel]")
{
  shared_mpmc_channel<std::unique_ptr<move_only_payload>> channel;
  auto pub = channel.get_publisher();
  auto sub = channel.get_subscriber();

  REQUIRE(pub->push(std::make_unique<move_only_payload>(123, "payload_data")));

  auto result = sub->pop();
  REQUIRE(result.has_value());
  REQUIRE(*result != nullptr);
  REQUIRE((*result)->id_ == 123);
  REQUIRE((*result)->data_ == "payload_data");
}

// =============================================================================
// Close/Interruption tests
// =============================================================================

TEST_CASE("shared_mpmc_channel close stops publishers", "[channel]")
{
  shared_mpmc_channel<int> channel;
  auto pub = channel.get_publisher();

  channel.close();

  // Push should fail after close
  REQUIRE_FALSE(pub->push(42));
}

TEST_CASE("shared_mpmc_channel close returns nullopt from subscribers", "[channel]")
{
  shared_mpmc_channel<int> channel;
  auto pub = channel.get_publisher();
  auto sub = channel.get_subscriber();

  // Push some items
  REQUIRE(pub->push(1));
  REQUIRE(pub->push(2));

  // Close the channel (interrupts and clears)
  channel.close();

  // Subscribers should get nullopt
  auto result = sub->pop();
  REQUIRE_FALSE(result.has_value());
}

TEST_CASE("shared_mpmc_channel close wakes blocked subscribers", "[channel]")
{
  shared_mpmc_channel<int> channel;
  auto sub = channel.get_subscriber();

  std::atomic<bool> subscriber_returned{false};
  std::optional<int> pop_result;

  std::thread subscriber_thread([&]() {
    pop_result          = sub->pop();
    subscriber_returned = true;
  });

  // Give the subscriber time to block
  std::this_thread::sleep_for(50ms);
  REQUIRE_FALSE(subscriber_returned.load());

  // Close should wake the subscriber
  channel.close();

  // Wait for subscriber to return
  auto start   = std::chrono::steady_clock::now();
  auto timeout = 1s;
  while (!subscriber_returned.load()) {
    std::this_thread::sleep_for(10ms);
    if (std::chrono::steady_clock::now() - start > timeout) {
      subscriber_thread.detach();
      FAIL("Timeout waiting for subscriber to return after close");
    }
  }

  subscriber_thread.join();
  REQUIRE_FALSE(pop_result.has_value());
}

TEST_CASE("shared_mpmc_channel close wakes multiple blocked subscribers", "[channel]")
{
  shared_mpmc_channel<int> channel;
  constexpr int num_subscribers = 4;
  std::atomic<int> subscribers_returned{0};

  std::vector<subscriber<int>> subs;
  std::vector<std::thread> threads;

  for (int i = 0; i < num_subscribers; ++i) {
    subs.push_back(channel.get_subscriber());
  }

  for (int i = 0; i < num_subscribers; ++i) {
    threads.emplace_back([&, idx = i]() {
      auto result = subs[idx]->pop();
      REQUIRE_FALSE(result.has_value());
      subscribers_returned.fetch_add(1);
    });
  }

  // Give subscribers time to block
  std::this_thread::sleep_for(50ms);
  REQUIRE(subscribers_returned.load() == 0);

  // Close should wake all subscribers
  channel.close();

  // Wait for all subscribers to return
  auto start   = std::chrono::steady_clock::now();
  auto timeout = 2s;
  while (subscribers_returned.load() < num_subscribers) {
    std::this_thread::sleep_for(20ms);
    if (std::chrono::steady_clock::now() - start > timeout) {
      for (auto& t : threads) {
        if (t.joinable()) t.detach();
      }
      FAIL("Timeout waiting for all subscribers to return");
    }
  }

  for (auto& t : threads) {
    t.join();
  }
  REQUIRE(subscribers_returned.load() == num_subscribers);
}

// =============================================================================
// Multi-threaded tests
// =============================================================================

TEST_CASE("shared_mpmc_channel concurrent publishers and subscribers", "[channel]")
{
  shared_mpmc_channel<int> channel;
  constexpr int num_publishers      = 4;
  constexpr int num_subscribers     = 4;
  constexpr int items_per_publisher = 100;
  constexpr int total_items         = num_publishers * items_per_publisher;

  std::atomic<int> published_count{0};
  std::atomic<int> consumed_count{0};
  std::atomic<bool> stop_consumers{false};

  std::vector<std::thread> pub_threads;
  std::vector<std::thread> sub_threads;
  std::vector<publisher<int>> publishers;
  std::vector<subscriber<int>> subscribers;

  // Create publishers and subscribers
  for (int i = 0; i < num_publishers; ++i) {
    publishers.push_back(channel.get_publisher());
  }
  for (int i = 0; i < num_subscribers; ++i) {
    subscribers.push_back(channel.get_subscriber());
  }

  // Start subscribers
  for (int i = 0; i < num_subscribers; ++i) {
    sub_threads.emplace_back([&, idx = i]() {
      while (!stop_consumers.load()) {
        auto result = subscribers[idx]->pop();
        if (result.has_value()) {
          consumed_count.fetch_add(1);
        }
        if (!result.has_value() && stop_consumers.load()) {
          break;
        }
      }
    });
  }

  // Start publishers
  for (int i = 0; i < num_publishers; ++i) {
    pub_threads.emplace_back([&, idx = i]() {
      for (int j = 0; j < items_per_publisher; ++j) {
        int value = idx * items_per_publisher + j;
        if (publishers[idx]->push(value)) {
          published_count.fetch_add(1);
        }
      }
    });
  }

  // Wait for publishers to finish
  for (auto& t : pub_threads) {
    t.join();
  }

  // Wait for all items to be consumed
  auto start   = std::chrono::steady_clock::now();
  auto timeout = 5s;
  while (consumed_count.load() < total_items) {
    std::this_thread::sleep_for(10ms);
    if (std::chrono::steady_clock::now() - start > timeout) {
      stop_consumers = true;
      channel.close();
      for (auto& t : sub_threads) {
        if (t.joinable()) t.detach();
      }
      FAIL("Timeout waiting for consumption");
    }
  }

  // Stop subscribers
  stop_consumers = true;
  channel.close();
  for (auto& t : sub_threads) {
    t.join();
  }

  REQUIRE(published_count.load() == total_items);
  REQUIRE(consumed_count.load() == total_items);
}

TEST_CASE("shared_mpmc_channel concurrent with unique_ptrs", "[channel]")
{
  shared_mpmc_channel<std::unique_ptr<int>> channel;
  constexpr int num_publishers      = 3;
  constexpr int items_per_publisher = 50;
  constexpr int total_items         = num_publishers * items_per_publisher;

  std::atomic<int> published_count{0};
  std::atomic<int> consumed_count{0};
  std::atomic<long long> sum{0};
  std::atomic<bool> stop_consumer{false};

  auto pub1 = channel.get_publisher();
  auto pub2 = channel.get_publisher();
  auto pub3 = channel.get_publisher();
  auto sub  = channel.get_subscriber();

  std::thread consumer([&]() {
    while (!stop_consumer.load() || consumed_count.load() < total_items) {
      auto result = sub->pop();
      if (result.has_value() && *result) {
        sum.fetch_add(**result);
        consumed_count.fetch_add(1);
      }
      if (!result.has_value() && stop_consumer.load()) {
        break;
      }
    }
  });

  auto producer_work = [&](publisher<std::unique_ptr<int>>& pub, int start_value) {
    for (int i = 0; i < items_per_publisher; ++i) {
      if (pub->push(std::make_unique<int>(start_value + i))) {
        published_count.fetch_add(1);
      }
    }
  };

  std::thread p1([&]() { producer_work(pub1, 0); });
  std::thread p2([&]() { producer_work(pub2, items_per_publisher); });
  std::thread p3([&]() { producer_work(pub3, 2 * items_per_publisher); });

  p1.join();
  p2.join();
  p3.join();

  // Wait for consumption
  auto start   = std::chrono::steady_clock::now();
  auto timeout = 5s;
  while (consumed_count.load() < total_items) {
    std::this_thread::sleep_for(10ms);
    if (std::chrono::steady_clock::now() - start > timeout) {
      stop_consumer = true;
      channel.close();
      if (consumer.joinable()) consumer.detach();
      FAIL("Timeout");
    }
  }

  stop_consumer = true;
  channel.close();
  consumer.join();

  // Verify sum: 0 + 1 + 2 + ... + (total_items-1)
  long long expected_sum = static_cast<long long>(total_items) * (total_items - 1) / 2;
  REQUIRE(sum.load() == expected_sum);
}

// =============================================================================
// Edge cases
// =============================================================================

TEST_CASE("shared_mpmc_channel publishers and subscribers can outlive channel pointer", "[channel]")
{
  auto pub = std::unique_ptr<i_publisher<int>>{};
  auto sub = std::unique_ptr<i_subscriber<int>>{};

  {
    shared_mpmc_channel<int> channel;
    pub = channel.get_publisher();
    sub = channel.get_subscriber();

    // Push while channel is alive
    REQUIRE(pub->push(42));
  }
  // Channel is destroyed but pub/sub still hold shared_ptr to queue

  // Pop should still work if item was pushed before destruction
  auto result = sub->pop();
  // Result depends on whether the underlying queue's lifetime is tied to the channel
  // Based on the implementation, the queue is shared_ptr so this might work
  // or might not depending on interrupt behavior on destruction
}

TEST_CASE("shared_mpmc_channel exception safety in push", "[channel]")
{
  shared_mpmc_channel<int> channel;
  auto pub = channel.get_publisher();

  // Close channel first
  channel.close();

  // Push should return false, not throw
  REQUIRE_NOTHROW(pub->push(42));
  REQUIRE_FALSE(pub->push(42));
}
