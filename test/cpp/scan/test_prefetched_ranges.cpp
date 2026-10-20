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

#include "catch.hpp"
#include "op/scan/prefetched_ranges.hpp"
#include "scan/test_utils.hpp"

#include <rmm/cuda_stream.hpp>
#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_buffer.hpp>
#include <rmm/mr/callback_memory_resource.hpp>

#include <cuda_runtime_api.h>

#include <cucascade/memory/memory_space.hpp>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <new>
#include <stdexcept>
#include <string>
#include <thread>
#include <tuple>
#include <utility>
#include <vector>

using sirius::op::scan::device_stream;
using sirius::op::scan::prefetched_ranges;

namespace {

using release_reason = prefetched_ranges::release_reason;

constexpr std::size_t kBufferBytes = 1ull << 20;
constexpr std::size_t kEntries     = 2;
constexpr std::size_t kPerEntry    = 2;

/// What the release hook saw, shared with the hook so it outlives the prefetch.
struct release_counts {
  std::array<std::atomic<int>, 4> by_reason{};
  std::atomic<std::uint64_t> wait_ns{0};
  std::atomic<std::size_t> bytes{0};

  [[nodiscard]] int of(release_reason reason) const
  {
    return by_reason[static_cast<std::size_t>(reason)].load();
  }
  [[nodiscard]] int total() const
  {
    int sum = 0;
    for (auto const& count : by_reason) {
      sum += count.load();
    }
    return sum;
  }
};

prefetched_ranges::release_hook counting_hook(std::shared_ptr<release_counts> counts)
{
  return [counts = std::move(counts)](prefetched_ranges::release_event event) noexcept {
    counts->by_reason[static_cast<std::size_t>(event.reason)].fetch_add(1);
    counts->wait_ns.store(event.wait_ns);
    counts->bytes.store(event.bytes);
  };
}

std::uint8_t pattern_of(std::size_t entry, std::size_t range)
{
  return static_cast<std::uint8_t>(0x10 * (entry + 1) + range + 1);
}

struct prefetched_ranges_fixture {
  std::unique_ptr<sirius::memory::sirius_memory_reservation_manager> memory =
    initialize_memory_manager(1);
  cucascade::memory::memory_space* space = memory->get_memory_space(Tier::GPU, 0);
  std::shared_ptr<device_stream> tier    = device_stream::create(0);
  std::shared_ptr<release_counts> counts = std::make_shared<release_counts>();
  /// Where make_ranges() put each buffer, entry-major.
  std::vector<void const*> addresses;

  /// kEntries x kPerEntry buffers of kBufferBytes, each filled with its own pattern on the tier
  /// stream, which is synchronized so the bytes are there before anyone calls set_ready.
  std::shared_ptr<prefetched_ranges> make_ranges()
  {
    REQUIRE(space != nullptr);
    std::vector<prefetched_ranges::device_ranges> ranges(kEntries);
    for (std::size_t k = 0; k < kEntries; ++k) {
      for (std::size_t j = 0; j < kPerEntry; ++j) {
        rmm::device_buffer buffer(kBufferBytes, tier->view(), space->get_default_allocator());
        REQUIRE(
          cudaMemsetAsync(buffer.data(), pattern_of(k, j), kBufferBytes, tier->view().value()) ==
          cudaSuccess);
        addresses.push_back(buffer.data());
        ranges[k].spans.emplace_back(static_cast<std::uint8_t const*>(buffer.data()),
                                     buffer.size());
        ranges[k].buffers.push_back(std::move(buffer));
      }
    }
    tier->view().synchronize();
    return std::make_shared<prefetched_ranges>(
      0, tier, std::move(ranges), kEntries * kPerEntry * kBufferBytes, counting_hook(counts));
  }
};

}  // namespace

TEST_CASE_METHOD(prefetched_ranges_fixture,
                 "take returns the buffers once, rebinds their stream and reports consumed",
                 "[scan][prefetched_ranges]")
{
  rmm::cuda_stream other{rmm::cuda_stream::flags::non_blocking};
  auto ranges = make_ranges();
  CHECK(ranges->entry_count() == kEntries);
  CHECK(ranges->device_id() == 0);
  CHECK(ranges->total_bytes() == kEntries * kPerEntry * kBufferBytes);

  // Not ready yet: nothing to take or peek at, and the attempt does not consume the object.
  CHECK_FALSE(ranges->take(other.view()).has_value());
  CHECK(ranges->peek().empty());
  CHECK(counts->total() == 0);

  CHECK(ranges->set_ready());
  CHECK_FALSE(ranges->set_ready());  // idempotent
  CHECK(ranges->current() == prefetched_ranges::state::ready);

  // Peeking shows the entries in place without consuming them.
  auto const peeked = ranges->peek();
  REQUIRE(peeked.size() == kEntries);
  CHECK(peeked[0].spans.front().data() == addresses.front());
  CHECK_FALSE(ranges->taken());
  CHECK(counts->total() == 0);

  auto taken = ranges->take(other.view());
  REQUIRE(taken.has_value());
  CHECK(ranges->taken());
  CHECK(ranges->entry_count() == 0);
  CHECK(counts->of(release_reason::consumed) == 1);
  CHECK(counts->total() == 1);
  CHECK(counts->bytes.load() == kEntries * kPerEntry * kBufferBytes);

  REQUIRE(taken->size() == kEntries);
  for (std::size_t k = 0; k < kEntries; ++k) {
    auto& entry = (*taken)[k];
    REQUIRE(entry.buffers.size() == kPerEntry);
    REQUIRE(entry.spans.size() == kPerEntry);
    for (std::size_t j = 0; j < kPerEntry; ++j) {
      CHECK(entry.buffers[j].data() == addresses[k * kPerEntry + j]);
      CHECK(entry.spans[j].data() == entry.buffers[j].data());
      CHECK(entry.spans[j].size() == kBufferBytes);
      CHECK(entry.buffers[j].stream() == other.view());
      std::vector<std::uint8_t> host(kBufferBytes);
      REQUIRE(
        cudaMemcpy(host.data(), entry.spans[j].data(), kBufferBytes, cudaMemcpyDeviceToHost) ==
        cudaSuccess);
      CHECK(std::ranges::all_of(host, [&](std::uint8_t b) { return b == pattern_of(k, j); }));
    }
  }

  CHECK_FALSE(ranges->take(other.view()).has_value());
  CHECK(ranges->peek().empty());
  ranges.reset();
  CHECK(counts->of(release_reason::consumed) == 1);
  CHECK(counts->total() == 1);

  // The consumer's free is stream-ordered on its own stream.
  taken.reset();
  other.synchronize();
}

TEST_CASE_METHOD(prefetched_ranges_fixture,
                 "a failed prefetch is reported, never taken, and released as failed",
                 "[scan][prefetched_ranges]")
{
  auto ranges = make_ranges();
  CHECK(ranges->set_failed(std::make_exception_ptr(std::runtime_error("first"))));
  CHECK_FALSE(ranges->set_failed(std::make_exception_ptr(std::runtime_error("second"))));
  CHECK_FALSE(ranges->set_ready());
  CHECK_FALSE(ranges->abandon());
  CHECK(ranges->current() == prefetched_ranges::state::failed);
  CHECK_FALSE(ranges->abandoned());
  REQUIRE(ranges->failure() != nullptr);
  CHECK_THROWS_WITH(std::rethrow_exception(ranges->failure()), "first");
  CHECK(ranges->wait_for(std::chrono::milliseconds(0)) == prefetched_ranges::state::failed);

  CHECK_FALSE(ranges->take(tier->view()).has_value());
  CHECK_FALSE(ranges->taken());
  CHECK(counts->total() == 0);

  ranges.reset();
  CHECK(counts->of(release_reason::failed) == 1);
  CHECK(counts->total() == 1);
}

TEST_CASE_METHOD(prefetched_ranges_fixture,
                 "abandon wakes a waiting consumer immediately",
                 "[scan][prefetched_ranges]")
{
  using clock = std::chrono::steady_clock;
  auto ranges = make_ranges();

  std::atomic<bool> waiting{false};
  prefetched_ranges::state seen{prefetched_ranges::state::pending};
  clock::time_point t_call{};
  clock::time_point woke{};
  std::thread consumer([&] {
    waiting.store(true);
    t_call = clock::now();
    seen   = ranges->wait_for(std::chrono::seconds(30));
    woke   = clock::now();
  });
  while (!waiting.load()) {
    std::this_thread::yield();
  }
  // Give the consumer time to block inside wait_for.  On a loaded machine the abandon may still
  // land before the consumer waits; that run then covers the already-settled path rather than
  // the notify_all wake-up, which is best-effort coverage here -- the bounds below hold either way.
  std::this_thread::sleep_for(std::chrono::milliseconds(50));
  auto const abandoned_at = clock::now();
  CHECK(ranges->abandon());
  CHECK_FALSE(ranges->abandon());
  consumer.join();

  CHECK(seen == prefetched_ranges::state::failed);
  // Far below the 30 s timeout: the abandon woke the consumer, the timeout did not.
  CHECK(woke - abandoned_at < std::chrono::seconds(10));
  CHECK(ranges->abandoned());
  REQUIRE(ranges->failure() != nullptr);
  CHECK_THROWS_WITH(std::rethrow_exception(ranges->failure()),
                    Catch::Matchers::ContainsSubstring("abandoned"));
  CHECK_FALSE(ranges->take(tier->view()).has_value());

  ranges.reset();
  CHECK(counts->of(release_reason::failed) == 1);
  CHECK(counts->total() == 1);
  auto const waited_ns = static_cast<std::uint64_t>(
    std::chrono::duration_cast<std::chrono::nanoseconds>(woke - t_call).count());
  CHECK(counts->wait_ns.load() > 0);
  CHECK(counts->wait_ns.load() <= waited_ns);
}

TEST_CASE_METHOD(prefetched_ranges_fixture,
                 "flag_wrong_device is reported at destruction",
                 "[scan][prefetched_ranges]")
{
  auto ranges = make_ranges();
  ranges->set_ready();
  ranges->flag_wrong_device();
  CHECK_FALSE(ranges->take(tier->view()).has_value());
  CHECK(counts->total() == 0);

  ranges.reset();
  CHECK(counts->of(release_reason::wrong_device) == 1);
  CHECK(counts->total() == 1);
}

TEST_CASE_METHOD(prefetched_ranges_fixture,
                 "take refuses a prefetch stamped for another device than the current one",
                 "[scan][prefetched_ranges]")
{
  // Stamped device 1 while the current device is 0.  No buffers, so no device-1 memory is needed
  // and the destructor has nothing to free on device 1 (it must not bind it either).
  REQUIRE(cudaSetDevice(0) == cudaSuccess);
  auto ranges = std::make_shared<prefetched_ranges>(
    1, tier, std::vector<prefetched_ranges::device_ranges>{}, 0, counting_hook(counts));
  ranges->set_ready();

  CHECK_FALSE(ranges->take(tier->view()).has_value());
  CHECK_FALSE(ranges->taken());
  CHECK(counts->total() == 0);

  ranges.reset();
  CHECK(counts->of(release_reason::wrong_device) == 1);
  CHECK(counts->total() == 1);
  int current = -1;
  CHECK(cudaGetDevice(&current) == cudaSuccess);
  CHECK(current == 0);
}

TEST_CASE_METHOD(prefetched_ranges_fixture,
                 "wait_for times out while pending and the wait is accounted",
                 "[scan][prefetched_ranges]")
{
  auto ranges      = make_ranges();
  auto const start = std::chrono::steady_clock::now();
  CHECK(ranges->wait_for(std::chrono::milliseconds(20)) == prefetched_ranges::state::pending);
  CHECK(std::chrono::steady_clock::now() - start >= std::chrono::milliseconds(20));
  CHECK(ranges->current() == prefetched_ranges::state::pending);

  ranges.reset();
  CHECK(counts->of(release_reason::unconsumed) == 1);
  CHECK(counts->total() == 1);
  CHECK(counts->wait_ns.load() >= 10'000'000ull);
}

TEST_CASE_METHOD(prefetched_ranges_fixture,
                 "the stream outlives the tier",
                 "[scan][prefetched_ranges]")
{
  // Each free records what its stream looked like: cudaSuccess / cudaErrorNotReady while the
  // stream exists, an error once it was destroyed -- which would mean _stream_owner was released
  // before _ranges.
  std::vector<cudaError_t> stream_at_free;
  rmm::mr::callback_memory_resource tracking(
    [](std::size_t bytes, rmm::cuda_stream_view, void*) -> void* {
      void* ptr = nullptr;
      if (cudaMalloc(&ptr, bytes) != cudaSuccess) { throw std::bad_alloc(); }
      return ptr;
    },
    [&stream_at_free](void* ptr, std::size_t, rmm::cuda_stream_view stream, void*) {
      stream_at_free.push_back(cudaStreamQuery(stream.value()));
      std::ignore = cudaFree(ptr);
    });

  std::vector<prefetched_ranges::device_ranges> entries(1);
  for (std::size_t j = 0; j < kPerEntry; ++j) {
    rmm::device_buffer buffer(kBufferBytes, tier->view(), tracking);
    entries[0].spans.emplace_back(static_cast<std::uint8_t const*>(buffer.data()), buffer.size());
    entries[0].buffers.push_back(std::move(buffer));
  }
  auto ranges = std::make_shared<prefetched_ranges>(
    0, tier, std::move(entries), kPerEntry * kBufferBytes, counting_hook(counts));
  std::weak_ptr<device_stream> stream = tier;

  // The tier lets go first; the prefetch still owns its allocation stream.
  tier.reset();
  CHECK_FALSE(stream.expired());
  ranges->set_ready();

  // The untaken buffers are freed on that stream, and only then is the stream destroyed.
  ranges.reset();
  CHECK(stream.expired());
  REQUIRE(stream_at_free.size() == kPerEntry);
  for (auto const status : stream_at_free) {
    CHECK((status == cudaSuccess || status == cudaErrorNotReady));
  }
  CHECK(counts->of(release_reason::unconsumed) == 1);
  CHECK(counts->total() == 1);
  CHECK(cudaDeviceSynchronize() == cudaSuccess);
  CHECK(cudaGetLastError() == cudaSuccess);
}

TEST_CASE_METHOD(prefetched_ranges_fixture,
                 "the destructor frees untaken buffers",
                 "[scan][prefetched_ranges]")
{
  REQUIRE(space != nullptr);
  auto const baseline = space->get_available_memory();
  auto ranges         = make_ranges();
  auto const held     = kEntries * kPerEntry * kBufferBytes;
  CHECK(space->get_available_memory() + held <= baseline + (1ull << 20));
  ranges->set_ready();

  ranges.reset();
  tier->view().synchronize();
  auto const after = space->get_available_memory();
  CHECK(after + (1ull << 20) >= baseline);
  CHECK(after <= baseline + (1ull << 20));
  CHECK(counts->of(release_reason::unconsumed) == 1);
  CHECK(counts->total() == 1);
}
