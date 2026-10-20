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

// The readahead's device tier with two GPUs: each split lands on the GPU it was placed on, and
// each GPU backs off on its own.

#include "catch.hpp"
#include "op/scan/gpu_ingestible_types.hpp"
#include "op/scan/prefetched_ranges.hpp"
#include "scan/test_utils.hpp"
#include "scan_manager/readahead_scan_manager.hpp"
#include "utils/local_ioctx.hpp"
#include "utils/readahead_test_utils.hpp"
#include "utils/sirius_test_env.hpp"

#include <cudf/io/text/byte_range_info.hpp>

#include <rmm/cuda_device.hpp>
#include <rmm/cuda_stream.hpp>

#include <cucascade/cudf/datasource.hpp>
#include <cucascade/io/io_context.hpp>
#include <cucascade/memory/memory_space.hpp>

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <thread>
#include <vector>

namespace {

namespace scan = sirius::op::scan;

using sirius::scan_manager::device_prefetch_options;
using sirius::scan_manager::prefetch_outcome_kind;
using sirius::scan_manager::readahead_scan_manager;
using sirius::test::temp_data_file;
using byte_range = cudf::io::text::byte_range_info;

constexpr std::size_t MiB        = std::size_t{1} << 20;
constexpr std::size_t kFileBytes = 8 * MiB;

struct device_split : scan::scan_info {
  using scan::scan_info::scan_info;
  [[nodiscard]] bool supports_device_prefetch() const noexcept override { return true; }
};

template <class Condition>
bool eventually(Condition condition, std::chrono::seconds timeout = std::chrono::seconds(10))
{
  auto const deadline = std::chrono::steady_clock::now() + timeout;
  while (!condition()) {
    if (std::chrono::steady_clock::now() >= deadline) { return false; }
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  return true;
}

struct two_gpu_fixture {
  std::unique_ptr<sirius::memory::sirius_memory_reservation_manager> memory =
    initialize_memory_manager(2);
  temp_data_file file{kFileBytes, temp_data_file::contents::positional};
  std::shared_ptr<cucascade::io::ioctx> local_ioctx = sirius::test::make_test_local_ioctx();
  std::shared_ptr<sirius::event::query_event_publisher> publisher =
    std::make_shared<sirius::event::query_event_publisher>();
  sirius::test::single_scan_query query;

  [[nodiscard]] std::shared_ptr<readahead_scan_manager> readahead(
    std::chrono::milliseconds backoff = std::chrono::milliseconds(100))
  {
    auto manager = std::make_shared<readahead_scan_manager>(
      *publisher,
      2,
      nullptr,
      device_prefetch_options{.fraction = 1.0, .memory = memory.get(), .backoff = backoff});
    manager->prepare_for_query(query.get());
    return manager;
  }

  /// A split over [1 MiB, 3 MiB) of the file, placed on @p device.
  [[nodiscard]] std::shared_ptr<device_split> on_gpu(int device) const
  {
    auto split = std::make_shared<device_split>(std::vector<scan::scan_info::fadvise_entry>{
      {cucascade::io::open_datasource(local_ioctx, file.path.string()),
       {byte_range(1 * MiB, 2 * MiB)}}});
    split->set_preferred_device(device);
    return split;
  }

  /// Takes @p split's prefetch on its own GPU and compares the bytes with the file.
  static void require_landed_on(device_split& split, int device)
  {
    auto ranges = split.prefetched();
    REQUIRE(ranges != nullptr);
    REQUIRE(ranges->current() == scan::prefetched_ranges::state::ready);
    REQUIRE(ranges->device_id() == device);

    rmm::cuda_set_device_raii bind{rmm::cuda_device_id{device}};
    rmm::cuda_stream consumer{rmm::cuda_stream::flags::non_blocking};
    auto taken = ranges->take(consumer.view());
    REQUIRE(taken.has_value());
    REQUIRE(taken->size() == 1);
    auto const& span = taken->front().spans.at(0);
    REQUIRE(span.size() == 2 * MiB);
    cudaPointerAttributes attributes{};
    REQUIRE(cudaPointerGetAttributes(&attributes, span.data()) == cudaSuccess);
    CHECK(attributes.device == device);

    std::vector<std::uint8_t> host(span.size());
    REQUIRE(cudaMemcpyAsync(
              host.data(), span.data(), span.size(), cudaMemcpyDeviceToHost, consumer.value()) ==
            cudaSuccess);
    consumer.synchronize();
    for (std::size_t i = 0; i < host.size(); ++i) {
      if (host[i] != temp_data_file::positional_byte(1 * MiB + i)) {
        FAIL("byte " << i << " on GPU " << device << " differs from the file");
      }
    }
    taken.reset();
    consumer.synchronize();
  }
};

std::uint64_t outcomes(readahead_scan_manager const& m, prefetch_outcome_kind kind)
{
  return m.counters().outcome(kind);
}

}  // namespace

TEST_CASE("two splits stamped on GPU 0 and GPU 1 land on their own devices",
          "[multi_gpu][scan_manager][device_prefetch]")
{
  // Skip before building the fixture: it constructs a two-GPU memory manager.
  if (!sirius::test::has_gpus(2)) { return; }
  two_gpu_fixture fx;
  auto ra    = fx.readahead();
  auto first = fx.on_gpu(0);
  auto other = fx.on_gpu(1);
  ra->register_scan_task(first, 1);
  ra->register_scan_task(other, 1);
  ra->start();
  REQUIRE(eventually([&] { return outcomes(*ra, prefetch_outcome_kind::device_prefetched) == 2; }));
  CHECK(ra->counters().device_attempts.load() == 2);

  two_gpu_fixture::require_landed_on(*first, 0);
  two_gpu_fixture::require_landed_on(*other, 1);

  ra->stop();
  CHECK(eventually([&] { return ra->available_tickets_for_testing() == 2; }));
  auto const line = ra->summary();
  INFO(line);
  CHECK(line.find(" consumed=2 ") != std::string::npos);
  CHECK(line.find(" parked=0]") != std::string::npos);
}

TEST_CASE("back-off on GPU 0 does not suppress GPU 1", "[multi_gpu][scan_manager][device_prefetch]")
{
  if (!sirius::test::has_gpus(2)) { return; }
  two_gpu_fixture fx;
  auto* space0 = fx.memory->get_memory_space(cucascade::memory::Tier::GPU, 0);
  REQUIRE(space0 != nullptr);
  auto hog = space0->make_reservation_or_null(space0->get_max_memory() - 1 * MiB);
  REQUIRE(hog != nullptr);

  auto ra      = fx.readahead(std::chrono::hours(1));
  auto refused = fx.on_gpu(0);
  auto waiting = fx.on_gpu(0);
  auto landed  = fx.on_gpu(1);

  ra->register_scan_task(refused, 1);
  ra->start();
  REQUIRE(eventually([&] { return outcomes(*ra, prefetch_outcome_kind::nothing_to_issue) == 1; }));
  CHECK(ra->counters().device_skipped_memory.load() == 1);

  ra->register_scan_task(waiting, 1);
  REQUIRE(eventually([&] { return outcomes(*ra, prefetch_outcome_kind::nothing_to_issue) == 2; }));
  CHECK(ra->counters().device_skipped_backoff.load() == 1);

  ra->register_scan_task(landed, 1);
  REQUIRE(eventually([&] { return outcomes(*ra, prefetch_outcome_kind::device_prefetched) == 1; }));
  CHECK(ra->counters().device_attempts.load() == 2);
  CHECK(ra->counters().device_skipped_backoff.load() == 1);
  CHECK(refused->prefetched() == nullptr);
  CHECK(waiting->prefetched() == nullptr);
  two_gpu_fixture::require_landed_on(*landed, 1);

  ra->stop();
}
