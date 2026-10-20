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

// The readahead's device tier on its own: reserving under the fraction, allocating device
// buffers, issuing the vectored device reads and settling them through the fan-in, against a real
// uring ioctx with and without a host cache.

#include "catch.hpp"
#include "exec/query_lifecycle_registry.hpp"
#include "op/scan/gpu_ingestible_types.hpp"
#include "op/scan/prefetched_ranges.hpp"
#include "scan/test_utils.hpp"
#include "scan_manager/device_prefetch.hpp"
#include "scan_manager/sirius_scan_manager.hpp"
#include "utils/local_ioctx.hpp"
#include "utils/parquet_test_sources.hpp"
#include "utils/readahead_test_utils.hpp"

#include <cudf/io/text/byte_range_info.hpp>

#include <rmm/cuda_stream.hpp>

#include <cucascade/cudf/datasource.hpp>
#include <cucascade/exec/semi_future.hpp>
#include <cucascade/io/io_context.hpp>
#include <cucascade/io/types.hpp>
#include <cucascade/memory/memory_space.hpp>

#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <future>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace {

namespace scan = sirius::op::scan;

using sirius::scan_manager::device_prefetch_issue;
using sirius::scan_manager::device_prefetch_issuer;
using sirius::scan_manager::device_prefetch_result;
using sirius::scan_manager::device_window;
using sirius::test::temp_data_file;
using byte_range     = cudf::io::text::byte_range_info;
using prefetch_state = scan::scan_info::prefetch_state;

constexpr std::size_t MiB        = std::size_t{1} << 20;
constexpr std::size_t kFileBytes = 8 * MiB;

/// A split that can decode from device copies of its hint ranges.
struct device_split : scan::scan_info {
  using scan::scan_info::scan_info;
  [[nodiscard]] bool supports_device_prefetch() const noexcept override { return true; }
};

/// The default hints: two ranges of one file, 3 MiB in all.
std::vector<scan::scan_info::fadvise_entry> two_ranges(
  std::shared_ptr<cucascade::io::datasource> const& ds)
{
  return {{ds, {byte_range(0, 1 * MiB), byte_range(3 * MiB, 2 * MiB)}}};
}

/// An ioctx whose every read fails, reported through the returned future.
class failing_context final : public cucascade::io::ioctx {
 public:
  ~failing_context() override { pre_destroy(); }

  [[nodiscard]] cucascade::io::io_context_type type() const noexcept override
  {
    return cucascade::io::io_context_type::restful;
  }
  void shutdown() noexcept override {}
  [[nodiscard]] bool supports(std::string_view) const noexcept override { return true; }
  [[nodiscard]] bool supports_device_read() const noexcept override { return true; }
  [[nodiscard]] bool supports_host_to_device_read() const noexcept override { return false; }
  [[nodiscard]] bool supports_vector_host_read() const noexcept override { return false; }
  [[nodiscard]] bool supports_device_range_read() const noexcept override { return true; }
  [[nodiscard]] bool prefers_bulk_io() const noexcept override { return false; }

  [[nodiscard]] std::vector<cucascade::io::byte_range> align_and_coalesce(
    std::span<cucascade::io::byte_range const> ranges,
    std::optional<std::size_t>) const noexcept override
  {
    return {ranges.begin(), ranges.end()};
  }

  std::size_t host_read_io(cucascade::io::io_object const&,
                           std::size_t,
                           std::size_t,
                           std::uint8_t*) override
  {
    throw std::runtime_error("injected device read failure");
  }

  cucascade::exec::semi_future<std::size_t> mixed_readv_async_io(
    cucascade::io::io_object const&,
    std::vector<cucascade::io::prepared_io_slice>&&) noexcept override
  {
    return cucascade::exec::make_semi_future<std::size_t>(
      std::make_exception_ptr(std::runtime_error("injected device read failure")));
  }

 protected:
  std::shared_ptr<cucascade::io::io_object> create_io_object(std::string path) override
  {
    return std::make_shared<sirius::test::bulk_object>(std::move(path), kFileBytes);
  }
};

/// What on_done reported, and whether it ran at all.
struct done_probe {
  std::promise<device_prefetch_result> promise;
  std::future<device_prefetch_result> future = promise.get_future();
  std::shared_ptr<std::atomic<int>> calls    = std::make_shared<std::atomic<int>>(0);

  sirius::exec::invocable<void(device_prefetch_result) noexcept> callback()
  {
    return [p = std::move(promise), calls = calls](device_prefetch_result result) mutable noexcept {
      calls->fetch_add(1);
      p.set_value(std::move(result));
    };
  }

  device_prefetch_result wait()
  {
    REQUIRE(future.wait_for(std::chrono::seconds(30)) == std::future_status::ready);
    return future.get();
  }
};

std::string message_of(std::exception_ptr const& error)
{
  try {
    std::rethrow_exception(error);
  } catch (std::exception const& e) {
    return e.what();
  } catch (...) {
    return "unknown";
  }
}

struct issuer_fixture {
  std::unique_ptr<sirius::memory::sirius_memory_reservation_manager> memory =
    initialize_memory_manager(1);
  cucascade::memory::memory_space* space = memory->get_memory_space(Tier::GPU, 0);
  temp_data_file file{kFileBytes, temp_data_file::contents::positional};
  std::shared_ptr<scan::device_stream> tier         = scan::device_stream::create(0);
  std::shared_ptr<device_window> window             = std::make_shared<device_window>();
  std::shared_ptr<cucascade::io::ioctx> local_ioctx = sirius::test::make_test_local_ioctx();

  issuer_fixture() { REQUIRE(space != nullptr); }

  [[nodiscard]] device_prefetch_issuer issuer(double fraction = 1.0, std::size_t max_parked = 4)
  {
    return device_prefetch_issuer(*space, tier, window, fraction, max_parked);
  }

  [[nodiscard]] std::shared_ptr<cucascade::io::datasource> local_datasource() const
  {
    return cucascade::io::open_datasource(local_ioctx, file.path.string());
  }

  /// Copies the device bytes of @p span back and compares them with the file at @p offset.
  void require_file_bytes(cudf::device_span<std::uint8_t const> span, std::size_t offset) const
  {
    std::vector<std::uint8_t> host(span.size());
    REQUIRE(cudaMemcpy(host.data(), span.data(), span.size(), cudaMemcpyDeviceToHost) ==
            cudaSuccess);
    for (std::size_t i = 0; i < host.size(); ++i) {
      if (host[i] != temp_data_file::positional_byte(offset + i)) {
        FAIL("byte " << i << " of the range at " << offset << " differs from the file");
      }
    }
  }
};

}  // namespace

TEST_CASE_METHOD(issuer_fixture,
                 "issue reads the hint ranges into device buffers",
                 "[scan_manager][device_prefetch]")
{
  // Declared before the datasource: the manager owns the cached ioctx it reads through.
  sirius::exec::query_lifecycle_registry lifecycle;
  std::unique_ptr<sirius::scan_manager::sirius_scan_manager> manager;
  std::shared_ptr<cucascade::io::datasource> ds;
  SECTION("without a host cache: the ioctx's vectored device read")
  {
    ds = local_datasource();
    REQUIRE_FALSE(ds->uses_fs_cache());
  }
  SECTION("through the fs_cache")
  {
    manager = std::make_unique<sirius::scan_manager::sirius_scan_manager>(
      lifecycle,
      sirius::test::dispose_on_idle_config(),
      *memory,
      sirius::test::single_gpu_index_for_prefetch());
    REQUIRE(manager->io_ctx()->cache() != nullptr);
    ds = manager->create_datasource(file.path.string());
    REQUIRE(ds->uses_fs_cache());
  }

  auto split  = std::make_shared<device_split>(two_ranges(ds));
  auto issuer = this->issuer();
  done_probe done;
  REQUIRE(issuer.issue(split, done.callback()) == device_prefetch_issue::issued);
  CHECK(split->get_prefetch_state() == prefetch_state::prefetched);

  auto const result = done.wait();
  CHECK(result.ok);
  CHECK(result.bytes == 3 * MiB);
  CHECK(result.error == nullptr);
  CHECK(done.calls->load() == 1);

  auto ranges = split->prefetched();
  REQUIRE(ranges != nullptr);
  CHECK(ranges->current() == scan::prefetched_ranges::state::ready);
  CHECK(ranges->device_id() == 0);
  CHECK(ranges->total_bytes() == 3 * MiB);
  CHECK(window->parked_splits.load() == 1);
  CHECK(window->parked_bytes.load() == 3 * MiB);

  rmm::cuda_stream consumer{rmm::cuda_stream::flags::non_blocking};
  auto taken = ranges->take(consumer.view());
  REQUIRE(taken.has_value());
  REQUIRE(taken->size() == 1);
  auto const& entry = taken->front();
  REQUIRE(entry.spans.size() == 2);
  CHECK(entry.spans[0].size() == 1 * MiB);
  CHECK(entry.spans[1].size() == 2 * MiB);
  require_file_bytes(entry.spans[0], 0);
  require_file_bytes(entry.spans[1], 3 * MiB);

  CHECK(window->parked_splits.load() == 0);
  CHECK(window->parked_bytes.load() == 0);
  CHECK(window->consumed.load() == 1);
  CHECK(window->consumed_bytes.load() == 3 * MiB);

  taken.reset();
  consumer.synchronize();
}

TEST_CASE_METHOD(issuer_fixture,
                 "an unsupported split is left to the host tier",
                 "[scan_manager][device_prefetch]")
{
  auto split  = std::make_shared<scan::scan_info>(two_ranges(local_datasource()));
  auto issuer = this->issuer();
  done_probe done;
  CHECK(issuer.issue(split, done.callback()) == device_prefetch_issue::unsupported);
  CHECK(split->prefetched() == nullptr);
  CHECK(split->get_prefetch_state() == prefetch_state::idle);
  CHECK(done.calls->load() == 0);

  SECTION("a split with nothing to read is not issued either")
  {
    auto empty = std::make_shared<device_split>(std::vector<scan::scan_info::fadvise_entry>{});
    CHECK(issuer.issue(empty, done.callback()) == device_prefetch_issue::no_ranges);
    CHECK(empty->prefetched() == nullptr);
  }
}

TEST_CASE_METHOD(issuer_fixture,
                 "a split too large for the fraction is refused without back-off",
                 "[scan_manager][device_prefetch]")
{
  auto split  = std::make_shared<device_split>(two_ranges(local_datasource()));
  auto issuer = this->issuer(1e-9);
  done_probe done;
  CHECK(issuer.issue(split, done.callback()) == device_prefetch_issue::too_large);
  CHECK(split->prefetched() == nullptr);
  CHECK(split->get_prefetch_state() == prefetch_state::idle);
  CHECK(window->parked_splits.load() == 0);
}

TEST_CASE_METHOD(issuer_fixture,
                 "memory pressure is a memory refusal",
                 "[scan_manager][device_prefetch]")
{
  // Reservations are admitted against the space's reservation limit: leave 1 MiB of it.
  auto const available = space->get_available_memory();
  auto hog             = space->make_reservation_or_null(space->get_max_memory() - 1 * MiB);
  REQUIRE(hog != nullptr);

  auto split  = std::make_shared<device_split>(two_ranges(local_datasource()));
  auto issuer = this->issuer();
  done_probe done;
  CHECK(issuer.issue(split, done.callback()) == device_prefetch_issue::memory);
  CHECK(split->prefetched() == nullptr);
  CHECK(split->get_prefetch_state() == prefetch_state::idle);
  CHECK(window->parked_splits.load() == 0);
  CHECK(done.calls->load() == 0);

  hog.reset();
  CHECK(space->get_available_memory() == available);
}

TEST_CASE_METHOD(issuer_fixture, "a full window refuses", "[scan_manager][device_prefetch]")
{
  auto ds     = local_datasource();
  auto first  = std::make_shared<device_split>(two_ranges(ds));
  auto second = std::make_shared<device_split>(two_ranges(ds));
  auto issuer = this->issuer(1.0, 1);

  done_probe first_done;
  REQUIRE(issuer.issue(first, first_done.callback()) == device_prefetch_issue::issued);
  CHECK(first_done.wait().ok);

  done_probe second_done;
  CHECK(issuer.issue(second, second_done.callback()) == device_prefetch_issue::window_full);
  CHECK(second->prefetched() == nullptr);
  CHECK(window->parked_splits.load() == 1);

  // The first split dies unconsumed: its slot releases the window.
  first.reset();
  CHECK(window->parked_splits.load() == 0);
  CHECK(window->parked_bytes.load() == 0);
  CHECK(window->unconsumed.load() == 1);
  tier->view().synchronize();
}

TEST_CASE_METHOD(issuer_fixture,
                 "a split the consumer reached is not issued",
                 "[scan_manager][device_prefetch]")
{
  auto split = std::make_shared<device_split>(two_ranges(local_datasource()));
  split->set_scan_stage(cucascade::io::cache::scan_stage::preparing);
  auto issuer = this->issuer();
  done_probe done;
  CHECK(issuer.issue(split, done.callback()) == device_prefetch_issue::fell_behind);
  CHECK(split->prefetched() == nullptr);
  CHECK(split->get_prefetch_state() == prefetch_state::idle);
  CHECK(window->parked_splits.load() == 0);
}

namespace {

/// Falls behind on the second check -- the one after the slot is installed -- and only once a
/// consumer has taken the slot, so that consumer is waiting when the issuer abandons.
struct overtaken_split final : device_split {
  using device_split::device_split;
  mutable std::atomic<int> checks{0};

  [[nodiscard]] bool has_fallen_behind() const noexcept override
  {
    if (checks.fetch_add(1) == 0) { return false; }
    auto const deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
    while (prefetched() != nullptr && std::chrono::steady_clock::now() < deadline) {
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
    return true;
  }
};

}  // namespace

TEST_CASE_METHOD(issuer_fixture,
                 "a consumer that overtakes after install is woken by abandon",
                 "[scan_manager][device_prefetch]")
{
  auto split  = std::make_shared<overtaken_split>(two_ranges(local_datasource()));
  auto issuer = this->issuer();

  std::optional<scan::prefetched_ranges::state> seen;
  bool abandoned = false;
  std::chrono::steady_clock::duration waited{};
  std::thread consumer([&] {
    auto const deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
    std::shared_ptr<scan::prefetched_ranges> taken;
    while (!taken && std::chrono::steady_clock::now() < deadline) {
      taken = split->take_prefetched_ranges();
      if (!taken) { std::this_thread::sleep_for(std::chrono::milliseconds(1)); }
    }
    if (!taken) { return; }
    auto const start = std::chrono::steady_clock::now();
    seen             = taken->wait_for(std::chrono::seconds(5));
    waited           = std::chrono::steady_clock::now() - start;
    abandoned        = taken->abandoned();
  });

  done_probe done;
  auto const outcome = issuer.issue(split, done.callback());
  consumer.join();

  CHECK(outcome == device_prefetch_issue::fell_behind);
  CHECK(split->checks.load() == 2);
  REQUIRE(seen.has_value());
  CHECK(*seen == scan::prefetched_ranges::state::failed);
  CHECK(abandoned);
  // Woken by the abandon, not by the timeout.
  CHECK(waited < std::chrono::seconds(1));
  CHECK(split->get_prefetch_state() == prefetch_state::attempted);
  CHECK(split->prefetched() == nullptr);
  CHECK(window->parked_splits.load() == 0);
  CHECK(window->parked_bytes.load() == 0);
  CHECK(window->failed.load() == 1);
  CHECK(done.calls->load() == 0);
  tier->view().synchronize();
}

TEST_CASE_METHOD(issuer_fixture, "a short read is a failure", "[scan_manager][device_prefetch]")
{
  // Crosses the end of the 8 MiB file: the backend clamps it and still succeeds.
  auto split  = std::make_shared<device_split>(std::vector<scan::scan_info::fadvise_entry>{
    {local_datasource(), {byte_range(7 * MiB, 2 * MiB)}}});
  auto issuer = this->issuer();
  done_probe done;
  REQUIRE(issuer.issue(split, done.callback()) == device_prefetch_issue::issued);
  auto const result = done.wait();
  CHECK_FALSE(result.ok);
  CHECK(result.bytes == 2 * MiB);
  REQUIRE(result.error != nullptr);
  CHECK_THAT(message_of(result.error), Catch::Matchers::ContainsSubstring("short device read"));
  auto ranges = split->prefetched();
  REQUIRE(ranges != nullptr);
  CHECK(ranges->current() == scan::prefetched_ranges::state::failed);
  CHECK_FALSE(ranges->abandoned());

  ranges.reset();
  split->discard_prefetched_ranges();
  CHECK(window->failed.load() == 1);
  CHECK(window->parked_splits.load() == 0);
  tier->view().synchronize();
}

TEST_CASE_METHOD(issuer_fixture,
                 "an errored read is a failure, never a throw",
                 "[scan_manager][device_prefetch]")
{
  auto failing = std::make_shared<failing_context>();
  std::shared_ptr<cucascade::io::datasource> ds =
    cucascade::io::open_datasource(failing, file.path.string());
  auto split  = std::make_shared<device_split>(two_ranges(ds));
  auto issuer = this->issuer();
  done_probe done;
  device_prefetch_issue outcome{};
  REQUIRE_NOTHROW(outcome = issuer.issue(split, done.callback()));
  REQUIRE(outcome == device_prefetch_issue::issued);
  auto const result = done.wait();
  CHECK_FALSE(result.ok);
  REQUIRE(result.error != nullptr);
  CHECK_THAT(message_of(result.error),
             Catch::Matchers::ContainsSubstring("injected device read failure"));
  auto ranges = split->prefetched();
  REQUIRE(ranges != nullptr);
  CHECK(ranges->current() == scan::prefetched_ranges::state::failed);
  tier->view().synchronize();
}

TEST_CASE_METHOD(issuer_fixture,
                 "device memory returns once the buffers are dropped",
                 "[scan_manager][device_prefetch]")
{
  auto const baseline = space->get_available_memory();
  {
    auto split  = std::make_shared<device_split>(two_ranges(local_datasource()));
    auto issuer = this->issuer();
    done_probe done;
    REQUIRE(issuer.issue(split, done.callback()) == device_prefetch_issue::issued);
    REQUIRE(done.wait().ok);
    // Landed and parked: the buffers stay counted after the reservation was released.
    CHECK(space->get_available_memory() + 3 * MiB <= baseline + 1 * MiB);

    rmm::cuda_stream consumer{rmm::cuda_stream::flags::non_blocking};
    auto taken = split->prefetched()->take(consumer.view());
    REQUIRE(taken.has_value());
    taken.reset();
    consumer.synchronize();
  }
  tier->view().synchronize();
  auto const after = space->get_available_memory();
  CHECK(after + 1 * MiB >= baseline);
  CHECK(after <= baseline + 1 * MiB);
}
