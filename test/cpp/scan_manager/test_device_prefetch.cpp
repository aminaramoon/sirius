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
#include <functional>
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
using byte_range = cudf::io::text::byte_range_info;

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

using read_future = cucascade::exec::semi_future<std::size_t>;

/// An ioctx with no storage behind it whose reads settle as the test scripts them: every read
/// returns the future @p respond makes from the logical bytes it asked for.  Nothing is written
/// into the destination buffers.
class scripted_context final : public cucascade::io::ioctx {
 public:
  using responder = std::function<read_future(std::size_t requested)>;

  explicit scripted_context(responder respond) : _respond(std::move(respond)) {}
  ~scripted_context() override { pre_destroy(); }

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

  read_future mixed_readv_async_io(
    cucascade::io::io_object const&,
    std::vector<cucascade::io::prepared_io_slice>&& slices) noexcept override
  {
    try {
      std::size_t requested = 0;
      for (auto const& slice : slices) {
        requested += slice.rng.size;
      }
      return _respond(requested);
    } catch (...) {
      return cucascade::exec::make_semi_future<std::size_t>(std::current_exception());
    }
  }

 protected:
  std::shared_ptr<cucascade::io::io_object> create_io_object(std::string path) override
  {
    return std::make_shared<sirius::test::bulk_object>(std::move(path), kFileBytes);
  }

 private:
  responder _respond;
};

/// Every read fails with @p message, settled before the read call returns.
std::shared_ptr<scripted_context> failing_context(
  std::string const& message = "injected device read failure")
{
  return std::make_shared<scripted_context>([message](std::size_t) {
    return cucascade::exec::make_semi_future<std::size_t>(
      std::make_exception_ptr(std::runtime_error(message)));
  });
}

/// Every read reports the bytes it asked for, settled before the read call returns.
std::shared_ptr<scripted_context> instant_context()
{
  return std::make_shared<scripted_context>([](std::size_t requested) {
    return cucascade::exec::make_semi_future<std::size_t>(requested);
  });
}

/// One read that stays in flight until the test opens the gate.
struct read_gate {
  cucascade::exec::promise<std::size_t> promise;
  std::atomic<std::size_t> requested{0};
  std::atomic<int> reads{0};

  /// Completes the read with the bytes it asked for.
  void open() { promise.set_value(requested.load()); }
  /// Fails the read, as an IO error would.
  void fail()
  {
    promise.set_exception(
      std::make_exception_ptr(std::runtime_error("injected device read failure")));
  }
};

std::shared_ptr<scripted_context> gated_context(std::shared_ptr<read_gate> const& gate)
{
  return std::make_shared<scripted_context>([gate](std::size_t requested) {
    gate->requested.store(requested);
    gate->reads.fetch_add(1);
    return gate->promise.get_semi_future();
  });
}

/// Every read hands back an invalid future: nothing to install a callback on.
std::shared_ptr<scripted_context> no_future_context()
{
  return std::make_shared<scripted_context>([](std::size_t) { return read_future{}; });
}

/// Release hooks run on whichever thread drops the last reference: poll for their effects.
template <class Condition>
bool eventually(Condition condition, std::chrono::seconds timeout = std::chrono::seconds(5))
{
  auto const deadline = std::chrono::steady_clock::now() + timeout;
  while (!condition()) {
    if (std::chrono::steady_clock::now() >= deadline) { return false; }
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  return true;
}

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
  double fraction = 0.0;
  SECTION("larger than the whole budget") { fraction = 1e-9; }
  SECTION("larger than the parked window's half of the budget")
  {
    // A 4 MiB budget parks at most 2 MiB: the 3 MiB split fits the budget, never the window.
    fraction = static_cast<double>(4 * MiB) / static_cast<double>(space->get_max_memory());
  }
  auto split  = std::make_shared<device_split>(two_ranges(local_datasource()));
  auto issuer = this->issuer(fraction);
  done_probe done;
  CHECK(issuer.issue(split, done.callback()) == device_prefetch_issue::too_large);
  CHECK(split->prefetched() == nullptr);
  CHECK(window->parked_splits.load() == 0);
  CHECK(done.calls->load() == 0);
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
  CHECK(window->parked_splits.load() == 0);
  CHECK(done.calls->load() == 0);

  hog.reset();
  CHECK(space->get_available_memory() == available);
}

TEST_CASE_METHOD(issuer_fixture,
                 "a reservation that would break the usage floor is a memory refusal",
                 "[scan_manager][device_prefetch]")
{
  // With fraction 0.5 at least half the limit must stay available.  Leave the floor plus 1 MiB:
  // the 3 MiB reservation itself is granted, but would take the space below the floor.
  auto const limit     = space->get_max_memory();
  auto const available = space->get_available_memory();
  auto const floor     = static_cast<std::size_t>(0.5 * static_cast<double>(limit));
  REQUIRE(available > floor + 1 * MiB);
  auto hog = space->make_reservation_or_null(available - floor - 1 * MiB);
  REQUIRE(hog != nullptr);
  REQUIRE(space->get_available_memory() < floor + 3 * MiB);
  REQUIRE(space->make_reservation_or_null(3 * MiB) != nullptr);

  auto split  = std::make_shared<device_split>(two_ranges(local_datasource()));
  auto issuer = this->issuer(0.5);
  done_probe done;
  CHECK(issuer.issue(split, done.callback()) == device_prefetch_issue::memory);
  CHECK(split->prefetched() == nullptr);
  CHECK(window->parked_splits.load() == 0);
  CHECK(window->parked_bytes.load() == 0);
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
  CHECK(eventually([&] { return window->unconsumed.load() == 1; }));
  CHECK(window->parked_splits.load() == 0);
  CHECK(window->parked_bytes.load() == 0);
  tier->view().synchronize();
}

TEST_CASE_METHOD(issuer_fixture,
                 "a split whose slot is occupied is not issued",
                 "[scan_manager][device_prefetch]")
{
  auto split    = std::make_shared<device_split>(two_ranges(local_datasource()));
  auto occupant = std::make_shared<scan::prefetched_ranges>(
    0, tier, std::vector<scan::prefetched_ranges::device_ranges>{}, 0, nullptr);
  REQUIRE(split->set_prefetched_ranges(occupant));

  auto issuer = this->issuer();
  done_probe done;
  CHECK(issuer.issue(split, done.callback()) == device_prefetch_issue::error);
  CHECK(split->prefetched() == occupant);
  CHECK(done.calls->load() == 0);
  CHECK(window->parked_splits.load() == 0);
  CHECK(window->parked_bytes.load() == 0);
  CHECK(window->unconsumed.load() == 1);
  CHECK(window->failed.load() == 0);

  split->discard_prefetched_ranges();
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
  CHECK(window->parked_splits.load() == 0);
}

namespace {

/// Falls behind on the second check -- the one after the slot is installed -- and only once a
/// consumer took the slot and is about to wait on it, so the issuer abandons under that consumer.
struct overtaken_split final : device_split {
  using device_split::device_split;
  mutable std::atomic<int> checks{0};
  std::atomic<bool> consumer_waiting{false};  ///< set by the consumer just before wait_for()

  [[nodiscard]] bool has_fallen_behind() const noexcept override
  {
    if (checks.fetch_add(1) == 0) { return false; }
    auto const deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
    while (!consumer_waiting.load() && std::chrono::steady_clock::now() < deadline) {
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
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
  std::thread consumer([&] {
    auto const deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
    std::shared_ptr<scan::prefetched_ranges> taken;
    while (!taken && std::chrono::steady_clock::now() < deadline) {
      taken = split->take_prefetched_ranges();
      if (!taken) { std::this_thread::sleep_for(std::chrono::milliseconds(1)); }
    }
    if (!taken) { return; }
    split->consumer_waiting.store(true);
    seen      = taken->wait_for(std::chrono::seconds(30));
    abandoned = taken->abandoned();
  });

  done_probe done;
  auto const outcome = issuer.issue(split, done.callback());
  consumer.join();

  CHECK(outcome == device_prefetch_issue::fell_behind);
  CHECK(split->checks.load() == 2);
  REQUIRE(seen.has_value());
  CHECK(*seen == scan::prefetched_ranges::state::failed);
  CHECK(abandoned);
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
  CHECK(eventually([&] { return window->failed.load() == 1; }));
  CHECK(window->parked_splits.load() == 0);
  CHECK(window->parked_bytes.load() == 0);
  tier->view().synchronize();
}

TEST_CASE_METHOD(issuer_fixture,
                 "an errored read is a failure, never a throw",
                 "[scan_manager][device_prefetch]")
{
  std::shared_ptr<cucascade::io::datasource> ds =
    cucascade::io::open_datasource(failing_context(), file.path.string());
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
                 "one errored hint fails the prefetch once every hint arrived",
                 "[scan_manager][device_prefetch]")
{
  auto const path = file.path.string();
  auto issuer     = this->issuer();
  done_probe done;

  SECTION("the errored read settles inline, the good one later")
  {
    auto gate  = std::make_shared<read_gate>();
    auto split = std::make_shared<device_split>(std::vector<scan::scan_info::fadvise_entry>{
      {cucascade::io::open_datasource(failing_context(), path), {byte_range(0, 1 * MiB)}},
      {cucascade::io::open_datasource(gated_context(gate), path), {byte_range(3 * MiB, 2 * MiB)}}});
    REQUIRE(issuer.issue(split, done.callback()) == device_prefetch_issue::issued);
    REQUIRE(gate->reads.load() == 1);
    // The second hint is still in flight: nothing is settled or reported yet.
    CHECK(done.calls->load() == 0);
    auto ranges = split->prefetched();
    REQUIRE(ranges != nullptr);
    CHECK(ranges->current() == scan::prefetched_ranges::state::pending);

    gate->open();
    auto const result = done.wait();
    CHECK_FALSE(result.ok);
    CHECK(result.bytes == 3 * MiB);
    REQUIRE(result.error != nullptr);
    CHECK(message_of(result.error) == "injected device read failure");
    CHECK(ranges->current() == scan::prefetched_ranges::state::failed);
    CHECK_FALSE(ranges->abandoned());
  }

  SECTION("every read settles inline: reported once, with the first error, before issue returns")
  {
    auto split = std::make_shared<device_split>(std::vector<scan::scan_info::fadvise_entry>{
      {cucascade::io::open_datasource(instant_context(), path), {byte_range(0, 1 * MiB)}},
      {cucascade::io::open_datasource(failing_context("first failure"), path),
       {byte_range(1 * MiB, 1 * MiB)}},
      {cucascade::io::open_datasource(failing_context("second failure"), path),
       {byte_range(3 * MiB, 1 * MiB)}}});
    REQUIRE(issuer.issue(split, done.callback()) == device_prefetch_issue::issued);
    CHECK(done.calls->load() == 1);
    auto const result = done.wait();
    CHECK_FALSE(result.ok);
    CHECK(result.bytes == 3 * MiB);
    REQUIRE(result.error != nullptr);
    CHECK(message_of(result.error) == "first failure");
    REQUIRE(split->prefetched() != nullptr);
    CHECK(split->prefetched()->current() == scan::prefetched_ranges::state::failed);
  }

  // The split died with its section: the failed prefetch was released once, as failed.
  CHECK(done.calls->load() == 1);
  CHECK(eventually([&] { return window->failed.load() == 1; }));
  CHECK(window->parked_splits.load() == 0);
  CHECK(window->unconsumed.load() == 0);
  tier->view().synchronize();
}

TEST_CASE_METHOD(issuer_fixture,
                 "a read that returns no future fails its hint once, never the call",
                 "[scan_manager][device_prefetch]")
{
  auto const path = file.path.string();
  auto split      = std::make_shared<device_split>(std::vector<scan::scan_info::fadvise_entry>{
    {cucascade::io::open_datasource(instant_context(), path), {byte_range(0, 1 * MiB)}},
    {cucascade::io::open_datasource(no_future_context(), path), {byte_range(3 * MiB, 2 * MiB)}}});
  auto issuer     = this->issuer();
  done_probe done;
  device_prefetch_issue outcome{};
  REQUIRE_NOTHROW(outcome = issuer.issue(split, done.callback()));
  REQUIRE(outcome == device_prefetch_issue::issued);
  auto const result = done.wait();
  CHECK(done.calls->load() == 1);
  CHECK_FALSE(result.ok);
  CHECK(result.bytes == 3 * MiB);
  REQUIRE(result.error != nullptr);
  CHECK_THAT(message_of(result.error), Catch::Matchers::ContainsSubstring("no future"));
  REQUIRE(split->prefetched() != nullptr);
  CHECK(split->prefetched()->current() == scan::prefetched_ranges::state::failed);
  CHECK_FALSE(split->prefetched()->abandoned());

  split.reset();
  CHECK(eventually([&] { return window->failed.load() == 1; }));
  CHECK(window->parked_splits.load() == 0);
  CHECK(window->parked_bytes.load() == 0);
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

// ---- through the readahead worker ---------------------------------------------------------------

namespace {

using cucascade::io::cache::scan_stage;
using sirius::scan_manager::device_prefetch_options;
using sirius::scan_manager::prefetch_outcome_kind;
using sirius::scan_manager::readahead_scan_manager;

/// A readahead over one scan operator (operator id 1) whose device tier runs on the fixture's
/// memory manager.
struct readahead_fixture : issuer_fixture {
  std::shared_ptr<sirius::event::query_event_publisher> publisher =
    std::make_shared<sirius::event::query_event_publisher>();
  sirius::test::single_scan_query query;

  [[nodiscard]] device_prefetch_options options(double fraction = 1.0) const
  {
    return {.fraction = fraction, .memory = memory.get()};
  }

  [[nodiscard]] std::shared_ptr<readahead_scan_manager> readahead(device_prefetch_options opts,
                                                                  std::size_t budget = 2)
  {
    auto manager = std::make_shared<readahead_scan_manager>(*publisher, budget, nullptr, opts);
    manager->prepare_for_query(query.get());
    return manager;
  }

  [[nodiscard]] static std::shared_ptr<device_split> on_gpu0(
    std::vector<scan::scan_info::fadvise_entry> hints)
  {
    auto split = std::make_shared<device_split>(std::move(hints));
    split->set_preferred_device(0);
    return split;
  }
};

std::uint64_t outcomes(readahead_scan_manager const& m, prefetch_outcome_kind kind)
{
  return m.counters().outcome(kind);
}

/// Once its only operator is closed and drained, the worker takes no further slot: its last idle
/// poll returns the one it took and the worker exits.  From then on the tickets out are exactly the
/// ones the splits' attempts and reads hold, so the test can count them.
bool worker_settled(readahead_scan_manager const& m, int available)
{
  return m.counters().operators_drained.load() == 1 && m.counters().idle_polls.load() >= 1 &&
         m.available_tickets_for_testing() == available;
}

}  // namespace

TEST_CASE_METHOD(readahead_fixture,
                 "the worker lands a split on the device ahead of its consumer",
                 "[scan_manager][device_prefetch][readahead]")
{
  auto ra    = readahead(options());
  auto split = on_gpu0(two_ranges(local_datasource()));
  ra->register_scan_task(split, 1);
  ra->mark_operator_closed(1);
  ra->start();
  REQUIRE(eventually([&] { return outcomes(*ra, prefetch_outcome_kind::device_prefetched) == 1; }));

  auto ranges = split->prefetched();
  REQUIRE(ranges != nullptr);
  CHECK(ranges->current() == scan::prefetched_ranges::state::ready);
  CHECK(ranges->device_id() == 0);
  CHECK(ra->counters().device_attempts.load() == 1);
  CHECK(ra->counters().device_bytes.load() == 3 * MiB);
  CHECK(outcomes(*ra, prefetch_outcome_kind::nothing_to_issue) == 0);

  // A landed copy holds no capacity: its ticket came back when the copy settled, and the
  // consumer's reading report joins the covered split instead of charging a cold read.
  REQUIRE(eventually([&] { return worker_settled(*ra, 2); }));
  split->set_scan_stage(scan_stage::reading);
  ra->update_scan_state(1, split.get(), scan_stage::reading);
  CHECK(ra->counters().cold_read_tickets.load() == 0);
  CHECK(ra->counters().borrowed.load() == 0);
  CHECK(ra->available_tickets_for_testing() == 2);
  split->set_scan_stage(scan_stage::disposed);
  ra->update_scan_state(1, split.get(), scan_stage::disposed);
  CHECK(ra->available_tickets_for_testing() == 2);

  ra->stop();
  CHECK(eventually([&] { return ra->available_tickets_for_testing() == 2; }));
  auto const line = ra->summary();
  INFO(line);
  CHECK(line.find("device[prefetched=1 ") != std::string::npos);
  CHECK(line.find(" MiB=3 ") != std::string::npos);
  CHECK(line.find(" parked=1]") != std::string::npos);

  ranges.reset();
  split.reset();
  ra.reset();
  tier->view().synchronize();
}

TEST_CASE_METHOD(readahead_fixture,
                 "memory pressure puts the device in back-off and the host tier runs",
                 "[scan_manager][device_prefetch][readahead]")
{
  auto hog = space->make_reservation_or_null(space->get_max_memory() - 1 * MiB);
  REQUIRE(hog != nullptr);
  auto opts    = options();
  opts.backoff = std::chrono::hours(1);
  auto ra      = readahead(opts);
  auto first   = on_gpu0(two_ranges(local_datasource()));
  auto second  = on_gpu0(two_ranges(local_datasource()));

  ra->register_scan_task(first, 1);
  ra->start();
  REQUIRE(eventually([&] { return outcomes(*ra, prefetch_outcome_kind::nothing_to_issue) == 1; }));
  CHECK(ra->counters().device_skipped_memory.load() == 1);
  CHECK(ra->counters().device_attempts.load() == 1);

  ra->register_scan_task(second, 1);
  REQUIRE(eventually([&] { return outcomes(*ra, prefetch_outcome_kind::nothing_to_issue) == 2; }));
  CHECK(ra->counters().device_skipped_backoff.load() == 1);
  CHECK(ra->counters().device_attempts.load() == 1);
  CHECK(first->prefetched() == nullptr);
  CHECK(second->prefetched() == nullptr);

  ra->stop();
  CHECK(ra->available_tickets_for_testing() == 2);
}

TEST_CASE_METHOD(readahead_fixture,
                 "a disposal on the device ends the back-off",
                 "[scan_manager][device_prefetch][readahead]")
{
  auto hog = space->make_reservation_or_null(space->get_max_memory() - 1 * MiB);
  REQUIRE(hog != nullptr);
  auto opts    = options();
  opts.backoff = std::chrono::hours(1);
  auto ra      = readahead(opts);
  auto ds      = local_datasource();
  auto refused = on_gpu0(two_ranges(ds));
  auto waiting = on_gpu0(two_ranges(ds));
  auto landed  = on_gpu0(two_ranges(ds));

  ra->register_scan_task(refused, 1);
  ra->start();
  REQUIRE(eventually([&] { return ra->counters().device_skipped_memory.load() == 1; }));

  // The memory is back, but only a disposal (or the timer) ends the back-off.
  hog.reset();
  ra->register_scan_task(waiting, 1);
  REQUIRE(eventually([&] { return outcomes(*ra, prefetch_outcome_kind::nothing_to_issue) == 2; }));
  CHECK(ra->counters().device_skipped_backoff.load() == 1);

  ra->update_scan_state(1, refused.get(), scan_stage::disposed);
  ra->register_scan_task(landed, 1);
  REQUIRE(eventually([&] { return outcomes(*ra, prefetch_outcome_kind::device_prefetched) == 1; }));
  CHECK(ra->counters().device_attempts.load() == 2);
  CHECK(landed->prefetched() != nullptr);

  ra->stop();
  landed.reset();
  ra.reset();
  tier->view().synchronize();
}

TEST_CASE_METHOD(readahead_fixture,
                 "an oversized split does not trigger back-off",
                 "[scan_manager][device_prefetch][readahead]")
{
  // A 4 MiB budget parks at most 2 MiB: the 3 MiB split is too large, the 1 MiB one is not.
  auto ra =
    readahead(options(static_cast<double>(4 * MiB) / static_cast<double>(space->get_max_memory())));
  auto ds        = local_datasource();
  auto oversized = on_gpu0(two_ranges(ds));
  auto normal    = on_gpu0({{ds, {byte_range(0, 1 * MiB)}}});

  ra->register_scan_task(oversized, 1);
  ra->register_scan_task(normal, 1);
  ra->start();
  REQUIRE(eventually([&] { return outcomes(*ra, prefetch_outcome_kind::device_prefetched) == 1; }));
  CHECK(ra->counters().device_skipped_too_large.load() == 1);
  CHECK(ra->counters().device_skipped_backoff.load() == 0);
  CHECK(ra->counters().device_attempts.load() == 2);
  CHECK(oversized->prefetched() == nullptr);
  CHECK(normal->prefetched() != nullptr);

  ra->stop();
  normal.reset();
  ra.reset();
  tier->view().synchronize();
}

TEST_CASE_METHOD(readahead_fixture,
                 "the parked window refuses the third split",
                 "[scan_manager][device_prefetch][readahead]")
{
  auto opts              = options();
  opts.max_parked_splits = 2;
  auto ra                = readahead(opts);
  auto ds                = local_datasource();
  std::vector<std::shared_ptr<device_split>> splits{
    on_gpu0(two_ranges(ds)), on_gpu0(two_ranges(ds)), on_gpu0(two_ranges(ds))};
  for (auto const& split : splits) {
    ra->register_scan_task(split, 1);
  }
  ra->start();
  REQUIRE(eventually([&] {
    return outcomes(*ra, prefetch_outcome_kind::device_prefetched) == 2 &&
           outcomes(*ra, prefetch_outcome_kind::nothing_to_issue) == 1;
  }));
  CHECK(ra->counters().device_skipped_window.load() == 1);
  CHECK(ra->counters().device_skipped_backoff.load() == 0);
  CHECK(splits[0]->prefetched() != nullptr);
  CHECK(splits[1]->prefetched() != nullptr);
  CHECK(splits[2]->prefetched() == nullptr);

  ra->stop();
  splits.clear();
  ra.reset();
  tier->view().synchronize();
}

TEST_CASE_METHOD(readahead_fixture,
                 "prefetch eviction mode releases the host staging once the device copy landed",
                 "[scan_manager][device_prefetch][readahead]")
{
  auto cfg           = sirius::test::dispose_on_idle_config();
  cfg.cache.eviction = cucascade::io::cache::eviction_policy::prefetch;
  cfg.apply_cache_mode();
  // Declared before the datasource: the manager owns the cached ioctx it reads through.
  sirius::exec::query_lifecycle_registry lifecycle;
  sirius::scan_manager::sirius_scan_manager manager{
    lifecycle, cfg, *memory, sirius::test::single_gpu_index_for_prefetch()};
  auto* cache = manager.io_ctx()->cache();
  REQUIRE(cache != nullptr);

  bool release = false;
  SECTION("released as soon as the device copy landed") { release = true; }
  SECTION("kept until the split dies") { release = false; }

  auto ds = manager.create_datasource(file.path.string());
  REQUIRE(ds->uses_fs_cache());
  auto hints = two_ranges(ds);
  ds->fadvise(hints.front().ranges, 0);
  REQUIRE(ds->prepare_prefetch(false) == cucascade::io::prepare_result::prepared);
  auto const claimed = cache->claimed_bytes();
  REQUIRE(claimed > 0);

  auto opts                 = options();
  opts.release_host_staging = release;
  auto ra                   = readahead(opts);
  auto split                = on_gpu0(std::move(hints));
  ra->register_scan_task(split, 1);
  ra->start();
  REQUIRE(eventually([&] { return outcomes(*ra, prefetch_outcome_kind::device_prefetched) == 1; }));
  REQUIRE(split->prefetched() != nullptr);
  CHECK(split->prefetched()->current() == scan::prefetched_ranges::state::ready);
  INFO("cache: " << cache->summary());

  if (release) {
    CHECK(sirius::test::claimed_after_sweep(*cache, claimed, std::chrono::milliseconds(5000)) == 0);
    // Released while the split is alive and its consumer has not started.
    CHECK(split->get_scan_stage() == scan_stage::none);
  } else {
    CHECK(sirius::test::claimed_after_sweep(*cache, claimed, std::chrono::milliseconds(300)) > 0);
  }

  ra->stop();
  split.reset();
  ds.reset();
  cache->evict(1);
  CHECK(sirius::test::claimed_after_sweep(*cache, claimed, std::chrono::milliseconds(5000)) == 0);
  ra.reset();
  tier->view().synchronize();
}

TEST_CASE_METHOD(readahead_fixture,
                 "without a host cache the device tier works and the host tier never spins",
                 "[scan_manager][device_prefetch][readahead]")
{
  auto ds = local_datasource();
  REQUIRE_FALSE(ds->uses_fs_cache());

  SECTION("device tier on")
  {
    auto opts              = options();
    opts.max_parked_splits = 1;
    auto ra                = readahead(opts);
    auto landed            = on_gpu0(two_ranges(ds));
    ra->register_scan_task(landed, 1);
    ra->start();
    REQUIRE(
      eventually([&] { return outcomes(*ra, prefetch_outcome_kind::device_prefetched) == 1; }));
    REQUIRE(landed->prefetched() != nullptr);
    CHECK(landed->prefetched()->current() == scan::prefetched_ranges::state::ready);
    // The landed copy returned its ticket at completion.
    CHECK(eventually([&] { return ra->available_tickets_for_testing() == 2; }));

    // The window is full while the first split is unconsumed, so the second falls through to
    // the host tier, which has nowhere to stage and gives up at once.
    // No memory retry is the evidence that it never waited.
    auto refused = on_gpu0(two_ranges(ds));
    ra->register_scan_task(refused, 1);
    REQUIRE(
      eventually([&] { return outcomes(*ra, prefetch_outcome_kind::nothing_to_issue) == 1; }));
    CHECK(ra->counters().device_skipped_window.load() == 1);
    CHECK(ra->counters().memory_retries.load() == 0);
    CHECK(refused->prefetched() == nullptr);

    ra->stop();
    CHECK(eventually([&] { return ra->available_tickets_for_testing() == 2; }));
    landed.reset();
    ra.reset();
    tier->view().synchronize();
  }

  SECTION("device tier off: unchanged")
  {
    auto ra    = readahead(device_prefetch_options{});
    auto split = on_gpu0(two_ranges(ds));
    ra->register_scan_task(split, 1);
    ra->start();
    REQUIRE(
      eventually([&] { return outcomes(*ra, prefetch_outcome_kind::nothing_to_issue) == 1; }));
    CHECK(ra->counters().memory_retries.load() == 0);
    CHECK(ra->counters().device_attempts.load() == 0);
    ra->stop();
    CHECK(ra->available_tickets_for_testing() == 2);
  }
}

TEST_CASE_METHOD(readahead_fixture,
                 "a failed device read through the worker hands its ticket back once",
                 "[scan_manager][device_prefetch][readahead]")
{
  std::shared_ptr<cucascade::io::ioctx> context;
  SECTION("the read errors") { context = failing_context(); }
  SECTION("the read returns no future") { context = no_future_context(); }

  auto ra    = readahead(options());
  auto split = on_gpu0(two_ranges(cucascade::io::open_datasource(context, file.path.string())));
  ra->register_scan_task(split, 1);
  ra->mark_operator_closed(1);
  ra->start();
  REQUIRE(eventually([&] { return outcomes(*ra, prefetch_outcome_kind::device_failed) == 1; }));
  CHECK(ra->counters().device_attempts.load() == 1);
  CHECK(ra->counters().device_bytes.load() == 0);
  REQUIRE(split->prefetched() != nullptr);
  CHECK(split->prefetched()->current() == scan::prefetched_ranges::state::failed);

  // No consumer had started, so the failed attempt released its ticket at completion...
  REQUIRE(eventually([&] { return worker_settled(*ra, 2); }));
  // ...and the consumer's own read is charged one when it starts, until disposal.
  split->set_scan_stage(scan_stage::reading);
  ra->update_scan_state(1, split.get(), scan_stage::reading);
  CHECK(ra->counters().cold_read_tickets.load() == 1);
  CHECK(ra->available_tickets_for_testing() == 1);
  split->set_scan_stage(scan_stage::disposed);
  ra->update_scan_state(1, split.get(), scan_stage::disposed);
  CHECK(ra->available_tickets_for_testing() == 2);

  ra->stop();
  CHECK(ra->available_tickets_for_testing() == 2);
  CHECK(outcomes(*ra, prefetch_outcome_kind::device_prefetched) == 0);
  CHECK(ra->summary().find(" failed=1 ") != std::string::npos);
  CHECK(ra->summary().find(" parked=1]") != std::string::npos);

  // The failed buffers stay parked until the split dies.
  split.reset();
  CHECK(eventually([&] { return ra->summary().find(" parked=0]") != std::string::npos; }));
  ra.reset();
  tier->view().synchronize();
}

TEST_CASE_METHOD(readahead_fixture,
                 "a failed device read hands its ticket to a consumer already reading",
                 "[scan_manager][device_prefetch][readahead]")
{
  auto gate = std::make_shared<read_gate>();
  auto ra   = readahead(options());
  auto split =
    on_gpu0(two_ranges(cucascade::io::open_datasource(gated_context(gate), file.path.string())));
  ra->register_scan_task(split, 1);
  ra->mark_operator_closed(1);
  ra->start();
  REQUIRE(eventually([&] { return gate->reads.load() == 1; }));
  REQUIRE(eventually([&] { return worker_settled(*ra, 1); }));

  // The consumer starts while the device read is in flight: it joins the attempt's ticket.
  split->set_scan_stage(scan_stage::reading);
  ra->update_scan_state(1, split.get(), scan_stage::reading);
  CHECK(ra->counters().cold_read_tickets.load() == 0);
  CHECK(ra->counters().borrowed.load() == 0);
  CHECK(ra->available_tickets_for_testing() == 1);

  // The copy did not land, so the consumer reads for itself, charged to the ticket the attempt
  // held: it stays out past the completion, without a second charge, until the split is disposed.
  gate->fail();
  REQUIRE(eventually([&] { return outcomes(*ra, prefetch_outcome_kind::device_failed) == 1; }));
  CHECK(ra->available_tickets_for_testing() == 1);
  ra->update_scan_state(1, split.get(), scan_stage::reading);
  CHECK(ra->counters().cold_read_tickets.load() == 0);
  CHECK(ra->available_tickets_for_testing() == 1);
  split->set_scan_stage(scan_stage::disposed);
  ra->update_scan_state(1, split.get(), scan_stage::disposed);
  CHECK(ra->available_tickets_for_testing() == 2);
  // A second disposal report releases nothing more.
  ra->update_scan_state(1, split.get(), scan_stage::disposed);
  CHECK(ra->available_tickets_for_testing() == 2);

  ra->stop();
  CHECK(ra->available_tickets_for_testing() == 2);
  split.reset();
  ra.reset();
  tier->view().synchronize();
}

TEST_CASE_METHOD(readahead_fixture,
                 "stop waits for an in-flight device read to return its ticket",
                 "[scan_manager][device_prefetch][readahead]")
{
  auto gate = std::make_shared<read_gate>();
  auto ra   = readahead(options());
  auto split =
    on_gpu0(two_ranges(cucascade::io::open_datasource(gated_context(gate), file.path.string())));
  ra->register_scan_task(split, 1);
  ra->start();
  REQUIRE(eventually([&] { return gate->reads.load() == 1; }));

  // The read lands while stop() is draining: the ticket must be back when stop() returns.
  std::thread opener([&] {
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
    gate->open();
  });
  ra->stop();
  opener.join();
  CHECK(ra->available_tickets_for_testing() == 2);
  CHECK(outcomes(*ra, prefetch_outcome_kind::device_prefetched) == 1);

  split.reset();
  ra.reset();
  tier->view().synchronize();
}

TEST_CASE_METHOD(readahead_fixture,
                 "a device copy that lands after its split was disposed is wasted",
                 "[scan_manager][device_prefetch][readahead]")
{
  auto gate = std::make_shared<read_gate>();
  auto ra   = readahead(options());
  auto split =
    on_gpu0(two_ranges(cucascade::io::open_datasource(gated_context(gate), file.path.string())));
  ra->register_scan_task(split, 1);
  ra->start();
  REQUIRE(eventually([&] { return gate->reads.load() == 1; }));

  split->set_scan_stage(scan_stage::disposed);
  gate->open();
  REQUIRE(eventually([&] { return outcomes(*ra, prefetch_outcome_kind::device_wasted) == 1; }));
  CHECK(outcomes(*ra, prefetch_outcome_kind::device_prefetched) == 0);

  ra->stop();
  CHECK(ra->available_tickets_for_testing() == 2);
  split.reset();
  ra.reset();
  tier->view().synchronize();
}
