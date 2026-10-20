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
#include "duckdb/common/exception.hpp"
#include "event/query_event_publisher.hpp"
#include "memory/topology_index.hpp"
#include "op/scan/gpu_ingestible_types.hpp"
#include "op/scan/prefetched_ranges.hpp"
#include "op/scan/sirius_gpu_scan_operator_data.hpp"
#include "scan/test_utils.hpp"
#include "scan_manager/config.hpp"
#include "scan_manager/load_balancing_scan_batch_coalescer.hpp"
#include "scan_manager/preparation.hpp"
#include "scan_manager/readahead_scan_manager.hpp"
#include "scan_manager/sirius_scan_manager.hpp"
#include "scan_manager/split_connector.hpp"
#include "utils/readahead_test_utils.hpp"

#include <cudf/io/text/byte_range_info.hpp>

#include <cucascade/cudf/datasource.hpp>
#include <cucascade/io/cache/config.hpp>
#include <cucascade/io/cache/fs_cache.hpp>
#include <cucascade/io/cache/types.hpp>
#include <cucascade/io/io_context.hpp>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <memory>
#include <optional>
#include <regex>
#include <stdexcept>
#include <stop_token>
#include <string>
#include <thread>
#include <utility>
#include <vector>

using sirius::scan_manager::scan_manager_config;
using sirius::scan_manager::sirius_scan_manager;

using sirius::test::claimed_after_sweep;
using sirius::test::dispose_on_idle_config;
using sirius::test::evictions_from;
using sirius::test::single_gpu_index_for_prefetch;
using sirius::test::temp_data_file;

TEST_CASE("Deferred scan inputs initialize prefetch only when admitted",
          "[cache][scan][scan_preparation]")
{
  using cucascade::io::prepare_result;
  using sirius::op::scan::scan_info;
  using sirius::op::scan::scan_operator_input;
  struct interrupting_info : scan_info {
    using scan_info::scan_info;
    bool* interrupted = nullptr;
    std::size_t estimated_bytes() const noexcept override
    {
      if (interrupted) *interrupted = true;
      return 4096;
    }
  };
  temp_data_file file(8ull << 20);
  auto memory = initialize_memory_manager(1);
  sirius::exec::query_lifecycle_registry scan_lifecycle;
  sirius_scan_manager manager{
    scan_lifecycle, dispose_on_idle_config(), *memory, single_gpu_index_for_prefetch()};
  auto* cache = manager.io_ctx()->cache();
  REQUIRE(cache != nullptr);
  auto ds = manager.create_datasource(file.path.string());
  REQUIRE(ds != nullptr);
  std::vector<scan_info::fadvise_entry> hints{{ds, {cudf::io::text::byte_range_info{0, 4096}}}};
  auto info      = std::make_shared<scan_info>(hints);
  auto publisher = std::make_shared<sirius::event::query_event_publisher>();
  // No operator is registered: trying to enqueue readahead must fail.
  auto readahead = std::make_shared<sirius::scan_manager::readahead_scan_manager>(*publisher, 1);
  bool fail_registration = false, discard = false, cached_cancel = false;
  SECTION("cached worker observes interruption after sizing an insert-delta input")
  {
    cached_cancel = true;
  }
  SECTION("cancelled construction leaves no cache request") { discard = true; }
  SECTION("admitted initialization prepares the buffers") { readahead.reset(); }
  SECTION("failed registration disposes a partially initialized request")
  {
    fail_registration = true;
  }
  if (cached_cancel) {
    using namespace sirius::scan_manager;
    struct single_input : databatch_provider {
      batch next;
      batch get_next_batch() override { return std::exchange(next, {}); }
    } provider;
    bool interrupted        = false;
    auto delta              = std::make_unique<interrupting_info>(hints);
    delta->interrupted      = &interrupted;
    provider.next.scan_info = std::move(delta);
    auto gate               = std::make_shared<preparation_gate>();
    gate->interrupted       = [&] { return interrupted; };
    auto connector          = std::make_shared<split_connector>();
    std::stop_source stop;
    load_balancing_scan_batch_coalescer::drain_cached_provider(
      provider, *connector, stop.get_token(), false, 0, 0, nullptr, false, false, nullptr, 0, gate);
    CHECK_THROWS_AS(connector->get_next_split(), duckdb::InterruptException);
    CHECK(gate->closed);
    CHECK(cache->generation_count(file.path.string()) == 0);
    CHECK(cache->claimed_bytes() == 0);
    CHECK(ds->prepare_prefetch(false) == prepare_result::nothing_to_prepare);
    return;
  }
  auto input = std::make_unique<scan_operator_input>(
    info, readahead, 0, 0, scan_operator_input::prefetch_start::deferred);
  CHECK(cache->generation_count(file.path.string()) == 0);
  CHECK(cache->claimed_bytes() == 0);
  CHECK(ds->prepare_prefetch(false) == prepare_result::nothing_to_prepare);
  if (!discard) {
    if (fail_registration) {
      CHECK_THROWS_AS(input->initialize_prefetch(), std::out_of_range);
    } else {
      REQUIRE_NOTHROW(input->initialize_prefetch());
      CHECK(ds->prepare_prefetch(false) == prepare_result::prepared);
    }
    CHECK(cache->generation_count(file.path.string()) == 1);
    CHECK(cache->claimed_bytes() > 0);
    CHECK_THROWS_AS(input->initialize_prefetch(), std::logic_error);
  }
  input.reset();
  CHECK(info->get_scan_stage() == cucascade::io::cache::scan_stage::disposed);
  if (discard) {
    CHECK(cache->generation_count(file.path.string()) == 0);
    CHECK(cache->claimed_bytes() == 0);
  } else {
    CHECK(ds->prepare_prefetch(false) == prepare_result::fallen_behind);
  }
}

TEST_CASE("cache.eviction idle reclaims a disposed scan input's chunks",
          "[cache][scan][eviction][dispose]")
{
  temp_data_file file(8ull << 20);
  auto memory = initialize_memory_manager(1);
  sirius::exec::query_lifecycle_registry scan_lifecycle;
  sirius_scan_manager manager{
    scan_lifecycle, dispose_on_idle_config(), *memory, single_gpu_index_for_prefetch()};
  auto* cache = manager.io_ctx()->cache();
  REQUIRE(cache != nullptr);
  REQUIRE(cache->is_armed());
  REQUIRE(cache->claimed_bytes() == 0);

  std::size_t claimed = 0;
  {
    auto ds = manager.create_datasource(file.path.string());
    REQUIRE(ds != nullptr);
    std::vector<sirius::op::scan::scan_info::fadvise_entry> hints;
    hints.push_back({ds, {cudf::io::text::byte_range_info{0, 4ll << 20}}});
    auto info  = std::make_shared<sirius::op::scan::scan_info>(std::move(hints));
    auto input = std::make_unique<sirius::op::scan::scan_operator_input>(info);
    claimed    = cache->claimed_bytes();
    REQUIRE(claimed > 0);
    input.reset();
    CHECK(info->get_scan_stage() == cucascade::io::cache::scan_stage::disposed);
  }

  // The evictor sweeps when a request reaches it; this insert claims no buffers of its own.
  auto other = manager.create_datasource(file.path.string());
  REQUIRE(other != nullptr);
  std::vector<cudf::io::text::byte_range_info> untouched;
  untouched.emplace_back(6ll << 20, 1ll << 20);
  other->fadvise(untouched, 0);

  INFO("cache: " << cache->summary());
  // Primary assertion: the disposed input's chunks were evicted. The claimed
  // total must drop; it reaches zero unless `other`'s hint shares a chunk with
  // the disposed range (chunk size is a host-pool property, not fixed here).
  auto const after = claimed_after_sweep(*cache, claimed, std::chrono::milliseconds(5000));
  CHECK(evictions_from(cache->summary()) > 0);
  CHECK(after < claimed);
}

TEST_CASE("scan inputs stamp the preferred device on their scan_info before publication",
          "[cache][scan][scan_preparation]")
{
  using sirius::op::scan::scan_info;
  using sirius::op::scan::scan_operator_input;
  auto publisher = std::make_shared<sirius::event::query_event_publisher>();

  SECTION("a placed split carries its device by the time it is published")
  {
    auto info = std::make_shared<scan_info>();
    // No operator is registered, so registration -- the publication -- throws; the stamp must
    // already be there when it does.
    auto readahead = std::make_shared<sirius::scan_manager::readahead_scan_manager>(*publisher, 1);
    auto input     = std::make_unique<scan_operator_input>(
      info, readahead, 0, 0, scan_operator_input::prefetch_start::deferred);
    CHECK_FALSE(info->preferred_device().has_value());
    CHECK_THROWS_AS(input->initialize_prefetch(), std::out_of_range);
    CHECK(info->preferred_device() == std::optional<int>{0});
  }

  SECTION("a split nobody placed stays unplaced")
  {
    auto info  = std::make_shared<scan_info>();
    auto input = std::make_unique<scan_operator_input>(
      info, nullptr, 0, std::nullopt, scan_operator_input::prefetch_start::deferred);
    CHECK_FALSE(info->preferred_device().has_value());
    REQUIRE_NOTHROW(input->initialize_prefetch());
    CHECK_FALSE(info->preferred_device().has_value());
  }
}

TEST_CASE("the prefetched slot is set once and taken once", "[cache][scan][scan_preparation]")
{
  using sirius::op::scan::prefetched_ranges;
  using sirius::op::scan::scan_info;
  using release_reason = prefetched_ranges::release_reason;

  // Empty prefetches: the slot only moves ownership, so no device memory is needed.
  auto unconsumed = std::make_shared<std::atomic<int>>(0);
  auto make       = [&] {
    return std::make_shared<prefetched_ranges>(
      0,
      nullptr,
      std::vector<prefetched_ranges::device_ranges>{},
      0,
      [unconsumed](prefetched_ranges::release_event event) noexcept {
        if (event.reason == release_reason::unconsumed) { unconsumed->fetch_add(1); }
      });
  };

  scan_info info;
  CHECK(info.prefetched() == nullptr);
  CHECK_FALSE(info.supports_device_prefetch());
  CHECK_FALSE(info.set_prefetched_ranges(nullptr));
  CHECK(info.prefetched() == nullptr);

  auto first = make();
  auto* raw  = first.get();
  CHECK(info.set_prefetched_ranges(std::move(first)));
  CHECK(info.prefetched().get() == raw);

  // A second install is refused and its argument dropped.
  CHECK_FALSE(info.set_prefetched_ranges(make()));
  CHECK(unconsumed->load() == 1);
  CHECK(info.prefetched().get() == raw);

  auto const& view = info;
  auto taken       = view.take_prefetched_ranges();
  CHECK(taken.get() == raw);
  CHECK(taken.use_count() == 1);
  CHECK(view.take_prefetched_ranges() == nullptr);
  CHECK(info.prefetched() == nullptr);
  view.discard_prefetched_ranges();  // an empty slot is a no-op

  taken.reset();
  CHECK(unconsumed->load() == 2);

  CHECK(info.set_prefetched_ranges(make()));
  view.discard_prefetched_ranges();
  CHECK(info.prefetched() == nullptr);
  CHECK(unconsumed->load() == 3);
}
