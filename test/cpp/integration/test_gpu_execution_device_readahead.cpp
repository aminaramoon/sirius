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

// The readahead's device tier end to end: parquet queries through Sirius with
// scan_manager.device_prefetching_fraction set, against plain DuckDB.  The splits must land on
// the GPU ahead of their scan tasks (the `device[...]` group of each query's `[readahead]` line),
// the parquet materializer must decode them from the device copies without falling back, and
// every result must match the CPU's.  With the fraction at 0 the same queries run with the device
// tier off.

#include "catch.hpp"
#include "utils/gpu_execution_fixture.hpp"
#include "utils/log_test_utils.hpp"
#include "utils/parquet_fixture_utils.hpp"
#include "utils/sirius_test_env.hpp"
#include "utils/transparent_execution_test_utils.hpp"

#include <duckdb.hpp>
#include <unistd.h>

#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <memory>
#include <regex>
#include <string>
#include <system_error>
#include <thread>
#include <vector>

namespace {

namespace fs = std::filesystem;

/// A scratch directory, removed with everything in it.
struct scratch_dir {
  fs::path path = fs::temp_directory_path() /
                  ("sirius_device_readahead_" + std::to_string(::getpid()));

  scratch_dir()
  {
    std::error_code ec;
    fs::remove_all(path, ec);
    fs::create_directories(path);
  }
  ~scratch_dir()
  {
    std::error_code ec;
    fs::remove_all(path, ec);
  }
  scratch_dir(scratch_dir const&)            = delete;
  scratch_dir& operator=(scratch_dir const&) = delete;
};

/// integration.yaml's memory and executor blocks, a scan batch small enough to cut the test file
/// into several splits, and the readahead with the device tier at @p fraction.  The readahead
/// budget is set in both configurations, so the device tier is the only difference.
void write_config(fs::path const& path, double fraction)
{
  std::ofstream out(path);
  out << R"(sirius:
  topology:
    num_gpus: 1
  memory:
    gpu:
      usage_limit_fraction: 0.4
      reservation_limit_fraction: 1.0
    host:
      capacity_bytes: 32000000000
      initial_number_pools: 10
      pool_size: 512
      block_size: 1048576
  executor:
    pipeline:
      num_threads: 4
    task_creator:
      num_threads: 2
    downgrade:
      num_threads: 1
      monitor_period: 10ms
    scan_manager:
      max_readahead_scans: 4
      readahead_strategy: eager
      device_prefetching_fraction: )"
      << fraction << R"(
      cache: { mode: cucs, eviction: prefetch }
  operator_params:
    scan_task_batch_size: 4000000
    max_sort_partition_bytes: 0
    hash_partition_bytes: 100000000
    concat_batch_bytes: 100000000
    max_build_hash_table_bytes: 90000000
)";
}

/// A Sirius-enabled DuckDB on its own configuration.  One Sirius context runs per process at a
/// time, so the shared environments are paused first; the test listener resumes the integration
/// environment for the next test that needs it.
struct device_readahead_db {
  std::unique_ptr<sirius::test::shared_test_env> env;
  std::unique_ptr<duckdb::Connection> con;

  explicit device_readahead_db(fs::path const& config)
  {
    for (auto* shared : {sirius::test::g_shared_env,
                         sirius::test::g_integration_env,
                         sirius::test::g_integration_env_2gpu}) {
      if (shared != nullptr && shared->is_active()) { shared->pause(); }
    }
    env = std::make_unique<sirius::test::shared_test_env>(config);
    con = std::make_unique<duckdb::Connection>(env->make_connection());
  }
  ~device_readahead_db()
  {
    con.reset();
    env.reset();
  }
  device_readahead_db(device_readahead_db const&)            = delete;
  device_readahead_db& operator=(device_readahead_db const&) = delete;

  void run_ok(std::string const& sql)
  {
    auto result = con->Query(sql);
    REQUIRE(result);
    INFO(sql << " -> " << (result->HasError() ? result->GetError() : std::string{"ok"}));
    REQUIRE_FALSE(result->HasError());
  }

  std::vector<std::vector<std::string>> rows(std::string const& sql)
  {
    auto result = con->Query(sql);
    REQUIRE(result);
    INFO(sql << " -> " << (result->HasError() ? result->GetError() : std::string{"ok"}));
    REQUIRE_FALSE(result->HasError());
    return sirius::test::collect_rows(*result);
  }
};

/// The `device[...]` group of one query's `[readahead]` line.
struct device_counts {
  std::uint64_t prefetched{0};
  std::uint64_t wait_for_prefetch{0};
  std::uint64_t wasted{0};
  std::uint64_t failed{0};
  std::uint64_t consumed{0};
};

std::vector<device_counts> device_groups(sirius::test::recording_log_sink const& sink,
                                         std::vector<std::string>& lines)
{
  static std::regex const re(
    R"(device\[prefetched=(\d+) wait_for_prefetch=(\d+) wasted=(\d+) failed=(\d+) consumed=(\d+) )");
  std::vector<device_counts> groups;
  lines.clear();
  for (auto const& record : sink.records()) {
    if (!record.message.starts_with("[readahead] issued=")) { continue; }
    lines.push_back(record.message);
    std::smatch m;
    if (!std::regex_search(record.message, m, re)) { continue; }
    groups.push_back({std::stoull(m[1].str()),
                      std::stoull(m[2].str()),
                      std::stoull(m[3].str()),
                      std::stoull(m[4].str()),
                      std::stoull(m[5].str())});
  }
  return groups;
}

}  // namespace

TEST_CASE("device-tier readahead lands parquet splits on the GPU and matches DuckDB",
          "[integration][readahead][device_prefetch]")
{
  double fraction = 0.0;
  SECTION("device tier on") { fraction = 0.6; }
  SECTION("device tier off") { fraction = 0.0; }

  scratch_dir dir;
  auto const config = dir.path / "sirius.yaml";
  write_config(config, fraction);
  auto const file = (dir.path / "device_readahead.parquet").string();

  device_readahead_db db(config);
  db.run_ok("SET gpu_execution = false;");
  db.run_ok(
    "COPY (SELECT range AS id, range * 2 AS v, 'k' || (range % 7) AS s, "
    "(range % 100)::DECIMAL(9,2) AS d FROM range(3000000)) TO " +
    sirius::test::sql_literal(file) + " (FORMAT PARQUET, ROW_GROUP_SIZE 200000);");
  db.run_ok("CREATE VIEW t AS SELECT * FROM read_parquet(" + sirius::test::sql_literal(file) +
            ");");

  // Recorded from here: the Sirius context installed its own sink when the database opened.
  sirius::test::scoped_recording_log_sink log("info");

  std::vector<std::string> const queries{
    "SELECT count(*), sum(v) FROM t",
    "SELECT count(*), sum(v), min(s), max(s) FROM t WHERE id BETWEEN 1000000 AND 1400000",
    "SELECT count(*), sum(v) FROM t WHERE s = 'k3'",
    "SELECT sum(d), min(d), max(d), count(d) FROM t",
    "SELECT count(*), sum(a.v) FROM t a JOIN t b ON a.id = b.id WHERE b.id % 1000 = 7",
  };

  auto const before = sirius::test::get_transparent_execution_stats(*db.con);
  for (auto const& sql : queries) {
    INFO(sql);
    db.run_ok("SET gpu_execution = true;");
    auto const gpu = db.rows(sql);
    db.run_ok("SET gpu_execution = false;");
    auto const cpu = db.rows(sql);
    CHECK(gpu == cpu);
  }
  auto const after = sirius::test::get_transparent_execution_stats(*db.con);
  auto const gpu_queries = after.executions - before.executions;
  CHECK(gpu_queries == queries.size());
  CHECK(after.runtime_fallbacks == before.runtime_fallbacks);

  // Each query logs its `[readahead]` line when its scan state drains, which may trail the
  // result by a moment.
  std::vector<std::string> lines;
  auto groups         = device_groups(log.sink(), lines);
  auto const deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
  while (lines.size() < gpu_queries && std::chrono::steady_clock::now() < deadline) {
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
    groups = device_groups(log.sink(), lines);
  }
  for (auto const& line : lines) {
    UNSCOPED_INFO(line);
  }
  REQUIRE_FALSE(lines.empty());
  REQUIRE(groups.size() == lines.size());

  std::uint64_t landed   = 0;
  std::uint64_t consumed = 0;
  for (auto const& g : groups) {
    CHECK(g.failed == 0);
    landed += g.prefetched + g.wait_for_prefetch;
    consumed += g.consumed;
  }
  auto const decodes   = after.device_route_decodes - before.device_route_decodes;
  auto const fallbacks = after.device_route_fallbacks - before.device_route_fallbacks;
  CHECK(fallbacks == 0);

  if (fraction > 0.0) {
    std::uint64_t prefetched = 0;
    for (auto const& g : groups) {
      prefetched += g.prefetched;
    }
    CHECK(prefetched > 0);
    CHECK(consumed > 0);
    CHECK(consumed <= landed);
    CHECK(decodes > 0);
  } else {
    CHECK(landed == 0);
    CHECK(consumed == 0);
    CHECK(decodes == 0);
    for (auto const& line : lines) {
      CHECK(line.find("device[prefetched=0 ") != std::string::npos);
    }
  }
}
