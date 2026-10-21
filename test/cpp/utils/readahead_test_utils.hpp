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

#pragma once

// Fixtures shared by the readahead / scan-input prefetch tests: a real on-disk data file, a
// scan manager configuration with a prefetching cache, cache-summary helpers, and a one-scan
// query that parks the readahead worker.

#include "memory/topology_index.hpp"
#include "op/sirius_physical_operator.hpp"
#include "pipeline/pipeline_build_context.hpp"
#include "pipeline/sirius_pipeline.hpp"
#include "planner/query.hpp"
#include "query_id.hpp"
#include "scan_manager/config.hpp"
#include "utils/telemetry_utils.hpp"

#include <cucascade/io/cache/config.hpp>
#include <cucascade/io/cache/fs_cache.hpp>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <memory>
#include <regex>
#include <string>
#include <system_error>
#include <thread>
#include <vector>

namespace sirius::test {

/// A one-GPU topology (device 0, NUMA node 0) for a scan manager under test.
inline std::shared_ptr<const sirius::memory::topology_index> single_gpu_index_for_prefetch()
{
  cucascade::memory::system_topology_info topology;
  topology.num_gpus = 1;
  cucascade::memory::gpu_topology_info gpu;
  gpu.id        = 0;
  gpu.numa_node = 0;
  topology.gpus.push_back(std::move(gpu));
  return std::make_shared<sirius::memory::topology_index>(topology, std::vector<int>{0});
}

/// A real on-disk file, so the uring backend opens it and the cache sizes its
/// slot table from a genuine object size.
struct temp_data_file {
  /// What the file holds: one repeated byte, or @ref positional_byte of each offset (so a read
  /// from the wrong offset shows).
  enum class contents : std::uint8_t { constant, positional };

  std::filesystem::path path;

  /// The byte at @p offset of a @c contents::positional file.
  [[nodiscard]] static constexpr std::uint8_t positional_byte(std::size_t offset) noexcept
  {
    return static_cast<std::uint8_t>((offset * 131) ^ (offset >> 12));
  }

  explicit temp_data_file(std::size_t bytes, contents fill = contents::constant)
  {
    path = std::filesystem::temp_directory_path() /
           ("sirius_scan_input_prefetch_" + std::to_string(::getpid()) + "_" +
            std::to_string(reinterpret_cast<std::uintptr_t>(this)));
    std::ofstream out(path, std::ios::binary);
    std::vector<char> data(bytes, 'z');
    if (fill == contents::positional) {
      for (std::size_t i = 0; i < bytes; ++i) {
        data[i] = static_cast<char>(positional_byte(i));
      }
    }
    out.write(data.data(), static_cast<std::streamsize>(bytes));
  }

  ~temp_data_file()
  {
    std::error_code ec;
    std::filesystem::remove(path, ec);
  }

  temp_data_file(temp_data_file const&)            = delete;
  temp_data_file& operator=(temp_data_file const&) = delete;
};

/// `cache.mode: cucs` with `cache.eviction: idle`.
inline sirius::scan_manager::scan_manager_config dispose_on_idle_config()
{
  sirius::scan_manager::scan_manager_config cfg;
  cfg.thread_pool.num_threads = 3;
  cfg.uring_reactors_per_gpu  = 1;
  cfg.cache.mode              = cucascade::io::cache::cache_mode::cucs;
  cfg.cache.eviction          = cucascade::io::cache::eviction_policy::idle;
  cfg.apply_cache_mode();
  return cfg;
}

/// Pull `global[... evictions=N ...]` out of fs_cache::summary().
inline std::uint64_t evictions_from(std::string const& summary)
{
  std::smatch m;
  static std::regex const re(R"(global\[[^\]]*evictions=(\d+))");
  if (!std::regex_search(summary, m, re)) { return 0; }
  return std::stoull(m[1].str());
}

/// Poll until claimed_bytes() drops below @p before; the evictor is a background thread.
inline std::size_t claimed_after_sweep(cucascade::io::cache::fs_cache& cache,
                                       std::size_t before,
                                       std::chrono::milliseconds budget)
{
  // The evictor may release the disposed request's chunks over several passes:
  // keep polling until everything is reclaimed or the budget runs out, and
  // return the lowest value seen so a partial sweep is still reported.
  auto const deadline = std::chrono::steady_clock::now() + budget;
  auto lowest         = before;
  while (std::chrono::steady_clock::now() < deadline) {
    auto const now = cache.claimed_bytes();
    lowest         = std::min(lowest, now);
    if (now == 0) { return 0; }
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
  }
  return std::min(lowest, cache.claimed_bytes());
}

/// Minimal concrete scan carrying only the operator id the readahead keys its
/// work queue on.
struct test_scan : sirius::op::sirius_physical_operator {
  test_scan()
    : sirius::op::sirius_physical_operator(sirius::op::SiriusPhysicalOperatorType::GPU_SCAN, {}, 0)
  {
    operator_id = 1;
  }
};

/// A query with one GPU_SCAN operator, which is what it takes to park the
/// worker: with no work queues at all it runs out of order and exits on its
/// own, and a worker that is already gone hides everything a second start does.
class single_scan_query {
 public:
  single_scan_query()
  {
    auto pipeline = std::make_shared<sirius::pipeline::sirius_pipeline>(_ctx);
    _build.set_pipeline_source(*pipeline, *_scan);
    auto const id = sirius::make_query_id(1);
    _query        = std::make_unique<sirius::planner::query>(
      std::vector<std::shared_ptr<sirius::pipeline::sirius_pipeline>>{pipeline},
      _telemetry->context(),
      id,
      sirius::telemetry::query_telemetry_info{quent::now_v7(), _telemetry->worker_id().raw(), id});
  }

  [[nodiscard]] const sirius::planner::query& get() const { return *_query; }

 private:
  std::shared_ptr<const sirius::telemetry::telemetry_context> _telemetry =
    sirius::test::make_test_telemetry_context();
  sirius::pipeline::pipeline_build_context _ctx{nullptr, true};
  sirius::pipeline::sirius_pipeline_build_state _build;
  std::unique_ptr<test_scan> _scan = std::make_unique<test_scan>();
  std::unique_ptr<sirius::planner::query> _query;
};

}  // namespace sirius::test
