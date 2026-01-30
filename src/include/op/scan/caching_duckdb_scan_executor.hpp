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

#pragma once

#include <string>

// sirius
#include <cucascade/data/cpu_data_representation.hpp>
#include <op/scan/duckdb_scan_executor.hpp>
// #include <op/scan/duckdb_scan_task_queue.hpp>

namespace sirius::op::scan {

//===----------------------------------------------------------------------===//
// DuckDB Scan Executor
//===----------------------------------------------------------------------===//

/**
 * @brief A task executor for duckdb scan tasks.
 *
 * This class extends the generic itask_executor simply by instantiating it with a
 * duckdb_scan_task_queue.
 *
 */
class caching_duckdb_scan_executor : public duckdb_scan_executor {
 public:
  //===----------Constructor----------===//
  explicit caching_duckdb_scan_executor(sirius::exec::thread_pool_config config)
    : duckdb_scan_executor(config)
  {
  }

  void cache_scan_results_for_query(const std::string& query);

  //===----------Methods----------===//
  /**
   * @brief Schedule a new task for execution.
   *
   * @param task The task to be scheduled.
   */
  void schedule(std::unique_ptr<sirius::parallel::itask> task) override;

  //===----------Fields----------===//
 private:
  struct scan_state {
    std::size_t pipeline_id{0};
    std::atomic<std::size_t> cursor{0};
    std::vector<std::shared_ptr<int>> results;
  };
  std::mutex _cache_mutex;
  std::size_t _query_hash = 0;
  std::unordered_map<std::size_t, scan_state> _cache;
};

}  // namespace sirius::op::scan
