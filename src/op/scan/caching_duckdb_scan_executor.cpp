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

// sirius
#include <op/scan/caching_duckdb_scan_executor.hpp>

// standard library
#include <mutex>

namespace sirius::op::scan {

void caching_duckdb_scan_executor::cache_scan_results_for_query(const std::string& query)
{
  // Compute a simple hash for the query
  std::hash<std::string> hasher;
  /// convert query to lower case
  std::string lower_query = query;
  std::transform(lower_query.begin(), lower_query.end(), lower_query.begin(), ::tolower);

  auto new_hash = hasher(lower_query);

  if (new_hash == _query_hash) {
    return;
  }

  _query_hash = new_hash;
  _cache.clear();
}

void caching_duckdb_scan_executor::schedule(std::unique_ptr<sirius::parallel::itask> task)
{
  std::lock_guard lock(_cache_mutex);
  duckdb_scan_executor::schedule(std::move(task));
}

}  // namespace sirius::op::scan
