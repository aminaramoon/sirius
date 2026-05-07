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

#include "op/scan/scan_plan.hpp"
#include "scan_manager/split_provider.hpp"

#include <cudf/column/column.hpp>

#include <cucascade/memory/memory_space.hpp>
#include <duckdb/planner/expression.hpp>

#include <atomic>
#include <cstddef>
#include <memory>
#include <vector>

namespace sirius::scan_manager {

/**
 * @brief Split provider backed by pre-pinned columns from a pinned_entry.
 *
 * The scan_manager builds the per-column chunk vectors in scan_plan D-order
 * (one entry per @c data_columns slot, looked up by name in the pinned entry).
 * @ref create_split() then assembles one @ref op::scan::scan_cached_operator_data
 * per chunk: each carries a zero-copy view-backed data_batch over the pinned
 * columns plus the filter expression and a shared scan_plan, and is pushed
 * into the connector by @ref split_provider::run.
 *
 * @par Inputs
 *   - @p columns_per_request[d] is the chunk vector for D-position @p d. All
 *     inner vectors must have the same size — that size is the number of
 *     emitted batches.
 *   - @p memory_space is captured into each emitted data_batch so memory
 *     accounting matches where the cached columns reside.
 *   - @p filter_expression and @p plan are forwarded unchanged on every
 *     emitted batch, mirroring the parquet path's per-split contract. The scan
 *     operator queries @c needs_output_assembly(*plan) to decide whether to
 *     reshape the cached batch — when false, the cached batch is forwarded
 *     straight through (no permute, no prune -> no copy).
 */
class cached_split_provider : public split_provider {
 public:
  cached_split_provider(std::vector<std::vector<std::shared_ptr<cudf::column>>> columns_per_request,
                        cucascade::memory::memory_space& memory_space,
                        std::shared_ptr<duckdb::Expression> filter_expression,
                        std::shared_ptr<op::scan::scan_plan const> plan);

 protected:
  /// \brief Thread-safe iterator: each call atomically claims the next
  ///        chunk index and returns its cached batch wrapped in a single-
  ///        element vector. Returns an empty vector after all chunks have
  ///        been served.
  std::vector<std::unique_ptr<op::operator_data>> create_split() override;

 private:
  std::vector<std::vector<std::shared_ptr<cudf::column>>> _columns_per_request;
  cucascade::memory::memory_space* _memory_space;
  std::shared_ptr<duckdb::Expression> _filter_expression;
  std::shared_ptr<op::scan::scan_plan const> _plan;
  std::size_t _num_batches{0};
  std::atomic<std::size_t> _next_batch_idx{0};
};

}  // namespace sirius::scan_manager
