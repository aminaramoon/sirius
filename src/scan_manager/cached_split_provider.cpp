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

#include "scan_manager/cached_split_provider.hpp"

#include "data/data_batch_utils.hpp"
#include "op/scan/parquet_scan_operator_data.hpp"
#include "op/sirius_physical_operator.hpp"

#include <cudf/column/column.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/table/table_view.hpp>

#include <cucascade/data/data_batch.hpp>
#include <cucascade/data/gpu_data_representation.hpp>

#include <stdexcept>
#include <utility>
#include <vector>

namespace sirius::scan_manager {

cached_split_provider::cached_split_provider(
  std::vector<std::vector<std::shared_ptr<cudf::column>>> columns_per_request,
  cucascade::memory::memory_space& memory_space,
  std::shared_ptr<duckdb::Expression> filter_expression,
  std::shared_ptr<op::scan::scan_plan const> plan)
  : _columns_per_request(std::move(columns_per_request)),
    _memory_space(&memory_space),
    _filter_expression(std::move(filter_expression)),
    _plan(std::move(plan))
{
  _num_batches = _columns_per_request.empty() ? 0 : _columns_per_request.front().size();

  // Sanity-check: every column must contribute the same number of chunks. Done in the
  // constructor so failure surfaces synchronously to the caller before run().
  for (auto const& col_chunks : _columns_per_request) {
    if (col_chunks.size() != _num_batches) {
      throw std::runtime_error(
        "[cached_split_provider] mismatched chunk count across requested columns");
    }
  }
}

std::vector<std::unique_ptr<op::operator_data>> cached_split_provider::create_split()
{
  // Atomic claim of the next batch index lets multiple workers run create_split
  // in parallel without a mutex.
  auto const batch_idx = _next_batch_idx.fetch_add(1, std::memory_order_relaxed);
  if (batch_idx >= _num_batches) { return {}; }

  std::vector<cudf::column_view> col_views;
  col_views.reserve(_columns_per_request.size());
  // Owner keeps the shared_ptr<column> chunks alive for the lifetime of the
  // emitted data_batch, so the table_view's pointers stay valid even though
  // the gpu_table_representation does not own the underlying memory.
  std::vector<std::shared_ptr<cudf::column>> owner;
  owner.reserve(_columns_per_request.size());
  std::size_t alloc_size = 0;
  for (auto const& col_chunks : _columns_per_request) {
    auto const& col_ptr = col_chunks[batch_idx];
    col_views.emplace_back(col_ptr->view());
    alloc_size += col_ptr->alloc_size();
    owner.push_back(col_ptr);
  }

  cudf::table_view view(col_views);
  auto gpu_repr = std::make_unique<cucascade::gpu_table_representation>(
    view, std::move(owner), alloc_size, *_memory_space);
  auto batch =
    std::make_shared<cucascade::data_batch>(::sirius::get_next_batch_id(), std::move(gpu_repr));

  std::vector<std::unique_ptr<op::operator_data>> out;
  out.push_back(std::make_unique<op::scan::scan_cached_operator_data>(
    std::move(batch), _filter_expression, _plan));
  return out;
}

}  // namespace sirius::scan_manager
