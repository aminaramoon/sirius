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

#include "data/data_batch_utils.hpp"
#include "op/sirius_physical_operator.hpp"
#include "scan_manager/cached_split_connector.hpp"
#include "scan_manager/split_connector.hpp"

#include <cudf/column/column.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/table/table_view.hpp>

#include <cucascade/data/data_batch.hpp>
#include <cucascade/data/gpu_data_representation.hpp>

#include <optional>
#include <utility>
#include <vector>

namespace sirius::scan_manager {

cached_split_connector::cached_split_connector(
  std::vector<std::vector<std::shared_ptr<cudf::column>>> columns_per_request,
  cucascade::memory::memory_space& memory_space,
  std::shared_ptr<duckdb::Expression> filter_expression,
  std::shared_ptr<op::scan::scan_plan const> plan)
  : _columns_per_request(std::move(columns_per_request)),
    _memory_space(&memory_space),
    _filter_expression(std::move(filter_expression)),
    _plan(std::move(plan))
{
}

std::optional<std::unique_ptr<op::operator_data>> cached_split_connector::get_next_split()
{
  std::size_t batch_idx = _next_batch_idx.fetch_add(1);
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
  return std::nullopt;
}

bool cached_split_connector::is_closed() const
{
  return _next_batch_idx.load() >= _columns_per_request[0].size();
}

bool cached_split_connector::has_more_splits() const { return !is_closed(); }

}  // namespace sirius::scan_manager
