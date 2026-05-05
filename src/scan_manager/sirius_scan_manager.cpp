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

#include "scan_manager/sirius_scan_manager.hpp"

#include "exec/scoped_dispatcher.hpp"
#include "log/logging.hpp"
#include "op/scan/parquet_scan_info.hpp"
#include "op/scan/scan_plan.hpp"
#include "op/scan/scan_utils.hpp"
#include "op/scan/sirius_gpu_parquet_scan_operator.hpp"
#include "op/scan/sirius_gpu_scan_operator.hpp"
#include "op/sirius_physical_operator_type.hpp"
#include "pipeline/sirius_pipeline.hpp"
#include "planner/query.hpp"
#include "scan_manager/cached_split_connector.hpp"
#include "scan_manager/metadata_split_connector.hpp"
#include "scan_manager/split_connector.hpp"

#include <algorithm>
#include <cassert>
#include <utility>

namespace sirius::scan_manager {

sirius_scan_manager::sirius_scan_manager(exec::thread_pool_config config)
  : _config(std::move(config)),
    _thread_pool(_config.num_threads, _config.thread_name_prefix, _config.cpu_affinity_list),
    _dispatcher(std::make_unique<exec::scoped_dispatcher>(_thread_pool, _config.num_threads))
{
}

sirius_scan_manager::~sirius_scan_manager() { stop(); }

void sirius_scan_manager::prepare_for_query(const sirius::planner::query& query)
{
  reset();

  SIRIUS_LOG_DEBUG("[sirius_scan_manager::prepare_for_query] pipelines={}",
                   query.get_pipelines().size());

  for (auto const& pipeline : query.get_pipelines()) {
    if (!pipeline) { continue; }
    auto source = pipeline->get_source();
    if (!source) { continue; }
    if (source->type != ::sirius::op::SiriusPhysicalOperatorType::GPU_SCAN) { continue; }

    auto* op = &source->Cast<op::scan::sirius_gpu_scan_operator>();
    if (_providers_by_op.contains(op)) { continue; }

    auto connector = create_connector_for(op);
    if (!connector) {
      SIRIUS_LOG_ERROR(
        "[sirius_scan_manager::prepare_for_query] failed to create split_connector for op_id={}",
        op->get_operator_id());
      throw std::runtime_error("failed to create split_connector for scan operator");
    }
    op->set_split_connector(std::move(connector));
    _scan_op_order.push_back(op);

    SIRIUS_LOG_TRACE("[sirius_scan_manager::prepare_for_query] registered op_id={}",
                     op->get_operator_id());
  }
}

std::unique_ptr<split_connector> sirius_scan_manager::create_connector_for(
  op::scan::sirius_gpu_scan_operator* op)
{
  return nullptr;

  // auto& table_info = op->get_table_info();

  // try {
  //   for (auto const& [pinned_name, entry] : _pinned_entries) {
  //     if (!table_info.matches_files(entry.file_paths)) { continue; }
  //     if (entry.memory_space == nullptr) {
  //       throw std::runtime_error("[sirius_scan_manager::create_provider_for] pinned entry '" +
  //                                pinned_name + "' has no memory_space");
  //     }

  //     // Build the canonical scan_plan once. Everything downstream — cached column
  //     // layout, filter pushdown indices, post-read assembly — reads from this.
  //     // Held by shared_ptr<const> so each emitted scan_cached_operator_data can
  //     // carry it to the GPU scan operator's per-task assembly check without copying.
  //     auto plan_shared = std::make_shared<op::scan::scan_plan const>(
  //       op::scan::build_scan_plan(info->column_ids,
  //                                 info->projection_ids,
  //                                 info->names,
  //                                 info->returned_types,
  //                                 op->get_types().size(),
  //                                 info->partition_indices));
  //     auto const& plan = *plan_shared;

  //     // Hive partitions on a cached scan would require per-chunk file_path metadata
  //     // that pinned entries don't carry today. Fall through to the parquet path,
  //     // which extracts partition values per file at read time.
  //     if (plan.has_partitions()) {
  //       SIRIUS_LOG_DEBUG(
  //         "[sirius_scan_manager::create_provider_for] pinned entry '{}' matches op_id={} but "
  //         "scan has hive partitions; falling through to parquet_split_provider",
  //         pinned_name,
  //         op->get_operator_id());
  //       break;
  //     }

  //     // Look up the pinned chunks for each D-position by name. data_columns is in
  //     // D-order, so columns_per_request[d] is the chunk vector for D-position d.
  //     std::vector<std::vector<std::shared_ptr<cudf::column>>> columns_per_request;
  //     columns_per_request.reserve(plan.data_columns.size());
  //     for (auto const& dc : plan.data_columns) {
  //       auto it = entry.data_batches_by_column.find(dc.name);
  //       if (it == entry.data_batches_by_column.end()) {
  //         throw std::runtime_error("[sirius_scan_manager::create_provider_for] pinned entry '" +
  //                                  pinned_name + "' missing column '" + dc.name +
  //                                  "' required by scan op");
  //       }
  //       columns_per_request.push_back(it->second);
  //     }

  //     // Filter expression: BoundReferences are in D-space, via plan.batch_position_by_column_id.
  //     // Same recipe parquet_split_provider uses, so the filter evaluates correctly against
  //     // the cached batch (which is in D-order by construction above).
  //     std::shared_ptr<duckdb::Expression> filter_expression;
  //     if (info->table_filters && !info->table_filters->filters.empty()) {
  //       auto duckdb_expression =
  //         op::convert_table_filters_to_expression(*info->table_filters,
  //                                                 info->column_ids,
  //                                                 info->returned_types,
  //                                                 plan.batch_position_by_column_id,
  //                                                 plan.partition_primary_indices);
  //       if (duckdb_expression) {
  //         filter_expression = std::shared_ptr<duckdb::Expression>(std::move(duckdb_expression));
  //       }
  //     }

  //     SIRIUS_LOG_DEBUG(
  //       "[sirius_scan_manager::create_provider_for] using cached_split_provider for op_id={} "
  //       "(pinned='{}' data_cols={} needs_assembly={})",
  //       op->get_operator_id(),
  //       pinned_name,
  //       columns_per_request.size(),
  //       op::scan::needs_output_assembly(plan));
  //     return nullptr;

  //     // return std::make_unique<cached_split_provider>(std::move(columns_per_request),
  //     //                                                *entry.memory_space,
  //     //                                                std::move(filter_expression),
  //     //                                                std::move(plan_shared));
  //   }
  // } catch (...) {
  //   SIRIUS_LOG_TRACE("not all the columns are pinned for this query");
  // }
  // return std::make_unique<metadata_split_connector>(op->get_ingestible(), *op);
}

void sirius_scan_manager::reset()
{
  _dispatcher->request_stop();
  _dispatcher->wait_for_all();
  _providers_by_op.clear();
  _scan_op_order.clear();
  _dispatcher = std::make_unique<exec::scoped_dispatcher>(_thread_pool, _config.num_threads);
}

void sirius_scan_manager::start() {}

void sirius_scan_manager::stop()
{
  reset();
  _thread_pool.stop();
}

void sirius_scan_manager::insert_pinned_entry(const std::string& name,
                                              std::vector<std::string> column_names,
                                              std::vector<std::string> file_paths,
                                              std::vector<std::unique_ptr<cudf::table>> data_tables,
                                              cucascade::memory::memory_space& memory_space)
{
  // Compute the total row count of the incoming tables before releasing them
  // (release() empties the table; num_rows() would then return 0).
  std::size_t new_num_rows = 0;
  for (auto const& table : data_tables) {
    if (table) { new_num_rows += static_cast<std::size_t>(table->num_rows()); }
  }

  auto existing_it = _pinned_entries.find(name);
  if (existing_it != _pinned_entries.end()) {
    if (existing_it->second.num_rows == new_num_rows) {
      // Same row count → merge unique columns into the existing entry.
      auto& entry = existing_it->second;
      for (auto& table : data_tables) {
        if (!table) { continue; }
        auto cols = table->release();
        if (cols.size() != column_names.size()) {
          throw std::runtime_error(
            "[sirius_scan_manager::insert_pinned_entry] table column count " +
            std::to_string(cols.size()) + " does not match column_names size " +
            std::to_string(column_names.size()));
        }
        for (std::size_t i = 0; i < cols.size(); ++i) {
          auto const& col_name = column_names[i];
          if (entry.data_batches_by_column.contains(col_name)) {
            // Already cached — drop the duplicate column.
            continue;
          }
          entry.data_batches_by_column[col_name].emplace_back(std::move(cols[i]));
        }
      }
      // Append any new column names to the entry's column_names list so its
      // metadata reflects the union of pinned columns.
      for (auto& cn : column_names) {
        if (std::find(entry.column_names.begin(), entry.column_names.end(), cn) ==
            entry.column_names.end()) {
          entry.column_names.push_back(std::move(cn));
        }
      }
      return;
    }
    // Row count differs → drop the stale entry and rebuild below.
    _pinned_entries.erase(existing_it);
  }

  pinned_entry entry;
  entry.column_names = std::move(column_names);
  entry.file_paths   = std::move(file_paths);
  entry.memory_space = &memory_space;
  entry.num_rows     = new_num_rows;

  for (auto& table : data_tables) {
    if (!table) { continue; }
    auto cols = table->release();
    if (cols.size() != entry.column_names.size()) {
      throw std::runtime_error("[sirius_scan_manager::insert_pinned_entry] table column count " +
                               std::to_string(cols.size()) + " does not match column_names size " +
                               std::to_string(entry.column_names.size()));
    }
    for (std::size_t i = 0; i < cols.size(); ++i) {
      entry.data_batches_by_column[entry.column_names[i]].emplace_back(std::move(cols[i]));
    }
  }

  _pinned_entries[name] = std::move(entry);
}

void sirius_scan_manager::remove_pinned_entry(const std::string& name)
{
  _pinned_entries.erase(name);
}

}  // namespace sirius::scan_manager
