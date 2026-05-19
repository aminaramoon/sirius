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
#include <data/data_batch_utils.hpp>
#include <expression_executor/gpu_expression_executor.hpp>
#include <log/logging.hpp>
#include <op/scan/parquet_scan_info.hpp>
#include <op/scan/parquet_scan_operator_data.hpp>
#include <op/scan/scan_plan.hpp>
#include <op/scan/sirius_gpu_parquet_scan_operator.hpp>
#include <op/sirius_physical_operator.hpp>
#include <pipeline/sirius_meta_pipeline.hpp>
#include <scan_manager/split_connector.hpp>

// cudf
#include <cudf/io/parquet.hpp>
#include <cudf/table/table.hpp>
#include <cudf/utilities/memory_resource.hpp>

// cucascade
#include <cucascade/data/data_batch.hpp>
#include <cucascade/data/gpu_data_representation.hpp>

// standard library
#include <stdexcept>
#include <utility>

namespace sirius::op::scan {

//===----------------------------------------------------------------------===//
// Constructor
//===----------------------------------------------------------------------===//
sirius_gpu_parquet_scan_operator::sirius_gpu_parquet_scan_operator(
  duckdb::vector<sirius::logical_type> types,
  duckdb::idx_t estimated_cardinality,
  std::unique_ptr<parquet_scan_info> scan_info)
  : sirius_physical_operator(
      SiriusPhysicalOperatorType::GPU_PARQUET_SCAN, std::move(types), estimated_cardinality),
    _split_connector(std::make_unique<scan_manager::split_connector>()),
    _scan_info(std::move(scan_info))
{
  _split_connector->close();
}

sirius_gpu_parquet_scan_operator::~sirius_gpu_parquet_scan_operator() = default;

//===----------------------------------------------------------------------===//
// Friend access — wired by sirius_scan_manager during prepare_for_query.
//===----------------------------------------------------------------------===//
std::unique_ptr<parquet_scan_info> sirius_gpu_parquet_scan_operator::take_scan_info()
{
  return std::move(_scan_info);
}

void sirius_gpu_parquet_scan_operator::set_split_connector(
  std::unique_ptr<scan_manager::split_connector> connector)
{
  _split_connector = std::move(connector);
}

//===----------------------------------------------------------------------===//
// Scheduling interface
//===----------------------------------------------------------------------===//
std::optional<task_creation_hint> sirius_gpu_parquet_scan_operator::get_next_task_hint(
  std::optional<std::size_t> downstream_request)
{
  if (_split_connector->is_closed()) { return std::nullopt; }
  // Returns READY even when the split queue is empty but not yet closed. The dispatched worker
  // then parks inside split_connector::get_next_split's cv until the scan_manager pushes a split
  // or closes the connector. This trades one parked worker per active scan for not needing a
  // scheduler-visible wake-up signal from the scan_manager — the task_creator schedules the scan
  // op once and the worker self-perpetuates inside task_creator's "loop until all_ports_empty"
  // branch (see task_creator::manager_loop default else-branch). The ALL_TASKS request preserves
  // that self-perpetuating behavior.
  //
  // Lifecycle requirement: any error/drain path that joins task_creator's worker pool MUST first
  // close the operator's split_connector (e.g. via sirius_scan_manager::close_all_connectors());
  // otherwise the parked worker hangs forever and bounded_pool::wait_all deadlocks. Normal
  // completion is fine — the split_provider closes the connector when it has no more splits.
  // Sources have no upstream; if a downstream operator has communicated how many tasks it needs
  // we cap at that — otherwise default to ALL_TASKS (preserves self-perpetuating scan behavior).
  std::size_t cap = downstream_request.value_or(task_creation_hint::ALL_TASKS);
  return task_creation_hint{TaskCreationHint::READY, this, cap};
}

bool sirius_gpu_parquet_scan_operator::all_ports_empty() { return _split_connector->is_closed(); }

std::unique_ptr<operator_data> sirius_gpu_parquet_scan_operator::get_next_task_input_data()
{
  auto next = _split_connector->get_next_split();
  if (!next.has_value()) { return nullptr; }
  auto data = std::move(*next);

  // Fire fadvise(immediate) on each slice's per-scan datasource — "this
  // task is being created; if your backend prefers immediate prefetch
  // (fast local IO), kick off the work now."  Honored only when the
  // ioctx's preferred_prefetching_mode is immediate; otherwise the call
  // is a no-op (the provider's opportunistic fadvise already covered slow
  // backends).  The matching disposable fires from prepare_for_processing.
  if (auto* scan_data = dynamic_cast<parquet_scan_data*>(data.get())) {
    auto const pipeline = get_pipeline();
    auto const pid      = pipeline ? pipeline->get_pipeline_id() : std::size_t{0};
    SIRIUS_LOG_INFO("[fadvise immediate] op='{}' op_id={} pipeline_id={} slices={}",
                    get_name(),
                    get_operator_id(),
                    pid,
                    scan_data->rg_slices.size());
    for (auto& slice : scan_data->rg_slices) {
      if (slice.datasource) {
        slice.datasource->fadvise(sirius::io::prefetching_mode::immediate, slice.ranges);
      }
    }
  }
  return data;
}

//===----------------------------------------------------------------------===//
// read_table_from_metadata()
//===----------------------------------------------------------------------===//
std::unique_ptr<cudf::table> sirius_gpu_parquet_scan_operator::read_table_from_metadata(
  parquet_scan_data& scan_data, rmm::cuda_stream_view stream)
{
  auto filter_expression = scan_data.filter_expression;

  // Build reader options for this partition's files and corresponding row groups. The
  // FileMetaData on each slice is the cached footer parsed once by parquet_split_provider;
  // passing it through avoids re-fetching footers from disk for every split of a file.
  std::vector<std::unique_ptr<cudf::io::datasource>> sources;
  std::vector<cudf::io::parquet::FileMetaData> metadatas;
  std::vector<std::vector<cudf::size_type>> rg_per_src;
  sources.reserve(scan_data.rg_slices.size());
  metadatas.reserve(scan_data.rg_slices.size());
  rg_per_src.reserve(scan_data.rg_slices.size());

  for (auto& slice : scan_data.rg_slices) {
    // Move the per-slice sirius_datasource out into the cudf sources
    // vector.  The split_provider always attached a datasource (either
    // sirius-backed or kvikio-fallback through the same interface); after
    // this point the slice no longer owns it, but no other call site
    // touches slice.datasource after execute() begins.
    sources.push_back(std::move(slice.datasource));
    metadatas.push_back(*slice.file_metadata);  // copy unavoidable: cudf takes by value
    rg_per_src.push_back(slice.row_group_indices);
  }
  auto opts = *scan_data.reader_options;
  opts.set_row_groups(std::move(rg_per_src));

  // Try to translate the filter to a *stream-local* cudf AST for filter pushdown. If not
  // successful, fall back to post-read DuckDB-expression evaluation.
  std::optional<gpu_expression_translator::translated_expression> ast_expression = std::nullopt;
  if (filter_expression) {
    auto name_resolver = [plan = scan_data.plan](duckdb::idx_t ref_index) -> std::string {
      return plan->batch_column_name(ref_index);
    };
    gpu_expression_translator translator(stream, cudf::get_current_device_resource_ref());
    ast_expression = translator.translate_expression_with_names(*filter_expression, name_resolver);
    if (ast_expression) {
      opts.set_filter(ast_expression->back());
      SIRIUS_LOG_DEBUG(
        "[sirius_gpu_parquet_scan_operator] Translated filter expression for parquet reader filter "
        "pushdown.");
    } else {
      SIRIUS_LOG_DEBUG(
        "[sirius_gpu_parquet_scan_operator] AST translation failed for parquet reader filter "
        "pushdown.");
    }
  }

  rmm::device_async_resource_ref mr_ref(scan_data.gpu_memory_space->get_default_allocator());
  auto [table, metadata] =
    cudf::io::read_parquet(std::move(sources), std::move(metadatas), opts, stream, mr_ref);

  SIRIUS_LOG_DEBUG(
    "[sirius_gpu_parquet_scan_operator] Read {} file(s) (first: {}) — {} rows, {} columns",
    scan_data.rg_slices.size(),
    scan_data.rg_slices.empty() ? "<none>" : scan_data.rg_slices.front().file_path,
    table->num_rows(),
    table->num_columns());

  // Apply the filter if it was not pushed down into the parquet scan.
  if (filter_expression && !ast_expression) {
    sirius::gpu_expression_executor gpu_expression_executor(
      filter_expression.get(), cudf::get_current_device_resource_ref(), stream);
    // Hold the source table in a separate variable so its destruction (and the
    // stream-ordered dealloc of its buffers) is sequenced after select()'s kernels
    // have been queued. Assigning select()'s result directly back to `table` would
    // destruct the source synchronously before the queued kernels read from it.
    auto input_table = std::move(table);
    table            = gpu_expression_executor.select(input_table->view());
    SIRIUS_LOG_DEBUG(
      "[sirius_gpu_parquet_scan_operator] Applied duckdb filter expression post parquet scan.");
  }

  // Reshape the reader's output to the scan_plan's D-order layout: reorder data
  // columns, drop pure-filter columns, inject hive-partition columns. Skipped
  // when the scan is a trivial identity (no partitions, 1:1 data layout) or a
  // SELECT count(*) shape — see needs_output_assembly().
  if (needs_output_assembly(*scan_data.plan)) {
    table =
      assemble_scan_output(*scan_data.plan, std::move(table), scan_data.partition_values, stream);
    SIRIUS_LOG_DEBUG("[sirius_gpu_parquet_scan_operator] Assembled scan output to plan layout.");
  }

  return std::move(table);
}

//===----------------------------------------------------------------------===//
// execute()
//===----------------------------------------------------------------------===//
std::unique_ptr<operator_data> sirius_gpu_parquet_scan_operator::execute(
  const operator_data& input_data, rmm::cuda_stream_view stream)
{
  std::unique_ptr<cudf::table> table;
  cucascade::memory::memory_space* mem_space = nullptr;

  if (auto const* scan_data = dynamic_cast<const parquet_scan_data*>(&input_data)) {
    if (!scan_data->gpu_memory_space) {
      throw std::runtime_error(
        "[sirius_gpu_parquet_scan_operator] execute() called with null gpu_memory_space in "
        "input_data.");
    }
    // The base-class execute signature is `const operator_data&`, but the
    // scan operator is a source: it consumes its input_data exactly once
    // and there are no other readers.  read_table_from_metadata needs a
    // mutable reference so it can move each slice's owning
    // sirius_datasource into cudf's sources vector.
    auto& mutable_scan_data = const_cast<parquet_scan_data&>(*scan_data);
    table                   = read_table_from_metadata(mutable_scan_data, stream);
    mem_space               = scan_data->gpu_memory_space;
  } else if (auto const* cached = dynamic_cast<const scan_cached_operator_data*>(&input_data)) {
    auto ro_batch    = cached->batch->to_read_only();
    auto& gpu_rep    = ro_batch.get_data()->cast<cucascade::gpu_table_representation>();
    auto cached_view = gpu_rep.get_table_view();

    // The cached path carries only a DuckDB-expression filter (AST translation /
    // reader pushdown is not applicable to an already-materialized cached table)
    // and a shared scan_plan that decides whether the cached batch needs to be
    // reshaped to the plan's output layout.
    auto const has_filter     = static_cast<bool>(cached->filter_expression);
    auto const needs_assembly = needs_output_assembly(*cached->plan);

    if (!has_filter && !needs_assembly) {
      // Fast path: forward the cached batch unchanged. Avoids any allocation or copy
      // since the cached batch's gpu_table_representation already views the pinned
      // columns and co-owns them via the shared_ptr<column> chunks. Assembly is a
      // no-op when the scan_plan's output_layout is identity over data_columns.
      std::vector<std::shared_ptr<cucascade::data_batch>> batches;
      batches.push_back(cached->batch);
      return std::make_unique<pipelineable_operator_data>(std::move(batches));
    }

    if (has_filter) {
      // select() consumes the view and produces an owning table containing every D-column
      // (filter columns included) restricted to the rows that pass the predicate.
      sirius::gpu_expression_executor gpu_expression_executor(
        cached->filter_expression.get(), cudf::get_current_device_resource_ref(), stream);
      table = gpu_expression_executor.select(cached_view);
      SIRIUS_LOG_DEBUG(
        "[sirius_gpu_parquet_scan_operator] Applied duckdb filter expression on cached batch.");
    } else {
      // No filter but assembly is needed (the plan permutes / drops columns relative to
      // data_columns). Materialize the cached view into an owning table so
      // assemble_scan_output can release the columns by D-position and reassemble them
      // in output_layout order.
      std::vector<std::unique_ptr<cudf::column>> owned_cols;
      owned_cols.reserve(cached_view.num_columns());
      for (cudf::size_type i = 0; i < cached_view.num_columns(); ++i) {
        owned_cols.push_back(std::make_unique<cudf::column>(
          cached_view.column(i), stream, cudf::get_current_device_resource_ref()));
      }
      table = std::make_unique<cudf::table>(std::move(owned_cols));
    }

    if (needs_assembly) {
      // The cached path is gated upstream against hive partitions, so partition_values is
      // always empty here — assemble_scan_output's PARTITION branch is never exercised.
      table =
        assemble_scan_output(*cached->plan, std::move(table), /*partition_values=*/{}, stream);
      SIRIUS_LOG_DEBUG("[sirius_gpu_parquet_scan_operator] Assembled cached batch to plan layout.");
    }

    mem_space = ro_batch.get_memory_space();
  } else {
    throw std::runtime_error(
      "[sirius_gpu_parquet_scan_operator] execute() called with unexpected operator_data type; "
      "expected parquet_scan_data or scan_cached_operator_data.");
  }

  // Wrap the GPU table in operator_data for the downstream pipeline.
  auto batch = sirius::make_data_batch(std::move(table), *mem_space);
  std::vector<std::shared_ptr<cucascade::data_batch>> batches;
  batches.push_back(std::move(batch));
  return std::make_unique<pipelineable_operator_data>(std::move(batches));
}

std::size_t sirius_gpu_parquet_scan_operator::no_history_peak_memory_estimate(
  const op::input_stats& stats) const
{
  if (stats.type == op::operator_data_type::SCAN_CACHED) { return stats.bytes; }
  return stats.bytes * 8;
}

}  // namespace sirius::op::scan
