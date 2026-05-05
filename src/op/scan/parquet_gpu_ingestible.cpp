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

// sirius
#include <expression_executor/gpu_expression_executor.hpp>
#include <expression_executor/gpu_expression_translator_internal.hpp>
#include <log/logging.hpp>
#include <op/scan/parquet_gpu_ingestible.hpp>
#include <op/scan/parquet_scan_operator_data.hpp>  // hybrid_scan_reader alias
#include <op/scan/parquet_schema_mapping.hpp>
#include <op/scan/scan_utils.hpp>
#include <sirius/exception.hpp>

// cudf
#include <cudf/io/datasource.hpp>
#include <cudf/io/parquet.hpp>
#include <cudf/io/parquet_io_utils.hpp>
#include <cudf/table/table.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>

// standard library
#include <stdexcept>
#include <unordered_set>
#include <utility>

namespace sirius::op::scan {

//===----------------------------------------------------------------------===//
// parquet_ingestible_table_info
//===----------------------------------------------------------------------===//
bool parquet_ingestible_table_info::is_injestible_with(std::string_view filename)
{
  constexpr std::string_view PARQUET_EXTENSION = ".parquet";
  return filename.ends_with(PARQUET_EXTENSION);
}

std::unique_ptr<io::post_filter_and_projection_info>
parquet_ingestible_table_info::get_filter_and_pojection_info() const
{
  // No table-wide post-filter info — parquet builds it per-split inside
  // parquet_gpu_ingestible::ensure_enumerated.
  return nullptr;
}

//===----------------------------------------------------------------------===//
// parquet_split_info
//===----------------------------------------------------------------------===//
parquet_split_info::parquet_split_info(
  std::string file_path,
  std::vector<cudf::size_type> row_group_indices,
  std::shared_ptr<cudf::io::parquet_reader_options> reader_options,
  std::shared_ptr<cudf::io::datasource> datasource)
  : file_path(std::move(file_path)),
    row_group_indices(std::move(row_group_indices)),
    reader_options(std::move(reader_options)),
    datasource(std::move(datasource))
{
}

//===----------------------------------------------------------------------===//
// parquet_post_filter_and_projection_info
//===----------------------------------------------------------------------===//
parquet_post_filter_and_projection_info::parquet_post_filter_and_projection_info(
  std::shared_ptr<duckdb::Expression> filter_expression,
  partition_inject_fn_t inject_fn,
  std::string file_path)
  : filter_expression(std::move(filter_expression)),
    inject_fn(std::move(inject_fn)),
    file_path(std::move(file_path))
{
}

//===----------------------------------------------------------------------===//
// parquet_gpu_ingestible
//===----------------------------------------------------------------------===//
parquet_gpu_ingestible::parquet_gpu_ingestible(std::unique_ptr<io::ingestible_table_info> info)
  : io::gpu_ingestible(std::move(info))
{
  auto* parquet_info = dynamic_cast<parquet_ingestible_table_info*>(_table_info.get());
  if (!parquet_info) {
    throw std::runtime_error(
      "[parquet_gpu_ingestible] Expected parquet_ingestible_table_info; got a different "
      "concrete type.");
  }

  // Mirror parquet_split_provider's name-presence requirement.
  bool const needs_names = !parquet_info->projection_ids.empty() ||
                           (parquet_info->table_filters &&
                            !parquet_info->table_filters->filters.empty()) ||
                           !parquet_info->partition_indices.empty();
  if (needs_names && parquet_info->names.empty()) {
    throw sirius::internal_exception(
      "[parquet_gpu_ingestible] Projection, filter pushdown, or hive partitions "
      "require column names to be provided.");
  }

  _file_paths             = parquet_info->file_paths;
  _approximate_batch_size = parquet_info->approximate_batch_size;

  // Canonical scan_plan — single source of truth for D-order, output layout, C→D map.
  _plan = std::make_shared<scan_plan const>(build_scan_plan(parquet_info->column_ids,
                                                            parquet_info->projection_ids,
                                                            parquet_info->names,
                                                            parquet_info->returned_types,
                                                            parquet_info->scan_output_arity,
                                                            parquet_info->partition_indices));

  // Post-read assembly closure (null when the plan is identity).
  if (auto inject_fn = _plan->build_inject_fn()) {
    _partition_inject_fn = std::move(inject_fn);
  }

  // DuckDB filter expression (partition-column filters dropped). AST translation
  // success is predicted once below; the actual filter — pushdown or post-read —
  // is evaluated per task with a task-local stream.
  if (parquet_info->table_filters && !parquet_info->table_filters->filters.empty()) {
    auto batch_column_map = _plan->make_batch_column_map();
    auto duckdb_expression =
      op::convert_table_filters_to_expression(*parquet_info->table_filters,
                                              parquet_info->column_ids,
                                              parquet_info->returned_types,
                                              batch_column_map,
                                              _plan->partition_primary_indices);
    if (duckdb_expression) { _duckdb_filter_expression = std::move(duckdb_expression); }
  }

  // Shared reader options — column-name projection set once.
  _reader_options = std::make_shared<cudf::io::parquet_reader_options>(
    cudf::io::parquet_reader_options::builder().build());
  if (_plan->is_projected()) {
    _reader_options->set_column_names(_plan->data_column_names());
  }

  // Predict AST translation success once. The translator's stream is only used
  // for scalar allocation; translation success is structurally determined by
  // the expression shape and the name resolver, so the result is portable to
  // any later task-local stream.
  if (_duckdb_filter_expression) {
    auto name_resolver = [this](duckdb::idx_t ref_index) -> std::string {
      return _plan->batch_column_name(ref_index);
    };
    gpu_expression_translator translator(cudf::get_default_stream(),
                                         cudf::get_current_device_resource_ref());
    auto ast =
      translator.translate_expression_with_names(*_duckdb_filter_expression, name_resolver);
    _can_pushdown = ast.has_value();
    if (_can_pushdown) {
      SIRIUS_LOG_DEBUG(
        "[parquet_gpu_ingestible] AST translation succeeded; filter will be pushed down "
        "into the parquet reader.");
    } else {
      SIRIUS_LOG_DEBUG(
        "[parquet_gpu_ingestible] AST translation failed; filter will be applied post-read.");
    }
  }
}

parquet_gpu_ingestible::~parquet_gpu_ingestible() = default;

//===----------------------------------------------------------------------===//
// get_next_split
//===----------------------------------------------------------------------===//
std::unique_ptr<io::scan_and_filter_metadata> parquet_gpu_ingestible::get_next_split()
{
  ensure_enumerated();

  std::lock_guard lock(_enumerate_mutex);
  if (_splits.empty()) { return nullptr; }
  auto next = std::move(_splits.front());
  _splits.pop_front();
  return next;
}

//===----------------------------------------------------------------------===//
// ensure_enumerated  (mirrors parquet_split_provider::run_batch, single-threaded)
//===----------------------------------------------------------------------===//
void parquet_gpu_ingestible::ensure_enumerated()
{
  std::lock_guard lock(_enumerate_mutex);
  if (_enumerated) { return; }
  _enumerated = true;
  for (auto const& file_path : _file_paths) {
    enumerate_file(file_path);
  }
}

void parquet_gpu_ingestible::enumerate_file(std::string const& file_path)
{
  auto stream = cudf::get_default_stream();

  //===----------Read metadata footer----------===//
  auto datasource    = cudf::io::datasource::create(file_path);
  auto footer_buffer = cudf::io::parquet::fetch_footer_to_host(*datasource);

  //===----------Parse metadata----------===//
  hybrid_scan_reader reader(
    cudf::host_span<uint8_t const>(footer_buffer->data(), footer_buffer->size()),
    *_reader_options);
  auto metadata = reader.parquet_metadata();

  //===----------AST-translate filter for row-group pruning----------===//
  std::optional<gpu_expression_translator::translated_expression> ast_expression = std::nullopt;
  auto pruning_options = *_reader_options;
  if (_duckdb_filter_expression) {
    auto name_resolver = [this](duckdb::idx_t ref_index) -> std::string {
      return _plan->batch_column_name(ref_index);
    };
    gpu_expression_translator translator(stream, cudf::get_current_device_resource_ref());
    ast_expression =
      translator.translate_expression_with_names(*_duckdb_filter_expression, name_resolver);
    if (ast_expression) {
      pruning_options.set_filter(ast_expression->back());
      SIRIUS_LOG_DEBUG(
        "[parquet_gpu_ingestible] Translated filter expression for row group pruning.");
    } else {
      SIRIUS_LOG_DEBUG("[parquet_gpu_ingestible] AST translation failed for row group pruning.");
    }
  }

  //===----------Resolve selected DuckDB columns to parquet column-chunk indices----------===//
  auto const data_column_names = _plan->data_column_names();
  std::vector<std::size_t> selected_chunk_indices;
  std::unordered_set<std::size_t> pure_filter_chunk_indices;
  if (_plan->is_projected()) {
    auto const pure_filter_positions = _plan->pure_filter_batch_positions();
    selected_chunk_indices.reserve(data_column_names.size());
    for (std::size_t k = 0; k < data_column_names.size(); ++k) {
      auto leaves = detail::leaf_indices_for_column(metadata, data_column_names[k]);
      if (leaves.empty()) {
        throw std::runtime_error("[parquet_gpu_ingestible] Projected column '" +
                                 data_column_names[k] +
                                 "' not found in parquet file: " + file_path);
      }
      bool const is_pure_filter = pure_filter_positions.count(k);
      for (auto const leaf : leaves) {
        selected_chunk_indices.push_back(leaf);
        if (is_pure_filter) { pure_filter_chunk_indices.insert(leaf); }
      }
    }
  }

  //===----------Row-group pruning----------===//
  auto row_group_indices = reader.all_row_groups(pruning_options);
  if (ast_expression) {
    auto const row_groups_before = row_group_indices.size();
    row_group_indices =
      reader.filter_row_groups_with_stats(row_group_indices, pruning_options, stream);
    auto const row_groups_after = row_group_indices.size();
    SIRIUS_LOG_DEBUG(
      "[parquet_gpu_ingestible] Row group pruning: file: {} before: {} after: {} (pruned {})",
      file_path,
      row_groups_before,
      row_groups_after,
      row_groups_before - row_groups_after);
  }

  //===----------Row-group partitioning by uncompressed bytes----------===//
  auto datasource_shared = std::shared_ptr<cudf::io::datasource>(std::move(datasource));

  std::size_t partition_uncompressed_bytes = 0;
  std::vector<cudf::size_type> partition_rg_indices;
  partition_rg_indices.reserve(row_group_indices.size());

  auto flush_partition = [&]() {
    if (partition_rg_indices.empty()) { return; }
    auto rg_indices              = std::move(partition_rg_indices);
    partition_rg_indices         = {};
    partition_uncompressed_bytes = 0;

    auto split_info = std::make_unique<parquet_split_info>(
      file_path, std::move(rg_indices), _reader_options, datasource_shared);

    // Carry filter_expression on the post-filter info ONLY when we can't push it
    // down into the reader; otherwise materialize_table applies it via pushdown
    // and we'd be applying the filter twice.
    std::shared_ptr<duckdb::Expression> post_filter_expression =
      _can_pushdown ? nullptr : _duckdb_filter_expression;

    std::unique_ptr<parquet_post_filter_and_projection_info> filter_info;
    bool const needs_post_work =
      static_cast<bool>(post_filter_expression) || static_cast<bool>(_partition_inject_fn);
    if (needs_post_work) {
      filter_info = std::make_unique<parquet_post_filter_and_projection_info>(
        std::move(post_filter_expression), _partition_inject_fn, file_path);
    }

    _splits.push_back(std::make_unique<io::scan_and_filter_metadata>(std::move(split_info),
                                                                     std::move(filter_info)));
  };

  auto accumulate_chunk = [&](cudf::io::parquet::ColumnChunk const& chunk, bool is_pure_filter) {
    auto const& column_metadata = chunk.meta_data;
    if (column_metadata.total_uncompressed_size > 0 && !is_pure_filter) {
      partition_uncompressed_bytes +=
        static_cast<std::size_t>(column_metadata.total_uncompressed_size);
    }
  };

  for (auto const rg_idx : row_group_indices) {
    auto const& row_group = metadata.row_groups[rg_idx];
    partition_rg_indices.push_back(rg_idx);

    if (_plan->is_projected()) {
      for (auto const chunk_idx : selected_chunk_indices) {
        accumulate_chunk(row_group.columns[chunk_idx],
                         pure_filter_chunk_indices.contains(chunk_idx));
      }
    } else {
      for (auto const& chunk : row_group.columns) {
        accumulate_chunk(chunk, false);
      }
    }

    if (partition_uncompressed_bytes >= _approximate_batch_size) { flush_partition(); }
  }

  // Trailing partition smaller than the target size.
  flush_partition();
}

//===----------------------------------------------------------------------===//
// materialize_table  (read portion of read_table_from_metadata, with AST pushdown
//                     when the filter is translatable)
//===----------------------------------------------------------------------===//
std::unique_ptr<cudf::table> parquet_gpu_ingestible::materialize_table(
  io::scan_info const& info,
  rmm::device_async_resource_ref /*mr*/,
  rmm::cuda_stream_view stream)
{
  auto const* split = dynamic_cast<parquet_split_info const*>(&info);
  if (!split) {
    throw std::runtime_error(
      "[parquet_gpu_ingestible] materialize_table received a non-parquet scan_info.");
  }

  auto opts = *split->reader_options;
  opts.set_source(cudf::io::source_info{split->datasource.get()});
  opts.set_row_groups({split->row_group_indices});

  // AST filter pushdown into the reader. Translation success was predicted
  // structurally at construction (_can_pushdown); we re-translate here so the
  // produced scalars live on the task-local stream.
  std::optional<gpu_expression_translator::translated_expression> ast_expression = std::nullopt;
  if (_can_pushdown && _duckdb_filter_expression) {
    auto name_resolver = [this](duckdb::idx_t ref_index) -> std::string {
      return _plan->batch_column_name(ref_index);
    };
    gpu_expression_translator translator(stream, cudf::get_current_device_resource_ref());
    ast_expression =
      translator.translate_expression_with_names(*_duckdb_filter_expression, name_resolver);
    if (!ast_expression) {
      // Translation success is structurally deterministic; reaching here means
      // the predicate at construction disagrees with this site, which would
      // silently drop the filter (post-info doesn't carry it when _can_pushdown).
      throw std::runtime_error(
        "[parquet_gpu_ingestible] AST translation succeeded at construction but failed at "
        "materialize time; this would silently drop the filter.");
    }
    opts.set_filter(ast_expression->back());
    SIRIUS_LOG_DEBUG(
      "[parquet_gpu_ingestible] Pushed translated filter expression into the parquet reader.");
  }

  auto [table, metadata] = cudf::io::read_parquet(opts, stream);

  SIRIUS_LOG_DEBUG("[parquet_gpu_ingestible] Read {} — {} rows, {} columns",
                   split->file_path,
                   table->num_rows(),
                   table->num_columns());

  return std::move(table);
}

//===----------------------------------------------------------------------===//
// post_filter_and_project  (post-read filter + assembly closure)
//===----------------------------------------------------------------------===//
std::unique_ptr<cudf::table> parquet_gpu_ingestible::post_filter_and_project(
  cudf::table_view input,
  io::post_filter_and_projection_info const& info,
  rmm::device_async_resource_ref mr,
  rmm::cuda_stream_view stream)
{
  auto const* parquet_info = dynamic_cast<parquet_post_filter_and_projection_info const*>(&info);
  if (!parquet_info) {
    throw std::runtime_error(
      "[parquet_gpu_ingestible] post_filter_and_project received a non-parquet info.");
  }

  std::unique_ptr<cudf::table> table;

  if (parquet_info->filter_expression) {
    sirius::gpu_expression_executor executor(parquet_info->filter_expression.get(), mr, stream);
    table = executor.select(input);
    SIRIUS_LOG_DEBUG(
      "[parquet_gpu_ingestible] Applied DuckDB filter expression post parquet scan.");
  } else {
    // No filter — materialize the input view into an owning table so inject_fn
    // can take ownership and rearrange columns.
    std::vector<std::unique_ptr<cudf::column>> cols;
    cols.reserve(input.num_columns());
    for (cudf::size_type i = 0; i < input.num_columns(); ++i) {
      cols.push_back(std::make_unique<cudf::column>(input.column(i), stream, mr));
    }
    table = std::make_unique<cudf::table>(std::move(cols));
  }

  if (parquet_info->inject_fn) {
    table = parquet_info->inject_fn(std::move(table), parquet_info->file_path, stream);
    SIRIUS_LOG_DEBUG("[parquet_gpu_ingestible] Applied scan_plan inject_fn.");
  }

  return table;
}

}  // namespace sirius::op::scan
