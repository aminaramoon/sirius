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

// sirius
#include <helper/logical_type.hpp>
#include <io/gpu_ingestible.hpp>
#include <op/scan/hive_partition.hpp>
#include <op/scan/scan_plan.hpp>
#include <sirius_config.hpp>

// duckdb
#include <duckdb/common/column_index.hpp>
#include <duckdb/common/multi_file/multi_file_data.hpp>
#include <duckdb/common/types.hpp>
#include <duckdb/common/vector.hpp>
#include <duckdb/planner/expression.hpp>
#include <duckdb/planner/table_filter.hpp>

// cudf
#include <cudf/io/datasource.hpp>
#include <cudf/io/parquet.hpp>
#include <cudf/io/text/byte_range_info.hpp>
#include <cudf/types.hpp>

// standard library
#include <cstddef>
#include <deque>
#include <memory>
#include <mutex>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace sirius::op::scan {

//===----------------------------------------------------------------------===//
// parquet_ingestible_table_info
//===----------------------------------------------------------------------===//
/**
 * @brief Concrete @c io::ingestible_table_info for parquet sources.
 *
 * Holds the same bind-data fields as the legacy @c parquet_scan_info struct;
 * @c parquet_gpu_ingestible interprets these to build the canonical scan_plan,
 * filter expression, and reader options shared across all splits.
 */
class parquet_ingestible_table_info : public io::ingestible_table_info {
 public:
  duckdb::vector<sirius::logical_type> returned_types;
  std::vector<std::string> file_paths;
  duckdb::vector<duckdb::ColumnIndex> column_ids;
  duckdb::vector<duckdb::idx_t> projection_ids;
  duckdb::vector<std::string> names;
  duckdb::unique_ptr<duckdb::TableFilterSet> table_filters;
  duckdb::vector<duckdb::HivePartitioningIndex> partition_indices;
  std::size_t scan_output_arity        = 0;
  std::size_t approximate_batch_size   = sirius::config::DEFAULT_SCAN_TASK_BATCH_SIZE;

  [[nodiscard]] io::ingestible_type type() const override
  {
    return io::ingestible_type::PARQUET;
  }

  bool is_injestible_with(std::string_view filename) override;

  [[nodiscard]] std::unique_ptr<io::post_filter_and_projection_info>
  get_filter_and_pojection_info() const override;
};

//===----------------------------------------------------------------------===//
// parquet_split_info
//===----------------------------------------------------------------------===//
/**
 * @brief Concrete @c io::scan_info — per-split data needed to materialize one
 *        row-group partition from one parquet file.
 *
 * Mirrors the per-task fields of @c parquet_scan_data minus the post-read
 * concerns (filter expression, scan_plan, partition_inject_fn), which travel
 * separately on the @c parquet_post_filter_and_projection_info.
 */
class parquet_split_info : public io::scan_info {
 public:
  parquet_split_info(std::string file_path,
                     std::vector<cudf::size_type> row_group_indices,
                     std::shared_ptr<cudf::io::parquet_reader_options> reader_options,
                     std::shared_ptr<cudf::io::datasource> datasource);

  [[nodiscard]] bool is_prefetchable() const override { return false; }

  [[nodiscard]] std::span<cudf::io::text::byte_range_info> get_prefetching_ranges()
    const override
  {
    return {};
  }

  std::string file_path;
  std::vector<cudf::size_type> row_group_indices;
  std::shared_ptr<cudf::io::parquet_reader_options> reader_options;
  std::shared_ptr<cudf::io::datasource> datasource;
};

//===----------------------------------------------------------------------===//
// parquet_post_filter_and_projection_info
//===----------------------------------------------------------------------===//
/**
 * @brief Concrete @c io::post_filter_and_projection_info for parquet splits.
 *
 * Carries the DuckDB filter expression (when present) and the post-read
 * assembly closure produced by @c scan_plan::build_inject_fn (when the plan
 * is non-identity). Either, both, or neither may be populated; @c nullptr in
 * the corresponding @c scan_and_filter_metadata indicates "no post-scan work".
 */
class parquet_post_filter_and_projection_info : public io::post_filter_and_projection_info {
 public:
  parquet_post_filter_and_projection_info(std::shared_ptr<duckdb::Expression> filter_expression,
                                          partition_inject_fn_t inject_fn,
                                          std::string file_path);

  std::shared_ptr<duckdb::Expression> filter_expression;
  partition_inject_fn_t inject_fn;
  std::string file_path;
};

//===----------------------------------------------------------------------===//
// parquet_gpu_ingestible
//===----------------------------------------------------------------------===//
/**
 * @brief @c io::gpu_ingestible implementation backed by parquet files.
 *
 * Construction mirrors @c parquet_split_provider's constructor: it builds the
 * canonical @c scan_plan, the post-read assembly closure, the DuckDB filter
 * expression, and the shared reader options.
 *
 * @c get_next_split mirrors @c parquet_split_provider::run_batch — but
 * synchronous and single-threaded. On the first call, it eagerly enumerates
 * splits across all files (read footer, AST-based row-group pruning,
 * row-group partitioning by @c approximate_batch_size); subsequent calls
 * dequeue the next split and return @c nullptr when the queue drains.
 *
 * @c materialize_table mirrors the read portion of
 * @c sirius_gpu_parquet_scan_operator::read_table_from_metadata, with one
 * intentional simplification: it does not attempt AST filter pushdown into
 * the parquet reader. The filter is always applied via
 * @c post_filter_and_project, which mirrors the rest of @c read_table_from_metadata
 * (DuckDB-expression filtering and post-read assembly).
 */
class parquet_gpu_ingestible : public io::gpu_ingestible {
 public:
  explicit parquet_gpu_ingestible(std::unique_ptr<io::ingestible_table_info> info);

  ~parquet_gpu_ingestible() override;

  std::unique_ptr<io::scan_and_filter_metadata> get_next_split() override;

  std::unique_ptr<cudf::table> materialize_table(io::scan_info const& info,
                                                 rmm::device_async_resource_ref mr,
                                                 rmm::cuda_stream_view stream) override;

  std::unique_ptr<cudf::table> post_filter_and_project(
    cudf::table_view input,
    io::post_filter_and_projection_info const& info,
    rmm::device_async_resource_ref mr,
    rmm::cuda_stream_view stream) override;

  [[nodiscard]] bool supports_prefetching() const override { return false; }

 private:
  /// One-shot, idempotent eager enumeration. Reads every file's footer, prunes
  /// row groups via AST-translated stats, partitions by approximate_batch_size,
  /// and pushes scan_and_filter_metadata onto _splits. Holds _enumerate_mutex.
  void ensure_enumerated();

  /// Enumerate splits for a single file (footer read + row-group partitioning).
  void enumerate_file(std::string const& file_path);

  // Fields populated at construction (mirror parquet_split_provider state).
  std::shared_ptr<scan_plan const> _plan;
  std::shared_ptr<duckdb::Expression> _duckdb_filter_expression;
  partition_inject_fn_t _partition_inject_fn;
  std::shared_ptr<cudf::io::parquet_reader_options> _reader_options;
  std::vector<std::string> _file_paths;
  std::size_t _approximate_batch_size = 0;

  /// True when the DuckDB filter expression can be translated to a cudf AST and
  /// therefore pushed down into the parquet reader during @c materialize_table.
  /// When false, the filter is applied post-read inside @c post_filter_and_project.
  /// Computed once at construction with @c cudf::get_default_stream — translation
  /// success is structurally determined by the expression and the name resolver
  /// (both fixed), so the prediction holds for the task-stream re-translation.
  bool _can_pushdown = false;

  // Pre-enumerated splits; produced lazily on first get_next_split().
  std::deque<std::unique_ptr<io::scan_and_filter_metadata>> _splits;
  std::mutex _enumerate_mutex;
  bool _enumerated = false;
};

}  // namespace sirius::op::scan
