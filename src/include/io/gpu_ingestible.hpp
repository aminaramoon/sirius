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

#include "io/types.hpp"

#include <cudf/io/text/byte_range_info.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/resource_ref.hpp>

#include <algorithm>
#include <cstdint>
#include <memory>
#include <span>
#include <string_view>
#include <utility>

namespace sirius::io {

/**
 * @brief Tag identifying which concrete @c gpu_ingestible implementation a
 *        given @c ingestible_table_info should produce.
 */
enum class ingestible_type : uint8_t {
  PARQUET,
  DUCKDB,
};

/**
 * @brief Polymorphic description of a post-scan filter and/or projection that
 *        must be applied to a materialized table.
 */
class post_filter_and_projection_info {
 public:
  virtual ~post_filter_and_projection_info() = default;
};

/**
 * @brief Opaque polymorphic payload handed to a @c gpu_ingestible on
 *        construction. Concrete implementations subclass this and interpret
 *        their own derived state. The carried @c type() drives
 *        @c make_gpu_ingestible's dispatch to the matching concrete ingestible.
 */
class ingestible_table_info {
 public:
  virtual ~ingestible_table_info() = default;

  [[nodiscard]] virtual ingestible_type type() const = 0;

  virtual bool is_injestible_with(std::string_view filename) = 0;

  /// @brief checks if the files in the table_info match the given files in the function
  /// @param files the files to check
  /// @return true if the files match, false otherwise
  [[nodiscard]] virtual bool matches_files(std::span<std::string> files) const = 0;

  [[nodiscard]] virtual std::unique_ptr<post_filter_and_projection_info>
  get_filter_and_pojection_info() const = 0;
};

/**
 * @brief Polymorphic description of the data needed to materialize a single
 *        split into a cuDF table.
 *
 * Carries optional prefetch hints: when @c is_prefetchable returns @c true,
 * @c get_prefetching_ranges yields the byte ranges the IO layer should pull
 * before the split is materialized.
 */
class scan_info {
 public:
  virtual ~scan_info() = default;

  [[nodiscard]] virtual bool is_prefetchable() const = 0;

  [[nodiscard]] virtual std::shared_ptr<sirius::io::sirius_io_object> get_io_object() const = 0;

  [[nodiscard]] virtual std::span<cudf::io::text::byte_range_info> get_prefetching_ranges()
    const = 0;
};

/**
 * @brief Per-split metadata returned from @c gpu_ingestible::get_next_split.
 *
 * Concrete wrapper that owns the scan description and, optionally, a
 * post-scan filter / projection description. A null filter pointer indicates
 * that no post-scan work is required.
 */
class scan_and_filter_metadata {
 public:
  scan_and_filter_metadata(std::vector<std::unique_ptr<scan_info>> scan,
                           std::unique_ptr<post_filter_and_projection_info> filter_and_project)
    : _scan(std::move(scan)), _filter_and_project(std::move(filter_and_project))
  {
  }

  [[nodiscard]] bool has_filter() const noexcept { return _filter_and_project != nullptr; }

  [[nodiscard]] const std::vector<std::unique_ptr<scan_info>>& get_scan_infos() const noexcept
  {
    return _scan;
  }

  [[nodiscard]] post_filter_and_projection_info const& filter_and_project() const noexcept
  {
    return *_filter_and_project;
  }

 private:
  std::vector<std::unique_ptr<scan_info>> _scan;
  std::unique_ptr<post_filter_and_projection_info> _filter_and_project;
};

/**
 * @brief Interface for sources that can be injected into a GPU scan pipeline.
 *
 * A @c gpu_ingestible produces splits of work, materializes each split into a
 * cuDF table on the GPU, and applies any post-scan filters and projections.
 * Implementations may opt in to prefetching by returning @c true from
 * @c supports_prefetching().
 */
class gpu_ingestible : public std::enable_shared_from_this<gpu_ingestible> {
 public:
  enum filter_state {
    UNFILTERED,
    ROWGROUP_FILTERED,
    ROW_FILTERED,
    ROW_FILTERED_AND_PROJECTED,
  };

  struct filtered_result {
    std::unique_ptr<cudf::table> table;
    filter_state state{UNFILTERED};
  };

  explicit gpu_ingestible(std::unique_ptr<ingestible_table_info> info)
    : _table_info(std::move(info))
  {
  }

  virtual ~gpu_ingestible() = default;

  gpu_ingestible(gpu_ingestible const&)            = delete;
  gpu_ingestible& operator=(gpu_ingestible const&) = delete;

  virtual std::unique_ptr<scan_info> process_next_metadata() = 0;

  /**
   * @brief Pull the next split of work.
   *
   * @return The split's metadata, or @c nullptr when the source is exhausted.
   *         Prefetch hints, if any, are reachable via the wrapped
   *         @c scan_info (see @c scan_info::is_prefetchable).
   */
  virtual std::unique_ptr<scan_and_filter_metadata> get_next_split(std::size_t batch_size) = 0;

  /**
   * @brief Materialize a split into a cuDF table on the GPU.
   */
  virtual filtered_result materialize_table(const std::vector<std::unique_ptr<scan_info>>& info,
                                            rmm::device_async_resource_ref mr,
                                            rmm::cuda_stream_view stream) = 0;

  /**
   * @brief Apply post-scan filters and projections to a materialized table.
   */
  virtual std::unique_ptr<cudf::table> post_filter_and_project(
    cudf::table_view input,
    filter_state filter_state,
    post_filter_and_projection_info const& info,
    rmm::device_async_resource_ref mr,
    rmm::cuda_stream_view stream) = 0;

  /**
   * @brief Whether this source supports prefetching the byte ranges returned
   *        from @c get_next_split().
   */
  [[nodiscard]] virtual bool supports_prefetching() const = 0;

  [[nodiscard]] const ingestible_table_info& table_info() const noexcept { return *_table_info; }

 protected:
  std::unique_ptr<ingestible_table_info> _table_info;
};

/**
 * @brief Construct the concrete @c gpu_ingestible that matches
 *        @c table_info->type().
 *
 * @throws std::runtime_error if no implementation is registered for the
 *         carried type.
 */
std::shared_ptr<gpu_ingestible> make_gpu_ingestible(
  std::unique_ptr<ingestible_table_info> table_info);

}  // namespace sirius::io
