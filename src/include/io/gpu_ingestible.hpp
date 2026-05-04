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

#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/resource_ref.hpp>

#include <memory>
#include <utility>

namespace sirius::io {

// `prefetching_ranges` is still concrete and supplied elsewhere — forward
// declared here because this interface only refers to it by type.
class prefetching_ranges;

/**
 * @brief Opaque polymorphic payload handed to a @c gpu_ingestible on
 *        construction. Concrete implementations subclass this and interpret
 *        their own derived state.
 */
class scan_info {
 public:
  virtual ~scan_info() = default;
};

/**
 * @brief Polymorphic description of a single unit of scan work, including
 *        whether any post-scan filter must be applied to the materialized
 *        table.
 */
class scan_and_filter_metadata {
 public:
  virtual ~scan_and_filter_metadata() = default;

  [[nodiscard]] virtual bool has_filter() const = 0;
};

/**
 * @brief Interface for sources that can be injected into a GPU scan pipeline.
 *
 * A @c gpu_ingestible produces splits of work, materializes each split into a
 * cuDF table on the GPU, and applies any post-scan filters and projections.
 * Implementations may opt in to prefetching by returning @c true from
 * @c supports_prefetching().
 */
class gpu_ingestible {
 public:
  explicit gpu_ingestible(std::unique_ptr<scan_info> info) : _scan_info(std::move(info)) {}

  virtual ~gpu_ingestible() = default;

  gpu_ingestible(gpu_ingestible const&)            = delete;
  gpu_ingestible& operator=(gpu_ingestible const&) = delete;

  /**
   * @brief Pull the next split of work, paired with the byte ranges that
   *        should be prefetched to satisfy it.
   *
   * @return A pair of (metadata, prefetching_ranges), or @c nullptr when the
   *         source is exhausted. Metadata is held via @c unique_ptr because
   *         @c scan_and_filter_metadata is an abstract base class.
   */
  virtual std::unique_ptr<std::pair<std::unique_ptr<scan_and_filter_metadata>, prefetching_ranges>>
  get_next_split() = 0;

  /**
   * @brief Materialize a split's metadata into a cuDF table on the GPU.
   */
  virtual std::unique_ptr<cudf::table> materialize_table(scan_and_filter_metadata const& metadata,
                                                         rmm::device_async_resource_ref mr,
                                                         rmm::cuda_stream_view stream) = 0;

  /**
   * @brief Apply post-scan filters and projections to a materialized table.
   */
  virtual std::unique_ptr<cudf::table> post_filter_and_project(
    cudf::table_view input,
    scan_and_filter_metadata const& metadata,
    rmm::device_async_resource_ref mr,
    rmm::cuda_stream_view stream) = 0;

  /**
   * @brief Whether this source supports prefetching the byte ranges returned
   *        from @c get_next_split().
   */
  [[nodiscard]] virtual bool supports_prefetching() const = 0;

 protected:
  std::unique_ptr<scan_info> _scan_info;
};

}  // namespace sirius::io
