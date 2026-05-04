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
#include "cucascade/data/gpu_data_representation.hpp"

#include <io/gpu_ingestible.hpp>
#include <op/sirius_physical_operator.hpp>

// cucascade
#include <cucascade/data/data_batch.hpp>
#include <cucascade/memory/memory_space.hpp>

// standard library
#include <cstddef>
#include <memory>
#include <optional>
#include <vector>

namespace sirius::op::scan {

//===----------------------------------------------------------------------===//
// scan_operator_input
//===----------------------------------------------------------------------===//
/**
 * @brief Input to a GPU scan task that materializes a fresh split through a
 *        @c io::gpu_ingestible implementation.
 *
 * Carries the per-split metadata pulled from the ingestible's
 * @c get_next_split. The operator's @c execute() reads the wrapped
 * @c scan_info to materialize the table and the wrapped
 * @c post_filter_and_projection_info (when present) to apply post-scan work.
 *
 * @c prepare_for_processing captures the task's reservation memory space into
 * @c gpu_memory_space; @c execute() reads it when wrapping the materialized
 * output table in a data_batch.
 */
class scan_operator_input : public op::operator_data {
 public:
  explicit scan_operator_input(std::unique_ptr<io::scan_and_filter_metadata> metadata)
    : metadata(std::move(metadata))
  {
  }

  std::optional<std::vector<::cucascade::data_batch_processing_handle>> prepare_for_processing(
    const ::cucascade::memory::memory_space* requested_memory_space,
    rmm::cuda_stream_view /*stream*/) override
  {
    gpu_memory_space = const_cast<cucascade::memory::memory_space*>(requested_memory_space);
    return std::vector<::cucascade::data_batch_processing_handle>{};
  }

  std::unique_ptr<io::scan_and_filter_metadata> metadata;
  cucascade::memory::memory_space* gpu_memory_space = nullptr;
};

//===----------------------------------------------------------------------===//
// scan_operator_with_pinned_table_input
//===----------------------------------------------------------------------===//
/**
 * @brief Input to a GPU scan task served from a pinned (cached) table.
 *
 * The pinned batch is forwarded as-is when @c filter_info is null; otherwise
 * @c execute() applies @c io::gpu_ingestible::post_filter_and_project to the
 * batch's table view and wraps the result in a new data_batch tied to the
 * memory space captured during @c prepare_for_processing.
 */
class scan_operator_with_pinned_table_input : public op::operator_data {
 public:
  scan_operator_with_pinned_table_input(
    std::shared_ptr<cucascade::data_batch> batch,
    std::unique_ptr<io::post_filter_and_projection_info> filter_info)
    : batch(std::move(batch)), filter_info(std::move(filter_info))
  {
  }

  std::optional<std::vector<::cucascade::data_batch_processing_handle>> prepare_for_processing(
    const ::cucascade::memory::memory_space* requested_memory_space,
    rmm::cuda_stream_view /*stream*/) override
  {
    gpu_memory_space = const_cast<cucascade::memory::memory_space*>(requested_memory_space);
    return std::vector<::cucascade::data_batch_processing_handle>{};
  }

  [[nodiscard]] std::size_t get_estimated_size_in_bytes() const override
  {
    return batch->get_data()->cast<cucascade::gpu_table_representation>().get_size_in_bytes();
  }

  std::shared_ptr<cucascade::data_batch> batch;
  std::unique_ptr<io::post_filter_and_projection_info> filter_info;
  cucascade::memory::memory_space* gpu_memory_space = nullptr;
};

}  // namespace sirius::op::scan
