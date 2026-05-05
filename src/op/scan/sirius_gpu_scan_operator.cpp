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
#include "io/gpu_ingestible.hpp"

#include <data/data_batch_utils.hpp>
#include <op/scan/sirius_gpu_scan_operator.hpp>
#include <op/scan/sirius_gpu_scan_operator_data.hpp>
#include <op/sirius_physical_operator.hpp>
#include <scan_manager/split_connector.hpp>

// cudf
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
sirius_gpu_scan_operator::sirius_gpu_scan_operator(
  duckdb::vector<sirius::logical_type> types,
  duckdb::idx_t estimated_cardinality,
  std::unique_ptr<io::ingestible_table_info> table_info)
  : sirius_physical_operator(
      SiriusPhysicalOperatorType::GPU_SCAN, std::move(types), estimated_cardinality),
    _ingestible(io::make_gpu_ingestible(std::move(table_info)))
{
}

sirius_gpu_scan_operator::~sirius_gpu_scan_operator() = default;

//===----------------------------------------------------------------------===//
// Scheduling interface
//===----------------------------------------------------------------------===//
std::optional<task_creation_hint> sirius_gpu_scan_operator::get_next_task_hint()
{
  if (_split_connector->is_closed()) { return std::nullopt; }
  return task_creation_hint{TaskCreationHint::READY, this};
}

bool sirius_gpu_scan_operator::all_ports_empty()
{
  return _split_connector->is_closed() && !_split_connector->has_more_splits();
}

std::unique_ptr<operator_data> sirius_gpu_scan_operator::get_next_task_input_data()
{
  auto next = _split_connector->get_next_split();
  if (!next.has_value()) { return nullptr; }
  return std::move(*next);
}

//===----------------------------------------------------------------------===//
// execute()
//===----------------------------------------------------------------------===//
std::unique_ptr<operator_data> sirius_gpu_scan_operator::execute(const operator_data& input_data,
                                                                 rmm::cuda_stream_view stream)
{
  auto mr = cudf::get_current_device_resource_ref();

  std::unique_ptr<cudf::table> table;
  cucascade::memory::memory_space* mem_space = nullptr;
  std::vector<std::shared_ptr<cucascade::data_batch>> batches;

  if (auto const* fresh = dynamic_cast<const scan_operator_input*>(&input_data)) {
    if (!fresh->gpu_memory_space) {
      throw std::runtime_error(
        "[sirius_gpu_scan_operator] scan_operator_input has null gpu_memory_space; "
        "prepare_for_processing must run before execute().");
    }
    auto const& metadata = *fresh->metadata;
    auto filtered_table  = _ingestible->materialize_table(metadata.get_scan_infos(), mr, stream);
    if (metadata.has_filter() &&
        filtered_table.state != io::gpu_ingestible::filter_state::ROW_FILTERED_AND_PROJECTED) {
      table = _ingestible->post_filter_and_project(filtered_table.table->view(),
                                                   filtered_table.state,
                                                   metadata.filter_and_project(),
                                                   mr,
                                                   stream);
    } else {
      table = std::move(filtered_table.table);
    }
    mem_space = fresh->gpu_memory_space;
  } else if (auto const* pinned =
               dynamic_cast<const scan_operator_with_pinned_table_input*>(&input_data)) {
    if (!pinned->filter_info) {
      // Fast path: no post-scan work — forward the pinned batch unchanged.
      batches.push_back(pinned->batch);
      return std::make_unique<pipelineable_operator_data>(std::move(batches));
    }
    if (!pinned->gpu_memory_space) {
      throw std::runtime_error(
        "[sirius_gpu_scan_operator] scan_operator_with_pinned_table_input has null "
        "gpu_memory_space; prepare_for_processing must run before execute().");
    }
    auto pinned_view = sirius::get_cudf_table_view(*pinned->batch);
    table            = _ingestible->post_filter_and_project(
      pinned_view, io::gpu_ingestible::UNFILTERED, *pinned->filter_info, mr, stream);
    mem_space = pinned->gpu_memory_space;
  } else {
    throw std::runtime_error(
      "[sirius_gpu_scan_operator] execute() called with unexpected operator_data type; "
      "expected scan_operator_input or scan_operator_with_pinned_table_input.");
  }

  auto batch = sirius::make_data_batch(std::move(table), *mem_space);
  batches.push_back(std::move(batch));
  return std::make_unique<pipelineable_operator_data>(std::move(batches));
}

}  // namespace sirius::op::scan
