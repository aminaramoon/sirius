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
    _split_connector(std::make_unique<scan_manager::split_connector>()),
    _table_info(std::move(table_info))
{
  _split_connector->close();
}

sirius_gpu_scan_operator::~sirius_gpu_scan_operator() = default;

//===----------------------------------------------------------------------===//
// Setup hooks
//===----------------------------------------------------------------------===//
std::unique_ptr<io::ingestible_table_info> sirius_gpu_scan_operator::take_table_info()
{
  return std::move(_table_info);
}

void sirius_gpu_scan_operator::set_ingestible(std::unique_ptr<io::gpu_ingestible> ingestible)
{
  _ingestible = std::move(ingestible);
}

void sirius_gpu_scan_operator::set_split_connector(
  std::unique_ptr<scan_manager::split_connector> connector)
{
  _split_connector = std::move(connector);
}

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
  if (!_ingestible) {
    throw std::runtime_error(
      "[sirius_gpu_scan_operator] execute() called before set_ingestible() was wired.");
  }

  auto mr = cudf::get_current_device_resource_ref();

  std::unique_ptr<cudf::table> table;
  cucascade::memory::memory_space* mem_space = nullptr;

  if (auto const* fresh = dynamic_cast<const scan_operator_input*>(&input_data)) {
    if (!_gpu_memory_space) {
      throw std::runtime_error(
        "[sirius_gpu_scan_operator] execute() called before set_gpu_memory_space() was wired.");
    }
    auto const& metadata = *fresh->metadata;
    table                = _ingestible->materialize_table(metadata.scan(), mr, stream);
    if (metadata.has_filter()) {
      auto input_table = std::move(table);
      table            = _ingestible->post_filter_and_project(
        input_table->view(), metadata.filter_and_project(), mr, stream);
    }
    mem_space = _gpu_memory_space;
  } else if (auto const* pinned =
               dynamic_cast<const scan_operator_with_pinned_table_input*>(&input_data)) {
    if (!pinned->filter_info) {
      // Fast path: no post-scan work — forward the pinned batch unchanged.
      std::vector<std::shared_ptr<cucascade::data_batch>> batches;
      batches.push_back(pinned->batch);
      return std::make_unique<pipelineable_operator_data>(std::move(batches));
    }
    auto pinned_view = sirius::get_cudf_table_view(*pinned->batch);
    table = _ingestible->post_filter_and_project(pinned_view, *pinned->filter_info, mr, stream);
    mem_space = pinned->batch->get_memory_space();
  } else {
    throw std::runtime_error(
      "[sirius_gpu_scan_operator] execute() called with unexpected operator_data type; "
      "expected scan_operator_input or scan_operator_with_pinned_table_input.");
  }

  auto batch = sirius::make_data_batch(std::move(table), *mem_space);
  std::vector<std::shared_ptr<cucascade::data_batch>> batches;
  batches.push_back(std::move(batch));
  return std::make_unique<pipelineable_operator_data>(std::move(batches));
}

}  // namespace sirius::op::scan
