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
#include <io/gpu_ingestible.hpp>
#include <op/scan/sirius_gpu_scan_operator_data.hpp>
#include <op/sirius_physical_operator.hpp>
#include <op/sirius_physical_operator_type.hpp>

// cucascade
#include <cucascade/memory/memory_space.hpp>

// standard library
#include <memory>
#include <optional>

namespace sirius::scan_manager {
class split_connector;
}  // namespace sirius::scan_manager

namespace sirius::op::scan {

//===----------------------------------------------------------------------===//
// sirius_gpu_scan_operator
//===----------------------------------------------------------------------===//
/**
 * @brief Generic GPU scan operator that delegates per-split materialization
 *        and post-scan filter / projection to a @c io::gpu_ingestible
 *        implementation.
 *
 * The operator's constructor consumes an @c io::ingestible_table_info and
 * builds the matching @c io::gpu_ingestible via @c io::make_gpu_ingestible.
 * Splits — one @c scan_operator_input per fresh-read partition or one
 * @c scan_operator_with_pinned_table_input per cached batch — are pushed
 * into the operator's bound @c split_connector by an external split
 * provider. The operator pulls splits via @c get_next_task_input_data, which
 * blocks inside @c split_connector::get_next_split until a split arrives or
 * the connector is closed.
 */
class sirius_gpu_scan_operator : public sirius_physical_operator {
 public:
  static constexpr SiriusPhysicalOperatorType TYPE = SiriusPhysicalOperatorType::GPU_SCAN;

  //===----------Constructor----------===//
  /**
   * @param types                  Output column types.
   * @param estimated_cardinality  Estimated row count.
   * @param table_info             Bind-data extracted upstream; consumed by
   *                               @c io::make_gpu_ingestible to build the
   *                               concrete ingestible owned by this operator.
   */
  sirius_gpu_scan_operator(duckdb::vector<sirius::logical_type> types,
                           duckdb::idx_t estimated_cardinality,
                           std::unique_ptr<io::ingestible_table_info> table_info);

  ~sirius_gpu_scan_operator() override;

  //===----------Source interface----------===//
  bool is_source() const override { return true; }

  //===----------Scheduling interface----------===//
  /**
   * @return nullopt once the bound split_connector is closed and drained;
   *         READY pointing at this operator otherwise.
   */
  std::optional<task_creation_hint> get_next_task_hint() override;

  /**
   * @return true once the bound split_connector is closed and drained.
   */
  [[nodiscard]] bool all_ports_empty() override;

  /**
   * @brief Pull the next operator_data input from the bound split_connector.
   *
   * Blocks inside split_connector::get_next_split until a split is available or
   * the connector is closed.
   *
   * @return the next split (either a @c scan_operator_input or a
   *         @c scan_operator_with_pinned_table_input), or nullptr when the
   *         connector is closed and drained.
   */
  std::unique_ptr<operator_data> get_next_task_input_data() override;

  //===----------Execution----------===//
  /**
   * @brief Produce a gpu_table_representation data batch from @p input_data.
   *
   * Two input shapes are supported:
   *   - @c scan_operator_input: a fresh split is materialized via
   *     @c io::gpu_ingestible::materialize_table; if the carried metadata
   *     reports @c has_filter, the result is then passed through
   *     @c io::gpu_ingestible::post_filter_and_project. The output table is
   *     wrapped in a data_batch using the memory space captured by the
   *     input's @c prepare_for_processing.
   *   - @c scan_operator_with_pinned_table_input: when @c filter_info is null
   *     the pinned batch is forwarded unchanged; otherwise the batch's view
   *     is run through @c io::gpu_ingestible::post_filter_and_project and the
   *     result is wrapped in a new batch tied to the memory space captured
   *     by the input's @c prepare_for_processing.
   *
   * @param input_data  Must be either @c scan_operator_input or
   *                    @c scan_operator_with_pinned_table_input.
   * @param stream      CUDA stream.
   * @return gpu_table_representation data batch wrapped as
   *         pipelineable_operator_data.
   * @throws std::runtime_error if @p input_data is of an unsupported type.
   */
  std::unique_ptr<operator_data> execute(const operator_data& input_data,
                                         rmm::cuda_stream_view stream) override;

  //===----------Connector access----------===//
  /// \brief Accessor used by the external split provider to push splits and
  ///        close the connector when production is done.
  scan_manager::split_connector* get_split_connector() noexcept { return _split_connector.get(); }

 private:
  //===----------Fields----------===//
  std::unique_ptr<scan_manager::split_connector> _split_connector;
  std::unique_ptr<io::gpu_ingestible> _ingestible;
};

}  // namespace sirius::op::scan
