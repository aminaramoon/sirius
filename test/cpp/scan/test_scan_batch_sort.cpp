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

#include "operator/operator_test_utils.hpp"

#include <cudf/column/column_factories.hpp>
#include <cudf/table/table.hpp>

#include <rmm/cuda_stream.hpp>

#include <catch.hpp>
#include <cucascade/cudf/gpu_data_representation.hpp>
#include <cucascade/cudf/host_data_representation.hpp>
#include <data/data_batch_utils.hpp>
#include <data/sirius_converter_registry.hpp>
#include <io/ioctx_resolver.hpp>
#include <op/scan/detail/read_retirement.hpp>
#include <op/scan/gpu_ingestible.hpp>
#include <op/scan/sirius_gpu_scan_operator.hpp>
#include <op/scan/sirius_gpu_scan_operator_data.hpp>
#include <planner/late_mat_plan_pass.hpp>

#include <cstdint>
#include <limits>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {
using namespace sirius::op::scan;
using sirius::test::operator_utils::get_resource_ref;

struct environment {
  std::unique_ptr<sirius::memory::sirius_memory_reservation_manager> manager =
    sirius::test::operator_utils::initialize_memory_manager();
  cucascade::memory::memory_space* gpu = manager->get_memory_space(cucascade::memory::Tier::GPU, 0);
  cucascade::memory::memory_space* host =
    manager->get_memory_space(cucascade::memory::Tier::HOST, 0);
  rmm::cuda_stream stream;
};

environment& env()
{
  sirius::test::operator_utils::ensure_converter_registry();
  static environment result;
  return result;
}

class fixture_table_info final : public ingestible_table_info {
 public:
  std::span<std::string const> column_names() const override { return {}; }
  std::span<std::string const> file_paths() const override { return {}; }
  std::string display_name() const override { return "batch-sort-fixture"; }
};

// Exercise the real GPU_SCAN tail, borrowing the real resident input. In particular this must
// copy the input, not surrender it, so an injected sort allocation failure can retry the split.
class copy_ingestible final : public gpu_ingestible {
 public:
  std::unique_ptr<cudf::table> post_filter_and_project(
    filtered_table&& source,
    cucascade::memory::memory_space const& space,
    ::cuda::stream_ref stream,
    bool,
    std::shared_ptr<const sirius::like_multiliteral_cache>,
    std::unique_ptr<cudf::column>*,
    std::span<std::size_t const>,
    sirius::expression_query_policy) override
  {
    return std::make_unique<cudf::table>(
      source.table.view(), stream, space.get_default_allocator());
  }
  std::unique_ptr<batch_coalescer> create_batch_coalescer() const override { return nullptr; }
  bool has_processed_all_metadata() const override { return true; }
  metadata_scan_task_t next_split_provider(sirius::io::ioctx_resolver) override { return {}; }
  filtered_table materialize_metadata_to_table(
    scan_info const&,
    cucascade::memory::memory_space const&,
    ::cuda::stream_ref,
    bool,
    std::shared_ptr<const sirius::like_multiliteral_cache>,
    sirius::expression_query_policy) override
  { throw std::logic_error("resident fixture does not read metadata"); }
  ingestible_table_info const& table_info() const noexcept override { return info; }
  std::vector<std::size_t> materialized_column_order() const override { return {0, 1}; }

 private:
  fixture_table_info info;
};

auto types()
{
  return duckdb::vector<sirius::logical_type>(2,
                                              sirius::logical_type::make(sirius::type_id::INTEGER));
}

sirius_gpu_scan_operator make_scan()
{ return sirius_gpu_scan_operator(types(), 10, std::make_shared<copy_ingestible>(), 1); }

std::shared_ptr<cucascade::data_batch> make_batch(std::vector<int32_t> const& keys,
                                                  std::vector<int32_t> const& payload,
                                                  std::vector<cudf::bitmask_type> masks = {})
{
  auto& e = env();
  REQUIRE(keys.size() == payload.size());
  std::vector<std::unique_ptr<cudf::column>> columns;
  for (auto const* values : {&keys, &payload}) {
    auto column = cudf::make_numeric_column(
      cudf::data_type{cudf::type_id::INT32},
      static_cast<cudf::size_type>(values->size()),
      masks.empty() ? cudf::mask_state::UNALLOCATED : cudf::mask_state::ALL_VALID,
      e.stream,
      get_resource_ref(*e.gpu));
    REQUIRE(cudaMemcpyAsync(column->mutable_view().data<int32_t>(),
                            values->data(),
                            values->size() * sizeof(int32_t),
                            cudaMemcpyHostToDevice,
                            e.stream.value()) == cudaSuccess);
    if (!masks.empty()) {
      REQUIRE(masks.size() == 2);
      REQUIRE(values->size() <= 32);
      REQUIRE(cudaMemcpyAsync(column->mutable_view().null_mask(),
                              &masks[columns.size()],
                              sizeof(cudf::bitmask_type),
                              cudaMemcpyHostToDevice,
                              e.stream.value()) == cudaSuccess);
      column->set_null_count(1);
    }
    columns.push_back(std::move(column));
  }
  // Host fixtures/masks may be temporaries; keep them alive through the upload.
  e.stream.synchronize();
  return sirius::make_data_batch(
    std::make_unique<cudf::table>(std::move(columns)), *e.gpu, e.stream, {});
}

std::unique_ptr<scan_operator_input> input_for(std::shared_ptr<cucascade::data_batch> batch)
{
  auto result = std::make_unique<scan_operator_input>(std::move(batch));
  pin_validation validation;
  validation.identity = validation.layout = validation.structure = {true, true};
  result->resident_contract_id                                   = 1;
  result->resident_validation                                    = validation;
  result->gpu_memory_space                                       = env().gpu;
  return result;
}

std::vector<int32_t> read_column(cudf::column_view column)
{
  auto& e = env();
  std::vector<int32_t> result(column.size());
  REQUIRE(cudaMemcpyAsync(result.data(),
                          column.data<int32_t>(),
                          result.size() * sizeof(int32_t),
                          cudaMemcpyDeviceToHost,
                          e.stream.value()) == cudaSuccess);
  e.stream.synchronize();
  return result;
}

void check_output(sirius::op::operator_data const& output,
                  std::vector<int32_t> const& keys,
                  std::vector<int32_t> const& payload)
{
  auto const* pipelineable = dynamic_cast<sirius::op::pipelineable_operator_data const*>(&output);
  REQUIRE(pipelineable);
  auto batches = pipelineable->get_read_only_batches();
  REQUIRE(batches.size() == 1);
  auto view = sirius::get_cudf_table_view(batches[0]);
  REQUIRE(view.num_columns() == 2);
  CHECK(read_column(view.column(0)) == keys);
  CHECK(read_column(view.column(1)) == payload);
}
}  // namespace

TEST_CASE("GPU scan batch sorting is off by default", "[scan_batch_sort][gpu]")
{
  auto scan = make_scan();
  CHECK(scan.batch_sort_keys.empty());
  auto input  = input_for(make_batch({3, 1, 2}, {30, 10, 20}));
  auto output = scan.execute(*input, env().stream);
  check_output(*output, {3, 1, 2}, {30, 10, 20});
}

TEST_CASE("GPU scan sorting permutes every column within each batch", "[scan_batch_sort][gpu]")
{
  auto scan            = make_scan();
  scan.batch_sort_keys = {{0}, {1, cudf::order::DESCENDING, cudf::null_order::AFTER}};
  auto first           = make_batch({3, 1, 2, 1, 0}, {30, 11, 20, 12, 0});
  auto input           = input_for(first);
  auto output          = scan.execute(*input, env().stream);
  check_output(*output, {0, 1, 1, 2, 3}, {0, 12, 11, 20, 30});
  // Source pin remains byte/value-identical; sorting never edits the input.
  auto ro = first->to_read_only();
  CHECK(read_column(sirius::get_cudf_table_view(ro).column(0)) ==
        std::vector<int32_t>{3, 1, 2, 1, 0});
  auto second        = input_for(make_batch({6, 4, 5}, {60, 40, 50}));
  auto second_output = scan.execute(*second, env().stream);
  check_output(*second_output, {4, 5, 6}, {40, 50, 60});
  CHECK(scan.output_order_preservation() == sirius::OrderPreservationType::NO_ORDER);
}

TEST_CASE("GPU scan sorting retains nullable key and payload row association",
          "[scan_batch_sort][gpu]")
{
  auto& e               = env();
  bool const descending = GENERATE(false, true);
  bool const null_first = GENERATE(false, true);
  auto scan             = make_scan();
  scan.batch_sort_keys  = {
    {0,
     descending ? cudf::order::DESCENDING : cudf::order::ASCENDING,
     descending != null_first ? cudf::null_order::BEFORE : cudf::null_order::AFTER}};
  // key row 0 is NULL; payload row 2 is NULL.
  auto input         = input_for(make_batch({3, 1, 2}, {30, 10, 20}, {6, 3}));
  auto output        = scan.execute(*input, e.stream);
  auto* pipelineable = dynamic_cast<sirius::op::pipelineable_operator_data*>(output.get());
  REQUIRE(pipelineable);
  auto batches = pipelineable->get_read_only_batches();
  REQUIRE(batches.size() == 1);
  auto view             = sirius::get_cudf_table_view(batches[0]);
  std::vector<int> rows = descending ? std::vector<int>{2, 1} : std::vector<int>{1, 2};
  if (null_first)
    rows.insert(rows.begin(), 0);
  else
    rows.push_back(0);
  std::vector<std::vector<int32_t>> original{{3, 1, 2}, {30, 10, 20}};
  std::vector<cudf::bitmask_type> original_masks{6, 3};
  for (int col = 0; col < 2; ++col) {
    auto values             = read_column(view.column(col));
    cudf::bitmask_type mask = 0;
    REQUIRE(cudaMemcpyAsync(&mask,
                            view.column(col).null_mask(),
                            sizeof(mask),
                            cudaMemcpyDeviceToHost,
                            e.stream.value()) == cudaSuccess);
    e.stream.synchronize();
    for (int row = 0; row < 3; ++row) {
      bool const valid = (original_masks[col] >> rows[row]) & 1;
      CHECK(bool((mask >> row) & 1) == valid);
      if (valid) CHECK(values[row] == original[col][rows[row]]);
    }
  }
}

TEST_CASE("GPU scan failed sort retirement retains assembly owners", "[scan_batch_sort][gpu]")
{
  auto scan             = make_scan();
  scan.batch_sort_keys  = {{0}};
  auto input            = input_for(make_batch({3, 1, 2}, {30, 10, 20}));
  auto const checkpoint = GENERATE(sirius::op::scan::detail::read_checkpoint::SCAN_SORT_ORDER,
                                   sirius::op::scan::detail::read_checkpoint::SCAN_SORT_GATHER);
  sirius::op::scan::detail::read_retirement_test_hooks hooks(checkpoint, true);
  REQUIRE_THROWS_AS(scan.execute(*input, env().stream), sirius::unjoined_gpu_work);
  REQUIRE(hooks.retained.has_value());
  auto retained = sirius::op::scan::detail::scan_operator_test_access::retained_assembly(hooks);
  REQUIRE(retained);
  if (checkpoint == sirius::op::scan::detail::read_checkpoint::SCAN_SORT_GATHER) {
    REQUIRE(retained->output);
    CHECK(read_column(retained->output->view().column(0)) == std::vector<int32_t>{1, 2, 3});
  } else {
    CHECK(retained->output == nullptr);  // The retained source/permutation precede the gather.
  }
  CHECK_FALSE(input->stolen_table_consumed);
}

TEST_CASE("GPU scan sort failures retain retryable resident input", "[scan_batch_sort][gpu]")
{
  auto const checkpoint = GENERATE(sirius::op::scan::detail::read_checkpoint::SCAN_SORT_ORDER,
                                   sirius::op::scan::detail::read_checkpoint::SCAN_SORT_GATHER);
  auto scan             = make_scan();
  scan.batch_sort_keys  = {{0}};
  auto input            = input_for(make_batch({3, 1, 2}, {30, 10, 20}));
  {
    sirius::op::scan::detail::read_retirement_test_hooks hooks(checkpoint);
    REQUIRE_THROWS_AS(scan.execute(*input, env().stream), rmm::out_of_memory);
    CHECK(hooks.checkpoint_reached);
  }
  CHECK_FALSE(input->stolen_table_consumed);
  REQUIRE(input->get_cached_batch());
  auto ro = input->get_cached_batch()->to_read_only();
  CHECK(read_column(sirius::get_cudf_table_view(ro).column(0)) == std::vector<int32_t>{3, 1, 2});
  auto output = scan.execute(*input, env().stream);
  check_output(*output, {1, 2, 3}, {10, 20, 30});
}

TEST_CASE("scan sort suppresses both converted-input steal arms", "[scan_batch_sort][gpu]")
{
  auto& e              = env();
  auto scan            = make_scan();
  scan.batch_sort_keys = {{0}};
  auto input_batch     = make_batch({3, 1, 2}, {30, 10, 20});
  {
    auto mut = input_batch->to_mutable();
    mut.convert_to<cucascade::host_data_representation>(
      sirius::converter_registry::get(), e.host, e.stream);
  }
  auto input = input_for(input_batch);
  // This is the flag the scan stamps at dequeue, before prepare_for_processing.
  input->preserve_converted_input = true;
  input->needs_carrier_conversion = GENERATE(false, true);
  input->prepare_for_processing(e.gpu, e.stream);
  CHECK_FALSE(input->converted_table_transferable());
  CHECK_FALSE(input->converted_table_steal_pending);
  CHECK_FALSE(input->stolen_table);
  {
    sirius::op::scan::detail::read_retirement_test_hooks hooks(
      sirius::op::scan::detail::read_checkpoint::SCAN_SORT_GATHER);
    REQUIRE_THROWS_AS(scan.execute(*input, e.stream), rmm::out_of_memory);
  }
  CHECK_FALSE(input->stolen_table_consumed);
  auto output = scan.execute(*input, e.stream);
  check_output(*output, {1, 2, 3}, {10, 20, 30});
}

TEST_CASE("scan batch sorting skips deferrals installed after planning", "[scan_batch_sort]")
{
  auto scan            = make_scan();
  scan.batch_sort_keys = {{0}};
  REQUIRE(scan.batch_sort_active());
  REQUIRE(
    sirius::planner::install_count_deferral(scan, sirius::late_mat::deferred_scan_output{{1}}));
  CHECK_FALSE(scan.batch_sort_active());
  sirius::op::input_stats stats;
  stats.bytes = stats.working_set_bytes = 128;
  stats.resident                        = true;
  CHECK(scan.no_history_peak_memory_estimate(stats) == 128);
  auto input    = input_for(make_batch({3, 1, 2}, {30, 10, 20}));
  input->origin = std::make_shared<const sirius::late_mat::scan_batch_origin>(
    sirius::late_mat::scan_batch_origin{.range = {10, 3}});
  sirius::op::scan::detail::read_retirement_test_hooks hooks(
    sirius::op::scan::detail::read_checkpoint::SCAN_SORT_GATHER);
  auto output = scan.execute(*input, env().stream);
  CHECK_FALSE(hooks.checkpoint_reached);
  auto* pipelineable = dynamic_cast<sirius::op::pipelineable_operator_data*>(output.get());
  REQUIRE(pipelineable);
  auto batches = pipelineable->get_read_only_batches();
  REQUIRE(batches.size() == 1);
  auto view = sirius::get_cudf_table_view(batches[0]);
  CHECK(read_column(view.column(0)) == std::vector<int32_t>{3, 1, 2});
  std::vector<std::uint64_t> rowids(3);
  REQUIRE(cudaMemcpyAsync(rowids.data(),
                          view.column(1).data<std::uint64_t>(),
                          rowids.size() * sizeof(std::uint64_t),
                          cudaMemcpyDeviceToHost,
                          env().stream.value()) == cudaSuccess);
  env().stream.synchronize();
  CHECK(rowids == std::vector<std::uint64_t>{10, 11, 12});
}

TEST_CASE("scan batch sorting cold estimate includes output permutation and scratch",
          "[scan_batch_sort]")
{
  auto scan = make_scan();
  sirius::op::input_stats stats;
  stats.bytes             = 128;
  stats.working_set_bytes = 128;
  stats.resident          = true;
  CHECK(scan.no_history_peak_memory_estimate(stats) == 128);
  scan.batch_sort_keys       = {{0}};
  auto const sorted_estimate = scan.no_history_peak_memory_estimate(stats);
  CHECK(sorted_estimate >= 2 * stats.bytes + 16 * sizeof(cudf::size_type));
  stats.bytes = std::numeric_limits<std::size_t>::max();
  CHECK(scan.no_history_peak_memory_estimate(stats) == std::numeric_limits<std::size_t>::max());
  stats.resident = false;
  CHECK(scan.no_history_peak_memory_estimate(stats) == std::numeric_limits<std::size_t>::max());
}
