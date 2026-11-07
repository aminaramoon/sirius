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

#include <catch.hpp>
#include <op/scan/parquet_gpu_ingestible.hpp>
#include <scan_manager/preparation_fixture.hpp>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace {
namespace scan   = sirius::op::scan;
namespace pq     = cudf::io::parquet;
using membership = std::vector<std::pair<size_t, size_t>>;
using interval   = std::pair<int64_t, int64_t>;

std::vector<uint8_t> encoded(int64_t value, pq::Type type)
{
  auto bits = static_cast<uint64_t>(value);
  std::vector<uint8_t> result(type == pq::Type::INT32 ? 4 : 8);
  for (size_t i = 0; i < result.size(); ++i)
    result[i] = (bits >> (8 * i)) & 0xff;
  return result;
}

// These are parsed-footer descriptors, not files: no filesystem or GPU is needed.
std::unique_ptr<scan::parquet_file_scan_info> file_info(size_t index,
                                                        std::vector<interval> const& bounds,
                                                        std::vector<int> const& selected,
                                                        pq::Type type = pq::Type::INT64)
{
  auto metadata = std::make_shared<pq::FileMetaData>();
  metadata->schema.resize(2);
  metadata->schema[0].name            = "schema";
  metadata->schema[0].num_children    = 1;
  metadata->schema[0].children_idx    = {1};
  metadata->schema[1].name            = "x";
  metadata->schema[1].type            = type;
  metadata->schema[1].parent_idx      = 0;
  metadata->schema[1].repetition_type = pq::FieldRepetitionType::OPTIONAL;
  for (auto const& [lo, hi] : bounds) {
    pq::RowGroup group;
    group.num_rows = 10;
    group.columns.resize(1);
    auto& column                           = group.columns.front();
    column.schema_idx                      = 1;
    column.meta_data.type                  = type;
    column.meta_data.path_in_schema        = {"x"};
    column.meta_data.num_values            = 10;
    column.meta_data.statistics.min_value  = encoded(lo, type);
    column.meta_data.statistics.max_value  = encoded(hi, type);
    column.meta_data.statistics.null_count = 0;
    metadata->row_groups.push_back(std::move(group));
    metadata->num_rows += 10;
  }
  auto file           = std::make_unique<scan::parquet_file_scan_info>();
  file->file_index    = index;
  file->file_path     = std::to_string(index) + ".parquet";
  file->file_metadata = std::move(metadata);
  for (int group : selected)
    file->row_groups.push_back({group, 7, 11, 3, 10});
  file->set_contract_payload(
    1, {{.contract_id = 1, .split_id = 0, .input_identity = file->file_path}}, {{}});
  return file;
}

membership members(scan::parquet_range_scan_info const& range)
{
  membership result;
  for (auto const& ref : range.row_groups)
    result.emplace_back(ref.file_slot, ref.row_group_slot);
  return result;
}

membership sorted(membership result)
{
  std::sort(result.begin(), result.end());
  return result;
}

std::unique_ptr<scan::batch_coalescer> coalescer(size_t cap)
{
  auto info                    = std::make_unique<scan::parquet_ingestible_table_info>();
  info->names                  = {"x"};
  info->returned_types         = {sirius::logical_type::make(sirius::type_id::INTEGER)};
  info->column_ids             = {duckdb::ColumnIndex(0)};
  info->projected_output_count = 1;
  info->contract_id            = 1;
  info->approximate_batch_size = cap;
  scan::parquet_gpu_ingestible reader(std::move(info));
  return reader.create_batch_coalescer();
}

// Keep production batching and certificates; omit footer IO as in preparation_parquet.
void omit_footers(scan::parquet_range_scan_info& range)
{
  for (auto& file : range.files)
    file->file_metadata.reset();
}

struct drained {
  std::vector<std::unique_ptr<scan::scan_info>> batches;
  membership groups;
  size_t preparation = 0;
};

drained drain(scan::parquet_range_scan_info& range, scan::batch_coalescer& batcher, size_t quantum)
{
  drained result;
  auto accept = [&](std::unique_ptr<scan::scan_info> batch) {
    auto& split = dynamic_cast<scan::parquet_split_info&>(*batch);
    CHECK(split.certificates().size() == split.rg_slices.size());
    CHECK(split.dependencies().size() == split.rg_slices.size());
    for (size_t i = 0; i < split.rg_slices.size(); ++i) {
      auto const& slice = split.rg_slices[i];
      CHECK(split.certificates()[i].input_identity == slice.file_path);
      CHECK(split.certificates()[i].contract_id == 1);
      for (auto group : slice.row_group_indices)
        result.groups.emplace_back(slice.file_index, group);
    }
    result.preparation += batch->preparation_bytes();
    result.batches.push_back(std::move(batch));
  };
  size_t cursor = 0;
  for (size_t calls = 0;; ++calls) {
    REQUIRE(calls < 1000);
    auto before = cursor;
    auto step   = batcher.advance(range, cursor, quantum);
    CHECK(cursor >= before);
    CHECK(cursor - before <= quantum);
    if (step.batch) accept(std::move(step.batch));
    size_t consumed = 0;
    for (size_t i = 0; i < cursor; ++i) {
      auto const& ref  = range.row_groups[i];
      auto const& file = *range.files[ref.file_slot];
      if (!file.row_groups.empty())
        consumed += file.row_groups[ref.row_group_slot].decode_working_bytes;
    }
    CHECK(result.preparation + batcher.retained_bytes() == consumed);
    if (step.finished) break;
  }
  CHECK(cursor == range.row_groups.size());
  for (auto& batch : batcher.flush())
    accept(std::move(batch));
  CHECK(result.preparation == range.preparation_bytes());
  CHECK(batcher.retained_bytes() == 0);
  CHECK(batcher.flush().empty());
  return result;
}
}  // namespace

TEST_CASE("Parquet range clusters selected noncontiguous groups across files",
          "[scan][parquet][range_cluster]")
{
  scan::parquet_range_scan_info range;
  range.files.push_back(file_info(8, {{30, 40}, {-999, -999}, {-10, 5}}, {0, 2}));
  range.files.push_back(file_info(3, {{-20, -15}, {4, 35}, {999, 999}}, {0, 1}));
  auto* first = range.files[0].get();
  range.cluster("x");
  CHECK(range.key == "x");
  CHECK(range.eligible);
  CHECK(range.reordered);
  CHECK(range.fallback_reason.empty());
  CHECK(members(range) == membership{{1, 0}, {0, 1}, {1, 1}, {0, 0}});
  CHECK(sorted(members(range)) == membership{{0, 0}, {0, 1}, {1, 0}, {1, 1}});
  CHECK(range.files[0].get() == first);
  CHECK(range.files[0]->row_groups[1].index == 2);
  CHECK(range.files[0]->row_groups[0].index == 0);
  CHECK(range.estimated_bytes() == 28);
  CHECK(range.estimated_working_set_bytes() == 44);
  CHECK(range.preparation_bytes() == 44);
  REQUIRE(range.row_groups.front().interval);
  CHECK(*range.row_groups.front().interval == interval{-20, -15});
}

TEST_CASE("Parquet range ordering is signed lexicographic and stable for overlapping intervals",
          "[scan][parquet][range_cluster]")
{
  auto const minimum = std::numeric_limits<int64_t>::min();
  auto const maximum = std::numeric_limits<int64_t>::max();
  scan::parquet_range_scan_info range;
  range.files.push_back(
    file_info(0, {{0, maximum}, {-5, 8}, {minimum, minimum}, {-5, 3}}, {0, 1, 2, 3}));
  range.files.push_back(file_info(1, {{-5, 3}, {maximum, maximum}, {minimum, maximum}}, {0, 1, 2}));
  range.cluster("x");
  REQUIRE(range.eligible);
  CHECK(members(range) == membership{{0, 2}, {1, 2}, {0, 3}, {1, 0}, {0, 1}, {0, 0}, {1, 1}});
  auto original = members(range);
  range.cluster("x");
  CHECK(members(range) == original);
}

TEST_CASE("Parquet range collect mode keeps eligible groups in original order",
          "[scan][parquet][range_cluster]")
{
  scan::parquet_range_scan_info range;
  range.files.push_back(file_info(0, {{20, 30}, {-10, 0}}, {0, 1}));
  range.files.push_back(file_info(1, {{-30, -20}}, {0}));
  range.cluster("x", false);
  CHECK(range.eligible);
  CHECK_FALSE(range.reordered);
  CHECK(range.fallback_reason.empty());
  CHECK(members(range) == membership{{0, 0}, {0, 1}, {1, 0}});
  CHECK(range.preparation_bytes() == 33);
}

TEST_CASE("Parquet range accepts signed INT32 endpoint statistics",
          "[scan][parquet][range_cluster]")
{
  scan::parquet_range_scan_info range;
  range.files.push_back(
    file_info(0,
              {{0, 4},
               {std::numeric_limits<int32_t>::min(), -1},
               {std::numeric_limits<int32_t>::max(), std::numeric_limits<int32_t>::max()}},
              {0, 1, 2},
              pq::Type::INT32));
  range.cluster("x");
  REQUIRE(range.eligible);
  CHECK(members(range) == membership{{0, 1}, {0, 0}, {0, 2}});
  CHECK(*range.row_groups[0].interval == interval{std::numeric_limits<int32_t>::min(), -1});
}

TEST_CASE("One unsupported selected parquet group prevents all range reordering",
          "[scan][parquet][range_cluster]")
{
  auto defect = GENERATE(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19);
  CAPTURE(defect);
  scan::parquet_range_scan_info range;
  range.files.push_back(file_info(0, {{20, 30}, {10, 12}}, {0, 1}));
  range.files.push_back(file_info(1, {{-30, -20}}, {0}));
  auto footer = std::make_shared<pq::FileMetaData>(*range.files[1]->file_metadata);
  auto& leaf  = footer->schema[1];
  auto& chunk = footer->row_groups[0].columns[0];
  auto& stats = chunk.meta_data.statistics;
  switch (defect) {
    case 0: stats.min_value.reset(); break;
    case 1: stats.max_value.reset(); break;
    case 2: stats.min_value = std::vector<uint8_t>{1, 2, 3}; break;
    case 3: stats.max_value = encoded(-40, pq::Type::INT64); break;
    case 4: leaf.converted_type = pq::ConvertedType::UINT_64; break;
    case 5: leaf.logical_type = pq::LogicalType(pq::IntType{64, false}); break;
    case 6: leaf.logical_type = pq::LogicalType(pq::LogicalType::DATE); break;
    case 7:
      leaf.name                      = "other";
      chunk.meta_data.path_in_schema = {"other"};
      break;
    case 8: chunk.meta_data.path_in_schema = {"nested", "x"}; break;
    case 9:
      leaf.type            = pq::Type::BYTE_ARRAY;
      chunk.meta_data.type = pq::Type::BYTE_ARRAY;
      break;
    case 10: footer->row_groups.clear(); break;
    case 11: range.files[1]->file_metadata.reset(); break;
    case 12: stats.null_count.reset(); break;
    case 13: stats.null_count = 10; break;
    case 14: stats.is_min_value_exact = false; break;
    case 15: stats.is_max_value_exact = false; break;
    case 16: chunk.meta_data.num_values = 9; break;
    case 17: stats.null_count = -1; break;
    case 18: leaf.repetition_type = pq::FieldRepetitionType::UNSPECIFIED; break;
    case 19: leaf.repetition_type = pq::FieldRepetitionType::REPEATED; break;
  }
  if (defect != 11) range.files[1]->file_metadata = std::move(footer);
  range.cluster("x");
  CHECK_FALSE(range.eligible);
  CHECK_FALSE(range.reordered);
  CHECK_FALSE(range.fallback_reason.empty());
  CHECK(members(range) == membership{{0, 0}, {0, 1}, {1, 0}});
  CHECK(range.preparation_bytes() == 33);
}

TEST_CASE("Parquet range accepts explicit signed integer annotations",
          "[scan][parquet][range_cluster]")
{
  auto type   = GENERATE(pq::Type::INT32, pq::Type::INT64);
  auto modern = GENERATE(false, true);
  scan::parquet_range_scan_info range;
  range.files.push_back(file_info(0, {{10, 20}, {-20, -10}}, {0, 1}, type));
  auto footer = std::make_shared<pq::FileMetaData>(*range.files[0]->file_metadata);
  if (modern)
    footer->schema[1].logical_type =
      pq::LogicalType(pq::IntType{static_cast<int8_t>(type == pq::Type::INT32 ? 32 : 64), true});
  else
    footer->schema[1].converted_type =
      type == pq::Type::INT32 ? pq::ConvertedType::INT_32 : pq::ConvertedType::INT_64;
  range.files[0]->file_metadata = std::move(footer);
  range.cluster("x");
  CHECK(range.eligible);
  CHECK(members(range) == membership{{0, 1}, {0, 0}});
}

TEST_CASE("Unselected parquet groups do not veto range clustering",
          "[scan][parquet][range_cluster]")
{
  scan::parquet_range_scan_info range;
  range.files.push_back(file_info(0, {{10, 20}, {0, 0}, {-20, -10}}, {0, 2}));
  auto footer = std::make_shared<pq::FileMetaData>(*range.files[0]->file_metadata);
  footer->row_groups[1].columns[0].meta_data.statistics = {};
  range.files[0]->file_metadata                         = std::move(footer);
  range.cluster("x");
  REQUIRE(range.eligible);
  CHECK(members(range) == membership{{0, 1}, {0, 0}});
}

TEST_CASE("Parquet range coalescing preserves membership certificates and its preparation ledger",
          "[scan][parquet][range_cluster][scan_preparation]")
{
  auto quantum = GENERATE(1u, 2u, 100u);
  scan::parquet_range_scan_info range;
  range.files.push_back(file_info(8, {{30, 40}, {90, 100}, {-10, 5}}, {0, 2}));
  range.files.push_back(file_info(3, {{-20, -15}, {4, 35}}, {0, 1}));
  range.cluster("x");
  REQUIRE(range.eligible);
  omit_footers(range);
  auto batcher = coalescer(22);
  auto result  = drain(range, *batcher, quantum);
  CHECK(result.groups == membership{{3, 0}, {8, 2}, {3, 1}, {8, 0}});
  REQUIRE(result.batches.size() == 2);
  for (auto const& batch : result.batches)
    CHECK(batch->preparation_bytes() == 22);
}

TEST_CASE("Parquet range coalescing inherits compatibility and row count boundaries",
          "[scan][parquet][range_cluster][scan_preparation]")
{
  auto boundary = GENERATE(0, 1, 2, 3, 4);
  auto quantum  = GENERATE(1u, 100u);
  CAPTURE(boundary, quantum);
  scan::parquet_range_scan_info range;
  range.files.push_back(file_info(0, {{0, 1}}, {0}));
  range.files.push_back(file_info(1, {{2, 3}}, {0}));
  auto& second = *range.files[1];
  switch (boundary) {
    case 0: second.original_schema = {1}; break;
    case 1: second.arrow_schema = "different"; break;
    case 2: second.partition_values = {"different"}; break;
    case 3:
      second.reader_options = std::make_shared<cudf::io::parquet_reader_options>(
        cudf::io::parquet_reader_options::builder().build());
      break;
    case 4:
      for (auto& file : range.files)
        file->row_groups[0].num_rows = int64_t{std::numeric_limits<cudf::size_type>::max()} / 2 + 1;
      break;
  }
  range.cluster("x");
  omit_footers(range);
  auto batcher = coalescer(1024);
  auto result  = drain(range, *batcher, quantum);
  CHECK(result.groups == membership{{0, 0}, {1, 0}});
  CHECK(result.batches.size() == 2);
}

TEST_CASE("Pruned files in a live parquet range do not add empty output batches",
          "[scan][parquet][range_cluster][scan_preparation]")
{
  scan::parquet_range_scan_info range;
  range.files.push_back(file_info(0, {{1, 2}}, {}));
  range.files.push_back(file_info(1, {{20, 30}, {-20, -10}}, {0, 1}));
  range.files.push_back(file_info(2, {{3, 4}}, {}));
  range.cluster("x");
  REQUIRE(range.eligible);
  CHECK(sorted(members(range)) == membership{{0, 0}, {1, 0}, {1, 1}, {2, 0}});
  omit_footers(range);
  auto batcher = coalescer(1024);
  auto result  = drain(range, *batcher, 1);
  CHECK(result.groups == membership{{1, 1}, {1, 0}});
  CHECK(result.batches.size() == 1);
}

TEST_CASE("All-pruned parquet range retains one certified typed empty split",
          "[scan][parquet][range_cluster][scan_preparation]")
{
  scan::parquet_range_scan_info range;
  range.files.push_back(file_info(7, {{1, 2}}, {}));
  range.files.push_back(file_info(9, {{3, 4}}, {}));
  range.files[0]->partition_values = {"first"};
  range.cluster("x");
  CHECK(members(range) == membership{{0, 0}, {1, 0}});
  CHECK(range.preparation_bytes() == 0);
  omit_footers(range);
  auto batcher = coalescer(1024);
  auto result  = drain(range, *batcher, 1);
  CHECK(result.groups.empty());
  REQUIRE(result.batches.size() == 1);
  auto const& split = dynamic_cast<scan::parquet_split_info const&>(*result.batches[0]);
  REQUIRE(split.rg_slices.size() == 1);
  CHECK(split.rg_slices[0].file_index == 7);
  CHECK(split.rg_slices[0].row_group_indices.empty());
  CHECK(split.partition_values == std::vector<std::string>{"first"});
}

TEST_CASE("Parquet range preparation drains beyond lookahead and releases aggregate ownership",
          "[scan][parquet][range_cluster][scan_preparation][coordinator]")
{
  namespace prep = sirius::test::preparation;
  using namespace sirius::scan_manager;
  auto cancel         = GENERATE(false, true);
  auto oversized_tail = GENERATE(false, true);
  CAPTURE(cancel, oversized_tail);
  auto options              = preparation_config{}.resolve(1);
  options.lookahead_batches = 1;
  options.drain_quantum     = 1;
  // This tests credit-driven progress, not wall-clock partial emission.
  options.partial_batch = partial_batch_options{50, std::chrono::hours(1), std::chrono::hours(1)};
  prep::fixture f(1, 1, k_interrupt_check_interval, options);
  auto guard    = f.shutdown_guard();
  auto out      = std::make_shared<prep::sink>();
  auto released = std::make_shared<std::atomic<size_t>>(0);
  struct tracked_range : scan::parquet_range_scan_info {
    std::shared_ptr<std::atomic<size_t>> released;
    ~tracked_range() override { ++*released; }
  };
  size_t published_bytes = 0;
  membership published_groups;
  f.coordinator.after_publication_for_testing([&](size_t, auto const& metadata, size_t bytes) {
    auto const& split = dynamic_cast<scan::parquet_split_info const&>(*metadata);
    std::lock_guard lock(out->mutex);
    published_bytes += bytes;
    for (auto const& slice : split.rg_slices)
      for (auto group : slice.row_group_indices)
        published_groups.emplace_back(slice.file_index, group);
    out->published.push_back(static_cast<int>(out->published.size()));
    out->pending_bytes.push_back(bytes);
    out->cv.notify_all();
  });
  preparation_coordinator::source source;
  source.coalescer = coalescer(22);
  source.try_claim = prep::single_job([released, oversized_tail] {
    auto range      = std::make_unique<tracked_range>();
    range->released = released;
    for (size_t i = 0; i < 3; ++i) {
      range->files.push_back(file_info(i, {{30, 40}, {20, 30}, {10, 20}, {0, 10}}, {0, 1, 2, 3}));
    }
    // This is last after stable interval sorting, and larger than the entire work window.
    // Final publication must still progress after the aggregate metadata slot retires.
    if (oversized_tail) range->files.back()->row_groups.front().decode_working_bytes = 33;
    range->cluster("x");
    omit_footers(*range);
    return range;
  });
  source.construct = [](auto batch) { return prep::make_publication(std::move(batch)); };
  source.close     = [out] { out->close(); };
  f.register_source(std::move(source), out);
  f.start();
  REQUIRE(out->await(1));
  if (cancel) {
    // Stop while the aggregate still owns unconsumed row groups behind one output credit.
    f.stop();
    CHECK(released->load() == 1);
    CHECK(f.coordinator.snapshot().phase == preparation_coordinator::lifecycle::quiescent);
    return;
  }
  auto const expected_batches = oversized_tail ? 7u : 6u;
  for (size_t n = 1; n <= expected_batches; ++n) {
    REQUIRE(out->await(n));
    out->consume();
  }
  REQUIRE(out->await_closed());
  f.finish();
  CHECK_FALSE(f.completion.has_error());
  CHECK_FALSE(f.owner_error);
  CHECK(released->load() == 1);
  // Credits are byte-based: the 11-byte underfilled boundary can coexist with one
  // oversized 33-byte final batch. Overshoot by one indivisible batch is intentional.
  CHECK(f.coordinator.snapshot().output_peak <= (oversized_tail ? 2 : 1));
  std::lock_guard lock(out->mutex);
  CHECK(out->published.size() == expected_batches);
  CHECK(published_bytes == (oversized_tail ? 154 : 132));
  CHECK(sorted(published_groups) == membership{{0, 0},
                                               {0, 1},
                                               {0, 2},
                                               {0, 3},
                                               {1, 0},
                                               {1, 1},
                                               {1, 2},
                                               {1, 3},
                                               {2, 0},
                                               {2, 1},
                                               {2, 2},
                                               {2, 3}});
}

TEST_CASE("Parquet range provider eagerly claims only when its output key is admitted",
          "[scan][parquet][range_cluster]")
{
  auto mode = GENERATE(0, 1, 2, 3);
  CAPTURE(mode);
  auto info                    = std::make_unique<scan::parquet_ingestible_table_info>();
  info->resolved_file_paths    = {"a.parquet", "b.parquet"};
  info->names                  = {"x"};
  info->returned_types         = {sirius::logical_type::make(sirius::type_id::INTEGER)};
  info->column_ids             = {duckdb::ColumnIndex(0)};
  info->projected_output_count = 1;
  if (mode == 1 || mode == 2) info->experimental_range_cluster_output_key = mode == 1 ? 0 : 99;
  if (mode == 3) info->experimental_range_cluster_audit_output_key = 0;
  scan::parquet_gpu_ingestible reader(std::move(info));
  size_t resolutions = 0;
  auto resolve       = [&](std::string_view) -> std::shared_ptr<cucascade::io::ioctx> {
    ++resolutions;
    return {};
  };
  CHECK_FALSE(reader.has_processed_all_metadata());
  auto first = reader.next_split_provider(resolve);
  REQUIRE(first);
  CHECK(reader.has_processed_all_metadata() == (mode == 1));
  auto second = reader.next_split_provider(resolve);
  CHECK(static_cast<bool>(second) == (mode != 1));
  CHECK(reader.has_processed_all_metadata());
  CHECK_FALSE(reader.next_split_provider(resolve));
  CHECK(resolutions == 0);
}

TEST_CASE("Parquet range preparation gate is absent from default bind descriptors",
          "[scan][parquet][range_cluster]")
{
  scan::parquet_ingestible_table_info info;
  CHECK_FALSE(info.experimental_range_cluster_output_key);
  CHECK_FALSE(info.experimental_range_cluster_audit_output_key);
}
