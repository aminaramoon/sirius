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

// The parquet consumer's device route: decoding from column chunks a device-tier readahead
// already landed on the GPU, with its checks and its fallbacks to the general route.

#include "catch.hpp"
#include "op/scan/parquet_gpu_ingestible.hpp"
#include "op/scan/parquet_materialize.hpp"
#include "op/scan/prefetched_ranges.hpp"
#include "op/scan/scan_plan.hpp"
#include "op/scan/table_scan/scan_contract.hpp"
#include "scan/test_utils.hpp"
#include "utils/local_ioctx.hpp"
#include "utils/parquet_fixture_utils.hpp"
#include "utils/parquet_test_sources.hpp"
#include "utils/table_filter_test_utils.hpp"
#include "utils/test_validation_utility.hpp"
#include "utils/utils.hpp"

#include <cudf/ast/expressions.hpp>
#include <cudf/io/datasource.hpp>
#include <cudf/io/parquet.hpp>
#include <cudf/io/text/byte_range_info.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/span.hpp>

#include <rmm/cuda_stream.hpp>
#include <rmm/device_buffer.hpp>

#include <cucascade/cudf/datasource.hpp>
#include <cucascade/io/io_context.hpp>
#include <cucascade/memory/memory_space.hpp>
#include <duckdb.hpp>
#include <duckdb/common/multi_file/multi_file_reader.hpp>

#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <numeric>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace {

namespace fs   = std::filesystem;
namespace scan = sirius::op::scan;

using byte_range = cudf::io::text::byte_range_info;
using scan::prefetched_chunk_spans;
using scan::prefetched_ranges;
using sirius::test::bulk_object;
using sirius::test::local_context;

constexpr cudf::size_type k_rows       = 10240;
constexpr cudf::size_type k_row_groups = 5;

/// `x` BIGINT (offset + i), `y` INTEGER, `z` VARCHAR; k_row_groups row groups.
fs::path write_rows(duckdb::Connection& con,
                    fs::path const& dir,
                    std::string const& table,
                    std::int64_t offset)
{
  sirius::test::require_ok(con.Query("DROP TABLE IF EXISTS " + table));
  sirius::test::require_ok(
    con.Query("CREATE TABLE " + table + " AS SELECT (i + " + std::to_string(offset) +
              ")::BIGINT AS x, (i * 7)::INTEGER AS y, ('v' || i::VARCHAR) AS z FROM range(" +
              std::to_string(k_rows) + ") t(i)"));
  auto path = dir / (table + ".parquet");
  sirius::test::require_ok(con.Query("COPY " + table + " TO '" + path.string() +
                                     "' (FORMAT PARQUET, ROW_GROUP_SIZE " +
                                     std::to_string(k_rows / k_row_groups) + ")"));
  return path;
}

/// Two parquet files with the same schema and different values, in a directory of their own
/// (test shards run in parallel processes).
struct parquet_files {
  sirius::test::scratch_dir scratch{"device_prefetch"};
  fs::path dir = scratch.path();
  fs::path first;
  fs::path second;

  parquet_files()
  {
    auto [db_owner, con] = sirius::make_test_db_and_connection();
    first                = write_rows(con, dir, "device_prefetch_first", 0);
    second               = write_rows(con, dir, "device_prefetch_second", 1'000'000);
  }
};

std::vector<cudf::size_type> all_row_groups(cudf::io::parquet::FileMetaData const& metadata)
{
  std::vector<cudf::size_type> groups(metadata.row_groups.size());
  std::iota(groups.begin(), groups.end(), 0);
  return groups;
}

/// A source over @p ioctx; with a route-only context nothing is ever read through it.
scan::parquet_source source_over(std::shared_ptr<cucascade::io::ioctx> const& ioctx,
                                 fs::path const& path,
                                 bool real_object = false)
{
  auto object = real_object ? ioctx->open_io_object(path.string())
                            : std::make_shared<bulk_object>(path.string(), fs::file_size(path));
  auto metadata =
    std::make_shared<cudf::io::parquet::FileMetaData const>(sirius::test::read_metadata(path));
  auto groups = all_row_groups(*metadata);
  REQUIRE(groups.size() == k_row_groups);
  return {std::make_shared<cucascade::io::datasource>(ioctx, std::move(object)),
          std::move(metadata),
          std::move(groups)};
}

/// What a device-tier prefetch would have landed for @p source under @p options.
struct staged_source {
  std::vector<byte_range> ranges;
  sirius::test::staged_chunks chunks;
};

staged_source stage(scan::parquet_source const& source,
                    fs::path const& path,
                    cudf::io::parquet_reader_options const& options,
                    ::cuda::stream_ref stream)
{
  staged_source out;
  out.ranges = scan::column_chunk_ranges(*source.metadata, options, source.row_group_indices);
  auto file  = cudf::io::datasource::create(path.string());
  out.chunks = sirius::test::stage_column_chunks(*file, out.ranges, stream);
  return out;
}

/// `x < 3000`: the first 3000 rows of the first file.
struct x_below_3000 {
  cudf::numeric_scalar<std::int64_t> limit{3000};
  cudf::ast::literal value{limit};
  cudf::ast::column_name_reference column{"x"};
  cudf::ast::operation expression{cudf::ast::ast_operator::LESS, column, value};
};

bool tables_equal(cudf::table_view const& lhs,
                  cudf::table_view const& rhs,
                  ::cuda::stream_ref stream)
{
  return sirius::test::expect_tables_equivalent_impl(lhs, rhs, stream);
}

}  // namespace

TEST_CASE("materialize_parquet decodes from prefetched device spans without a bulk backend",
          "[scan][parquet][device_prefetch]")
{
  parquet_files const files;
  auto const stream  = cudf::get_default_stream();
  auto const mr      = cudf::get_current_device_resource_ref();
  auto const options = cudf::io::parquet_reader_options::builder().build();

  std::vector<scan::parquet_source> const sources{
    source_over(std::make_shared<local_context>(), files.first)};
  REQUIRE_FALSE(scan::prefers_bulk_materialize(sources, options));
  auto const staged = stage(sources.front(), files.first, options, stream);
  std::vector<std::vector<byte_range>> const ranges{staged.ranges};
  std::vector<prefetched_chunk_spans> prefetched;
  prefetched.emplace_back(staged.chunks.spans);
  REQUIRE(scan::can_materialize_from_prefetched(sources, ranges, prefetched));

  auto const from_device =
    scan::materialize_parquet(sources, options, ranges, prefetched, stream, mr);
  REQUIRE(from_device);
  CHECK(from_device->num_rows() == k_rows);
  CHECK(from_device->num_columns() == 3);

  auto const expected =
    cudf::io::read_parquet(
      cudf::io::parquet_reader_options::builder(cudf::io::source_info{files.first.string()})
        .build(),
      stream)
      .tbl;
  CHECK(tables_equal(from_device->view(), expected->view(), stream));

  // The 5-argument call over a real local backend takes the general route.
  auto const local_files = sirius::test::make_test_local_ioctx();
  std::vector<scan::parquet_source> const readable{
    source_over(local_files, files.first, /*real_object=*/true)};
  REQUIRE_FALSE(scan::prefers_bulk_materialize(readable, options));
  auto const general = scan::materialize_parquet(readable, options, {}, stream, mr);
  CHECK(tables_equal(from_device->view(), general->view(), stream));
}

TEST_CASE("two sources decode from prefetched spans", "[scan][parquet][device_prefetch]")
{
  parquet_files const files;
  auto const stream  = cudf::get_default_stream();
  auto const mr      = cudf::get_current_device_resource_ref();
  auto const options = cudf::io::parquet_reader_options::builder().build();
  auto const local   = std::make_shared<local_context>();

  std::vector<scan::parquet_source> const sources{source_over(local, files.first),
                                                  source_over(local, files.second)};
  auto const first  = stage(sources[0], files.first, options, stream);
  auto const second = stage(sources[1], files.second, options, stream);
  std::vector<std::vector<byte_range>> const ranges{first.ranges, second.ranges};
  std::vector<prefetched_chunk_spans> prefetched;
  prefetched.emplace_back(first.chunks.spans);
  prefetched.emplace_back(second.chunks.spans);

  auto const from_device =
    scan::materialize_parquet(sources, options, ranges, prefetched, stream, mr);
  REQUIRE(from_device);
  CHECK(from_device->num_rows() == 2 * k_rows);

  auto const expected =
    cudf::io::read_parquet(
      cudf::io::parquet_reader_options::builder(cudf::io::source_info{std::vector<std::string>{
                                                  files.first.string(), files.second.string()}})
        .build(),
      stream)
      .tbl;
  CHECK(tables_equal(from_device->view(), expected->view(), stream));
}

TEST_CASE("prefetched spans that do not line up are rejected before decoding",
          "[scan][parquet][device_prefetch]")
{
  parquet_files const files;
  auto const stream  = cudf::get_default_stream();
  auto const mr      = cudf::get_current_device_resource_ref();
  auto const options = cudf::io::parquet_reader_options::builder().build();

  std::vector<scan::parquet_source> const sources{
    source_over(std::make_shared<local_context>(), files.first)};
  auto const staged = stage(sources.front(), files.first, options, stream);
  std::vector<std::vector<byte_range>> const ranges{staged.ranges};
  auto const& spans = staged.chunks.spans;
  REQUIRE(spans.size() > 1);

  SECTION("a span dropped")
  {
    std::vector<prefetched_chunk_spans> dropped;
    dropped.emplace_back(std::span(spans).first(spans.size() - 1));
    CHECK_FALSE(scan::can_materialize_from_prefetched(sources, ranges, dropped));
    CHECK_THROWS_AS(scan::materialize_parquet(sources, options, ranges, dropped, stream, mr),
                    std::invalid_argument);
    // Without the split's ranges only the reader's own enumeration can tell.
    CHECK(scan::can_materialize_from_prefetched(sources, {}, dropped));
    CHECK_THROWS_AS(scan::materialize_parquet(sources, options, {}, dropped, stream, mr),
                    scan::prefetched_mismatch);
  }

  SECTION("a span of the wrong size with the right count")
  {
    auto resized = spans;
    resized[0]   = cudf::device_span<std::uint8_t const>(resized[0].data(), resized[0].size() - 1);
    std::vector<prefetched_chunk_spans> wrong;
    wrong.emplace_back(resized);
    CHECK_FALSE(scan::can_materialize_from_prefetched(sources, ranges, wrong));
    CHECK(scan::can_materialize_from_prefetched(sources, {}, wrong));
    CHECK_THROWS_AS(scan::materialize_parquet(sources, options, {}, wrong, stream, mr),
                    scan::prefetched_mismatch);
  }

  SECTION("an empty entry")
  {
    std::vector<prefetched_chunk_spans> empty(1);
    CHECK_FALSE(scan::can_materialize_from_prefetched(sources, {}, empty));
    CHECK_THROWS_AS(scan::materialize_parquet(sources, options, {}, empty, stream, mr),
                    std::invalid_argument);
  }
}

TEST_CASE("a pushed-down filter on a projected column decodes from device spans",
          "[scan][parquet][device_prefetch]")
{
  parquet_files const files;
  auto const stream = cudf::get_default_stream();
  auto const mr     = cudf::get_current_device_resource_ref();
  x_below_3000 const filter;
  auto const options =
    sirius::test::filtered_options({files.first.string()}, {"x", "z"}, filter.expression);

  std::vector<scan::parquet_source> const sources{
    source_over(std::make_shared<local_context>(), files.first)};
  auto const staged = stage(sources.front(), files.first, options, stream);
  std::vector<std::vector<byte_range>> const ranges{staged.ranges};
  std::vector<prefetched_chunk_spans> prefetched;
  prefetched.emplace_back(staged.chunks.spans);

  auto const from_device =
    scan::materialize_parquet(sources, options, ranges, prefetched, stream, mr);
  auto const expected = cudf::io::read_parquet(options, stream).tbl;
  CHECK(from_device->num_rows() == 3000);
  CHECK(tables_equal(from_device->view(), expected->view(), stream));
}

TEST_CASE("a filter column outside the projection falls back instead of crashing",
          "[scan][parquet][device_prefetch]")
{
  parquet_files const files;
  auto const stream = cudf::get_default_stream();
  auto const mr     = cudf::get_current_device_resource_ref();
  x_below_3000 const filter;
  // Staged for the projection alone: the filter column's chunks are not on the device.
  auto staging_options =
    cudf::io::parquet_reader_options::builder(cudf::io::source_info{files.first.string()}).build();
  staging_options.set_column_names({"y", "z"});
  auto const decode_options =
    sirius::test::filtered_options({files.first.string()}, {"y", "z"}, filter.expression);

  std::vector<scan::parquet_source> const sources{
    source_over(std::make_shared<local_context>(), files.first)};
  auto const staged = stage(sources.front(), files.first, staging_options, stream);
  std::vector<std::vector<byte_range>> const ranges{staged.ranges};
  std::vector<prefetched_chunk_spans> prefetched;
  prefetched.emplace_back(staged.chunks.spans);

  REQUIRE(scan::can_materialize_from_prefetched(sources, ranges, prefetched));
  CHECK_THROWS_AS(
    scan::materialize_parquet(sources, decode_options, ranges, prefetched, stream, mr),
    scan::prefetched_mismatch);
}

namespace {

std::shared_ptr<cucascade::io::datasource> route_only_datasource(std::string const& name)
{
  return std::make_shared<cucascade::io::datasource>(std::make_shared<local_context>(),
                                                     std::make_shared<bulk_object>(name, 1024));
}

/// Slices a, b, c: a and c have a datasource, row groups and ranges; b (when @p b_empty) has no
/// row groups and no ranges, as an all-pruned file in a bundle would.
std::unique_ptr<scan::parquet_split_info> three_slice_split(bool b_empty)
{
  auto split =
    std::make_unique<scan::parquet_split_info>(std::vector<scan::scan_info::fadvise_entry>{});
  split->rg_slices.emplace_back(
    nullptr, "a", std::vector<cudf::size_type>{0}, 0, 0, 0, route_only_datasource("a"), 0);
  split->rg_slices.emplace_back(
    nullptr,
    "b",
    b_empty ? std::vector<cudf::size_type>{} : std::vector<cudf::size_type>{0},
    0,
    0,
    0,
    route_only_datasource("b"),
    1);
  split->rg_slices.emplace_back(
    nullptr, "c", std::vector<cudf::size_type>{0, 1}, 0, 0, 0, route_only_datasource("c"), 2);
  split->set_column_chunk_byte_ranges(
    {{byte_range(0, 16), byte_range(16, 32)},
     b_empty ? std::vector<byte_range>{} : std::vector<byte_range>{byte_range(4, 4)},
     {byte_range(100, 8)}});
  return split;
}

prefetched_ranges::device_ranges fake_entry(std::vector<std::size_t> const& sizes)
{
  prefetched_ranges::device_ranges entry;
  for (auto const size : sizes) {
    entry.buffers.emplace_back(size, cudf::get_default_stream());
  }
  for (auto const& buffer : entry.buffers) {
    entry.spans.emplace_back(static_cast<std::uint8_t const*>(buffer.data()), buffer.size());
  }
  return entry;
}

}  // namespace

TEST_CASE("prefetched_spans_per_slice mirrors the split's fadvise entries",
          "[scan][parquet][device_prefetch]")
{
  auto const split = three_slice_split(/*b_empty=*/true);

  std::vector<prefetched_ranges::device_ranges> entries;
  entries.push_back(fake_entry({16, 32}));
  entries.push_back(fake_entry({8}));
  auto const mapped = split->prefetched_spans_per_slice(entries);
  REQUIRE(mapped.has_value());
  REQUIRE(mapped->size() == 3);
  REQUIRE((*mapped)[0].size() == 2);
  CHECK((*mapped)[0][0].data() == entries[0].spans[0].data());
  CHECK((*mapped)[0][1].data() == entries[0].spans[1].data());
  CHECK((*mapped)[1].empty());
  REQUIRE((*mapped)[2].size() == 1);
  CHECK((*mapped)[2][0].data() == entries[1].spans[0].data());

  SECTION("a span of the wrong size")
  {
    std::vector<prefetched_ranges::device_ranges> wrong;
    wrong.push_back(fake_entry({16, 31}));
    wrong.push_back(fake_entry({8}));
    CHECK_FALSE(split->prefetched_spans_per_slice(wrong).has_value());
  }
  SECTION("a chunk missing from an entry")
  {
    std::vector<prefetched_ranges::device_ranges> wrong;
    wrong.push_back(fake_entry({16}));
    wrong.push_back(fake_entry({8}));
    CHECK_FALSE(split->prefetched_spans_per_slice(wrong).has_value());
  }
  SECTION("an extra entry")
  {
    std::vector<prefetched_ranges::device_ranges> wrong;
    wrong.push_back(fake_entry({16, 32}));
    wrong.push_back(fake_entry({8}));
    wrong.push_back(fake_entry({8}));
    CHECK_FALSE(split->prefetched_spans_per_slice(wrong).has_value());
  }
  SECTION("a missing entry")
  {
    std::vector<prefetched_ranges::device_ranges> wrong;
    wrong.push_back(fake_entry({16, 32}));
    CHECK_FALSE(split->prefetched_spans_per_slice(wrong).has_value());
  }
}

TEST_CASE("supports_device_prefetch requires every slice to be decodable from device spans",
          "[scan][parquet][device_prefetch]")
{
  CHECK(three_slice_split(/*b_empty=*/false)->supports_device_prefetch());
  CHECK_FALSE(three_slice_split(/*b_empty=*/true)->supports_device_prefetch());

  auto split = three_slice_split(/*b_empty=*/false);
  SECTION("the ingestible opted out")
  {
    split->device_prefetch_allowed = false;
    CHECK_FALSE(split->supports_device_prefetch());
  }
  SECTION("no slices")
  {
    split->rg_slices.clear();
    split->set_column_chunk_byte_ranges({});
    CHECK_FALSE(split->supports_device_prefetch());
  }
  SECTION("a slice without a datasource")
  {
    split->rg_slices[2].datasource.reset();
    CHECK_FALSE(split->supports_device_prefetch());
  }
  SECTION("a slice without ranges")
  {
    split->set_column_chunk_byte_ranges(
      {{byte_range(0, 16), byte_range(16, 32)}, {byte_range(4, 4)}, {}});
    CHECK_FALSE(split->supports_device_prefetch());
  }
  SECTION("user virtual columns")
  {
    auto plan = std::make_shared<scan::scan_plan>();
    CHECK(split->supports_device_prefetch());
    plan->virtual_columns.push_back({0,
                                     "filename",
                                     sirius::logical_type::make(sirius::type_id::VARCHAR),
                                     scan::scan_plan::parquet_virtual_column_kind::FILENAME,
                                     0});
    split->plan = std::move(plan);
    CHECK_FALSE(split->supports_device_prefetch());
  }
}

namespace {

/// What the release hook saw, shared with the hook so it outlives the prefetch.
struct release_counts {
  std::array<std::atomic<int>, 4> by_reason{};

  [[nodiscard]] int of(prefetched_ranges::release_reason reason) const
  {
    return by_reason[static_cast<std::size_t>(reason)].load();
  }
};

/// How the consumer fixture's scan is set up.
struct consumer_shape {
  bool two_files                 = false;  ///< a second same-schema file bundled into the split
  bool filename_column           = false;  ///< also project the `filename` virtual column
  bool filter_outside_projection = false;  ///< `y < 21000` with y not projected: 3000 rows/file
};

/// One parquet split built by the real split provider and coalescer over a local uring ioctx,
/// and the ingestible that decodes it.
struct consumer_fixture {
  sirius::test::scratch_dir scratch{"device_prefetch_consumer"};
  std::vector<fs::path> files;
  std::shared_ptr<scan::physical_check_counters> counters =
    std::make_shared<scan::physical_check_counters>();
  std::unique_ptr<sirius::memory::sirius_memory_reservation_manager> memory =
    initialize_memory_manager(1);
  cucascade::memory::memory_space* space      = memory->get_memory_space(Tier::GPU, 0);
  std::shared_ptr<cucascade::io::ioctx> ioctx = sirius::test::make_test_local_ioctx();
  std::shared_ptr<scan::device_stream> tier   = scan::device_stream::create(0);
  std::shared_ptr<release_counts> counts      = std::make_shared<release_counts>();
  std::shared_ptr<scan::parquet_gpu_ingestible> ingestible;
  std::unique_ptr<scan::scan_info> owned_split;
  scan::parquet_split_info* split = nullptr;
  rmm::cuda_stream stream;

  consumer_fixture() : consumer_fixture(consumer_shape{}) {}

  explicit consumer_fixture(consumer_shape shape)
  {
    REQUIRE(space != nullptr);
    {
      auto [db_owner, con] = sirius::make_test_db_and_connection();
      files.push_back(write_rows(con, scratch.path(), "device_prefetch_consumer", 0));
      if (shape.two_files) {
        files.push_back(write_rows(con, scratch.path(), "device_prefetch_consumer_2", 1'000'000));
      }
    }
    auto info         = std::make_unique<scan::parquet_ingestible_table_info>();
    info->contract_id = 53;
    for (auto const& path : files) {
      info->resolved_file_paths.push_back(path.string());
    }
    info->names = {"x", "y", "z"};
    info->returned_types.push_back(sirius::logical_type::make(sirius::type_id::BIGINT));
    info->returned_types.push_back(sirius::logical_type::make(sirius::type_id::INTEGER));
    info->returned_types.push_back(sirius::logical_type::make(sirius::type_id::VARCHAR));
    info->column_ids.push_back(duckdb::ColumnIndex(0));
    info->column_ids.push_back(duckdb::ColumnIndex(2));
    if (shape.filename_column) {
      info->column_ids.push_back(
        duckdb::ColumnIndex(duckdb::MultiFileReader::COLUMN_IDENTIFIER_FILENAME));
      info->virtual_columns = {{duckdb::MultiFileReader::COLUMN_IDENTIFIER_FILENAME,
                                "filename",
                                sirius::logical_type::make(sirius::type_id::VARCHAR),
                                scan::scan_plan::parquet_virtual_column_kind::FILENAME}};
    }
    info->projected_output_count = info->column_ids.size();
    if (shape.filter_outside_projection) {
      // Filter columns follow the projected outputs in column_ids.
      info->column_ids.push_back(duckdb::ColumnIndex(1));
      auto filters = duckdb::make_uniq<duckdb::TableFilterSet>();
      filters->PushFilter(duckdb::ProjectionIndex(info->column_ids.size() - 1),
                          sirius::test::constant_filter(duckdb::ExpressionType::COMPARE_LESSTHAN,
                                                        duckdb::Value::INTEGER(21000)));
      info->table_filters = std::move(filters);
    }
    info->approximate_batch_size = std::size_t{1} << 30;
    info->profiles->counters     = counters;
    ingestible                   = scan::make_ingestible(std::move(info));

    // One metadata task per file; the coalescer bundles them into one split.
    auto coalescer = ingestible->create_batch_coalescer();
    std::vector<std::unique_ptr<scan::scan_info>> batches;
    for (std::size_t i = 0; i < files.size(); ++i) {
      auto task = ingestible->next_split_provider(
        [this](std::string_view) -> std::shared_ptr<cucascade::io::ioctx> { return ioctx; });
      REQUIRE(task);
      for (auto& batch : coalescer->push(task({}))) {
        batches.push_back(std::move(batch));
      }
    }
    for (auto& batch : coalescer->flush()) {
      batches.push_back(std::move(batch));
    }
    REQUIRE(batches.size() == 1);
    owned_split = std::move(batches.front());
    split       = dynamic_cast<scan::parquet_split_info*>(owned_split.get());
    REQUIRE(split != nullptr);
    REQUIRE(split->rg_slices.size() == files.size());
    // The filter's statistics keep the first two row groups (rows 0..4095) of each file.
    auto const kept = shape.filter_outside_projection ? std::size_t{2} : std::size_t{k_row_groups};
    for (auto const& slice : split->rg_slices) {
      REQUIRE(slice.row_group_indices.size() == kept);
    }
  }

  scan::filtered_table materialize()
  {
    return ingestible->materialize_metadata_to_table(*split, *space, stream, false, nullptr);
  }

  /// The split's hint ranges staged on the tier stream, one entry per fadvise hint.  With
  /// @p short_first_span the first span is one byte shorter than its buffer.
  std::shared_ptr<prefetched_ranges> landed(int device_id, bool short_first_span = false)
  {
    std::vector<prefetched_ranges::device_ranges> entries;
    std::size_t total = 0;
    auto const ranges = split->column_chunk_byte_ranges();
    for (std::size_t i = 0; i < split->rg_slices.size(); ++i) {
      if (!split->rg_slices[i].datasource || ranges[i].empty()) { continue; }
      auto source = cudf::io::datasource::create(std::string{split->rg_slices[i].file_path});
      auto staged = sirius::test::stage_column_chunks(*source, ranges[i], tier->view());
      for (auto const& range : ranges[i]) {
        total += static_cast<std::size_t>(range.size());
      }
      entries.push_back({std::move(staged.buffers), std::move(staged.spans)});
    }
    REQUIRE(entries.size() == split->fadvise_hints().size());
    if (short_first_span) {
      auto& first = entries.front().spans.front();
      first       = cudf::device_span<std::uint8_t const>(first.data(), first.size() - 1);
    }
    tier->view().synchronize();
    return std::make_shared<prefetched_ranges>(
      device_id,
      tier,
      std::move(entries),
      total,
      [counts = counts](prefetched_ranges::release_event event) noexcept {
        counts->by_reason[static_cast<std::size_t>(event.reason)].fetch_add(1);
      });
  }

  void install(std::shared_ptr<prefetched_ranges> prefetch)
  {
    REQUIRE(split->set_prefetched_ranges(std::move(prefetch)));
  }
};

using release_reason = prefetched_ranges::release_reason;

}  // namespace

TEST_CASE_METHOD(consumer_fixture,
                 "the parquet consumer decodes a landed prefetch and falls back otherwise",
                 "[scan][parquet][device_prefetch]")
{
  CHECK(split->device_prefetch_allowed);
  CHECK(split->supports_device_prefetch());

  // Today's route over local files: read_parquet.
  auto const baseline = materialize();
  REQUIRE(baseline.table.num_rows() == k_rows);
  CHECK(counters->device_route_decodes.load() == 0);

  SECTION("a ready prefetch on the task's GPU is decoded from the device, once")
  {
    auto prefetch = landed(0);
    prefetch->set_ready();
    std::weak_ptr<prefetched_ranges> const watch = prefetch;
    install(std::move(prefetch));

    auto const decoded = materialize();
    CHECK(counters->device_route_decodes.load() == 1);
    CHECK(counters->device_route_fallbacks.load() == 0);
    CHECK(counts->of(release_reason::consumed) == 1);
    CHECK(watch.expired());
    CHECK(split->prefetched() == nullptr);
    CHECK(tables_equal(decoded.table.view(), baseline.table.view(), stream));

    // Re-entry (an OOM retry) finds the slot empty and reads for itself.
    auto const again = materialize();
    CHECK(counters->device_route_decodes.load() == 1);
    CHECK(tables_equal(again.table.view(), baseline.table.view(), stream));
  }

  SECTION("a prefetch on another GPU is discarded without waiting")
  {
    // Stamped device 1 against the task's device 0; no buffers, so no device-1 memory is
    // needed.  It is still pending: a wait would take the full bound.
    install(std::make_shared<prefetched_ranges>(
      1,
      tier,
      std::vector<prefetched_ranges::device_ranges>{},
      0,
      [counts = counts](prefetched_ranges::release_event event) noexcept {
        counts->by_reason[static_cast<std::size_t>(event.reason)].fetch_add(1);
      }));

    auto const start   = std::chrono::steady_clock::now();
    auto const decoded = materialize();
    CHECK(std::chrono::steady_clock::now() - start < std::chrono::seconds(30));
    CHECK(counts->of(release_reason::wrong_device) == 1);
    CHECK(counters->device_route_decodes.load() == 0);
    CHECK(tables_equal(decoded.table.view(), baseline.table.view(), stream));
  }

  SECTION("a failed or abandoned prefetch is read through the general route")
  {
    auto failed = landed(0);
    failed->set_failed(std::make_exception_ptr(std::runtime_error("injected read failure")));
    install(std::move(failed));
    auto const after_failure = materialize();
    CHECK(tables_equal(after_failure.table.view(), baseline.table.view(), stream));

    auto abandoned = landed(0);
    abandoned->abandon();
    install(std::move(abandoned));
    auto const after_abandon = materialize();
    CHECK(tables_equal(after_abandon.table.view(), baseline.table.view(), stream));

    CHECK(counts->of(release_reason::failed) == 2);
    CHECK(counters->device_route_decodes.load() == 0);
    CHECK(counters->device_route_fallbacks.load() == 0);
  }

  SECTION("spans the reader does not expect fall back to the general route, counted")
  {
    // The split's own ranges and the landed spans agree with each other (so the mapping and the
    // shape check pass) but not with the reader's enumeration: only the exact check catches it.
    std::vector<std::vector<byte_range>> ranges(split->column_chunk_byte_ranges().begin(),
                                                split->column_chunk_byte_ranges().end());
    auto const first       = ranges.front().front();
    ranges.front().front() = byte_range(first.offset(), first.size() - 1);
    split->set_column_chunk_byte_ranges(std::move(ranges));
    auto prefetch = landed(0);
    prefetch->set_ready();
    install(std::move(prefetch));

    auto const decoded = materialize();
    CHECK(counters->device_route_fallbacks.load() == 1);
    CHECK(counters->device_route_decodes.load() == 0);
    CHECK(counts->of(release_reason::consumed) == 1);
    CHECK(tables_equal(decoded.table.view(), baseline.table.view(), stream));
  }

  SECTION("spans that do not line up with the split are not decoded")
  {
    auto prefetch = landed(0, /*short_first_span=*/true);
    prefetch->set_ready();
    install(std::move(prefetch));

    auto const decoded = materialize();
    CHECK(counters->device_route_decodes.load() == 0);
    CHECK(counters->device_route_fallbacks.load() == 0);
    CHECK(counts->of(release_reason::unconsumed) == 1);
    CHECK(counts->of(release_reason::consumed) == 0);
    CHECK(tables_equal(decoded.table.view(), baseline.table.view(), stream));
  }
}

TEST_CASE_METHOD(consumer_fixture,
                 "a prefetch the split cannot decode from is released unconsumed",
                 "[scan][parquet][device_prefetch]")
{
  auto prefetch = landed(0);
  prefetch->set_ready();
  install(std::move(prefetch));

  SECTION("the split opted out of the device route")
  {
    split->device_prefetch_allowed = false;
    auto const decoded             = materialize();
    CHECK(counts->of(release_reason::unconsumed) == 1);
    CHECK(counters->device_route_decodes.load() == 0);
    CHECK(decoded.table.num_rows() == k_rows);
  }

  SECTION("every row group was pruned")
  {
    for (auto& slice : split->rg_slices) {
      slice.row_group_indices.clear();
    }
    auto const decoded = materialize();
    CHECK(counts->of(release_reason::unconsumed) == 1);
    CHECK(counters->device_route_decodes.load() == 0);
    CHECK(decoded.table.num_rows() == 0);
  }
}

TEST_CASE_METHOD(consumer_fixture,
                 "a consumer waiting on a pending prefetch decodes it once it lands",
                 "[scan][parquet][device_prefetch]")
{
  auto const baseline = materialize();
  auto prefetch       = landed(0);
  install(prefetch);

  // Lands only after the consumer emptied the slot -- its last step before it waits -- so the
  // consumer took the prefetch while it was still pending.
  std::atomic<bool> consumer_took{false};
  std::thread producer([prefetch, &consumer_took, this] {
    auto const deadline = std::chrono::steady_clock::now() + std::chrono::seconds(30);
    while (split->prefetched() != nullptr && std::chrono::steady_clock::now() < deadline) {
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    consumer_took.store(split->prefetched() == nullptr);
    prefetch->set_ready();
  });
  auto const decoded = materialize();
  producer.join();
  prefetch.reset();

  REQUIRE(consumer_took.load());
  CHECK(counters->device_route_decodes.load() == 1);
  CHECK(counters->device_route_fallbacks.load() == 0);
  CHECK(counts->of(release_reason::consumed) == 1);
  CHECK(tables_equal(decoded.table.view(), baseline.table.view(), stream));
}

namespace {

struct two_file_consumer_fixture : consumer_fixture {
  two_file_consumer_fixture() : consumer_fixture(consumer_shape{.two_files = true}) {}
};

struct filename_consumer_fixture : consumer_fixture {
  filename_consumer_fixture() : consumer_fixture(consumer_shape{.filename_column = true}) {}
};

struct filtered_consumer_fixture : consumer_fixture {
  filtered_consumer_fixture() : consumer_fixture(consumer_shape{.filter_outside_projection = true})
  {
  }
};

}  // namespace

TEST_CASE_METHOD(two_file_consumer_fixture,
                 "a two-file split decodes from its device prefetch",
                 "[scan][parquet][device_prefetch]")
{
  CHECK(split->supports_device_prefetch());
  auto const baseline = materialize();
  REQUIRE(baseline.table.num_rows() == 2 * k_rows);

  auto prefetch = landed(0);
  prefetch->set_ready();
  install(std::move(prefetch));
  auto const decoded = materialize();
  CHECK(counters->device_route_decodes.load() == 1);
  CHECK(counters->device_route_fallbacks.load() == 0);
  CHECK(counts->of(release_reason::consumed) == 1);
  CHECK(tables_equal(decoded.table.view(), baseline.table.view(), stream));
}

TEST_CASE_METHOD(filename_consumer_fixture,
                 "a prefetch for a scan with user virtual columns is released unconsumed",
                 "[scan][parquet][device_prefetch]")
{
  CHECK_FALSE(split->supports_device_prefetch());
  auto const baseline = materialize();

  auto prefetch = landed(0);
  prefetch->set_ready();
  install(std::move(prefetch));
  auto const decoded = materialize();
  CHECK(counts->of(release_reason::unconsumed) == 1);
  CHECK(counters->device_route_decodes.load() == 0);
  CHECK(tables_equal(decoded.table.view(), baseline.table.view(), stream));
}

TEST_CASE_METHOD(filtered_consumer_fixture,
                 "a filter column outside the projection reads correctly through the ingestible",
                 "[scan][parquet][device_prefetch]")
{
  auto const baseline = materialize();

  auto prefetch = landed(0);
  prefetch->set_ready();
  install(std::move(prefetch));
  auto const decoded = materialize();
  // The split's ranges come from the reader options, which read the filter column too: the staged
  // chunks cover it, so the split decodes from the device.
  CHECK(counters->device_route_decodes.load() == 1);
  CHECK(counters->device_route_fallbacks.load() == 0);
  CHECK(counts->of(release_reason::consumed) == 1);
  CHECK(decoded.state == baseline.state);
  CHECK(tables_equal(decoded.table.view(), baseline.table.view(), stream));
}
