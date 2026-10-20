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

/**
 * @file parquet_test_sources.hpp
 * @brief Parquet files, footers, staged column chunks and stand-in IO contexts for tests of the
 *        parquet materialize routes.
 */

#include <cudf/ast/expressions.hpp>
#include <cudf/io/datasource.hpp>
#include <cudf/io/experimental/hybrid_scan.hpp>
#include <cudf/io/parquet.hpp>
#include <cudf/io/text/byte_range_info.hpp>
#include <cudf/utilities/span.hpp>

#include <rmm/device_buffer.hpp>

#include <cuda/stream>

#include <catch.hpp>
#include <cucascade/exec/semi_future.hpp>
#include <cucascade/io/byte_range.hpp>
#include <cucascade/io/io_context.hpp>
#include <cucascade/io/types.hpp>
#include <duckdb.hpp>

#include <cstdint>
#include <cstring>
#include <filesystem>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <vector>

namespace sirius::test {

inline void require_ok(std::unique_ptr<duckdb::QueryResult> result)
{
  REQUIRE(result);
  INFO((result->HasError() ? result->GetError() : ""));
  REQUIRE_FALSE(result->HasError());
}

/// An empty scratch directory under the system temp directory, named after @p tag.
inline std::filesystem::path fresh_tmp_dir(std::string const& tag)
{
  auto dir = std::filesystem::temp_directory_path() / ("sirius_pr6_schema_" + tag);
  std::error_code ec;
  std::filesystem::remove_all(dir, ec);
  std::filesystem::create_directories(dir);
  return dir;
}

/// Run @p create_sql (which creates @p table) and copy the table to `<dir>/<table>.parquet`.
inline std::filesystem::path write_parquet(duckdb::Connection& con,
                                           std::filesystem::path const& dir,
                                           std::string const& table,
                                           std::string const& create_sql)
{
  require_ok(con.Query("DROP TABLE IF EXISTS " + table));
  require_ok(con.Query(create_sql));

  auto path = dir / (table + ".parquet");
  require_ok(con.Query("COPY " + table + " TO '" + path.string() + "' (FORMAT PARQUET)"));
  return path;
}

/// The serialized footer of the parquet file behind @p source.
inline std::unique_ptr<cudf::io::datasource::buffer> read_parquet_footer(
  cudf::io::datasource& source)
{
  auto constexpr footer_tail_size = sizeof(cudf::io::parquet::file_ender_s);
  auto const file_size            = source.size();
  REQUIRE(file_size >= footer_tail_size);

  auto tail = source.host_read(file_size - footer_tail_size, footer_tail_size);

  std::uint32_t footer_size = 0;
  std::memcpy(&footer_size, tail->data(), sizeof(footer_size));
  REQUIRE(file_size >= footer_tail_size + footer_size);

  return source.host_read(file_size - footer_tail_size - footer_size, footer_size);
}

/// The parsed footer of the parquet file at @p path.
inline cudf::io::parquet::FileMetaData read_metadata(std::filesystem::path const& path)
{
  auto source = cudf::io::datasource::create(path.string());
  auto footer = read_parquet_footer(*source);
  auto opts   = cudf::io::parquet_reader_options::builder().build();
  cudf::io::parquet::experimental::hybrid_scan_reader reader{
    cudf::host_span<std::uint8_t const>(footer->data(), footer->size()), opts};
  return reader.parquet_metadata();
}

/// Reader options over @p paths projecting @p columns, with @p filter pushed down.
inline cudf::io::parquet_reader_options filtered_options(std::vector<std::string> const& paths,
                                                         std::vector<std::string> columns,
                                                         cudf::ast::operation const& filter)
{
  auto options =
    cudf::io::parquet_reader_options::builder(cudf::io::source_info{paths}).filter(filter).build();
  options.set_column_names(std::move(columns));
  return options;
}

/// One file's column chunks staged on the device, which is the shape the bulk
/// materialize route hands the hybrid scan readers.
struct staged_chunks {
  std::vector<rmm::device_buffer> buffers;
  std::vector<cudf::device_span<std::uint8_t const>> spans;
};

/// Copy each of @p ranges of @p source into its own device buffer, in order.
inline staged_chunks stage_column_chunks(cudf::io::datasource& source,
                                         std::span<cudf::io::text::byte_range_info const> ranges,
                                         ::cuda::stream_ref stream)
{
  staged_chunks staged;
  staged.buffers.reserve(ranges.size());
  for (auto const& range : ranges) {
    auto const host = source.host_read(static_cast<std::size_t>(range.offset()),
                                       static_cast<std::size_t>(range.size()));
    staged.buffers.emplace_back(host->data(), host->size(), stream);
  }
  staged.spans.reserve(staged.buffers.size());
  for (auto const& buffer : staged.buffers) {
    staged.spans.emplace_back(static_cast<std::uint8_t const*>(buffer.data()), buffer.size());
  }
  return staged;
}

/// An io_object with a fixed path and size and no backing storage.
class bulk_object final : public cucascade::io::io_object {
 public:
  explicit bulk_object(std::string path = "bulk", std::size_t size = 0)
    : _path(std::move(path)), _size(size)
  {
  }

  [[nodiscard]] std::string const& raw_file_cache_id() const noexcept override { return _path; }
  [[nodiscard]] std::string const& object_path() const noexcept override { return _path; }
  [[nodiscard]] std::size_t size() const noexcept override { return _size; }

 private:
  std::string _path;
  std::size_t _size;
};

/// Minimal ioctx that answers @c prefers_bulk_io() with @p PrefersBulkIo and never serves a
/// read: tests over it only route, validate schemas, or decode from chunks they staged
/// themselves; none reaches the backend.
template <bool PrefersBulkIo>
class route_only_context final : public cucascade::io::ioctx {
 public:
  ~route_only_context() override { pre_destroy(); }

  [[nodiscard]] cucascade::io::io_context_type type() const noexcept override
  {
    return cucascade::io::io_context_type::restful;
  }
  void shutdown() noexcept override {}
  [[nodiscard]] bool supports(std::string_view) const noexcept override { return true; }
  [[nodiscard]] bool supports_device_read() const noexcept override { return false; }
  [[nodiscard]] bool supports_host_to_device_read() const noexcept override { return false; }
  [[nodiscard]] bool supports_vector_host_read() const noexcept override { return false; }
  [[nodiscard]] bool supports_device_range_read() const noexcept override { return false; }
  [[nodiscard]] bool prefers_bulk_io() const noexcept override { return PrefersBulkIo; }

  [[nodiscard]] std::vector<cucascade::io::byte_range> align_and_coalesce(
    std::span<cucascade::io::byte_range const> ranges,
    std::optional<std::size_t>) const noexcept override
  {
    return {ranges.begin(), ranges.end()};
  }

  std::size_t host_read_io(cucascade::io::io_object const&,
                           std::size_t,
                           std::size_t size,
                           std::uint8_t*) override
  {
    return size;
  }

  cucascade::exec::semi_future<std::size_t> mixed_readv_async_io(
    cucascade::io::io_object const&,
    std::vector<cucascade::io::prepared_io_slice>&& slices) noexcept override
  {
    std::size_t bytes = 0;
    for (auto const& slice : slices) {
      bytes += slice.size();
    }
    return cucascade::exec::make_semi_future<std::size_t>(bytes);
  }

 protected:
  std::shared_ptr<cucascade::io::io_object> create_io_object(std::string path) override
  {
    return std::make_shared<bulk_object>(std::move(path));
  }
};

/// A backend that prefers bulk IO -- the one thing `prefers_bulk_materialize` asks a source's
/// datasource.
using bulk_context = route_only_context<true>;

/// The same stand-in as @ref bulk_context, but a backend that does not prefer bulk IO (as a
/// local-file backend does not), so `materialize_parquet` without device spans takes the general
/// route over it.
using local_context = route_only_context<false>;

}  // namespace sirius::test
