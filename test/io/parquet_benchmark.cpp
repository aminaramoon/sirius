#include "io/prefetching_cache.hpp"
#include "io/sirius_datasource.hpp"
#include "io/uring/uring_ioctx.hpp"

#include <cudf/io/datasource.hpp>
#include <cudf/io/experimental/hybrid_scan.hpp>
#include <cudf/io/parquet.hpp>
#include <cudf/io/parquet_metadata.hpp>
#include <cudf/io/parquet_schema.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/table/table.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/traits.hpp>

#include <rmm/cuda_stream.hpp>
#include <rmm/mr/cuda_async_memory_resource.hpp>
#include <rmm/mr/per_device_resource.hpp>

#include <fcntl.h>
#include <spdlog/common.h>
#include <spdlog/spdlog.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

// 4 columns used by the classic TPC-H lineitem aggregations (Q1, Q6, …).
static const std::vector<std::string> COLUMNS = {
  "l_orderkey",
  "l_extendedprice",
  "l_discount",
  "l_shipdate",
};

enum class DataSource { cudf, uring };

static DataSource parse_source(std::string_view s)
{
  if (s == "cudf") return DataSource::cudf;
  if (s == "uring") return DataSource::uring;
  throw std::invalid_argument(std::string("unknown datasource: ") + std::string(s) +
                              "  (expected: cudf | uring)");
}

static bool drop_caches()
{
  ::sync();
  int fd = ::open("/proc/sys/vm/drop_caches", O_WRONLY);
  if (fd < 0) return false;
  bool ok = ::write(fd, "3", 1) == 1;
  ::close(fd);
  return ok;
}

static void usage(char const* prog)
{
  std::cerr << "usage: " << prog << " <cudf|uring> <num_rows>\n"
            << "  cudf     – cudf default (mmap/pread)\n"
            << "  uring    – O_DIRECT io_uring + DMA to GPU\n"
            << "  num_rows – rows to read (0 = all)\n";
}

int main(int argc, char** argv)
{
  if (argc != 3) {
    usage(argv[0]);
    return 1;
  }

  DataSource source;
  try {
    source = parse_source(argv[1]);
  } catch (std::invalid_argument const& e) {
    std::cerr << e.what() << "\n";
    usage(argv[0]);
    return 1;
  }

  long long num_rows_arg = std::stoll(argv[2]);
  if (num_rows_arg < 0) {
    std::cerr << "num_rows must be >= 0\n";
    return 1;
  }
  size_t num_rows = static_cast<size_t>(num_rows_arg);  // 0 means all

  std::string path = "/home/aaramoon/Documents/tpch/sf100/parquet/lineitem_indexed.parquet";

  spdlog::set_level(spdlog::level::info);  // show per-read trace

  std::cout << "Source : " << argv[1] << "\n"
            << "Rows   : " << (num_rows == 0 ? "all" : std::to_string(num_rows)) << "\n"
            << "Columns: ";
  for (auto const& c : COLUMNS)
    std::cout << c << "  ";
  std::cout << "\n\n";

  bool can_drop = drop_caches();
  if (!can_drop) std::cout << "WARNING: cannot drop caches (run as root for cold results)\n\n";

  cudaFree(nullptr);

  rmm::mr::cuda_async_memory_resource async_mr;
  rmm::mr::set_current_device_resource(&async_mr);

  auto time_ms = [](auto fn) -> double {
    auto t0 = std::chrono::high_resolution_clock::now();
    fn();
    return std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - t0)
      .count();
  };

  // Approximate device-memory footprint of a materialized table: data buffer
  // for fixed-width columns + null-mask buffer where present.  Variable-width
  // columns (strings/lists) are not in our schema so we skip their children.
  auto table_bytes = [](cudf::table_view const& tv) -> size_t {
    size_t bytes = 0;
    for (cudf::size_type i = 0; i < tv.num_columns(); ++i) {
      auto const& col = tv.column(i);
      if (cudf::is_fixed_width(col.type()))
        bytes += static_cast<size_t>(col.size()) * cudf::size_of(col.type());
      if (col.nullable()) bytes += cudf::bitmask_allocation_size_bytes(col.size());
    }
    return bytes;
  };

  auto log_table = [&](cudf::io::table_with_metadata const& t) {
    auto bytes = table_bytes(t.tbl->view());
    std::cout << "Schema : " << t.tbl->num_columns() << " columns, " << t.tbl->num_rows()
              << " rows, " << std::fixed << std::setprecision(2)
              << static_cast<double>(bytes) / (1024.0 * 1024.0) << " MiB materialized\n\n";
  };
  (void)log_table;

  // Read Parquet footer metadata once using cudf's default datasource so that
  // neither the cudf nor the uring timed path pays the cost of a metadata scan.
  std::vector<uint8_t> footer_buf;
  {
    auto probe_sources = cudf::io::make_datasources(cudf::io::source_info{{path}});
    auto& probe_ds     = *probe_sources.front();
    auto file_size     = probe_ds.size();
    cudf::io::parquet::file_ender_s ender{};
    probe_ds.host_read(
      file_size - sizeof(ender), sizeof(ender), reinterpret_cast<uint8_t*>(&ender));
    footer_buf.resize(ender.footer_len);
    probe_ds.host_read(
      file_size - sizeof(ender) - ender.footer_len, ender.footer_len, footer_buf.data());
  }

  auto scan_opts = cudf::io::parquet_reader_options::builder().columns(COLUMNS).build();
  cudf::io::parquet::experimental::hybrid_scan_reader scanner{
    cudf::host_span<uint8_t const>{footer_buf.data(), footer_buf.size()}, scan_opts};
  auto file_metadata = scanner.parquet_metadata();

  // Walk row groups in order until we cover the requested row count, then use
  // hybrid_scan to ask which file byte ranges those row groups depend on for
  // the selected columns.
  std::vector<cudf::size_type> selected_row_groups;
  if (num_rows == 0) {
    selected_row_groups = scanner.all_row_groups(scan_opts);
  } else {
    int64_t accumulated = 0;
    for (cudf::size_type i = 0; i < static_cast<cudf::size_type>(file_metadata.row_groups.size()) &&
                                accumulated < static_cast<int64_t>(num_rows);
         ++i) {
      selected_row_groups.push_back(i);
      accumulated += file_metadata.row_groups[i].num_rows;
    }
  }
  auto byte_ranges = scanner.all_column_chunks_byte_ranges(selected_row_groups, scan_opts);

  // Coalesce adjacent / overlapping ranges so the allowed-range filter sees a
  // minimal, non-overlapping set.  Column chunks within a row group are stored
  // contiguously per column, so neighboring ranges frequently touch.
  {
    std::sort(byte_ranges.begin(), byte_ranges.end(), [](auto const& a, auto const& b) {
      return a.offset() < b.offset();
    });
    std::vector<cudf::io::text::byte_range_info> coalesced;
    coalesced.reserve(byte_ranges.size());
    for (auto const& br : byte_ranges) {
      if (br.size() <= 0) continue;
      if (!coalesced.empty()) {
        auto& back       = coalesced.back();
        int64_t back_end = back.offset() + back.size();
        if (br.offset() <= back_end) {
          int64_t br_end = br.offset() + br.size();
          if (br_end > back_end)
            back = cudf::io::text::byte_range_info{back.offset(), br_end - back.offset()};
          continue;
        }
      }
      coalesced.push_back(br);
    }
    byte_ranges = std::move(coalesced);
  }

  size_t total_range_bytes = 0;
  for (auto const& br : byte_ranges)
    total_range_bytes += static_cast<size_t>(br.size());
  std::cout << "Hybrid scan: " << selected_row_groups.size() << " row group(s), "
            << byte_ranges.size() << " byte range(s), " << std::fixed << std::setprecision(2)
            << static_cast<double>(total_range_bytes) / (1024.0 * 1024.0) << " MiB total\n\n";

  // Build read options (no source — provided separately as a datasource
  // vector).
  auto read_opts_builder = cudf::io::parquet_reader_options::builder().columns(COLUMNS);
  if (num_rows > 0) read_opts_builder.num_rows(num_rows);
  auto read_opts = read_opts_builder.build();

  // cudaMemcpyBatchAsync rejects the legacy / per-thread default streams with
  // cudaMemLocation hints — pass an explicit stream so all H2D copies inside
  // the datasource and cudf share the same one.
  rmm::cuda_stream stream;

  // Timed run.
  double ms = 0;
  if (source == DataSource::cudf) {
    auto sources = cudf::io::make_datasources(cudf::io::source_info{{path}});
    ms           = time_ms([&] {
      auto tbl =
        cudf::io::read_parquet(std::move(sources), {file_metadata}, read_opts, stream.view());
    });
  } else {
    // Size the buffer pool to fit the working set, plus headroom.
    constexpr uint32_t POOL_MAX_SLABS       = 20;
    constexpr size_t INFLIGHT_BUDGET_CHUNKS = 2048;
    sirius::io::buffer_pool pool(POOL_MAX_SLABS);

    auto io_ctx = std::make_shared<sirius::io::uring_ioctx>();
    // io_ctx->initialize_cache(pool, INFLIGHT_BUDGET_CHUNKS);
    auto io_obj = io_ctx->create_io_object(path);
    auto ds     = std::make_unique<sirius::io::sirius_datasource>(io_ctx, io_obj);
    // io_ctx->cache()->insert(*io_obj, /*metadata=*/nullptr, {});
    std::this_thread::sleep_for(std::chrono::milliseconds(1200));

    std::vector<std::unique_ptr<cudf::io::datasource>> sources;
    sources.push_back(std::move(ds));

    ms = time_ms([&] {
      auto tbl =
        cudf::io::read_parquet(std::move(sources), {file_metadata}, read_opts, stream.view());
    });

    // std::cout << "cache summary : " << io_ctx->cache()->summary() << std::endl;
  }

  std::cout << std::fixed << std::setprecision(1) << ms << " ms\n";
  return 0;
}
