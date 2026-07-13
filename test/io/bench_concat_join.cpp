/*
 * Copyright 2025, Sirius Contributors.
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

// bench_concat_join.cpp
//
// Measures the overhead of probe-side concatenation vs. streaming joins for
// the orders ⋈ lineitem join from TPC-H Q9 (join key: o_orderkey = l_orderkey).
//
// Build side: orders (o_orderkey, o_orderdate) — read once, hash table built once.
// Probe side: N_BATCHES lineitem batches (~100 MB of decoded column data each).
//             Columns: l_orderkey, l_suppkey, l_partkey, l_extendedprice, l_discount, l_quantity
//
// Three scenarios (I/O is done up-front; only join work is timed):
//
//   BM1 — concat:     cudf::concatenate(N_BATCHES batches) then one inner_join
//   BM2 — sequential: inner_join each batch on one stream, one after another
//   BM3 — parallel:   N_PARALLEL_STREAMS extra streams, each handling
//                     N_BATCHES/N_PARALLEL_STREAMS batches; main stream waits
//                     on all completion events before the stop event is recorded
//
// Usage:
//   build/release/test/io/concat_join_benchmark <tpch_parquet_dir> [warmup=2] [iters=5]
//   e.g.: build/release/test/io/concat_join_benchmark /data/tpch/sf100/parquet 2 5
//
// Output rows are printed to prevent the compiler from eliding join work.
// Timings are GPU-event-based (CUDA event elapsed time, in ms).

#include <cudf/concatenate.hpp>
#include <cudf/copying.hpp>
#include <cudf/io/parquet.hpp>
#include <cudf/io/parquet_metadata.hpp>
#include <cudf/join/hash_join.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/cuda_stream.hpp>
#include <rmm/cuda_stream_view.hpp>
#include <rmm/mr/cuda_memory_resource.hpp>
#include <rmm/mr/per_device_resource.hpp>
#include <rmm/mr/pool_memory_resource.hpp>

#include <cuda_runtime.h>
#include <nvtx3/nvtx3.hpp>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <memory>
#include <numeric>
#include <string>
#include <vector>

namespace fs = std::filesystem;

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

static constexpr int N_BATCHES          = 10;
static constexpr int N_PARALLEL_STREAMS = 5;

static_assert(N_BATCHES % N_PARALLEL_STREAMS == 0,
              "N_BATCHES must be divisible by N_PARALLEL_STREAMS");

// Q9 probe columns (l_orderkey must be index 0 — used as the join key)
static const std::vector<std::string> LINEITEM_COLS = {
  "l_orderkey",
  "l_suppkey",
  "l_partkey",
  "l_extendedprice",
  "l_discount",
  "l_quantity",
};

// Q9 build columns (o_orderkey must be index 0 — used as the join key)
static const std::vector<std::string> ORDERS_COLS = {
  "o_orderkey",
  "o_orderdate",
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

// Estimated decoded bytes for fixed-width columns in a table view.
static size_t table_bytes(cudf::table_view const& tv)
{
  size_t b = 0;
  for (int i = 0; i < tv.num_columns(); ++i) {
    auto const& col = tv.column(i);
    if (cudf::is_fixed_width(col.type()))
      b += static_cast<size_t>(col.size()) * cudf::size_of(col.type());
  }
  return b;
}

// Read a single-file parquet with selected columns and a list of row groups.
static std::unique_ptr<cudf::table> read_parquet_rgs(const std::string& path,
                                                     const std::vector<std::string>& columns,
                                                     std::vector<cudf::size_type> rgs,
                                                     rmm::cuda_stream_view stream,
                                                     rmm::device_async_resource_ref const& mr)
{
  auto opts = cudf::io::parquet_reader_options::builder(cudf::io::source_info{path})
                .column_names(columns)
                .row_groups({std::move(rgs)})  // outer dim = per-file; one file here
                .build();
  auto result = cudf::io::read_parquet(opts, stream, mr);
  return std::move(result.tbl);
}

// Read an entire parquet file in `n_chunks` row-group groups that are then
// concatenated. Reading all row groups in one read_parquet call makes cudf
// allocate decode scratch for the whole file at once and can exhaust device
// memory on large files (e.g. orders SF100).
static std::unique_ptr<cudf::table> read_parquet_whole_chunked(
  const std::string& path,
  const std::vector<std::string>& columns,
  int total_rgs,
  int n_chunks,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref const& mr)
{
  std::vector<std::vector<cudf::size_type>> chunk_rgs(n_chunks);
  for (int rg = 0; rg < total_rgs; ++rg) {
    int c = std::min(rg * n_chunks / total_rgs, n_chunks - 1);
    chunk_rgs[c].push_back(static_cast<cudf::size_type>(rg));
  }

  std::vector<std::unique_ptr<cudf::table>> parts;
  parts.reserve(n_chunks);
  for (int c = 0; c < n_chunks; ++c) {
    if (chunk_rgs[c].empty()) continue;
    parts.push_back(read_parquet_rgs(path, columns, chunk_rgs[c], stream, mr));
    stream.synchronize();  // free this chunk's decode scratch before the next
  }

  std::vector<cudf::table_view> views;
  views.reserve(parts.size());
  for (auto& p : parts)
    views.push_back(p->view());
  return cudf::concatenate(views, stream, mr);
}

// ---------------------------------------------------------------------------
// GPU event timer
// ---------------------------------------------------------------------------

struct GpuTimer {
  cudaEvent_t start_{}, stop_{};

  GpuTimer()
  {
    cudaEventCreate(&start_);
    cudaEventCreate(&stop_);
  }
  ~GpuTimer()
  {
    cudaEventDestroy(start_);
    cudaEventDestroy(stop_);
  }

  void record_start(cudaStream_t s) { cudaEventRecord(start_, s); }
  void record_stop(cudaStream_t s) { cudaEventRecord(stop_, s); }

  // Blocks until stop is recorded; returns elapsed ms.
  float elapsed_ms()
  {
    cudaEventSynchronize(stop_);
    float ms = 0.f;
    cudaEventElapsedTime(&ms, start_, stop_);
    return ms;
  }
};

// ---------------------------------------------------------------------------
// Core join step: probe one batch, gather output, return row count.
// ---------------------------------------------------------------------------

static int64_t probe_and_gather(const cudf::hash_join& ht,
                                const cudf::table& build_table,
                                const cudf::table& probe_batch,
                                rmm::cuda_stream_view stream,
                                rmm::device_async_resource_ref const& mr)
{
  // Column 0 of each table is the join key (l_orderkey / o_orderkey).
  cudf::table_view probe_keys = probe_batch.view().select({0});

  auto [left_idx, right_idx] = ht.inner_join(probe_keys, std::nullopt, stream, mr);

  // Wrap device_uvector results as column_views for cudf::gather.
  auto as_col = [](const rmm::device_uvector<cudf::size_type>& v) {
    return cudf::column_view{cudf::data_type{cudf::type_id::INT32},
                             static_cast<cudf::size_type>(v.size()),
                             v.data(),
                             /*null_mask=*/nullptr,
                             /*null_count=*/0};
  };

  // Gather both sides to produce the full output (prevents elision of join work).
  auto left_out = cudf::gather(
    probe_batch.view(), as_col(*left_idx), cudf::out_of_bounds_policy::DONT_CHECK, stream, mr);
  auto right_out = cudf::gather(
    build_table.view(), as_col(*right_idx), cudf::out_of_bounds_policy::DONT_CHECK, stream, mr);
  return static_cast<int64_t>(left_out->num_rows());
}

// ---------------------------------------------------------------------------
// BM1: concatenate all batches, then one inner_join
// ---------------------------------------------------------------------------

static double bm1_concat_join(const cudf::hash_join& ht,
                              const cudf::table& build_table,
                              const std::vector<std::unique_ptr<cudf::table>>& batches,
                              rmm::cuda_stream_view stream,
                              rmm::device_async_resource_ref const& mr,
                              int64_t& out_rows)
{
  nvtx3::scoped_range r{"BM1:concat_join"};
  GpuTimer timer;
  timer.record_start(stream.value());

  std::vector<cudf::table_view> views;
  views.reserve(batches.size());
  for (auto& b : batches)
    views.push_back(b->view());

  auto combined = cudf::concatenate(views, stream, mr);
  out_rows      = probe_and_gather(ht, build_table, *combined, stream, mr);

  timer.record_stop(stream.value());
  return timer.elapsed_ms();
}

// ---------------------------------------------------------------------------
// BM2: sequential per-batch joins on a single stream
// ---------------------------------------------------------------------------

static double bm2_sequential(const cudf::hash_join& ht,
                             const cudf::table& build_table,
                             const std::vector<std::unique_ptr<cudf::table>>& batches,
                             rmm::cuda_stream_view stream,
                             rmm::device_async_resource_ref const& mr,
                             int64_t& out_rows)
{
  nvtx3::scoped_range r{"BM2:sequential"};
  GpuTimer timer;
  timer.record_start(stream.value());

  out_rows = 0;
  for (auto& b : batches)
    out_rows += probe_and_gather(ht, build_table, *b, stream, mr);

  timer.record_stop(stream.value());
  return timer.elapsed_ms();
}

// ---------------------------------------------------------------------------
// BM3: N_PARALLEL_STREAMS side streams, each handling N_BATCHES/N_PARALLEL_STREAMS batches.
//       Main stream waits on all completion events.
// ---------------------------------------------------------------------------

static double bm3_parallel_streams(const cudf::hash_join& ht,
                                   const cudf::table& build_table,
                                   const std::vector<std::unique_ptr<cudf::table>>& batches,
                                   rmm::cuda_stream_view main_stream,
                                   rmm::device_async_resource_ref const& mr,
                                   int64_t& out_rows)
{
  nvtx3::scoped_range r{"BM3:parallel_streams"};
  constexpr int batches_per_stream = N_BATCHES / N_PARALLEL_STREAMS;

  std::vector<rmm::cuda_stream> side_streams(N_PARALLEL_STREAMS);
  std::vector<cudaEvent_t> done_events(N_PARALLEL_STREAMS);
  for (auto& ev : done_events)
    cudaEventCreateWithFlags(&ev, cudaEventDisableTiming);

  std::vector<int64_t> per_stream_rows(N_PARALLEL_STREAMS, 0LL);

  GpuTimer timer;
  timer.record_start(main_stream.value());

  for (int s = 0; s < N_PARALLEL_STREAMS; ++s) {
    int first = s * batches_per_stream;
    for (int b = 0; b < batches_per_stream; ++b)
      per_stream_rows[s] +=
        probe_and_gather(ht, build_table, *batches[first + b], side_streams[s].view(), mr);
    cudaEventRecord(done_events[s], side_streams[s].value());
  }

  // Main stream waits for all side streams before the stop event.
  for (auto& ev : done_events)
    cudaStreamWaitEvent(main_stream.value(), ev, 0);

  timer.record_stop(main_stream.value());
  float ms = timer.elapsed_ms();  // synchronizes on stop event

  out_rows = std::accumulate(per_stream_rows.begin(), per_stream_rows.end(), 0LL);

  for (auto& ev : done_events)
    cudaEventDestroy(ev);

  return static_cast<double>(ms);
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

static void usage(const char* prog)
{
  std::cerr << "usage: " << prog << " <tpch_parquet_dir> [warmup=2] [iters=5]\n"
            << "  tpch_parquet_dir  — directory containing orders.parquet and lineitem.parquet\n"
            << "  warmup            — warmup iterations (default 2)\n"
            << "  iters             — timed iterations (default 5)\n";
}

int main(int argc, char** argv)
{
  if (argc < 2 || argc > 4) {
    usage(argv[0]);
    return 1;
  }

  fs::path dir = argv[1];
  int warmup   = argc >= 3 ? std::stoi(argv[2]) : 2;
  int iters    = argc >= 4 ? std::stoi(argv[3]) : 5;

  std::string orders_path   = (dir / "orders.parquet").string();
  std::string lineitem_path = (dir / "lineitem.parquet").string();

  for (auto& p : {orders_path, lineitem_path}) {
    if (!fs::exists(p)) {
      std::cerr << "file not found: " << p << "\n";
      return 1;
    }
  }

  // -------------------------------------------------------------------------
  // Initialize CUDA and RMM
  // -------------------------------------------------------------------------
  cudaFree(nullptr);  // force context init
  // A pool over cudaMalloc (cudaMallocAsync misbehaves on some driver/OS
  // configs — allocations fail well below the reported free memory). Explicit
  // 40 GiB max so pool growth is never pinned to a low default.
  rmm::mr::cuda_memory_resource cuda_mr;
  rmm::mr::pool_memory_resource pool_mr(
    cuda::mr::any_resource<cuda::mr::device_accessible>{cuda_mr},
    /*initial_pool_size=*/2ULL * 1024 * 1024 * 1024,
    /*maximum_pool_size=*/40ULL * 1024 * 1024 * 1024);
  rmm::mr::set_current_device_resource(
    cuda::mr::any_resource<cuda::mr::device_accessible>{rmm::device_async_resource_ref{pool_mr}});
  auto mr = cudf::get_current_device_resource_ref();

  rmm::cuda_stream main_stream_owner;
  auto main_stream = main_stream_owner.view();

  // -------------------------------------------------------------------------
  // 1. Read orders — build side
  // -------------------------------------------------------------------------
  std::cout << "Reading orders (build side)...\n";
  auto orders_meta   = cudf::io::read_parquet_metadata(cudf::io::source_info{orders_path});
  int orders_rgs_cnt = static_cast<int>(orders_meta.num_rowgroups());
  auto orders_table  = read_parquet_whole_chunked(
    orders_path, ORDERS_COLS, orders_rgs_cnt, /*n_chunks=*/16, main_stream, mr);
  main_stream.synchronize();

  std::cout << "  " << orders_table->num_rows() << " rows, " << std::fixed << std::setprecision(1)
            << static_cast<double>(table_bytes(orders_table->view())) / (1 << 20)
            << " MiB decoded\n";

  // -------------------------------------------------------------------------
  // 2. Build hash table on o_orderkey (column index 0)
  // -------------------------------------------------------------------------
  std::cout << "Building hash table on o_orderkey...\n";
  cudf::table_view build_keys = orders_table->view().select({0});
  cudf::hash_join ht(build_keys, cudf::null_equality::UNEQUAL, main_stream);
  main_stream.synchronize();
  std::cout << "  done.\n\n";

  // -------------------------------------------------------------------------
  // 3. Select N_BATCHES consecutive ~100 MB runs of lineitem row groups. Only
  //    the first (N_BATCHES × ~100 MB) of lineitem is read, NOT the whole file.
  // -------------------------------------------------------------------------
  const size_t PROBE_BYTES_PER_ROW = LINEITEM_COLS.size() * 8;  // all 8-byte cols
  const size_t TARGET_BATCH_BYTES  = 100ULL * 1024 * 1024;

  auto li_meta    = cudf::io::read_parquet_metadata(cudf::io::source_info{lineitem_path});
  int total_rgs   = static_cast<int>(li_meta.num_rowgroups());
  int64_t li_rows = li_meta.num_rows();
  double rows_pr  = static_cast<double>(li_rows) / total_rgs;  // rows per row group
  int rgs_per_batch =
    std::max(1, static_cast<int>(std::ceil((TARGET_BATCH_BYTES / PROBE_BYTES_PER_ROW) / rows_pr)));

  std::cout << "lineitem: " << li_rows << " rows, " << total_rgs << " row groups\n";

  if (total_rgs < N_BATCHES * rgs_per_batch) {
    std::cerr << "Error: lineitem has " << total_rgs << " row groups; need "
              << N_BATCHES * rgs_per_batch << " for " << N_BATCHES << " × ~"
              << TARGET_BATCH_BYTES / (1 << 20) << " MiB batches\n";
    return 1;
  }

  std::vector<std::vector<cudf::size_type>> batch_rgs(N_BATCHES);
  for (int b = 0; b < N_BATCHES; ++b)
    for (int j = 0; j < rgs_per_batch; ++j)
      batch_rgs[b].push_back(static_cast<cudf::size_type>(b * rgs_per_batch + j));

  // -------------------------------------------------------------------------
  // 4. Pre-read all lineitem batches into GPU memory
  // -------------------------------------------------------------------------
  std::cout << "Pre-reading " << N_BATCHES << " lineitem batches into GPU memory...\n";
  std::vector<std::unique_ptr<cudf::table>> li_batches;
  li_batches.reserve(N_BATCHES);

  for (int i = 0; i < N_BATCHES; ++i) {
    li_batches.push_back(
      read_parquet_rgs(lineitem_path, LINEITEM_COLS, batch_rgs[i], main_stream, mr));
    main_stream.synchronize();
    auto rows  = li_batches.back()->num_rows();
    auto bytes = table_bytes(li_batches.back()->view());
    std::cout << "  batch[" << i << "]: " << batch_rgs[i].size() << " row group(s), " << rows
              << " rows, " << std::fixed << std::setprecision(1)
              << static_cast<double>(bytes) / (1 << 20) << " MiB\n";
  }

  // -------------------------------------------------------------------------
  // 5. Benchmark helper
  // -------------------------------------------------------------------------
  auto run_bench = [&](const std::string& label, auto fn) {
    std::cout << "\n=== " << label << " ===\n";
    std::vector<double> times;
    times.reserve(warmup + iters);
    int64_t last_rows = 0;

    for (int i = 0; i < warmup + iters; ++i) {
      int64_t rows = 0;
      double ms    = fn(rows);
      if (i >= warmup) {
        times.push_back(ms);
        last_rows = rows;
      }
    }

    double sum = std::accumulate(times.begin(), times.end(), 0.0);
    double avg = sum / static_cast<double>(times.size());
    double mn  = *std::min_element(times.begin(), times.end());
    double mx  = *std::max_element(times.begin(), times.end());

    std::cout << "  output rows : " << last_rows << "\n"
              << "  avg=" << std::fixed << std::setprecision(2) << avg << " ms"
              << "  min=" << mn << " ms"
              << "  max=" << mx << " ms"
              << "  (" << iters << " iters, " << warmup << " warmup)\n";
  };

  // -------------------------------------------------------------------------
  // 6. Run benchmarks
  // -------------------------------------------------------------------------
  run_bench("BM1: concat(" + std::to_string(N_BATCHES) + " batches) → 1 join", [&](int64_t& rows) {
    return bm1_concat_join(ht, *orders_table, li_batches, main_stream, mr, rows);
  });

  run_bench("BM2: " + std::to_string(N_BATCHES) + " sequential joins (1 stream)",
            [&](int64_t& rows) {
              return bm2_sequential(ht, *orders_table, li_batches, main_stream, mr, rows);
            });

  run_bench("BM3: " + std::to_string(N_BATCHES) + " joins on " +
              std::to_string(N_PARALLEL_STREAMS) + " streams (" +
              std::to_string(N_BATCHES / N_PARALLEL_STREAMS) + " batches/stream)",
            [&](int64_t& rows) {
              return bm3_parallel_streams(ht, *orders_table, li_batches, main_stream, mr, rows);
            });

  return 0;
}
