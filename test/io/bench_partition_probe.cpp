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

// bench_partition_probe.cpp
//
// Measures the overhead of NOT partitioning the probe side of a partitioned
// hash join (orders ⋈ lineitem, TPC-H Q9, key o_orderkey = l_orderkey).
//
// Common setup (not timed): the build side (orders) is hash-partitioned into
// N partitions (cudf::hash_partition on o_orderkey, MURMUR3, seed 0 — same as
// Sirius' gpu_partition_impl), and one cudf::hash_join is built per partition.
//
// Two scenarios are then compared on the probe/join phase:
//
//   BM_A — unpartitioned probe:
//     The probe side is left unpartitioned. Because a probe row could match
//     ANY build partition, every probe batch must be probed against ALL N
//     build-partition hash tables → N passes over the probe data.
//
//   BM_B — partitioned probe + concat:
//     Each probe batch is hash-partitioned into N partitions with the SAME
//     hash/seed as the build side, so probe partition i can only match build
//     partition i. The same-partition slices from all batches are concatenated
//     (capped at CONCAT_CAP_BYTES ≈ 1 GB per chunk), then each chunk is probed
//     against its build-partition hash table → 1 pass. BM_B's timed region
//     includes the probe-side partition + concat cost (the tradeoff vs. BM_A).
//
// Both scenarios produce the same total output row count (each probe row
// matches exactly one build partition), which validates correctness.
//
// Usage:
//   build/release/test/io/partition_probe_benchmark <tpch_parquet_dir> \
//       [n_partitions=8] [n_probe_batches=10] [warmup=2] [iters=5]

#include <cudf/concatenate.hpp>
#include <cudf/copying.hpp>
#include <cudf/hashing.hpp>
#include <cudf/io/parquet.hpp>
#include <cudf/io/parquet_metadata.hpp>
#include <cudf/join/hash_join.hpp>
#include <cudf/partitioning.hpp>
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
#include <cmath>
#include <cstdint>
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

// Cap for the per-partition probe concat in BM_B (~1 GB). Same-partition slices
// are packed into chunks no larger than this, then each chunk is joined.
static constexpr size_t CONCAT_CAP_BYTES = 1024ULL * 1024 * 1024;

// Q9 probe columns — l_orderkey must be index 0 (join key).
static const std::vector<std::string> LINEITEM_COLS = {
  "l_orderkey",
  "l_suppkey",
  "l_partkey",
  "l_extendedprice",
  "l_discount",
  "l_quantity",
};
// Q9 build columns — o_orderkey must be index 0 (join key).
static const std::vector<std::string> ORDERS_COLS = {"o_orderkey", "o_orderdate"};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

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

static std::unique_ptr<cudf::table> read_parquet_rgs(const std::string& path,
                                                     const std::vector<std::string>& columns,
                                                     std::vector<cudf::size_type> rgs,
                                                     rmm::cuda_stream_view stream,
                                                     rmm::device_async_resource_ref const& mr)
{
  auto opts = cudf::io::parquet_reader_options::builder(cudf::io::source_info{path})
                .column_names(columns)
                .row_groups({std::move(rgs)})
                .build();
  return std::move(cudf::io::read_parquet(opts, stream, mr).tbl);
}

// Read an entire parquet file, but in `n_chunks` row-group groups that are then
// concatenated — reading all row groups in a single read_parquet call makes
// cudf allocate decode scratch for the whole file at once and can exhaust
// device memory on large files (e.g. orders SF100).
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
    stream.synchronize();  // bound peak: free this chunk's decode scratch before the next
  }

  std::vector<cudf::table_view> views;
  views.reserve(parts.size());
  for (auto& p : parts)
    views.push_back(p->view());
  return cudf::concatenate(views, stream, mr);
}

// Wrap a device_uvector<size_type> as an INT32 column_view for cudf::gather.
static cudf::column_view idx_col(const rmm::device_uvector<cudf::size_type>& v)
{
  return cudf::column_view{cudf::data_type{cudf::type_id::INT32},
                           static_cast<cudf::size_type>(v.size()),
                           v.data(),
                           /*null_mask=*/nullptr,
                           /*null_count=*/0};
}

// Split a hash_partition result (reordered table + offsets) into per-partition
// non-owning table_views over the reordered table.
static std::vector<cudf::table_view> slice_partitions(cudf::table_view const& reordered,
                                                      const std::vector<cudf::size_type>& offsets,
                                                      int n,
                                                      rmm::cuda_stream_view stream)
{
  std::vector<cudf::size_type> slice_idx;
  slice_idx.reserve(n * 2);
  for (int i = 0; i < n; ++i) {
    slice_idx.push_back(offsets[i]);
    slice_idx.push_back(i == n - 1 ? reordered.num_rows() : offsets[i + 1]);
  }
  return cudf::slice(reordered, slice_idx, stream);
}

// Probe `probe_keys` (column 0 of probe_view) against hash table `ht` whose
// build side is `build_view`; gather both sides and return output row count.
static int64_t probe_one(const cudf::hash_join& ht,
                         cudf::table_view const& build_view,
                         cudf::table_view const& probe_view,
                         rmm::cuda_stream_view stream,
                         rmm::device_async_resource_ref const& mr)
{
  if (probe_view.num_rows() == 0) return 0;
  cudf::table_view probe_keys = probe_view.select({0});
  auto [l_idx, r_idx]         = ht.inner_join(probe_keys, std::nullopt, stream, mr);
  auto l_out =
    cudf::gather(probe_view, idx_col(*l_idx), cudf::out_of_bounds_policy::DONT_CHECK, stream, mr);
  auto r_out =
    cudf::gather(build_view, idx_col(*r_idx), cudf::out_of_bounds_policy::DONT_CHECK, stream, mr);
  return static_cast<int64_t>(l_out->num_rows());
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
  float elapsed_ms()
  {
    cudaEventSynchronize(stop_);
    float ms = 0.f;
    cudaEventElapsedTime(&ms, start_, stop_);
    return ms;
  }
};

// ---------------------------------------------------------------------------
// BM_A: unpartitioned probe — probe every batch against all N hash tables.
// ---------------------------------------------------------------------------
static double bm_a_unpartitioned(const std::vector<std::unique_ptr<cudf::hash_join>>& hts,
                                 const std::vector<cudf::table_view>& build_parts,
                                 const std::vector<std::unique_ptr<cudf::table>>& probe_batches,
                                 rmm::cuda_stream_view stream,
                                 rmm::device_async_resource_ref const& mr,
                                 int64_t& out_rows)
{
  nvtx3::scoped_range r{"BM_A:unpartitioned_probe"};
  GpuTimer timer;
  timer.record_start(stream.value());

  out_rows = 0;
  int n    = static_cast<int>(hts.size());
  for (int p = 0; p < n; ++p)
    for (auto& b : probe_batches)
      out_rows += probe_one(*hts[p], build_parts[p], b->view(), stream, mr);

  timer.record_stop(stream.value());
  return timer.elapsed_ms();
}

// ---------------------------------------------------------------------------
// BM_B: partition each probe batch (same hash as build), concat same-partition
//       slices into ≤ CONCAT_CAP_BYTES chunks, probe each chunk against its
//       build-partition hash table.
// ---------------------------------------------------------------------------
static double bm_b_partitioned(const std::vector<std::unique_ptr<cudf::hash_join>>& hts,
                               const std::vector<cudf::table_view>& build_parts,
                               const std::vector<std::unique_ptr<cudf::table>>& probe_batches,
                               int n_partitions,
                               rmm::cuda_stream_view stream,
                               rmm::device_async_resource_ref const& mr,
                               int64_t& out_rows)
{
  nvtx3::scoped_range r{"BM_B:partitioned_probe"};
  GpuTimer timer;
  timer.record_start(stream.value());

  // 1. Hash-partition every probe batch. Keep the reordered tables alive and
  //    collect per-partition slice views (grouped by partition index).
  std::vector<std::unique_ptr<cudf::table>> reordered_batches;
  reordered_batches.reserve(probe_batches.size());
  std::vector<std::vector<cudf::table_view>> slices_by_part(n_partitions);

  for (auto& b : probe_batches) {
    auto [reordered, offsets] = cudf::hash_partition(b->view(),
                                                     /*columns_to_hash=*/{0},  // l_orderkey
                                                     n_partitions,
                                                     cudf::hash_id::HASH_MURMUR3,
                                                     cudf::DEFAULT_HASH_SEED,
                                                     stream,
                                                     mr);
    auto parts                = slice_partitions(reordered->view(), offsets, n_partitions, stream);
    for (int p = 0; p < n_partitions; ++p)
      slices_by_part[p].push_back(parts[p]);
    reordered_batches.push_back(std::move(reordered));
  }

  // 2. For each partition, pack slices into ≤ CONCAT_CAP_BYTES chunks, concat,
  //    and probe against that partition's build hash table.
  out_rows = 0;
  for (int p = 0; p < n_partitions; ++p) {
    std::vector<cudf::table_view> chunk;
    size_t chunk_bytes = 0;

    auto flush = [&]() {
      if (chunk.empty()) return;
      auto concatenated = cudf::concatenate(chunk, stream, mr);
      out_rows += probe_one(*hts[p], build_parts[p], concatenated->view(), stream, mr);
      chunk.clear();
      chunk_bytes = 0;
    };

    for (auto& slice : slices_by_part[p]) {
      size_t sb = table_bytes(slice);
      if (chunk_bytes + sb > CONCAT_CAP_BYTES && !chunk.empty()) flush();
      chunk.push_back(slice);
      chunk_bytes += sb;
    }
    flush();
  }

  timer.record_stop(stream.value());
  return timer.elapsed_ms();
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
static void usage(const char* prog)
{
  std::cerr << "usage: " << prog
            << " <tpch_parquet_dir> [n_partitions=8] [n_probe_batches=10] "
               "[warmup=2] [iters=5]\n"
            << "  tpch_parquet_dir — dir with orders.parquet and lineitem.parquet\n"
            << "  n_partitions     — hash partitions for build & probe (default 8)\n"
            << "  n_probe_batches  — number of ~100 MB lineitem batches (default 10)\n";
}

int main(int argc, char** argv)
{
  if (argc < 2 || argc > 6) {
    usage(argv[0]);
    return 1;
  }
  fs::path dir      = argv[1];
  int n_partitions  = argc >= 3 ? std::stoi(argv[2]) : 8;
  int n_probe_batch = argc >= 4 ? std::stoi(argv[3]) : 10;
  int warmup        = argc >= 5 ? std::stoi(argv[4]) : 2;
  int iters         = argc >= 6 ? std::stoi(argv[5]) : 5;

  if (n_partitions < 2) {
    std::cerr << "n_partitions must be >= 2\n";
    return 1;
  }

  std::string orders_path   = (dir / "orders.parquet").string();
  std::string lineitem_path = (dir / "lineitem.parquet").string();
  for (auto& p : {orders_path, lineitem_path}) {
    if (!fs::exists(p)) {
      std::cerr << "file not found: " << p << "\n";
      return 1;
    }
  }

  cudaFree(nullptr);
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
  auto stream = main_stream_owner.view();

  // -------------------------------------------------------------------------
  // Build side: read all of orders, hash-partition on o_orderkey, build a
  // hash table per partition.
  // -------------------------------------------------------------------------
  std::cout << "Reading orders (build side)...\n";
  auto orders_meta   = cudf::io::read_parquet_metadata(cudf::io::source_info{orders_path});
  int orders_rgs_cnt = static_cast<int>(orders_meta.num_rowgroups());
  auto orders_table  = read_parquet_whole_chunked(
    orders_path, ORDERS_COLS, orders_rgs_cnt, /*n_chunks=*/16, stream, mr);
  stream.synchronize();
  std::cout << "  " << orders_table->num_rows() << " rows, " << std::fixed << std::setprecision(1)
            << static_cast<double>(table_bytes(orders_table->view())) / (1 << 20)
            << " MiB decoded\n";

  std::cout << "Hash-partitioning build side into " << n_partitions << " partitions...\n";
  auto [build_reordered, build_offsets] = cudf::hash_partition(orders_table->view(),
                                                               /*columns_to_hash=*/{0},
                                                               n_partitions,
                                                               cudf::hash_id::HASH_MURMUR3,
                                                               cudf::DEFAULT_HASH_SEED,
                                                               stream,
                                                               mr);
  auto build_parts = slice_partitions(build_reordered->view(), build_offsets, n_partitions, stream);
  stream.synchronize();

  std::cout << "Building " << n_partitions << " hash tables...\n";
  std::vector<std::unique_ptr<cudf::hash_join>> hts;
  hts.reserve(n_partitions);
  for (int p = 0; p < n_partitions; ++p) {
    cudf::table_view keys = build_parts[p].select({0});
    hts.push_back(std::make_unique<cudf::hash_join>(keys, cudf::null_equality::UNEQUAL, stream));
    std::cout << "  partition[" << p << "]: " << build_parts[p].num_rows() << " build rows\n";
  }
  stream.synchronize();

  // -------------------------------------------------------------------------
  // Probe side: read N ~100 MB lineitem batches. Each batch is a consecutive
  // run of row groups sized to ≈ TARGET_BATCH_BYTES — only the first
  // (n_probe_batch × ~100 MB) of lineitem is read, NOT the whole 22 GB file.
  // -------------------------------------------------------------------------
  const size_t PROBE_BYTES_PER_ROW = LINEITEM_COLS.size() * 8;  // all 8-byte cols
  const size_t TARGET_BATCH_BYTES  = 100ULL * 1024 * 1024;

  auto li_meta   = cudf::io::read_parquet_metadata(cudf::io::source_info{lineitem_path});
  int total_rgs  = static_cast<int>(li_meta.num_rowgroups());
  double rows_pr = static_cast<double>(li_meta.num_rows()) / total_rgs;  // rows per row group
  int rgs_per_batch =
    std::max(1, static_cast<int>(std::ceil((TARGET_BATCH_BYTES / PROBE_BYTES_PER_ROW) / rows_pr)));

  if (total_rgs < n_probe_batch * rgs_per_batch) {
    std::cerr << "Error: lineitem has " << total_rgs << " row groups; need "
              << n_probe_batch * rgs_per_batch << " for " << n_probe_batch << " × ~"
              << TARGET_BATCH_BYTES / (1 << 20) << " MiB batches\n";
    return 1;
  }

  std::cout << "\nReading " << n_probe_batch << " lineitem probe batches (~"
            << TARGET_BATCH_BYTES / (1 << 20) << " MiB, " << rgs_per_batch
            << " row groups each)...\n";
  std::vector<std::vector<cudf::size_type>> batch_rgs(n_probe_batch);
  for (int b = 0; b < n_probe_batch; ++b)
    for (int j = 0; j < rgs_per_batch; ++j)
      batch_rgs[b].push_back(static_cast<cudf::size_type>(b * rgs_per_batch + j));

  std::vector<std::unique_ptr<cudf::table>> probe_batches;
  probe_batches.reserve(n_probe_batch);
  size_t total_probe_bytes = 0;
  for (int i = 0; i < n_probe_batch; ++i) {
    probe_batches.push_back(
      read_parquet_rgs(lineitem_path, LINEITEM_COLS, batch_rgs[i], stream, mr));
    stream.synchronize();
    size_t bytes = table_bytes(probe_batches.back()->view());
    total_probe_bytes += bytes;
    std::cout << "  batch[" << i << "]: " << probe_batches.back()->num_rows() << " rows, "
              << std::fixed << std::setprecision(1) << static_cast<double>(bytes) / (1 << 20)
              << " MiB\n";
  }
  std::cout << "  total probe: " << std::fixed << std::setprecision(1)
            << static_cast<double>(total_probe_bytes) / (1 << 20) << " MiB\n";

  // -------------------------------------------------------------------------
  // Benchmark runner
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
    double avg =
      std::accumulate(times.begin(), times.end(), 0.0) / static_cast<double>(times.size());
    double mn = *std::min_element(times.begin(), times.end());
    double mx = *std::max_element(times.begin(), times.end());
    std::cout << "  output rows : " << last_rows << "\n"
              << "  avg=" << std::fixed << std::setprecision(2) << avg << " ms  min=" << mn
              << " ms  max=" << mx << " ms  (" << iters << " iters, " << warmup << " warmup)\n";
  };

  run_bench("BM_A: unpartitioned probe → probe all " + std::to_string(n_partitions) +
              " hash tables (" + std::to_string(n_partitions) + " passes)",
            [&](int64_t& rows) {
              return bm_a_unpartitioned(hts, build_parts, probe_batches, stream, mr, rows);
            });

  run_bench(
    "BM_B: partition probe + concat(≤1GB) → probe matching partition (1 pass)", [&](int64_t& rows) {
      return bm_b_partitioned(hts, build_parts, probe_batches, n_partitions, stream, mr, rows);
    });

  return 0;
}
