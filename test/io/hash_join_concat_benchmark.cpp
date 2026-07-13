// hash_join_concat_benchmark
// -----------------------------------------------------------------------------
// Measures the overhead of concatenating the probe side of a hash join, modeled
// on TPC-H Q9's orders |X| lineitem join (o_orderkey = l_orderkey).
//
//   * Build side  : `orders`   -> a single cudf::hash_join, exactly the way
//                                 sirius_physical_hash_join builds it in
//                                 BUILD_PROBE mode
//                                 (cudf::hash_join(keys, null_equality::UNEQUAL)).
//   * Probe side  : `lineitem`, read in ~100 MiB chunks and replicated
//                   `num_passes` times (default 10) so we probe "10x lineitem".
//
// Three strategies are timed over the same resident probe batches:
//
//   1. CONCAT       : concatenate all probe batches into one table, then a
//                     single inner_join + gather.
//   2. LOOP_1STREAM : inner_join + gather per batch on one stream, producing one
//                     output batch per probe batch.
//   3. STREAMS_N    : same per-batch joins, round-robined across N streams
//                     (default 5), synchronized back to the main stream at the
//                     end via CUDA events.
//
// The build hash table is constructed once and reused across all strategies and
// iterations. Only the probe-time work (concat / join / gather) is timed; I/O
// and hash-table construction are excluded.
// -----------------------------------------------------------------------------

#include <cudf/column/column_view.hpp>
#include <cudf/concatenate.hpp>
#include <cudf/copying.hpp>
#include <cudf/io/datasource.hpp>
#include <cudf/io/parquet.hpp>
#include <cudf/join/hash_join.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/traits.hpp>

#include <rmm/cuda_stream.hpp>
#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/mr/cuda_async_memory_resource.hpp>
#include <rmm/mr/per_device_resource.hpp>

#include <cuda_runtime.h>
#include <glob.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

// Columns read from each table. Column 0 is always the join key; the remaining
// columns are the Q9 payload gathered out of the join.
const std::vector<std::string> ORDERS_COLUMNS   = {"o_orderkey", "o_orderdate"};
const std::vector<std::string> LINEITEM_COLUMNS = {
  "l_orderkey", "l_partkey", "l_suppkey", "l_quantity", "l_extendedprice", "l_discount"};

// Expand a path or shell glob to a sorted list of file paths.
std::vector<std::string> expand_paths(std::string const& spec)
{
  if (spec.find('*') == std::string::npos) return {spec};
  std::vector<std::string> paths;
  glob_t g{};
  if (::glob(spec.c_str(), GLOB_TILDE, nullptr, &g) == 0) {
    paths.reserve(g.gl_pathc);
    for (size_t i = 0; i < g.gl_pathc; ++i)
      paths.emplace_back(g.gl_pathv[i]);
  }
  ::globfree(&g);
  std::sort(paths.begin(), paths.end());
  return paths;
}

// Approximate device footprint of a materialized fixed-width table.
size_t table_bytes(cudf::table_view const& tv)
{
  size_t bytes = 0;
  for (cudf::size_type i = 0; i < tv.num_columns(); ++i) {
    auto const& col = tv.column(i);
    if (cudf::is_fixed_width(col.type()))
      bytes += static_cast<size_t>(col.size()) * cudf::size_of(col.type());
    if (col.nullable()) bytes += cudf::bitmask_allocation_size_bytes(col.size());
  }
  return bytes;
}

cudf::io::parquet_reader_options make_read_opts(std::vector<std::string> const& paths,
                                                std::vector<std::string> const& columns)
{
  return cudf::io::parquet_reader_options::builder(cudf::io::source_info{paths})
    .column_names(columns)
    .build();
}

// Read the whole file set into a single device table (build side).
std::unique_ptr<cudf::table> read_full(std::vector<std::string> const& paths,
                                       std::vector<std::string> const& columns,
                                       rmm::cuda_stream_view stream)
{
  auto opts = make_read_opts(paths, columns);
  return cudf::io::read_parquet(opts, stream, cudf::get_current_device_resource_ref()).tbl;
}

// Read the file set in ~chunk_bytes chunks and return each chunk as a resident
// device table.
std::vector<std::unique_ptr<cudf::table>> read_chunks(std::vector<std::string> const& paths,
                                                      std::vector<std::string> const& columns,
                                                      size_t chunk_bytes,
                                                      rmm::cuda_stream_view stream)
{
  auto opts = make_read_opts(paths, columns);
  cudf::io::chunked_parquet_reader reader(
    chunk_bytes, opts, stream, cudf::get_current_device_resource_ref());
  std::vector<std::unique_ptr<cudf::table>> chunks;
  while (reader.has_next()) {
    auto chunk = reader.read_chunk();
    if (chunk.tbl->num_rows() == 0) continue;
    chunks.push_back(std::move(chunk.tbl));
  }
  stream.synchronize();
  return chunks;
}

// Column indices gathered from each side after the join (everything but the key
// column). The gathered columns are what a real Q9 join would carry downstream.
std::vector<cudf::size_type> payload_indices(int num_columns)
{
  std::vector<cudf::size_type> idxs(num_columns - 1);
  std::iota(idxs.begin(), idxs.end(), 1);  // skip column 0 (the key)
  return idxs;
}

// Build a column_view over a device_uvector<size_type> so it can drive gather.
cudf::column_view as_gather_map(rmm::device_uvector<cudf::size_type> const& map)
{
  return cudf::column_view(cudf::data_type{cudf::type_id::INT32},
                           static_cast<cudf::size_type>(map.size()),
                           map.data(),
                           nullptr,
                           0);
}

// Probe `hj` with `probe` and materialize the inner-join output, mirroring the
// BUILD_PROBE path in sirius_physical_hash_join: inner_join -> gather both
// sides. Returns the number of output rows. All transient results are released
// when the returned table goes out of scope in the caller.
std::unique_ptr<cudf::table> probe_and_gather(cudf::hash_join const& hj,
                                              cudf::table_view const& build_full,
                                              std::vector<cudf::size_type> const& build_payload,
                                              cudf::table_view const& probe_full,
                                              std::vector<cudf::size_type> const& probe_payload,
                                              rmm::cuda_stream_view stream)
{
  auto probe_keys = probe_full.select({0});
  auto [left_idx, right_idx] =
    hj.inner_join(probe_keys, {}, stream, cudf::get_current_device_resource_ref());

  // Left/probe payload columns.
  auto left_cols = probe_full.select(probe_payload);
  auto left_out  = cudf::gather(left_cols,
                               as_gather_map(*left_idx),
                               cudf::out_of_bounds_policy::DONT_CHECK,
                               stream,
                               cudf::get_current_device_resource_ref());
  // Right/build payload columns.
  auto right_cols = build_full.select(build_payload);
  auto right_out  = cudf::gather(right_cols,
                                as_gather_map(*right_idx),
                                cudf::out_of_bounds_policy::DONT_CHECK,
                                stream,
                                cudf::get_current_device_resource_ref());

  std::vector<std::unique_ptr<cudf::column>> cols = left_out->release();
  for (auto& c : right_out->release())
    cols.push_back(std::move(c));
  return std::make_unique<cudf::table>(std::move(cols), stream);
}

struct timing {
  double min_ms = 0;
  double avg_ms = 0;
  int64_t rows  = 0;
};

template <typename Fn>
timing time_it(int iters, Fn&& fn)
{
  timing t;
  double total = 0;
  t.min_ms     = std::numeric_limits<double>::max();
  for (int i = 0; i < iters; ++i) {
    auto t0  = std::chrono::high_resolution_clock::now();
    t.rows   = fn();
    double ms = std::chrono::duration<double, std::milli>(
                  std::chrono::high_resolution_clock::now() - t0)
                  .count();
    t.min_ms = std::min(t.min_ms, ms);
    total += ms;
  }
  t.avg_ms = total / iters;
  return t;
}

void usage(char const* prog)
{
  std::cerr << "usage: " << prog
            << " <orders_path|glob> <lineitem_path|glob>"
               " [num_passes=10] [chunk_mb=100] [num_streams=5] [iters=3]\n";
}

}  // namespace

int main(int argc, char** argv)
{
  if (argc < 3 || argc > 7) {
    usage(argv[0]);
    return 1;
  }
  std::string orders_spec   = argv[1];
  std::string lineitem_spec = argv[2];
  int num_passes            = argc > 3 ? std::stoi(argv[3]) : 10;
  size_t chunk_mb           = argc > 4 ? std::stoull(argv[4]) : 100;
  int num_streams           = argc > 5 ? std::stoi(argv[5]) : 5;
  int iters                 = argc > 6 ? std::stoi(argv[6]) : 3;
  if (num_passes < 1 || num_streams < 1 || iters < 1) {
    std::cerr << "num_passes, num_streams and iters must be >= 1\n";
    return 1;
  }

  auto orders_paths   = expand_paths(orders_spec);
  auto lineitem_paths = expand_paths(lineitem_spec);
  if (orders_paths.empty() || lineitem_paths.empty()) {
    std::cerr << "no files matched for orders or lineitem\n";
    return 1;
  }

  cudaFree(nullptr);  // force-init the CUDA context before we time anything
  rmm::mr::cuda_async_memory_resource async_mr;
  // set_current_device_resource takes an owning any_resource by value; wrap a
  // non-owning ref to our stack resource (kept alive for the run).
  rmm::mr::set_current_device_resource(
    cuda::mr::any_resource<cuda::mr::device_accessible>{rmm::device_async_resource_ref{async_mr}});

  rmm::cuda_stream main_stream;

  // ---- Build side: orders -> hash table (built once, reused everywhere) ----
  auto orders = read_full(orders_paths, ORDERS_COLUMNS, main_stream.view());
  main_stream.synchronize();
  cudf::table_view orders_view = orders->view();
  auto orders_payload          = payload_indices(orders_view.num_columns());

  auto build_keys = orders_view.select({0});
  cudf::hash_join hj(build_keys, cudf::null_equality::UNEQUAL, main_stream.view());
  main_stream.synchronize();

  // ---- Probe side: lineitem in ~chunk_mb chunks, replicated num_passes times ----
  size_t chunk_bytes = chunk_mb * (1ull << 20);
  std::vector<std::unique_ptr<cudf::table>> probe_batches;
  for (int p = 0; p < num_passes; ++p) {
    auto chunks = read_chunks(lineitem_paths, LINEITEM_COLUMNS, chunk_bytes, main_stream.view());
    for (auto& c : chunks)
      probe_batches.push_back(std::move(c));
  }
  if (probe_batches.empty()) {
    std::cerr << "no probe batches were read from lineitem\n";
    return 1;
  }

  std::vector<cudf::table_view> probe_views;
  probe_views.reserve(probe_batches.size());
  size_t probe_rows = 0, probe_res_bytes = 0;
  for (auto const& b : probe_batches) {
    probe_views.push_back(b->view());
    probe_rows += b->num_rows();
    probe_res_bytes += table_bytes(b->view());
  }
  auto probe_payload = payload_indices(probe_views.front().num_columns());

  std::cout << std::fixed << std::setprecision(2);
  std::cout << "Build (orders)   : " << orders_view.num_rows() << " rows, "
            << table_bytes(orders_view) / (1024.0 * 1024.0) << " MiB\n";
  std::cout << "Probe (lineitem) : " << probe_batches.size() << " batches, " << probe_rows
            << " rows, " << probe_res_bytes / (1024.0 * 1024.0) << " MiB resident ("
            << num_passes << "x, ~" << chunk_mb << " MiB/chunk)\n";
  std::cout << "Streams          : " << num_streams << ", iters: " << iters << "\n\n";

  // ---- Extra streams + events for strategy 3 ----
  std::vector<rmm::cuda_stream> streams(num_streams);
  std::vector<cudaEvent_t> events(num_streams);
  for (auto& e : events)
    cudaEventCreateWithFlags(&e, cudaEventDisableTiming);

  // Warm-up (not timed): one join on the first batch so lazy allocations and
  // any first-touch costs don't land in the measured strategies.
  {
    auto t = probe_and_gather(
      hj, orders_view, orders_payload, probe_views.front(), probe_payload, main_stream.view());
    main_stream.synchronize();
  }

  // ---- Strategy 1: CONCAT then single join ----
  auto concat_t = time_it(iters, [&]() -> int64_t {
    auto big   = cudf::concatenate(probe_views, main_stream.view());
    auto out   = probe_and_gather(
      hj, orders_view, orders_payload, big->view(), probe_payload, main_stream.view());
    int64_t n = out->num_rows();
    main_stream.synchronize();
    return n;
  });

  // ---- Strategy 2: per-batch join on a single stream ----
  auto loop_t = time_it(iters, [&]() -> int64_t {
    int64_t n = 0;
    std::vector<std::unique_ptr<cudf::table>> outs;
    outs.reserve(probe_views.size());
    for (auto const& pv : probe_views) {
      auto out = probe_and_gather(
        hj, orders_view, orders_payload, pv, probe_payload, main_stream.view());
      n += out->num_rows();
      outs.push_back(std::move(out));
    }
    main_stream.synchronize();
    return n;
  });

  // ---- Strategy 3: per-batch join round-robined across N streams ----
  auto streams_t = time_it(iters, [&]() -> int64_t {
    int64_t n = 0;
    std::vector<std::unique_ptr<cudf::table>> outs;
    outs.reserve(probe_views.size());
    for (size_t i = 0; i < probe_views.size(); ++i) {
      auto& s  = streams[i % num_streams];
      auto out = probe_and_gather(
        hj, orders_view, orders_payload, probe_views[i], probe_payload, s.view());
      n += out->num_rows();
      outs.push_back(std::move(out));
    }
    // Synchronize all worker streams back to the main stream via events, then
    // block on the main stream only.
    for (int s = 0; s < num_streams; ++s) {
      cudaEventRecord(events[s], streams[s].value());
      cudaStreamWaitEvent(main_stream.value(), events[s], 0);
    }
    main_stream.synchronize();
    return n;
  });

  for (auto& e : events)
    cudaEventDestroy(e);

  auto report = [](char const* name, timing const& t) {
    std::cout << std::left << std::setw(16) << name << " min " << std::right << std::setw(9)
              << t.min_ms << " ms   avg " << std::setw(9) << t.avg_ms << " ms   out_rows "
              << t.rows << "\n";
  };
  std::cout << "Results (lower is better):\n";
  report("1 CONCAT", concat_t);
  report("2 LOOP_1STREAM", loop_t);
  report("3 STREAMS_N", streams_t);

  if (!(concat_t.rows == loop_t.rows && loop_t.rows == streams_t.rows)) {
    std::cerr << "\nWARNING: output row counts differ across strategies "
              << "(concat=" << concat_t.rows << " loop=" << loop_t.rows
              << " streams=" << streams_t.rows << ")\n";
  }
  return 0;
}
