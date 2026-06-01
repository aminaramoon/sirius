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

#include "scan_manager/sirius_scan_manager.hpp"

#include "exec/thread_pool.hpp"
#include "io/cache/prefetching_cache.hpp"
#include "io/kvikio/kvikio_context.hpp"
#include "io/uring/uring_ioctx.hpp"
#include "log/logging.hpp"
#include "op/scan/parquet_scan_info.hpp"
#include "op/scan/scan_plan.hpp"
#include "op/scan/scan_utils.hpp"
#include "op/scan/sirius_gpu_parquet_scan_operator.hpp"
#include "op/sirius_physical_operator_type.hpp"
#include "pipeline/sirius_pipeline.hpp"
#include "planner/query.hpp"
#include "scan_manager/cached_split_provider.hpp"
#include "scan_manager/parquet_split_provider.hpp"
#include "scan_manager/pipeline_ordered_prefetching_manager.hpp"
#include "scan_manager/split_connector.hpp"
#include "scan_manager/split_provider.hpp"

#include <cucascade/memory/fixed_size_host_memory_resource.hpp>
#include <cucascade/memory/memory_reservation_manager.hpp>
#include <cucascade/memory/memory_space.hpp>

#include <algorithm>
#include <exception>
#include <stdexcept>
#include <utility>

namespace sirius::scan_manager {

sirius_scan_manager::sirius_scan_manager(
  scan_manager_config config, cucascade::memory::memory_reservation_manager& reservation_manager)
  : _config(std::move(config)),
    _thread_pool(_config.thread_pool.num_threads,
                 _config.thread_pool.thread_name_prefix,
                 _config.thread_pool.cpu_affinity_list),
    _dispatcher(
      std::make_unique<exec::scoped_dispatcher>(_thread_pool, _config.thread_pool.num_threads))
{
  // Pull the host memory resource out of the reservation manager.  Both
  // the uring backend's bounce-slot pool and the prefetching_cache's
  // buffer pool need a fixed_size_host_memory_resource; we use the
  // first HOST-tier space's resource for both.  Null is OK on the
  // kvikio path — the cache falls back to metadata-only construction.
  cucascade::memory::fixed_size_host_memory_resource* host_mr = nullptr;
  {
    auto host_spaces =
      reservation_manager.get_memory_spaces_for_tier(cucascade::memory::Tier::HOST);
    if (!host_spaces.empty()) {
      // get_memory_spaces_for_tier returns const memory_space*, but
      // get_memory_resource_as is const so the cast through the public
      // accessor is fine.
      host_mr = host_spaces[0]
                  ->get_memory_resource_as<cucascade::memory::fixed_size_host_memory_resource>();
    }
  }

  // scan_manager always owns an io_ctx: sirius_datasource (uring) on the
  // fast path, kvikio_context as the universal fallback so the rest of the
  // scan path (parquet_split_provider, scan tasks) always has an ioctx to
  // talk to.  kvikio_context wraps cudf::io::datasource so the read path
  // is identical from the caller's point of view.
  if (_config.use_sirius_datasource) {
    if (host_mr == nullptr) {
      throw std::runtime_error(
        "[sirius_scan_manager] use_sirius_datasource is true but the reservation "
        "manager has no HOST-tier fixed_size_host_memory_resource");
    }
    _io_ctx = std::make_shared<sirius::io::uring::uring_ioctx>(_config.uring_n_reactors, *host_mr);
    SIRIUS_LOG_DEBUG("[sirius_scan_manager] sirius_datasource enabled (uring_ioctx n_reactors={})",
                     _config.uring_n_reactors);
  } else {
    // kvikio's cudf datasource doesn't model per-device file handles, so
    // it only works correctly when there's a single active GPU.  Fail
    // fast on multi-GPU rather than silently mis-routing IO.
    auto gpu_spaces = reservation_manager.get_memory_spaces_for_tier(cucascade::memory::Tier::GPU);
    if (gpu_spaces.size() > 1) {
      throw std::runtime_error(
        "[sirius_scan_manager] kvikio_context fallback (use_sirius_datasource=false) "
        "does not support multi-GPU; reservation_manager reports " +
        std::to_string(gpu_spaces.size()) +
        " GPU memory spaces.  Enable use_sirius_datasource for multi-GPU runs.");
    }
    _io_ctx = std::make_shared<sirius::io::kvikio_context>();
    SIRIUS_LOG_DEBUG(
      "[sirius_scan_manager] sirius_datasource disabled — using kvikio_context fallback");
  }

  // The scan_manager owns the buffer pool and the cache.  The cache
  // takes both as construction-time dependencies and is single-life: it
  // arms itself iff (pool != nullptr && io_ctx->supports_vector_host_read()
  // && budget > 0); otherwise it's a metadata-only store.
  if (host_mr != nullptr) {
    auto const slab_bytes =
      host_mr->get_block_size() *
      static_cast<std::size_t>(sirius::io::cache::buffer_pool::CHUNKS_PER_SLAB);
    auto const max_slabs =
      static_cast<uint32_t>((_config.prefetch_buffer_pool_bytes + slab_bytes - 1) / slab_bytes);
    _buffer_pool = std::make_unique<sirius::io::cache::buffer_pool>(
      *host_mr, max_slabs, /*initial_slabs=*/max_slabs);
  }

  // Build the prefetching cache on the ioctx.  Budget=0 keeps the
  // cache unarmed (no background threads); we pass that whenever the
  // user has disabled prefetching so the construction is always
  // unconditional and there's no "is the cache present" branch to
  // worry about in callers.
  size_t const budget = _config.enable_prefetch_cache ? _config.prefetch_inflight_budget_chunks : 0;
  _io_ctx->initialize_cache(_buffer_pool.get(), budget);

  if (_io_ctx->cache() && _io_ctx->cache()->is_armed()) {
    SIRIUS_LOG_DEBUG("[sirius_scan_manager] prefetch cache armed (inflight_chunks={})",
                     _config.prefetch_inflight_budget_chunks);
  } else {
    SIRIUS_LOG_DEBUG("[sirius_scan_manager] prefetch cache unarmed");
  }
}

sirius_scan_manager::~sirius_scan_manager()
{
  if (_io_ctx && _io_ctx->cache()) {
    SIRIUS_LOG_INFO("[sirius_scan_manager] cache summary: {}", _io_ctx->cache()->summary());
  }
  // Drain the dispatcher (and the worker pool) first so no in-flight
  // metadata-scan / sequencer task can still be reaching into the
  // cache via _io_ctx when we tear it down below.
  stop();
  // Tear down the cache before releasing the buffer_pool — the cache's
  // worker holds a raw _pool pointer, and its destructor drains
  // in-flight IO before returning, so the pool stays alive long
  // enough for callbacks to release their chunks safely.
  if (_io_ctx) { _io_ctx->shutdown_cache(); }
  _buffer_pool.reset();
}

void sirius_scan_manager::prepare_for_query(const sirius::planner::query& query)
{
  reset();

  if (_io_ctx && _io_ctx->cache()) {
    SIRIUS_LOG_INFO("[sirius_scan_manager] cache summary: {}", _io_ctx->cache()->summary());
  }

  // Advance the cache age so the evictor can score this query's inserts
  // against entries left over from prior queries.
  // refresh_cache() removed — the cache no longer has a query-epoch
  // notion.  Per-file aging in the eviction queue handles staleness
  // without scan_manager involvement.

  SIRIUS_LOG_DEBUG("[sirius_scan_manager::prepare_for_query] scan_operators={}",
                   query.get_scan_operators().size());

  // Build a fresh sequencer for this query.  Slots are added by
  // create_provider_for() whenever it chooses the parquet path; the
  // cached path skips slot allocation since there's no IO to prefetch.
  // The sequencer task piggy-backs on the dispatcher's injected
  // stop_token, so reset()/request_stop on the dispatcher tears it
  // down without a side-channel stop_source.
  _prefetch_manager = std::make_unique<pipeline_ordered_prefetching_manager>();

  for (auto const& scan_op : query.get_scan_operators()) {
    if (scan_op->type != ::sirius::op::SiriusPhysicalOperatorType::GPU_PARQUET_SCAN) { continue; }

    auto* op = &scan_op->Cast<op::scan::sirius_gpu_parquet_scan_operator>();
    if (_providers_by_op.find(op) != _providers_by_op.end()) { continue; }

    auto provider = create_provider_for(op);
    if (!provider) {
      // No scan_info parked on the operator (e.g. tests construct the operator
      // directly). Skip — caller is responsible for the connector.
      continue;
    }
    op->set_split_connector(std::make_unique<split_connector>());
    _providers_by_op.emplace(op, std::move(provider));
    _scan_op_order.push_back(op);

    SIRIUS_LOG_DEBUG("[sirius_scan_manager::prepare_for_query] registered op_id={}",
                     op->get_operator_id());
  }

  if (_scan_op_order.empty()) { return; }

  // Spawn the sequencer task on the dispatcher.  Must happen after all
  // slots have been added (create_provider_for allocates them above) so
  // the task captures the full slot list in insertion order.  Safe to
  // skip when no slots were added (purely-cached query) — the manager
  // is fine with zero slots.
  _prefetch_manager->register_ranges(*_dispatcher);

  start_metadata_processing();
}

std::unique_ptr<split_provider> sirius_scan_manager::create_provider_for(
  op::scan::sirius_gpu_parquet_scan_operator* op)
{
  auto info = op->take_scan_info();
  if (!info) { return nullptr; }

  // If a pinned entry's file paths match this operator's scan_info, build the same
  // scan_plan the parquet path would build and serve the scan from cache.
  auto matches_scan_info = [&info](const pinned_entry& entry) {
    if (entry.file_paths.size() != info->file_paths.size()) { return false; }
    auto sorted_a = entry.file_paths;
    auto sorted_b = info->file_paths;
    std::sort(sorted_a.begin(), sorted_a.end());
    std::sort(sorted_b.begin(), sorted_b.end());
    return sorted_a == sorted_b;
  };
  try {
    for (auto const& [pinned_name, entry] : _pinned_entries) {
      if (!matches_scan_info(entry)) { continue; }
      if (entry.memory_space == nullptr) {
        throw std::runtime_error("[sirius_scan_manager::create_provider_for] pinned entry '" +
                                 pinned_name + "' has no memory_space");
      }

      // Build the canonical scan_plan once. Everything downstream — cached column
      // layout, filter pushdown indices, post-read assembly — reads from this.
      // Held by shared_ptr<const> so each emitted scan_cached_operator_data can
      // carry it to the GPU scan operator's per-task assembly check without copying.
      auto plan_shared = std::make_shared<op::scan::scan_plan const>(
        op::scan::build_scan_plan(info->column_ids,
                                  info->projection_ids,
                                  info->names,
                                  info->returned_types,
                                  op->get_types().size(),
                                  info->partition_indices));
      auto const& plan = *plan_shared;

      // Hive partitions on a cached scan would require per-chunk file_path metadata
      // that pinned entries don't carry today. Fall through to the parquet path,
      // which extracts partition values per file at read time.
      if (plan.has_partitions()) {
        SIRIUS_LOG_DEBUG(
          "[sirius_scan_manager::create_provider_for] pinned entry '{}' matches op_id={} but "
          "scan has hive partitions; falling through to parquet_split_provider",
          pinned_name,
          op->get_operator_id());
        break;
      }

      // Filter expression: BoundReferences are in D-space, via plan.batch_position_by_column_id.
      // Same recipe parquet_split_provider uses, so the filter evaluates correctly against
      // the cached batch (which is in D-order by construction above). Built before the
      // tier-specific assembly so both branches share the same filter.
      std::shared_ptr<duckdb::Expression> filter_expression;
      if (info->table_filters && !info->table_filters->filters.empty()) {
        auto duckdb_expression =
          op::convert_table_filters_to_expression(*info->table_filters,
                                                  info->column_ids,
                                                  info->returned_types,
                                                  plan.batch_position_by_column_id,
                                                  plan.partition_primary_indices);
        if (duckdb_expression) {
          filter_expression = std::shared_ptr<duckdb::Expression>(std::move(duckdb_expression));
        }
      }

      if (entry.tier == cucascade::memory::Tier::HOST) {
        // Map each D-position to its index inside the captured host chunk. column_names
        // is in capture order, so we look up the requested data column by name. A missing
        // column means the user pinned a subset that doesn't cover this scan — fall back
        // to the parquet path so the query still succeeds.
        std::vector<std::size_t> column_indices;
        column_indices.reserve(plan.data_columns.size());
        for (auto const& dc : plan.data_columns) {
          auto it = std::find(entry.column_names.begin(), entry.column_names.end(), dc.name);
          if (it == entry.column_names.end()) {
            throw std::runtime_error("[sirius_scan_manager::create_provider_for] pinned entry '" +
                                     pinned_name + "' missing column '" + dc.name +
                                     "' required by scan op");
          }
          column_indices.push_back(
            static_cast<std::size_t>(std::distance(entry.column_names.begin(), it)));
        }

        SIRIUS_LOG_DEBUG(
          "[sirius_scan_manager::create_provider_for] using host cached_split_provider for "
          "op_id={} (pinned='{}' data_cols={} chunks={} needs_assembly={})",
          op->get_operator_id(),
          pinned_name,
          column_indices.size(),
          entry.host_chunks.size(),
          op::scan::needs_output_assembly(plan));

        return std::make_unique<cached_split_provider>(entry.host_chunks,
                                                       std::move(column_indices),
                                                       *entry.memory_space,
                                                       std::move(filter_expression),
                                                       std::move(plan_shared));
      }

      // Look up the pinned chunks for each D-position by name. data_columns is in
      // D-order, so columns_per_request[d] is the chunk vector for D-position d.
      std::vector<std::vector<std::shared_ptr<cudf::column>>> columns_per_request;
      columns_per_request.reserve(plan.data_columns.size());
      for (auto const& dc : plan.data_columns) {
        auto it = entry.data_batches_by_column.find(dc.name);
        if (it == entry.data_batches_by_column.end()) {
          throw std::runtime_error("[sirius_scan_manager::create_provider_for] pinned entry '" +
                                   pinned_name + "' missing column '" + dc.name +
                                   "' required by scan op");
        }
        columns_per_request.push_back(it->second);
      }

      SIRIUS_LOG_DEBUG(
        "[sirius_scan_manager::create_provider_for] using cached_split_provider for op_id={} "
        "(pinned='{}' data_cols={} needs_assembly={})",
        op->get_operator_id(),
        pinned_name,
        columns_per_request.size(),
        op::scan::needs_output_assembly(plan));

      return std::make_unique<cached_split_provider>(std::move(columns_per_request),
                                                     *entry.memory_space,
                                                     std::move(filter_expression),
                                                     std::move(plan_shared));
    }
  } catch (...) {
    SIRIUS_LOG_TRACE("not all the columns are pinned for this query");
  }
  auto const pipeline    = op->get_pipeline();
  auto const pipeline_id = pipeline ? pipeline->get_pipeline_id() : std::size_t{0};
  // Allocate a sequencer slot for this pipeline so its opportunistic
  // fadvise calls are serialised with the other parquet pipelines in
  // this query.  Slots are gated on the cache being armed: a
  // metadata-only cache (no buffer pool, or unarmed io_ctx) doesn't
  // honor opportunistic fadvise, so a slot would only add unnecessary
  // sequencer overhead.  Null slot keeps the provider on the legacy
  // direct-fadvise path (also a no-op when the cache is unarmed).
  pipeline_ordered_prefetching_manager::pipeline_slot* prefetch_slot = nullptr;
  if (_prefetch_manager && _io_ctx && _io_ctx->cache() && _io_ctx->cache()->is_armed()) {
    prefetch_slot = _prefetch_manager->add_pipeline_slot(pipeline_id);
  }
  return std::make_unique<parquet_split_provider>(
    info->returned_types,
    info->file_paths,
    info->column_ids,
    info->projection_ids,
    info->names,
    op->get_types().size(),
    std::move(info->table_filters),
    info->partition_indices,
    info->approximate_batch_size,
    parquet_split_provider::DEFAULT_MAX_FILE_PROCESSED,
    _io_ctx,
    op->get_name(),
    op->get_operator_id(),
    pipeline_id,
    prefetch_slot);
}

void sirius_scan_manager::start_metadata_processing()
{
  for (auto* op : _scan_op_order) {
    auto it = _providers_by_op.find(op);
    if (it == _providers_by_op.end()) { continue; }
    auto* connector = op->get_split_connector();
    if (connector == nullptr) { continue; }

    try {
      // run() is fire-and-forget: it enqueues workers and returns immediately.
      // Worker exceptions ride on connector.close(exception_ptr) and surface
      // when the consumer drains via get_next_split().
      it->second->run(*_dispatcher, *connector);
    } catch (const std::exception& e) {
      SIRIUS_LOG_ERROR("[sirius_scan_manager] driver: provider failed to start: {}", e.what());
      // Synchronous failure inside run() (e.g. scheduler.enqueue throwing)
      // bypasses the worker error path, so forward it through the connector
      // here. close() is idempotent and keeps the first stored exception.
      connector->close(std::current_exception());
    }
  }
}

void sirius_scan_manager::reset()
{
  // Dispatcher::request_stop fires its internal stop_token, which the
  // sequencer task polls every SEQUENCER_POLL_INTERVAL ms — so it bails
  // mid-pipeline without a side-channel stop_source.
  _dispatcher->request_stop();
  _dispatcher->wait_for_all();
  _scan_op_order.clear();
  _providers_by_op.clear();
  // Providers (whose run_batch holds slot pointers) and the sequencer
  // task have both finished by now, so it's safe to tear down the
  // manager and the slots it owns.
  _prefetch_manager.reset();
  _dispatcher =
    std::make_unique<exec::scoped_dispatcher>(_thread_pool, _config.thread_pool.num_threads);
}

void sirius_scan_manager::start() {}

void sirius_scan_manager::stop()
{
  reset();
  _thread_pool.stop();
}

void sirius_scan_manager::insert_pinned_entry(const std::string& name,
                                              std::vector<std::string> column_names,
                                              std::vector<std::string> file_paths,
                                              std::vector<std::unique_ptr<cudf::table>> data_tables,
                                              cucascade::memory::memory_space& memory_space)
{
  // Compute the total row count of the incoming tables before releasing them
  // (release() empties the table; num_rows() would then return 0).
  std::size_t new_num_rows = 0;
  for (auto const& table : data_tables) {
    if (table) { new_num_rows += static_cast<std::size_t>(table->num_rows()); }
  }

  auto existing_it = _pinned_entries.find(name);
  if (existing_it != _pinned_entries.end()) {
    if (existing_it->second.num_rows == new_num_rows) {
      // Same row count → merge unique columns into the existing entry.
      auto& entry = existing_it->second;
      for (auto& table : data_tables) {
        if (!table) { continue; }
        auto cols = table->release();
        if (cols.size() != column_names.size()) {
          throw std::runtime_error(
            "[sirius_scan_manager::insert_pinned_entry] table column count " +
            std::to_string(cols.size()) + " does not match column_names size " +
            std::to_string(column_names.size()));
        }
        for (std::size_t i = 0; i < cols.size(); ++i) {
          auto const& col_name = column_names[i];
          if (entry.data_batches_by_column.contains(col_name)) {
            // Already cached — drop the duplicate column.
            continue;
          }
          entry.data_batches_by_column[col_name].emplace_back(std::move(cols[i]));
        }
      }
      // Append any new column names to the entry's column_names list so its
      // metadata reflects the union of pinned columns.
      for (auto& cn : column_names) {
        if (std::find(entry.column_names.begin(), entry.column_names.end(), cn) ==
            entry.column_names.end()) {
          entry.column_names.push_back(std::move(cn));
        }
      }
      return;
    }
    // Row count differs → drop the stale entry and rebuild below.
    _pinned_entries.erase(existing_it);
  }

  pinned_entry entry;
  entry.column_names = std::move(column_names);
  entry.file_paths   = std::move(file_paths);
  entry.tier         = cucascade::memory::Tier::GPU;
  entry.memory_space = &memory_space;
  entry.num_rows     = new_num_rows;

  for (auto& table : data_tables) {
    if (!table) { continue; }
    auto cols = table->release();
    if (cols.size() != entry.column_names.size()) {
      throw std::runtime_error("[sirius_scan_manager::insert_pinned_entry] table column count " +
                               std::to_string(cols.size()) + " does not match column_names size " +
                               std::to_string(entry.column_names.size()));
    }
    for (std::size_t i = 0; i < cols.size(); ++i) {
      entry.data_batches_by_column[entry.column_names[i]].emplace_back(std::move(cols[i]));
    }
  }

  _pinned_entries[name] = std::move(entry);
}

void sirius_scan_manager::insert_pinned_entry_host(
  const std::string& name,
  std::vector<std::string> column_names,
  std::vector<std::string> file_paths,
  std::vector<std::shared_ptr<cucascade::host_data_representation>> host_chunks,
  cucascade::memory::memory_space& memory_space)
{
  // The host-tier path captures one chunk per emitted batch; each chunk holds every
  // pinned column. Re-insert always replaces — there is no per-column merge analog
  // to the GPU path because the chunk-vs-column dimensions are flipped.
  std::size_t new_num_rows = 0;
  for (auto const& chunk : host_chunks) {
    if (!chunk) { continue; }
    auto const& host_table = chunk->get_host_table();
    if (host_table && !host_table->columns.empty()) {
      new_num_rows += static_cast<std::size_t>(host_table->columns.front().num_rows);
    }
  }

  pinned_entry entry;
  entry.column_names = std::move(column_names);
  entry.file_paths   = std::move(file_paths);
  entry.tier         = cucascade::memory::Tier::HOST;
  entry.memory_space = &memory_space;
  entry.num_rows     = new_num_rows;
  entry.host_chunks  = std::move(host_chunks);

  _pinned_entries[name] = std::move(entry);
}

void sirius_scan_manager::remove_pinned_entry(const std::string& name)
{
  _pinned_entries.erase(name);
}

}  // namespace sirius::scan_manager
