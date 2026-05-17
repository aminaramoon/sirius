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

#include "io/prefetching_cache.hpp"

#include "io/io_context.hpp"

#include <cuda_runtime.h>

#include <spdlog/spdlog.h>

#include <algorithm>
#include <cassert>
#include <memory>
#include <stdexcept>

namespace sirius::io {

// ===========================================================================
// buffer_pool
// ===========================================================================

buffer_pool::buffer_pool(cucascade::memory::fixed_size_host_memory_resource& mr,
                         uint32_t max_slabs,
                         uint32_t initial_slabs)
  : _mr(mr), _chunk_bytes(mr.get_block_size()), _max_slabs(max_slabs)
{
  std::unique_lock lk(_mtx);
  auto const n = std::min(initial_slabs, _max_slabs);
  for (uint32_t i = 0; i < n; ++i) {
    if (!grow_locked()) break;
  }
}

buffer_pool::~buffer_pool() = default;

bool buffer_pool::grow_locked()
{
  if (_allocations.size() >= _max_slabs) return false;

  auto const bytes = slab_bytes();
  cucascade::memory::fixed_multiple_blocks_allocation alloc;
  try {
    alloc = _mr.allocate_multiple_blocks(bytes);
  } catch (std::exception const& e) {
    spdlog::warn("buffer_pool: allocate_multiple_blocks({:.0f}MB) failed: {}",
                 static_cast<double>(bytes) / (1024.0 * 1024.0),
                 e.what());
    return false;
  }
  if (!alloc) return false;

  auto blocks = alloc->get_blocks();
  _free_list.reserve(_free_list.size() + blocks.size());
  for (auto* p : blocks)
    _free_list.push_back(p);
  auto const n = static_cast<uint32_t>(blocks.size());
  _allocations.push_back(std::move(alloc));
  _total_chunks.fetch_add(n, std::memory_order_relaxed);
  _total_free.fetch_add(n, std::memory_order_relaxed);

  spdlog::debug("buffer_pool: allocated slab {} ({} chunks, {:.0f}MB)",
                _allocations.size(),
                n,
                static_cast<double>(bytes) / (1024.0 * 1024.0));
  return true;
}

std::byte* buffer_pool::allocate()
{
  std::unique_lock lk(_mtx);
  if (_free_list.empty() && !grow_locked()) return nullptr;
  std::byte* p = _free_list.back();
  _free_list.pop_back();
  _total_free.fetch_sub(1, std::memory_order_relaxed);
  return p;
}

size_t buffer_pool::allocate_bulk(size_t n, std::vector<std::byte*>& out)
{
  if (n == 0) return 0;

  std::unique_lock lk(_mtx);
  size_t got = 0;
  while (got < n) {
    auto take = std::min(n - got, _free_list.size());
    if (take > 0) {
      out.insert(out.end(), _free_list.end() - static_cast<ptrdiff_t>(take), _free_list.end());
      _free_list.resize(_free_list.size() - take);
      got += take;
    }
    if (got == n) break;
    if (!grow_locked()) break;
  }
  _total_free.fetch_sub(static_cast<uint32_t>(got), std::memory_order_relaxed);
  return got;
}

void buffer_pool::deallocate_bulk(std::vector<std::byte*>& out)
{
  if (out.empty()) return;
  std::unique_lock lk(_mtx);
  _free_list.insert(_free_list.end(), out.begin(), out.end());
  _total_free.fetch_add(static_cast<uint32_t>(out.size()), std::memory_order_relaxed);
  lk.unlock();
  out.clear();
}

void buffer_pool::deallocate(std::byte* p)
{
  std::unique_lock lk(_mtx);
  _free_list.push_back(p);
  _total_free.fetch_add(1, std::memory_order_relaxed);
}

void buffer_pool::reclaim_all()
{
  std::unique_lock lk(_mtx);
  _free_list.clear();
  // The slabs we already pulled from the upstream resource stay live; rebuild
  // the free list from their per-block pointers.  No upstream traffic.
  size_t total = 0;
  for (auto const& alloc : _allocations) {
    auto blocks = alloc->get_blocks();
    _free_list.reserve(_free_list.size() + blocks.size());
    for (auto* p : blocks)
      _free_list.push_back(p);
    total += blocks.size();
  }
  _total_free.store(static_cast<uint32_t>(total), std::memory_order_relaxed);
  _total_chunks.store(static_cast<uint32_t>(total), std::memory_order_relaxed);
}

// ===========================================================================
// pinned_view
// ===========================================================================

namespace {

// Carries a strong ref to the entry until the host callback fires, so the
// entry can't be destroyed mid-flight even if the cache map drops it.
struct release_callback_args {
  std::shared_ptr<cache_entry> entry;
};

// cudaStreamAddCallback signature (deprecated but used deliberately).
// Unlike cudaLaunchHostFunc, this callback fires even when the stream is
// already in an error state — so a poisoned stream cannot leak the entry's
// pin and its pool chunks forever.  Status is intentionally ignored: the
// read pin must be released regardless of whether the H2D copy succeeded.
void CUDART_CB release_read_host_callback(cudaStream_t /*stream*/,
                                          cudaError_t /*status*/,
                                          void* p) noexcept
{
  std::unique_ptr<release_callback_args> args(static_cast<release_callback_args*>(p));
  args->entry->state.release_read();
}

}  // namespace

pinned_view::pinned_view(std::shared_ptr<cache_entry> entry,
                         duckdb_moodycamel::ConcurrentQueue<eviction_candidate>& candidate_queue,
                         cudaStream_t stream)
  : _entry(nullptr), _candidate_queue(&candidate_queue), _stream(stream)
{
  if (!entry) return;
  if (!entry->state.try_acquire_read()) return;
  _entry = std::move(entry);
}

pinned_view::~pinned_view() { unpin(); }

pinned_view::pinned_view(pinned_view&& o) noexcept
  : _entry(std::move(o._entry)), _candidate_queue(o._candidate_queue), _stream(o._stream)
{
  o._entry.reset();
}

pinned_view& pinned_view::operator=(pinned_view&& o) noexcept
{
  if (this != &o) {
    unpin();
    _entry = std::move(o._entry);
    o._entry.reset();
    _candidate_queue = o._candidate_queue;
    _stream          = o._stream;
  }
  return *this;
}

void pinned_view::unpin()
{
  if (!_entry) return;

  // Always enqueue an eviction candidate — the evictor's state.get_state()
  // gate skips entries still in_use, so a candidate whose deferred
  // release_read is still pending on a stream callback simply isn't evicted
  // until the callback runs.  Silent post: no semaphore release.  The
  // evictor drains candidates on its next poll tick (EVICTOR_POLL_INTERVAL)
  // or whenever a chunk request wakes it — whichever comes first.
  _candidate_queue->enqueue({std::weak_ptr<cache_entry>(_entry)});

  if (_stream == nullptr) {
    // Synchronous path (host reads): no async ops outstanding, release now.
    _entry->state.release_read();
    _entry.reset();
  } else {
    // Async path (device reads): defer release_read via a host callback so
    // it fires only after the caller's stream reaches this point.  Any
    // cudaMemcpyAsync submitted earlier against this entry's pinned chunks
    // therefore completes before pin_count drops to zero — at which point
    // the entry transitions in_use → cached and becomes evictable.
    auto args   = std::make_unique<release_callback_args>();
    args->entry = std::move(_entry);
    // cudaStreamAddCallback (not cudaLaunchHostFunc): fires on error too,
    // so the release_read() is guaranteed even if the user's stream is
    // poisoned by an unrelated failure.  Otherwise a single stream error
    // would permanently leak this entry's pin and its pinned chunks.
    cudaStreamAddCallback(_stream, &release_read_host_callback, args.release(), 0);
  }
}

size_t pinned_view::num_chunks() const noexcept { return _entry ? _entry->chunks.size() : 0; }

std::span<const std::byte> pinned_view::operator[](size_t i) const noexcept
{
  if (!_entry || i >= _entry->chunks.size()) return {};
  auto phys_size   = static_cast<size_t>(_entry->physical_range.size());
  auto chunk_bytes = _entry->chunk_bytes;
  auto chunk_start = i * chunk_bytes;
  auto chunk_sz    = std::min(chunk_bytes, phys_size - chunk_start);
  return {_entry->chunks[i], chunk_sz};
}

cudf::io::text::byte_range_info pinned_view::logical_range() const noexcept
{
  if (!_entry) return {0, 0};
  return _entry->logical_range;
}

cudf::io::text::byte_range_info pinned_view::physical_range() const noexcept
{
  if (!_entry) return {0, 0};
  return _entry->physical_range;
}

size_t pinned_view::size() const noexcept
{
  return _entry ? static_cast<size_t>(_entry->logical_range.size()) : 0;
}

std::vector<cudf::io::datasource::non_owning_buffer> pinned_view::slice(size_t offset,
                                                                        size_t size) const
{
  std::vector<cudf::io::datasource::non_owning_buffer> result;
  if (!_entry || size == 0) return result;

  // Physical range starts at a potentially earlier (aligned) offset.
  // The delta tells us where logical byte 0 sits inside the physical buffer.
  auto phys_off    = static_cast<size_t>(_entry->physical_range.offset());
  auto logical_off = static_cast<size_t>(_entry->logical_range.offset());
  auto phys_size   = static_cast<size_t>(_entry->physical_range.size());

  // Convert logical [offset, offset+size) to physical byte position
  // within the chunked buffer.
  size_t phys_start = (offset - logical_off) + (logical_off - phys_off);
  size_t remaining  = size;

  // Walk the chunks that span [phys_start, phys_start + size).
  auto const chunk_bytes = _entry->chunk_bytes;
  size_t chunk_idx       = phys_start / chunk_bytes;
  size_t off_in_chunk    = phys_start % chunk_bytes;

  while (remaining > 0 && chunk_idx < _entry->chunks.size()) {
    auto chunk_avail =
      std::min(chunk_bytes - off_in_chunk, phys_size - chunk_idx * chunk_bytes - off_in_chunk);
    auto n  = std::min(remaining, chunk_avail);
    auto* p = reinterpret_cast<uint8_t const*>(_entry->chunks[chunk_idx]) + off_in_chunk;

    // Coalesce with the previous slice if this chunk is contiguous with the
    // tail of the previous slice in the pinned host address space.  Adjacent
    // chunks within the same slab are virtually contiguous (the slab is one
    // cudaHostAlloc), which is the common case for a freshly-filled pool.
    if (!result.empty()) {
      auto const& last = result.back();
      if (last.data() + last.size() == p) {
        result.back() = cudf::io::datasource::non_owning_buffer(last.data(), last.size() + n);
        remaining -= n;
        ++chunk_idx;
        off_in_chunk = 0;
        continue;
      }
    }

    result.emplace_back(p, n);
    remaining -= n;
    ++chunk_idx;
    off_in_chunk = 0;
  }

  return result;
}

// ===========================================================================
// prefetching_cache — construction / destruction
// ===========================================================================

prefetching_cache::prefetching_cache(cucascade::memory::fixed_size_host_memory_resource& mr,
                                     uint32_t max_slabs,
                                     uint32_t initial_slabs)
  : _pool_mr(&mr),
    _pool_max_slabs(max_slabs),
    _pool_initial_slabs(std::min(initial_slabs, max_slabs))
{
  // Separators point past the (empty) list initially; every bucket is empty.
  _lru_buckets.fill(_lru_list.end());
  // Threads start dormant — _active is false until reset(non-null) flips it.
  // Both still run so they can be re-armed cheaply without thread churn
  // between queries.
  _evictor_thread = std::jthread([this](std::stop_token st) { evictor_loop(std::move(st)); });
  _worker_thread  = std::jthread([this](std::stop_token st) { worker_loop(std::move(st)); });
}

prefetching_cache::~prefetching_cache()
{
  _worker_thread.request_stop();
  _evictor_thread.request_stop();
  _work_seq.fetch_add(1, std::memory_order_release);
  _work_seq.notify_all();
  _request_sem.release();
  {
    std::unique_lock lk(_ctrl_mtx);
    _active = false;
    _ctrl_cv.notify_all();
  }

  // Wake readers waiting on entries that were loading when shutdown began.
  // Outstanding backend requests may still resolve via their request_context
  // safety net, but waiters should not depend on that happening.
  abort_pending_entries();

  // Join explicitly so all cache-owned queues and counters outlive the worker
  // and evictor loops. std::jthread would join during member destruction, but
  // doing it here lets us abort any entries that raced into loading just before
  // the worker observed stop.
  if (_worker_thread.joinable()) { _worker_thread.join(); }
  if (_evictor_thread.joinable()) { _evictor_thread.join(); }

  abort_pending_entries();
}

void prefetching_cache::reset(std::shared_ptr<sirius_ioctx> io_ctx, size_t inflight_budget_chunks)
{
  // ----- Phase 1: take the cache offline ---------------------------------
  // Flip _active to false so:
  //   - insert() stops scheduling prefetch work (it still records metadata)
  //   - the worker, when it loops back, parks on _ctrl_cv
  // After flipping, drain the work queue ourselves so anything the worker
  // hasn't picked up yet doesn't sit around with a stale io_ctx attached.
  {
    std::unique_lock lk(_ctrl_mtx);
    _active = false;
    _ctrl_cv.notify_all();
  }
  _work_seq.fetch_add(1, std::memory_order_release);
  _work_seq.notify_all();

  work_item drained;
  while (_work_queue.try_dequeue(drained)) {
    for (auto& entry : drained.entries) {
      if (entry) entry->state.try_cancel_queued();
    }
  }

  // ----- Phase 2: wait for in-flight backend IO --------------------------
  // The worker may have dispatched host_read_ranges_async_io but the
  // completion callback hasn't fired yet.  Those callbacks touch the pool
  // and cache entries via the admission_control::slot captured in their
  // closure; draining the admission_control's outstanding slots therefore
  // proves no callback is mid-flight.  If we're transitioning null → null
  // there's no budget to drain.
  if (_inflight_budget) { _inflight_budget->wait_for_idle(); }

  // ----- Phase 3: drain the eviction side --------------------------------
  // Candidates are weak refs; just empty the queue.  Pending chunk
  // requests would otherwise have the evictor block on backpressure for
  // entries that are about to be wiped — fulfil them with an exception so
  // any worker still parked in fut.get() unblocks.
  eviction_candidate cand;
  while (_candidate_queue.try_dequeue(cand)) {}

  eviction_request ereq;
  while (_request_queue.try_dequeue(ereq)) {
    ereq.promise.set_exception(
      std::make_exception_ptr(std::runtime_error("eviction aborted — cache reset")));
  }

  // The LRU list holds raw cache_entry* pointers into the file_cache map;
  // drop it before the entries themselves go away.
  _lru_list.clear();
  _lru_buckets.fill(_lru_list.end());
  _last_seen_age = 0;

  // ----- Phase 4: drop all file entries ----------------------------------
  // After waiting for in-flight IO, no callback references a cache_entry
  // anymore — clearing the map releases every shared_ptr we own.
  {
    std::unique_lock map_lk(_map_mtx);
    _file_cache.clear();
  }

  // ----- Phase 5: pool + budget lifecycle + re-arm -----------------------
  std::unique_lock lk(_ctrl_mtx);
  _io_ctx = io_ctx.get();
  if (_io_ctx != nullptr) {
    if (_pool) {
      _pool->reclaim_all();
    } else if (_pool_mr != nullptr) {
      _pool = std::make_unique<buffer_pool>(*_pool_mr, _pool_max_slabs, _pool_initial_slabs);
    }
    // Rebuild the admission_control with the caller-supplied budget.  No
    // outstanding slots survive across reset (Phase 2 drained them), so
    // dropping the old instance is safe.
    _inflight_budget = std::make_unique<admission_control>(inflight_budget_chunks);
    _active          = true;
    _ctrl_cv.notify_all();
    _work_seq.fetch_add(1, std::memory_order_release);
    _work_seq.notify_all();
  } else {
    // Detached: hand the slabs back to the upstream cucascade resource and
    // release the admission_control.
    _pool.reset();
    _inflight_budget.reset();
    // _active stays false until a future reset(non-null) re-arms us.
  }
}

void prefetching_cache::enqueue_work(work_item item)
{
  _work_queue.enqueue(std::move(item));
  _work_seq.fetch_add(1, std::memory_order_release);
  _work_seq.notify_one();
}

void prefetching_cache::release_chunks(cache_entry& entry)
{
  if (_pool) {
    for (auto* p : entry.chunks)
      _pool->deallocate(p);
  }
  entry.chunks.clear();
}

void prefetching_cache::abort_pending_entries() noexcept
{
  try {
    std::shared_lock map_lk(_map_mtx);
    for (auto const& [_, file_ptr] : _file_cache) {
      if (!file_ptr) { continue; }
      std::shared_lock file_lk(file_ptr->mtx);
      for (auto const& entry : file_ptr->entries) {
        if (entry) { entry->state.try_abort_pending(); }
      }
    }
  } catch (...) {
    // Destructors must not throw; best effort wake-up only.
  }
}

// ===========================================================================
// find_entry — binary search + hit/miss classification
// ===========================================================================

std::shared_ptr<cache_entry> prefetching_cache::find_entry(
  const std::vector<std::shared_ptr<cache_entry>>& entries, size_t offset, size_t size)
{
  // upper_bound: first entry whose offset > requested offset.  The candidate
  // is pos-1 (the last entry whose offset <= requested offset).
  auto pos =
    std::upper_bound(entries.begin(), entries.end(), offset, [](size_t off, auto const& e) {
      return off < static_cast<size_t>(e->logical_range.offset());
    });

  // No entry starts at or before our offset — nothing covers us.
  if (pos == entries.begin()) {
    _full_miss_count.fetch_add(1, std::memory_order_relaxed);
    return nullptr;
  }
  --pos;
  auto entry_end = static_cast<size_t>((*pos)->logical_range.offset()) +
                   static_cast<size_t>((*pos)->logical_range.size());

  // Candidate ends before our offset — no overlap at all.
  if (entry_end <= offset) {
    _full_miss_count.fetch_add(1, std::memory_order_relaxed);
    return nullptr;
  }
  // Candidate overlaps but doesn't fully contain the requested tail.
  if (offset + size > entry_end) {
    _partial_miss_count.fetch_add(1, std::memory_order_relaxed);
    return nullptr;
  }
  return *pos;
}

// ===========================================================================
// register_metadata
// ===========================================================================

void prefetching_cache::register_metadata(sirius_io_object& obj,
                                          std::shared_ptr<sirius_io_object_metadata> metadata)
{
  // Match insert's contract: caller must own @p obj through a shared_ptr.
  auto obj_sp     = obj.shared_from_this();
  auto const& key = obj.raw_file_cache_id();
  auto file_size  = obj.size();

  std::unique_lock map_lk(_map_mtx);
  auto [it, inserted] = _file_cache.try_emplace(key, nullptr);
  if (inserted) it->second = std::make_unique<file_entry>();
  auto& file = *it->second;

  std::unique_lock file_lk(file.mtx);
  map_lk.unlock();

  file.io_obj    = std::move(obj_sp);
  file.file_size = file_size;
  // Symmetric with insert(): a null @p metadata leaves any existing
  // metadata in place rather than clobbering it.
  if (metadata) { file.metadata = std::move(metadata); }
}

// ===========================================================================
// insert
// ===========================================================================

void prefetching_cache::insert(sirius_io_object& obj,
                               std::shared_ptr<sirius_io_object_metadata> metadata,
                               const std::vector<cudf::io::text::byte_range_info>& ranges)
{
  assert(std::is_sorted(ranges.begin(),
                        ranges.end(),
                        [](auto const& a, auto const& b) { return a.offset() < b.offset(); }) &&
         "ranges must be sorted by offset");

  // shared_from_this() throws std::bad_weak_ptr if @p obj isn't owned by a
  // shared_ptr — the contract is enforced at the call site, no null check
  // needed here.
  auto obj_sp     = obj.shared_from_this();
  auto const& key = obj.raw_file_cache_id();
  auto file_size  = obj.size();

  // Snapshot the lifecycle once.  Either piece going missing (no pool /
  // no ioctx) puts insert() into the "metadata only" path the user asked
  // for: still record the file entry + metadata so subsequent reads can
  // observe it, but skip building per-range cache_entries that would need
  // the pool's chunk_bytes and the ioctx's compute_physical_range.
  buffer_pool* pool    = nullptr;
  sirius_ioctx* io_ctx = nullptr;
  bool active          = false;
  {
    std::unique_lock lk(_ctrl_mtx);
    pool   = _pool.get();
    io_ctx = _io_ctx;
    active = _active;
  }

  std::unique_lock map_lk(_map_mtx);
  auto [it, inserted] = _file_cache.try_emplace(key, nullptr);
  if (inserted) it->second = std::make_unique<file_entry>();
  auto& file = *it->second;

  std::unique_lock file_lk(file.mtx);
  map_lk.unlock();

  file.io_obj    = obj_sp;
  file.file_size = file_size;
  // A nullptr @p metadata means "no new metadata to store" — preserve whatever
  // a previous insert() left in place so callers can skip re-supplying it on
  // every prefetch insert.
  if (metadata) { file.metadata = std::move(metadata); }

  if (!active || pool == nullptr || io_ctx == nullptr) {
    // Dormant cache: file entry + metadata are stored; per-range
    // prefetching needs the pool and ioctx and waits for the next reset.
    return;
  }

  // _cache_age only advances via refresh_cache().  Sample it here so every
  // range stamped in this call shares the same request_ts — they belong to
  // the same epoch.
  auto tick = _cache_age.load(std::memory_order_relaxed);

  // Every range in the insert becomes a prefetch request, regardless of the
  // entry's current state.  The worker decides at dispatch time whether the
  // entry actually needs loading (via state check in try_start_loading).
  // This lets inserts that land on an evicted-then-inserted-again range
  // trigger a fresh prefetch.
  std::vector<std::shared_ptr<cache_entry>> new_entries;

  std::vector<std::shared_ptr<cache_entry>> merged;
  merged.reserve(file.entries.size() + ranges.size());
  new_entries.reserve(ranges.size());

  auto ex_it  = file.entries.begin();
  auto ex_end = file.entries.end();

  for (auto const& logical : ranges) {
    auto off = logical.offset();
    while (ex_it != ex_end && (*ex_it)->logical_range.offset() < off) {
      merged.push_back(std::move(*ex_it));
      ++ex_it;
    }
    if (ex_it != ex_end && (*ex_it)->logical_range.offset() == off) {
      auto& existing = *ex_it;
      existing->n_total_request.fetch_add(1, std::memory_order_relaxed);
      // n_request: same epoch → bump; different epoch → reset to 1.
      auto prev_ts = existing->request_ts.load(std::memory_order_acquire);
      if (prev_ts == tick)
        existing->n_request.fetch_add(1, std::memory_order_relaxed);
      else
        existing->n_request.store(1, std::memory_order_relaxed);
      existing->request_ts.store(tick, std::memory_order_release);
      new_entries.push_back(existing);
      merged.push_back(std::move(existing));
      ++ex_it;
    } else {
      auto physical = io_ctx->compute_physical_range(logical, file_size);
      auto e        = std::make_shared<cache_entry>(logical, physical, pool->chunk_bytes());
      e->n_total_request.fetch_add(1, std::memory_order_relaxed);
      e->n_request.store(1, std::memory_order_relaxed);
      e->request_ts.store(tick, std::memory_order_release);
      new_entries.push_back(e);
      merged.push_back(std::move(e));
    }
  }
  // Forward any trailing existing entries.
  for (; ex_it != ex_end; ++ex_it)
    merged.push_back(std::move(*ex_it));

  file.entries = std::move(merged);

  file_lk.unlock();

  if (!new_entries.empty())
    enqueue_work(prefetch_req{key, std::move(obj_sp), std::move(new_entries)});
}

// ===========================================================================
// read — non-blocking, single range by offset
// ===========================================================================

pinned_view prefetching_cache::read(const sirius_io_object& obj,
                                    size_t offset,
                                    size_t size,
                                    cudaStream_t stream)
{
  auto const& key = obj.raw_file_cache_id();

  std::shared_lock map_lk(_map_mtx);
  auto it = _file_cache.find(key);
  if (it == _file_cache.end()) {
    _full_miss_count.fetch_add(1, std::memory_order_relaxed);
    return {};
  }
  auto& file = *it->second;

  std::shared_lock file_lk(file.mtx);
  map_lk.unlock();

  auto entry = find_entry(file.entries, offset, size);
  if (!entry) return {};

  // Dispatch on state:
  //   cached / in_use  → pin immediately (stamps consumption_ts).
  //   loading          → wait for the worker to resolve the load, then retry.
  //   queued / evicting / empty → return empty (caller falls back).
  bool waited_on_loading = false;
  while (true) {
    auto st = entry->state.get_state();
    if (st == entry_state::cached || st == entry_state::in_use) {
      pinned_view view{entry, _candidate_queue, stream};
      if (view) {
        _hit_count.fetch_add(1, std::memory_order_relaxed);
        if (waited_on_loading) _hit_after_wait.fetch_add(1, std::memory_order_relaxed);
        entry->consumption_ts.store(_cache_age.load(std::memory_order_relaxed),
                                    std::memory_order_release);
        entry->n_request.fetch_sub(1, std::memory_order_relaxed);
        return view;
      }
      // try_acquire_read lost a race; re-observe the state.
      continue;
    }
    if (st == entry_state::loading) {
      waited_on_loading = true;
      // Release file_lk across the wait: the entry is pinned by our local
      // shared_ptr, and file.mtx no longer protects anything we touch while
      // parked.  Holding it shared would stall every concurrent insert() on
      // this file (and, under writer-preference shared_mutex, every reader
      // queued behind that insert).
      file_lk.unlock();
      entry->state.wait_while_loading();
      file_lk.lock();
      continue;
    }
    _partial_miss_count.fetch_add(1, std::memory_order_relaxed);
    entry->n_request.fetch_sub(1, std::memory_order_relaxed);
    return {};
  }
}

std::shared_ptr<sirius_io_object_metadata> prefetching_cache::get_metadata(
  const sirius_io_object& obj) const
{
  auto const& key = obj.raw_file_cache_id();

  std::shared_lock map_lk(_map_mtx);
  auto it = _file_cache.find(key);
  if (it == _file_cache.end()) { return nullptr; }
  auto& file = *it->second;

  std::shared_lock file_lk(file.mtx);
  return file.metadata;
}

void prefetching_cache::refresh_cache()
{
  _cache_age.fetch_add(1, std::memory_order_relaxed);

  // Drain any pending prefetch work.  The entries in each drained item were
  // queued for a previous epoch; cancel them (queued → empty) so future
  // inserts can re-queue them fresh.
  work_item item;
  while (_work_queue.try_dequeue(item)) {
    for (auto& entry : item.entries) {
      entry->state.try_cancel_queued();
    }
  }

  spdlog::info("cache being refreshed: closing — {}", summary());
}

std::string prefetching_cache::summary() const
{
  auto hits            = _hit_count.load(std::memory_order_relaxed);
  auto hits_after_wait = _hit_after_wait.load(std::memory_order_relaxed);
  auto partial         = _partial_miss_count.load(std::memory_order_relaxed);
  auto full            = _full_miss_count.load(std::memory_order_relaxed);
  auto worker_skip     = _worker_skipped_reader_resolved.load(std::memory_order_relaxed);
  auto evicted_entries = _evicted_entries.load(std::memory_order_relaxed);
  auto evicted_chunks  = _evicted_chunks.load(std::memory_order_relaxed);
  auto total           = hits + partial + full;
  auto pct             = [&](uint64_t n) {
    return total > 0 ? (100.0 * static_cast<double>(n) / static_cast<double>(total)) : 0.0;
  };
  auto s = fmt::format(
    "prefetching_cache: age={} {} reads ({} hit {:.1f}% [of which {} waited "
    "on loading], {} partial-miss {:.1f}%, {} full-miss {:.1f}%); worker "
    "skipped "
    "(reader-resolved) {}; evicted {} entries / {} chunks; pool {}/{} "
    "chunks free",
    _cache_age.load(std::memory_order_relaxed),
    total,
    hits,
    pct(hits),
    hits_after_wait,
    partial,
    pct(partial),
    full,
    pct(full),
    worker_skip,
    evicted_entries,
    evicted_chunks,
    _pool ? _pool->free_count() : 0U,
    _pool ? _pool->total_chunks() : 0U);
  spdlog::info("{}", s);
  return s;
}

// ===========================================================================
// LRU helpers (evictor-thread only)
// ===========================================================================

int prefetching_cache::lru_bucket_for(cache_entry const& entry, int64_t age) noexcept
{
  int64_t n   = entry.n_total_request.load(std::memory_order_relaxed);
  int64_t idx = static_cast<int64_t>(N_LRU_BUCKETS - 1) + n - age;
  if (idx < 0) return 0;
  if (idx > static_cast<int64_t>(N_LRU_BUCKETS - 1)) return static_cast<int>(N_LRU_BUCKETS - 1);
  return static_cast<int>(idx);
}

std::list<cache_entry*>::iterator prefetching_cache::lru_insert(cache_entry* entry, int bucket)
{
  // Insert just before the separator that ends this bucket — the new
  // element becomes the bucket's new tail.
  auto new_it = _lru_list.insert(_lru_buckets[bucket], entry);
  // Any lower separator that used to coincide with the insertion point
  // now sits *after* the new element.  If we don't patch them, the new
  // element would appear to belong to a lower bucket.
  auto boundary = _lru_buckets[bucket];
  for (int i = bucket - 1; i >= 0; --i) {
    if (_lru_buckets[i] == boundary)
      _lru_buckets[i] = new_it;
    else
      break;
  }
  return new_it;
}

std::list<cache_entry*>::iterator prefetching_cache::lru_erase(std::list<cache_entry*>::iterator it)
{
  auto next = std::next(it);
  for (auto& sep : _lru_buckets) {
    if (sep == it) sep = next;
  }
  return _lru_list.erase(it);
}

void prefetching_cache::age_lru_buckets()
{
  if (_lru_list.empty()) return;
  // Shift left: bucket i absorbs what used to be bucket i+1.
  for (size_t i = 0; i + 1 < N_LRU_BUCKETS; ++i)
    _lru_buckets[i] = _lru_buckets[i + 1];
  // _lru_buckets[N-1] stays at end() by invariant.  Pull the last real
  // separator back so bucket N-2 ends at the very last element, i.e.
  // bucket N-1 contains exactly one entry (the newest).
  auto last                       = std::prev(_lru_list.end());
  _lru_buckets[N_LRU_BUCKETS - 2] = last;
  // Clamp any earlier separator that still points at end() — without this,
  // separators would be non-monotonic (end() > last).
  for (int i = static_cast<int>(N_LRU_BUCKETS) - 3; i >= 0; --i) {
    if (_lru_buckets[i] == _lru_list.end())
      _lru_buckets[i] = last;
    else
      break;
  }
}

void prefetching_cache::drain_candidates_into_lru()
{
  int64_t age = _cache_age.load(std::memory_order_relaxed);
  eviction_candidate cand;
  while (_candidate_queue.try_dequeue(cand)) {
    auto sp = cand.entry.lock();
    if (!sp) continue;
    auto* raw = sp.get();
    if (raw->in_lru) continue;  // already filed
    int bucket = lru_bucket_for(*raw, age);
    lru_insert(raw, bucket);
    raw->in_lru = true;
  }
}

// ===========================================================================
// evictor_loop
// ===========================================================================

void prefetching_cache::evictor_loop(std::stop_token stop)
{
  std::stop_callback stop_cb(stop, [this] { _request_sem.release(); });

  // Free one cached entry's chunks back to the pool.  Returns the number
  // of chunks freed, or 0 if the entry isn't eligible right now.  Caller
  // removes the entry from the LRU on non-zero return.
  //
  // The state-machine gate (state == cached, pin_count == 0) is the only
  // in-use check needed: pinned_view::unpin defers release_read via a host
  // callback on the reader's stream, so an entry stays in_use until any
  // cudaMemcpyAsync the reader submitted has completed.
  auto try_evict_raw = [this](cache_entry* entry, int64_t age) -> size_t {
    if (entry->state.get_state() != entry_state::cached) return 0;
    if (!entry->is_consumed_or_stale(static_cast<uint64_t>(age))) return 0;
    if (!entry->state.try_start_evicting()) return 0;
    size_t n = entry->chunks.size();
    // The pool can only be torn down after reset() has waited for the
    // worker and in-flight IO to drain, so any cached entry still in the
    // LRU points into a live pool — but assert for safety in case a future
    // refactor races the evictor against pool teardown.
    if (_pool) { _pool->deallocate_bulk(entry->chunks); }
    entry->state.mark_evicted();
    _evicted_entries.fetch_add(1, std::memory_order_relaxed);
    _evicted_chunks.fetch_add(n, std::memory_order_relaxed);
    return n;
  };

  // Catch the evictor up with the live cache age by shifting bucket
  // separators.  After N_LRU_BUCKETS shifts the LRU is saturated, so
  // further aging is a no-op.
  auto catch_up_age = [this]() {
    int64_t age   = _cache_age.load(std::memory_order_relaxed);
    int64_t delta = age - _last_seen_age;
    if (delta <= 0) return;
    auto shifts = std::min<int64_t>(delta, static_cast<int64_t>(N_LRU_BUCKETS));
    for (int64_t i = 0; i < shifts; ++i)
      age_lru_buckets();
    _last_seen_age = age;
  };

  while (!stop.stop_requested()) {
    // Step 1: if a backpressure request is already waiting, service it now.
    eviction_request req;
    bool have_request = _request_queue.try_dequeue(req);

    if (!have_request) {
      // Step 2: wait for either a new request or the poll timeout.
      bool signalled = _request_sem.try_acquire_for(EVICTOR_POLL_INTERVAL);
      if (stop.stop_requested()) break;
      if (signalled) {
        have_request = _request_queue.try_dequeue(req);
      } else {
        // Idle tick: pull fresh candidates into the LRU and age buckets
        // so the list is organised when a request eventually lands.
        drain_candidates_into_lru();
        catch_up_age();
        continue;
      }
    }

    if (!have_request) continue;  // spurious wake

    size_t needed            = req.n_chunks_needed;
    size_t reclaimed         = 0;
    bool cache_aged_mid_wait = false;

    // Keep trying to satisfy the request.  Only give up (set_exception) when
    // the cache ages mid-wait — that's the signal that this request's epoch
    // is stale and the caller is moving on.
    while (reclaimed < needed) {
      drain_candidates_into_lru();
      catch_up_age();

      int64_t age = _cache_age.load(std::memory_order_relaxed);

      // --- One eviction walk -----------------------------------------------
      auto it           = _lru_list.begin();
      size_t cur_bucket = 0;
      while (it != _lru_list.end() && reclaimed < needed) {
        while (cur_bucket < N_LRU_BUCKETS && it == _lru_buckets[cur_bucket])
          ++cur_bucket;
        if (cur_bucket >= N_LRU_BUCKETS) break;

        cache_entry* entry = *it;

        if (!entry->is_consumed_or_stale(static_cast<uint64_t>(age))) {
          ++it;
          continue;
        }

        int intended = lru_bucket_for(*entry, age);
        if (intended <= static_cast<int>(cur_bucket)) {
          size_t freed = try_evict_raw(entry, age);
          if (freed > 0) {
            entry->in_lru = false;
            it            = lru_erase(it);
            reclaimed += freed;
          } else {
            ++it;
          }
        } else {
          it = lru_erase(it);
          lru_insert(entry, intended);
        }
      }

      if (reclaimed >= needed) break;

      // Couldn't satisfy yet.  Wait 50ms to let more candidates arrive (and
      // to observe any refresh_cache that happens during the wait).  If the
      // cache ages during the wait, the request is stale — bail.
      int64_t age_before_wait = age;
      std::this_thread::sleep_for(std::chrono::milliseconds(50));
      if (stop.stop_requested()) break;

      int64_t age_after_wait = _cache_age.load(std::memory_order_relaxed);
      if (age_after_wait != age_before_wait) {
        cache_aged_mid_wait = true;
        break;
      }
      // Else: loop and try again.
    }

    if (reclaimed >= needed) {
      req.promise.set_value();
    } else if (cache_aged_mid_wait) {
      req.promise.set_exception(
        std::make_exception_ptr(std::runtime_error("eviction aborted — cache aged during wait")));
    } else {
      // stop_requested mid-wait — fulfil with exception so the worker
      // unblocks cleanly during shutdown.
      req.promise.set_exception(
        std::make_exception_ptr(std::runtime_error("eviction aborted — cache shutting down")));
    }
  }

  // A worker may be blocked in fut.get() after enqueueing a request while the
  // stop token is already set. Drain anything the loop did not pick up and
  // resolve it with an exception so the worker can exit.
  eviction_request req;
  while (_request_queue.try_dequeue(req)) {
    req.promise.set_exception(
      std::make_exception_ptr(std::runtime_error("eviction aborted — cache shutting down")));
  }
}

// ===========================================================================
// worker_loop
// ===========================================================================

void prefetching_cache::worker_loop(std::stop_token stop)
{
  std::stop_callback stop_cb(stop, [this] {
    _work_seq.fetch_add(1, std::memory_order_release);
    _work_seq.notify_one();
  });

  while (!stop.stop_requested()) {
    // ---- Lifecycle gate ----------------------------------------------------
    // The cache may be dormant (no io_ctx attached yet, or temporarily
    // detached by reset()).  Park on _ctrl_cv until reset() arms us again
    // rather than spinning on the work queue.
    {
      std::unique_lock lk(_ctrl_mtx);
      _ctrl_cv.wait(lk, [&] { return _active || stop.stop_requested(); });
    }
    if (stop.stop_requested()) break;

    work_item item;
    if (!_work_queue.try_dequeue(item)) {
      auto seq = _work_seq.load(std::memory_order_acquire);
      if (!_work_queue.try_dequeue(item)) {
        _work_seq.wait(seq, std::memory_order_relaxed);
        continue;
      }
    }

    // Snapshot pool + io_ctx under the lifecycle lock so we can't race with
    // a reset() that's about to clear them.  If the cache was detached
    // between the gate above and here, the snapshot is null and we
    // cancel-queued every entry so a re-arm sees them as fresh.
    buffer_pool* pool    = nullptr;
    sirius_ioctx* io_ctx = nullptr;
    {
      std::unique_lock lk(_ctrl_mtx);
      if (!_active) {
        for (auto& e : item.entries)
          if (e) e->state.try_cancel_queued();
        continue;
      }
      pool   = _pool.get();
      io_ctx = _io_ctx;
    }
    if (pool == nullptr || io_ctx == nullptr) {
      for (auto& e : item.entries)
        if (e) e->state.try_cancel_queued();
      continue;
    }

    // ---- Phase 1: compute an upper bound on chunks for the whole item ------
    // Peek at each entry's state to skip ones that obviously don't need
    // loading (already cached / loading / etc.).  Peeks are racy, but only
    // as an optimisation — entries that transition out from under us get
    // filtered again in Phase 4 by try_start_loading.
    std::vector<size_t> per_entry_chunks(item.entries.size(), 0);
    size_t upper_bound_chunks = 0;
    auto const chunk_bytes    = pool->chunk_bytes();
    for (size_t i = 0; i < item.entries.size(); ++i) {
      auto const& e = item.entries[i];
      if (e->n_request.load(std::memory_order_acquire) <= 0) continue;
      auto st = e->state.get_state();
      if (st != entry_state::empty && st != entry_state::queued) continue;
      auto phys_size      = static_cast<size_t>(e->physical_range.size());
      auto n              = (phys_size + chunk_bytes - 1) / chunk_bytes;
      per_entry_chunks[i] = n;
      upper_bound_chunks += n;
    }

    if (upper_bound_chunks == 0) continue;

    // ---- Phase 2: allocate chunks up-front, with at most one eviction wait.
    // No entry has been transitioned to loading yet, so if anything here
    // fails we can abandon the work_item cleanly — readers see the original
    // state (empty/cached/etc.) and take their normal paths.
    std::vector<std::byte*> ptrs;
    ptrs.reserve(upper_bound_chunks);
    auto got                = pool->allocate_bulk(upper_bound_chunks, ptrs);
    bool eviction_requested = false;
    if (got < upper_bound_chunks) {
      size_t shortfall = upper_bound_chunks - got;
      eviction_request req;
      req.n_chunks_needed = shortfall;
      auto fut            = req.promise.get_future();
      _request_queue.enqueue(std::move(req));
      _request_sem.release();
      eviction_requested = true;

      try {
        fut.get();
      } catch (...) {
        pool->deallocate_bulk(ptrs);
        continue;
      }

      auto extra = pool->allocate_bulk(shortfall, ptrs);
      if (extra < shortfall) {
        pool->deallocate_bulk(ptrs);
        continue;
      }
    }

    // ---- Phase 3: reserve the inflight budget for the upper bound ----------
    // We may end up using less than upper_bound (if entries race in Phase 4),
    // but admission_control::slot is a fixed reservation — we hold the full
    // amount until IO completes.  This is a conservative over-reservation;
    // any excess chunks are returned to the pool immediately in Phase 4.
    // _active==true (checked above under _ctrl_mtx) guarantees the budget
    // unique_ptr is live; reset() can't drop it without first waiting for
    // wait_for_idle, which only returns after this work item's callback
    // (and every other outstanding callback) has released its slot.
    auto budget_slot = _inflight_budget->acquire(upper_bound_chunks, stop);
    if (!budget_slot) {
      pool->deallocate_bulk(ptrs);
      break;
    }

    // ---- Phase 4: per-entry try_start_loading + chunk assignment -----------
    // This is the first and only place the state machine transitions into
    // loading.  If the CAS loses (state raced past empty/queued), we return
    // that entry's pre-allocated chunks to the pool — no state cleanup needed
    // because we never touched the state.
    std::vector<std::shared_ptr<cache_entry>> batch;
    batch.reserve(item.entries.size());
    size_t ptr_idx = 0;
    for (size_t i = 0; i < item.entries.size(); ++i) {
      auto n = per_entry_chunks[i];
      if (n == 0) continue;

      auto const& e      = item.entries[i];
      auto return_chunks = [&] {
        for (size_t j = 0; j < n; ++j)
          pool->deallocate(ptrs[ptr_idx + j]);
        ptr_idx += n;
      };

      if (e->n_request.load(std::memory_order_acquire) <= 0) {
        _worker_skipped_reader_resolved.fetch_add(1, std::memory_order_relaxed);
        return_chunks();
        continue;
      }
      if (!e->state.try_start_loading()) {
        return_chunks();
        continue;
      }

      e->chunks.assign(ptrs.begin() + ptr_idx, ptrs.begin() + ptr_idx + n);
      ptr_idx += n;
      batch.push_back(e);
    }

    if (batch.empty()) {
      // Every entry raced or resolved under us; budget_slot releases on scope
      // exit, pool chunks already returned.
      continue;
    }

    // ---- Phase 5: build IO ranges and dispatch -----------------------------
    std::vector<cudf::io::text::byte_range_info> io_ranges;
    std::vector<cudf::host_span<std::byte>> io_dsts;
    for (auto const& e : batch) {
      auto phys_off          = static_cast<size_t>(e->physical_range.offset());
      auto phys_size         = static_cast<size_t>(e->physical_range.size());
      auto const chunk_bytes = e->chunk_bytes;
      for (size_t i = 0; i < e->chunks.size(); ++i) {
        auto off = phys_off + i * chunk_bytes;
        auto sz  = std::min(chunk_bytes, phys_size - i * chunk_bytes);
        io_ranges.emplace_back(static_cast<int64_t>(off), static_cast<int64_t>(sz));
        io_dsts.emplace_back(e->chunks[i], sz);
      }
    }

    // io_completion_handler is std::function (copy-constructible), so the
    // move-only slot must be wrapped in a shared_ptr to survive the copy.
    auto slot_holder = std::make_shared<admission_control::slot>(std::move(budget_slot));

    {
      auto& obj_ref = *item.io_obj;
      // The admission_control slot captured in the lambda below is the
      // single source of truth for "this IO is still in flight": its
      // destructor (firing when the closure is destroyed at callback
      // exit) calls back into admission_control::release, which is what
      // reset() waits on via admission_control::wait_for_idle.  As long
      // as the slot is alive, reset() blocks before touching the pool or
      // io_ctx — so the callback can safely use both.
      io_ctx->host_read_ranges_async_io(
        obj_ref,
        io_ranges,
        io_dsts,
        [pool,
         batch  = std::move(batch),
         slot   = std::move(slot_holder),
         io_obj = std::move(item.io_obj),
         key    = std::move(item.file_key)](size_t /*bytes*/, std::exception_ptr ep) {
          if (ep) {
            try {
              std::rethrow_exception(std::move(ep));
            } catch (std::exception const& ex) {
              spdlog::error("prefetching_cache: IO failed for {}: {}", key, ex.what());
            }
            for (auto const& e : batch) {
              pool->deallocate_bulk(e->chunks);
              e->state.try_mark_load_failed();
            }
          } else {
            for (auto const& e : batch)
              e->state.try_mark_cached();
          }
          // `slot` and `io_obj` destruct here — budget is returned to
          // admission_control (which wakes any wait_for_idle waiter once
          // the last slot drops), and the file handle is released only
          // after the IO backend is done with it.
        });
    }
  }
}

}  // namespace sirius::io
