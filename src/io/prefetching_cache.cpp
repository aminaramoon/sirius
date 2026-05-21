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
#include <cmath>
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

pinned_view::pinned_view(std::shared_ptr<cache_entry> entry, cudaStream_t stream)
  : _entry(nullptr), _stream(stream)
{
  if (!entry) return;
  if (!entry->state.try_acquire_read()) return;
  _entry = std::move(entry);
}

pinned_view::~pinned_view() { unpin(); }

pinned_view::pinned_view(pinned_view&& o) noexcept : _entry(std::move(o._entry)), _stream(o._stream)
{
  o._entry.reset();
}

pinned_view& pinned_view::operator=(pinned_view&& o) noexcept
{
  if (this != &o) {
    unpin();
    _entry = std::move(o._entry);
    o._entry.reset();
    _stream = o._stream;
  }
  return *this;
}

void pinned_view::unpin()
{
  if (!_entry) return;

  // The wrapper this entry came from is already in the eviction queue
  // (pushed at insert time).  We just need to release the read pin so
  // the entry transitions back from in_use to cached and becomes
  // evictable.  No candidate-queue push.

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
// file_demand
// ===========================================================================

uint64_t file_demand::register_request() noexcept
{
  uint64_t cur = _packed.load(std::memory_order_relaxed);
  while (true) {
    auto v          = unpack(cur);
    auto next_stamp = v.stamp + 1;
    auto next_n =
      static_cast<uint16_t>(std::min<uint32_t>(static_cast<uint32_t>(v.n_pending) + 1, N_MASK));
    auto next = pack(next_stamp, next_n);
    if (_packed.compare_exchange_weak(
          cur, next, std::memory_order_acq_rel, std::memory_order_relaxed)) {
      return next_stamp;
    }
  }
}

void file_demand::unregister_request() noexcept
{
  uint64_t cur = _packed.load(std::memory_order_relaxed);
  while (true) {
    auto v = unpack(cur);
    if (v.n_pending == 0) return;  // clamped no-op
    auto next = pack(v.stamp, static_cast<uint16_t>(v.n_pending - 1));
    if (_packed.compare_exchange_weak(
          cur, next, std::memory_order_acq_rel, std::memory_order_relaxed)) {
      return;
    }
  }
}

file_demand::view file_demand::load(std::memory_order ord) const noexcept
{
  return unpack(_packed.load(ord));
}

// ===========================================================================
// prefetching_handle — cancel / release
// ===========================================================================

void prefetching_handle::cancel() noexcept
{
  if (!_alive) return;
  // Atomically flip alive to false exactly once. Exchange guarantees we only
  // call notify_disposed() for the first caller when multiple threads race.
  bool was_alive = _alive->exchange(false, std::memory_order_acq_rel);
  if (was_alive && _cache) { _cache->notify_disposed(); }
}

void prefetching_handle::release() noexcept
{
  if (_demand == nullptr) return;
  _demand->unregister_request();
  _demand = nullptr;
  _cache  = nullptr;
}

// ===========================================================================
// prefetching_cache — construction / destruction
// ===========================================================================

namespace {
// armed iff every dependency the prefetch machinery needs is present.
bool compute_armed(buffer_pool* pool, sirius_ioctx const* io_ctx, size_t budget) noexcept
{
  return pool != nullptr && io_ctx != nullptr && io_ctx->supports_vector_host_read() && budget > 0;
}
}  // namespace

prefetching_cache::prefetching_cache(buffer_pool* pool,
                                     sirius_ioctx* io_ctx,
                                     size_t inflight_budget_chunks)
  : _pool(pool),
    _io_ctx(io_ctx),
    _armed(compute_armed(pool, _io_ctx, inflight_budget_chunks)),
    _inflight_budget(_armed ? std::make_unique<admission_control>(inflight_budget_chunks) : nullptr)
{
  // Threads only run when the cache is armed.  When unarmed nobody
  // consumes the work queue, and insert() short-circuits before
  // enqueuing anything anyway; read() simply misses on every lookup.
  if (_armed) {
    _evictor_thread   = std::jthread([this](std::stop_token st) { evictor_loop(std::move(st)); });
    _allocator_thread = std::jthread([this](std::stop_token st) { allocator_loop(std::move(st)); });
    _io_dispatch_thread =
      std::jthread([this](std::stop_token st) { io_dispatch_loop(std::move(st)); });
  }
}

prefetching_cache::~prefetching_cache()
{
  if (!_armed) {
    // Nothing to drain — no threads, no in-flight IO.  The file_cache map
    // and LRU list will be destroyed by member destruction.
    return;
  }

  // Stop the threads.  Stop-token requests + waitpoint wakeups (work_seq,
  // request_sem) unblock both loops; each one rechecks the token at every
  // wait point and exits promptly.
  _allocator_thread.request_stop();
  _evictor_thread.request_stop();
  _work_seq.fetch_add(1, std::memory_order_release);
  _work_seq.notify_all();
  _request_sem.release();

  // Wake readers waiting on entries that were loading when shutdown began.
  // Outstanding backend requests may still resolve via their request_context
  // safety net, but waiters shouldn't depend on that happening.
  abort_pending_entries();

  // Drain whatever the worker / evictor left behind so the loops don't
  // block forever waiting on the queues we'll never re-fill.
  work_item drained;
  while (_work_queue.try_dequeue(drained)) {
    for (auto& entry : drained.entries) {
      if (entry) entry->state.try_cancel_queued();
    }
  }

  eviction_request ereq;
  while (_request_queue.try_dequeue(ereq)) {
    ereq.promise.set_exception(
      std::make_exception_ptr(std::runtime_error("eviction aborted — cache shutting down")));
  }

  // Join.  After joins, no new IO will be dispatched, but the reactor
  // (owned by the ioctx that owns this cache) may still be processing
  // IOs we submitted earlier; their completion callbacks hold a slot
  // in _inflight_budget.
  if (_allocator_thread.joinable()) { _allocator_thread.join(); }
  if (_evictor_thread.joinable()) { _evictor_thread.join(); }

  // Stop the dispatch thread only after the allocator is fully joined, so
  // no more work_items can be pushed to _io_dispatch_queue.  The stop_callback
  // registered in io_dispatch_loop fires notify_all on _io_dispatch_cv, which
  // wakes the thread.  The thread then drains the queue and exits.
  _io_dispatch_thread.request_stop();
  if (_io_dispatch_thread.joinable()) { _io_dispatch_thread.join(); }

  // Wait for every in-flight IO callback to release its admission slot.
  // Each callback captures the raw _pool pointer; the owner of the pool
  // (scan_manager) is required to keep the pool alive until @c
  // sirius_ioctx::shutdown_cache returns — which only happens after
  // this destructor runs — so callbacks observing the pool here are
  // safe.  The owning ioctx is similarly required to stay alive while
  // this destructor is in progress (the scan_manager holds a
  // @c shared_ptr<sirius_ioctx> across @c shutdown_cache).
  //
  // Wrapped in try/catch: a destructor must be noexcept and a hanging
  // wait is preferable to a UAF.
  try {
    _inflight_budget->wait_for_idle();
  } catch (...) {
    // Best-effort drain.
  }

  abort_pending_entries();
}

void prefetching_cache::enqueue_work(work_item item)
{
  _work_queue.enqueue(std::move(item));
  _work_seq.fetch_add(1, std::memory_order_release);
  _work_seq.notify_one();
}

void prefetching_cache::notify_disposed() noexcept
{
  // Wake the evictor so it can immediately reclaim any memory that was
  // pre-allocated (queued/allocated state) for the cancelled request.
  // The evictor checks the alive flag on every eviction queue pop and will
  // free queued/allocated/cached entries for the cancelled wrapper.
  _request_sem.release();
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
        if (!entry) continue;
        auto old_st = entry->state.try_abort_pending();
        if (old_st == entry_state::allocated && _pool) {
          _pool->deallocate_bulk(entry->chunks);
          entry->chunks.clear();
        }
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
// insert
// ===========================================================================

prefetching_handle prefetching_cache::insert(
  sirius_io_object& obj, const std::vector<cudf::io::text::byte_range_info>& ranges)
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

  std::unique_lock map_lk(_map_mtx);
  auto [it, inserted] = _file_cache.try_emplace(key, nullptr);
  if (inserted) it->second = std::make_unique<file_entry>();
  auto& file = *it->second;

  std::unique_lock file_lk(file.mtx);
  map_lk.unlock();

  file.io_obj    = obj_sp;
  file.file_size = file_size;

  if (!_armed) {
    // Unarmed cache: record the file_entry so subsequent reads can find
    // metadata, but skip the per-range cache_entries — there's no pool
    // to allocate chunks from and no worker to dispatch IO.
    return {};
  }

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
      // Coalesce: reuse the existing entry.  Per-entry demand counters
      // are gone (we now count demand per-file via request_state).
      new_entries.push_back(*ex_it);
      merged.push_back(std::move(*ex_it));
      ++ex_it;
    } else {
      auto physical = _io_ctx->compute_physical_range(logical, file_size);
      auto e        = std::make_shared<cache_entry>(logical, physical, _pool->chunk_bytes());
      new_entries.push_back(e);
      merged.push_back(std::move(e));
    }
  }
  // Forward any trailing existing entries.
  for (; ex_it != ex_end; ++ex_it)
    merged.push_back(std::move(*ex_it));

  file.entries = std::move(merged);

  // Register this request with the file's demand counter (stamp+=1,
  // n_pending+=1 saturating).  The new stamp is snapshotted on the
  // eviction-queue wrapper so the evictor can later detect requests
  // overtaken by newer activity on the same file.
  uint64_t const new_stamp  = file.demand.register_request();
  file_demand* const demand = &file.demand;

  file_lk.unlock();

  // Shared flag.  cancel() flips it; the worker checks it on dequeue and
  // before dispatch; the evictor checks it on pop to short-circuit aging
  // on cancelled requests.
  auto alive = std::make_shared<std::atomic<bool>>(true);

  if (new_entries.empty()) {
    // Every range coalesced with an existing entry and no fresh prefetch
    // was scheduled — nothing for the worker to do, but we still hand
    // back a handle so the caller's per-file demand counter decrements
    // when the request is done.
    return prefetching_handle{std::move(alive), demand, this};
  }

  // Enqueue the eviction-queue wrapper BEFORE the work_item.  The wrapper
  // sits in the queue from now until evicted (whether or not IO ever
  // dispatches), so cancellation by handle drop doesn't need a separate
  // enqueue path: the evictor pops, sees n_pending==0 (after handle
  // dtor) or !alive, and drops the entries.
  _eviction_queue.enqueue(prefetch_request{new_entries, demand, new_stamp, alive});

  enqueue_work(prefetch_req{key, std::move(obj_sp), std::move(new_entries), alive});
  return prefetching_handle{std::move(alive), demand, this};
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
  //   cached / in_use   → pin immediately.
  //   allocated/loading → wait for the pipeline to resolve the load, then retry.
  //   queued / evicting / empty → return empty (caller falls back).
  bool waited_on_load = false;
  while (true) {
    auto st = entry->state.get_state();
    if (st == entry_state::cached || st == entry_state::in_use) {
      pinned_view view{entry, stream};
      if (view) {
        _hit_count.fetch_add(1, std::memory_order_relaxed);
        if (waited_on_load) _hit_after_wait.fetch_add(1, std::memory_order_relaxed);
        return view;
      }
      // try_acquire_read lost a race; re-observe the state.
      continue;
    }
    if (st == entry_state::allocated || st == entry_state::loading) {
      waited_on_load = true;
      // Release file_lk across the wait: the entry is pinned by our local
      // shared_ptr, and file.mtx no longer protects anything we touch while
      // parked.  Holding it shared would stall every concurrent insert() on
      // this file (and, under writer-preference shared_mutex, every reader
      // queued behind that insert).
      file_lk.unlock();
      entry->state.wait_while_pending();
      file_lk.lock();
      continue;
    }
    _partial_miss_count.fetch_add(1, std::memory_order_relaxed);
    return {};
  }
}

std::string prefetching_cache::summary() const
{
  auto hits            = _hit_count.load(std::memory_order_relaxed);
  auto hits_after_wait = _hit_after_wait.load(std::memory_order_relaxed);
  auto partial         = _partial_miss_count.load(std::memory_order_relaxed);
  auto full            = _full_miss_count.load(std::memory_order_relaxed);
  auto evicted_entries = _evicted_entries.load(std::memory_order_relaxed);
  auto evicted_chunks  = _evicted_chunks.load(std::memory_order_relaxed);
  auto total           = hits + partial + full;
  auto pct             = [&](uint64_t n) {
    return total > 0 ? (100.0 * static_cast<double>(n) / static_cast<double>(total)) : 0.0;
  };
  auto s = fmt::format(
    "prefetching_cache: {} reads ({} hit {:.1f}% [of which {} waited on "
    "loading], {} partial-miss {:.1f}%, {} full-miss {:.1f}%); evicted "
    "{} entries / {} chunks; pool {}/{} chunks free",
    total,
    hits,
    pct(hits),
    hits_after_wait,
    partial,
    pct(partial),
    full,
    pct(full),
    evicted_entries,
    evicted_chunks,
    _pool ? _pool->free_count() : 0U,
    _pool ? _pool->total_chunks() : 0U);
  spdlog::info("{}", s);
  return s;
}

// ===========================================================================
// evictor_loop — dispose-signal + on-demand eviction (no periodic pressure)
// ===========================================================================

void prefetching_cache::evictor_loop(std::stop_token stop)
{
  std::stop_callback stop_cb(stop, [this] { _request_sem.release(); });

  // Evict a single cached entry (cached → evicting → empty, returns chunks).
  // Returns the number of chunks freed (0 if the entry is not evictable).
  auto try_evict_cached = [this](cache_entry& entry) -> size_t {
    if (entry.state.get_state() != entry_state::cached) return 0;
    if (!entry.state.try_start_evicting()) return 0;
    size_t n = entry.chunks.size();
    if (_pool) { _pool->deallocate_bulk(entry.chunks); }
    entry.state.mark_evicted();
    _evicted_entries.fetch_add(1, std::memory_order_relaxed);
    _evicted_chunks.fetch_add(n, std::memory_order_relaxed);
    return n;
  };

  // Walk a wrapper's entries, evicting everything that can be reclaimed.
  // Handles all pre-data states (queued, allocated) in addition to cached.
  //
  //   queued    → empty         (no chunks to free)
  //   allocated → empty         (return pre-allocated chunks to pool)
  //   cached    → evicting → empty (return loaded chunks to pool)
  //   loading / in_use / evicting / empty → skip (in-flight or already gone)
  //
  // Returns:
  //   .freed:    total chunk count freed
  //   .leftover: entries still pinned (cached + pin_count>0); re-enqueued by caller
  struct walk_result {
    size_t freed{0};
    std::vector<std::shared_ptr<cache_entry>> leftover;
  };
  auto walk_and_evict = [&](prefetch_request& req) -> walk_result {
    walk_result r;
    r.leftover.reserve(req.entries.size());
    for (auto& e : req.entries) {
      if (!e) continue;
      auto st = e->state.get_state();

      if (st == entry_state::queued) {
        // If the CAS loses, the worker just moved the entry to allocated —
        // re-observe and fall through to handle the new state.
        if (e->state.try_cancel_queued()) continue;
        st = e->state.get_state();
      }

      if (st == entry_state::allocated) {
        if (e->state.try_cancel_allocated()) {
          size_t n = e->chunks.size();
          if (_pool) { _pool->deallocate_bulk(e->chunks); }
          r.freed += n;
        }
        continue;
      }

      if (st == entry_state::cached) {
        size_t freed = try_evict_cached(*e);
        if (freed > 0) {
          r.freed += freed;
        } else if (e->state.get_state() == entry_state::cached) {
          // Still cached but pin_count > 0 — reader holds it.  Retry later.
          r.leftover.push_back(std::move(e));
        }
        continue;
      }

      if (st == entry_state::in_use) {
        // Reader is active; keep the entry in leftover so we retry eviction
        // once the pin is released and the state returns to cached.
        r.leftover.push_back(std::move(e));
        continue;
      }
      // loading / evicting / empty: in-flight or already gone; drop.
    }
    return r;
  };

  // Drain the eviction queue, processing all disposed/stale/exhausted wrappers.
  // Called on every wakeup (both dispose signals and allocation requests).
  // Returns the total chunks freed so the allocation-request path can use it.
  auto evict_disposed_entries = [&]() -> size_t {
    size_t reclaimed        = 0;
    size_t const n_to_check = _eviction_queue.size_approx();
    for (size_t i = 0; i < n_to_check; ++i) {
      if (stop.stop_requested()) break;
      prefetch_request wrapper;
      if (!_eviction_queue.try_dequeue(wrapper)) break;

      auto const v         = wrapper.demand->load();
      bool const cancelled = !wrapper.alive->load(std::memory_order_acquire);
      bool const stale     = v.stamp > wrapper.stamp_at_insert + REQUEST_STALE_THRESHOLD;
      bool const exhausted = (v.n_pending == 0);

      if (cancelled || stale || exhausted) {
        auto r = walk_and_evict(wrapper);
        reclaimed += r.freed;
        if (!r.leftover.empty()) {
          wrapper.entries = std::move(r.leftover);
          _eviction_queue.enqueue(std::move(wrapper));
        }
      } else {
        // Live wrapper: age by one (clamped), rotate to tail.
        wrapper.demand->unregister_request();
        _eviction_queue.enqueue(std::move(wrapper));
      }
    }
    return reclaimed;
  };

  // Walk the FIFO until at least @p needed chunks have been reclaimed.
  // Called when the allocator has an outstanding eviction_request.
  auto evict_chunks = [&](size_t needed) -> size_t {
    size_t reclaimed             = 0;
    size_t empty_ticks           = 0;
    size_t const queue_size_hint = std::max<size_t>(_eviction_queue.size_approx(), 1);

    while (reclaimed < needed) {
      if (stop.stop_requested()) break;

      prefetch_request wrapper;
      if (!_eviction_queue.try_dequeue(wrapper)) {
        std::this_thread::sleep_for(EVICTOR_POLL_INTERVAL);
        if (stop.stop_requested()) break;
        if (++empty_ticks > 4) break;
        continue;
      }
      empty_ticks = 0;

      auto const v         = wrapper.demand->load();
      bool const cancelled = !wrapper.alive->load(std::memory_order_acquire);
      bool const stale     = v.stamp > wrapper.stamp_at_insert + REQUEST_STALE_THRESHOLD;
      bool const exhausted = (v.n_pending == 0);

      if (cancelled || stale || exhausted) {
        auto r = walk_and_evict(wrapper);
        reclaimed += r.freed;
        if (!r.leftover.empty()) {
          wrapper.entries = std::move(r.leftover);
          _eviction_queue.enqueue(std::move(wrapper));
        }
      } else {
        wrapper.demand->unregister_request();
        _eviction_queue.enqueue(std::move(wrapper));
        if (reclaimed == 0 && ++empty_ticks > queue_size_hint) {
          std::this_thread::sleep_for(EVICTOR_POLL_INTERVAL);
          if (stop.stop_requested()) break;
          empty_ticks = 0;
        }
      }
    }
    return reclaimed;
  };

  while (!stop.stop_requested()) {
    // Wait for any signal: allocation shortfall OR handle dispose.
    // Both paths release _request_sem.
    bool signalled = _request_sem.try_acquire_for(EVICTOR_POLL_INTERVAL);
    if (stop.stop_requested()) break;
    if (!signalled) continue;  // timeout — no work queued

    // Always scan for disposed entries first so memory is reclaimed
    // immediately after cancel() fires, before fulfilling any allocation
    // request that may have arrived at the same time.
    evict_disposed_entries();

    // Fulfil any pending on-demand allocation request.
    eviction_request chunk_req;
    if (!_request_queue.try_dequeue(chunk_req)) continue;

    size_t const needed    = chunk_req.n_chunks_needed;
    size_t const reclaimed = evict_chunks(needed);

    if (reclaimed >= needed) {
      chunk_req.promise.set_value();
    } else {
      chunk_req.promise.set_exception(std::make_exception_ptr(
        std::runtime_error("eviction aborted — couldn't satisfy chunk request")));
    }
  }

  // Drain any pending chunk requests on shutdown so allocators in fut.get() unblock.
  eviction_request req;
  while (_request_queue.try_dequeue(req)) {
    req.promise.set_exception(
      std::make_exception_ptr(std::runtime_error("eviction aborted — cache shutting down")));
  }
}

// ===========================================================================
// allocator_loop
// ===========================================================================

void prefetching_cache::allocator_loop(std::stop_token stop)
{
  std::stop_callback stop_cb(stop, [this] {
    _work_seq.fetch_add(1, std::memory_order_release);
    _work_seq.notify_one();
  });

  // _pool is a const member baked at construction.  Threads only run
  // when _armed (which implies pool != nullptr), so no per-iteration
  // null checks are needed.
  buffer_pool* const pool = _pool;

  while (!stop.stop_requested()) {
    // Dequeue.  Park on _work_seq when the queue is empty; the dtor
    // bumps _work_seq + notify_all (and the stop_callback above re-bumps
    // it) so a stopped worker observes the token here and exits.
    work_item item;
    if (!_work_queue.try_dequeue(item)) {
      auto seq = _work_seq.load(std::memory_order_acquire);
      if (stop.stop_requested()) break;
      if (!_work_queue.try_dequeue(item)) {
        _work_seq.wait(seq, std::memory_order_relaxed);
        continue;
      }
    }
    if (stop.stop_requested()) {
      for (auto& e : item.entries)
        if (e) e->state.try_cancel_queued();
      break;
    }

    // ---- Cancellation gate -------------------------------------------------
    // The caller-side prefetching_handle may have flipped alive to false
    // between insert() and now (e.g. fadvise(disposable) fired before the
    // worker drained the queue).  alive is never null on a dequeued item:
    // insert() always constructs one before enqueuing.
    if (!item.alive->load(std::memory_order_acquire)) {
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
      auto st       = e->state.get_state();
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
    auto got = pool->allocate_bulk(upper_bound_chunks, ptrs);
    if (got < upper_bound_chunks) {
      // Check stop before enqueueing: the evictor's exit drain runs only
      // for requests already in the queue when it stops.  An enqueue that
      // races past that drain would never be resolved and we'd hang here.
      if (stop.stop_requested()) {
        pool->deallocate_bulk(ptrs);
        for (auto& e : item.entries)
          if (e) e->state.try_cancel_queued();
        break;
      }
      size_t shortfall = upper_bound_chunks - got;
      eviction_request req;
      req.n_chunks_needed = shortfall;
      auto fut            = req.promise.get_future();
      _request_queue.enqueue(std::move(req));
      _request_sem.release();

      try {
        fut.get();
      } catch (...) {
        pool->deallocate_bulk(ptrs);
        continue;
      }

      // After the wait — re-check stop.  If the evictor resolved us
      // mid-shutdown, drop the work cleanly instead of pushing IO into
      // a tearing-down system.
      if (stop.stop_requested()) {
        pool->deallocate_bulk(ptrs);
        for (auto& e : item.entries)
          if (e) e->state.try_cancel_queued();
        break;
      }

      auto extra = pool->allocate_bulk(shortfall, ptrs);
      if (extra < shortfall) {
        pool->deallocate_bulk(ptrs);
        continue;
      }
    }

    // ---- Phase 4: queued → allocated + chunk assignment --------------------
    // Claim each entry (queued → allocated) and assign pre-allocated chunks.
    // IO dispatch (budget gating and loading transition) happens in
    // io_dispatch_loop.  If try_allocate fails, the entry raced past queued;
    // return its pre-allocated chunks and skip.
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

      if (!e->state.try_allocate()) {
        return_chunks();
        continue;
      }

      // Chunks assigned while in allocated state — visible to io_dispatch_loop.
      e->chunks.assign(ptrs.begin() + ptr_idx, ptrs.begin() + ptr_idx + n);
      ptr_idx += n;
      batch.push_back(e);
    }

    if (batch.empty()) {
      // Every entry raced past queued; pre-allocated chunks already returned.
      continue;
    }

    // ---- Hand off allocated entries to io_dispatch_loop -------------------
    work_item dispatch{item.file_key, item.io_obj, std::move(batch), item.alive};
    {
      std::lock_guard lk(_io_dispatch_mtx);
      _io_dispatch_queue.push_back(std::move(dispatch));
    }
    _io_dispatch_cv.notify_one();
  }
}

// ===========================================================================
// io_dispatch_loop
// ===========================================================================

void prefetching_cache::io_dispatch_loop(std::stop_token stop)
{
  std::stop_callback stop_cb(stop, [this] { _io_dispatch_cv.notify_all(); });

  buffer_pool* const pool    = _pool;
  sirius_ioctx* const io_ctx = _io_ctx;

  while (!stop.stop_requested()) {
    // ---- Wait for work from allocator_loop ---------------------------------
    work_item item;
    {
      std::unique_lock lk(_io_dispatch_mtx);
      _io_dispatch_cv.wait(lk,
                           [&] { return !_io_dispatch_queue.empty() || stop.stop_requested(); });
      if (stop.stop_requested()) {
        // Drain any remaining items so allocator_loop doesn't block.
        while (!_io_dispatch_queue.empty()) {
          auto& front = _io_dispatch_queue.front();
          for (auto& e : front.entries) {
            if (!e) continue;
            if (e->state.try_cancel_allocated()) {
              if (pool) { pool->deallocate_bulk(e->chunks); }
            }
          }
          _io_dispatch_queue.pop_front();
        }
        break;
      }
      item = std::move(_io_dispatch_queue.front());
      _io_dispatch_queue.pop_front();
    }

    // ---- Cancellation check ------------------------------------------------
    // The handle may have been cancelled between allocator_loop's push and now.
    // Return chunks for all allocated entries and skip.
    if (!item.alive->load(std::memory_order_acquire)) {
      for (auto& e : item.entries) {
        if (!e) continue;
        if (e->state.try_cancel_allocated()) {
          if (pool) { pool->deallocate_bulk(e->chunks); }
        }
      }
      continue;
    }

    // ---- allocated → loading + build batch ---------------------------------
    // try_start_loading() (allocated → loading) establishes exclusive ownership
    // of the entry's chunks via an acq_rel CAS.  Reading e->chunks BEFORE
    // this CAS would race with the evictor's try_cancel_allocated() which
    // clears the vector.  We therefore count chunks and build the batch in
    // a single pass, reading chunks only after the CAS succeeds.
    // Entries where try_start_loading fails were cancelled by the evictor,
    // which already freed their chunks.
    std::vector<std::shared_ptr<cache_entry>> batch;
    batch.reserve(item.entries.size());
    size_t total_chunks = 0;
    for (auto& e : item.entries) {
      if (!e) continue;
      if (!e->state.try_start_loading()) continue;
      total_chunks += e->chunks.size();  // safe: CAS above gives us sole ownership
      batch.push_back(e);
    }

    if (batch.empty()) {
      // All entries cancelled; nothing to dispatch.
      continue;
    }

    // ---- Phase 3: acquire inflight budget ----------------------------------
    // Budget is sized to the exact chunks we own (not all allocated entries,
    // since some may have been cancelled between allocator_loop and here).
    auto budget_slot = _inflight_budget->acquire(total_chunks, stop);
    if (!budget_slot) {
      // Stop requested while waiting for budget.  Return chunks and exit.
      for (auto& e : batch) {
        pool->deallocate_bulk(e->chunks);
        e->state.try_mark_load_failed();
      }
      break;
    }

    // ---- Phase 5: dispatch IO in sub-batches -------------------------------
    // SUBBATCH at 16 trades dispatch count for HOL latency: a 600-range
    // request becomes ~38 sub-dispatches; first-byte latency for the
    // head reader drops from max(600 IOs) to max(16 IOs).
    constexpr size_t SUBBATCH = 16;

    auto slot_holder = std::make_shared<admission_control::slot>(std::move(budget_slot));
    auto& obj_ref    = *item.io_obj;

    for (size_t sb_start = 0; sb_start < batch.size(); sb_start += SUBBATCH) {
      // Cancellation re-check between sub-batches: a fadvise(disposable)
      // that fires after we dispatched some sub-batches can still cancel
      // the remaining ones.  Already-dispatched callbacks complete normally.
      if (!item.alive->load(std::memory_order_acquire)) {
        for (size_t i = sb_start; i < batch.size(); ++i) {
          pool->deallocate_bulk(batch[i]->chunks);
          batch[i]->state.try_mark_load_failed();
        }
        break;
      }

      size_t const sb_end = std::min(sb_start + SUBBATCH, batch.size());

      std::vector<cudf::io::text::byte_range_info> sb_ranges;
      std::vector<cudf::host_span<std::byte>> sb_dsts;
      for (size_t i = sb_start; i < sb_end; ++i) {
        auto const& e          = batch[i];
        auto phys_off          = static_cast<size_t>(e->physical_range.offset());
        auto phys_size         = static_cast<size_t>(e->physical_range.size());
        auto const chunk_bytes = e->chunk_bytes;
        for (size_t c = 0; c < e->chunks.size(); ++c) {
          auto off = phys_off + c * chunk_bytes;
          auto sz  = std::min(chunk_bytes, phys_size - c * chunk_bytes);
          sb_ranges.emplace_back(static_cast<int64_t>(off), static_cast<int64_t>(sz));
          sb_dsts.emplace_back(e->chunks[c], sz);
        }
      }

      std::vector<std::shared_ptr<cache_entry>> sb_batch(
        batch.begin() + static_cast<std::ptrdiff_t>(sb_start),
        batch.begin() + static_cast<std::ptrdiff_t>(sb_end));

      io_ctx->host_read_ranges_async_io(
        obj_ref,
        sb_ranges,
        sb_dsts,
        [pool,
         batch  = std::move(sb_batch),
         slot   = slot_holder,  // shared across sub-batches
         io_obj = item.io_obj,  // copy, not move — used by later sub-batches
         key    = item.file_key]   // copy, not move
        (size_t /*bytes*/, std::exception_ptr ep) {
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
