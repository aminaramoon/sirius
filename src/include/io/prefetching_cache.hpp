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

#include "io/admission_control.hpp"
#include "io/prefetch_types.hpp"

#include <concurrentqueue.h>

#include <atomic>
#include <condition_variable>
#include <deque>
#include <memory>
#include <semaphore>
#include <shared_mutex>
#include <stop_token>
#include <thread>
#include <unordered_map>
#include <vector>

namespace sirius::io {

class sirius_ioctx;

// Forward declaration needed: prefetching_handle holds a back-pointer to the
// cache so cancel() can signal the evictor directly.
class prefetching_cache;

// ---------------------------------------------------------------------------
// prefetching_handle
// ---------------------------------------------------------------------------

/**
 * @brief Cancellation token returned from @c prefetching_cache::insert.
 *
 * The cache and the caller jointly hold a @c shared_ptr<atomic<bool>>.  The
 * caller flips the flag to false (via @c cancel) when the consumer no longer
 * needs the prefetched data; the cache's pipeline checks the flag before
 * doing work on the corresponding @c work_item and skips when cancelled.
 *
 * When @c cancel is called the handle also signals the cache's evictor so
 * it can immediately reclaim any memory that was already allocated for this
 * request but has not yet been dispatched to IO.
 *
 * Lifetime:
 *   - The handle returned by @c insert outlives the @c work_item it
 *     references (both hold the same @c shared_ptr).  Either side dropping
 *     does not invalidate the other.
 *   - A default-constructed handle is "empty": @c cancel is a no-op and
 *     @c operator bool() returns false.  Returned by @c insert when the
 *     cache is dormant or no new prefetch work was scheduled.
 */
class prefetching_handle {
 public:
  prefetching_handle() = default;

  ~prefetching_handle() { release(); }

  prefetching_handle(prefetching_handle const&)            = delete;
  prefetching_handle& operator=(prefetching_handle const&) = delete;

  prefetching_handle(prefetching_handle&& o) noexcept
    : _alive(std::move(o._alive)), _demand(o._demand), _cache(o._cache)
  {
    o._demand = nullptr;
    o._cache  = nullptr;
  }

  prefetching_handle& operator=(prefetching_handle&& o) noexcept
  {
    if (this != &o) {
      release();
      _alive    = std::move(o._alive);
      _demand   = o._demand;
      _cache    = o._cache;
      o._demand = nullptr;
      o._cache  = nullptr;
    }
    return *this;
  }

  /// Mark the prefetch as no longer wanted.  Idempotent; safe on an empty
  /// handle.  Thread-safe with respect to the pipeline's cancellation checks:
  /// concurrent calls to @c cancel() on the same handle from multiple threads
  /// are safe (the alive flag is an atomic).  However, @c cancel() is NOT safe
  /// concurrent with move-construction or move-assignment on the same handle
  /// instance — handles are scoped per-caller and are not moved while in use.
  /// Flips the alive flag and signals the cache's evictor so it can reclaim
  /// any pre-allocated memory for this request immediately.
  /// Note: the per-file n_request counter is decremented on destruction (or
  /// move-out), not on cancel.
  void cancel() noexcept;

  /// True iff this handle is bound to a real prefetch request (i.e. came
  /// from an @c insert that scheduled work).
  explicit operator bool() const noexcept { return _alive != nullptr; }

 private:
  friend class prefetching_cache;
  prefetching_handle(std::shared_ptr<std::atomic<bool>> alive,
                     file_demand* demand,
                     prefetching_cache* cache) noexcept
    : _alive(std::move(alive)), _demand(demand), _cache(cache)
  {
  }

  /// Drops this handle's contribution to the file's @c file_demand
  /// counter via @c unregister_request (clamped).  Idempotent: clears
  /// @c _demand after the call so repeat release()s (move-out, then
  /// dtor) are no-ops.
  void release() noexcept;

  /// Shared with the corresponding @c work_item / @c prefetch_request.
  /// true == still wanted; false == cancelled, worker / evictor should drop.
  std::shared_ptr<std::atomic<bool>> _alive;

  /// Non-owning pointer into the source @c file_entry's @c file_demand.
  /// The cache keeps file_entries alive for its lifetime; handles never
  /// outlive the cache, so this raw pointer is stable.
  file_demand* _demand{nullptr};

  /// Non-owning back-pointer to the owning cache.  Used by @c cancel() to
  /// signal the evictor.  The cache outlives all handles: handles are vended
  /// by the cache, and callers must not retain them past cache destruction
  /// (i.e. past @c sirius_ioctx::shutdown_cache()).  @c cancel() clears this
  /// to nullptr after signalling; @c release() clears it unconditionally on
  /// destruction or move-out.
  prefetching_cache* _cache{nullptr};
};

// ---------------------------------------------------------------------------
// prefetching_cache
// ---------------------------------------------------------------------------
//
// Locking hierarchy:
//   Level 0: _map_mtx          — protects _file_cache map
//   Level 1: file_entry::mtx   — protects one file's entry vector
//   (independent): cache_entry atomics — lock-free

class prefetching_cache {
  // The cache only accepts new prefetch requests through
  // sirius_datasource::fadvise — that's the single entry point for the
  // fadvise(speculative/immediate/disposable) protocol.  Friending the
  // datasource keeps insert() out of the public API while still letting
  // fadvise dispatch through it.
  friend class sirius_datasource;
  // prefetching_handle calls notify_disposed() on cancel — needs access to
  // the private method.
  friend class prefetching_handle;

 public:
  /// Construct the cache with all of its dependencies wired up at
  /// construction time.  The cache is "armed" — workers start, fadvise
  /// inserts schedule prefetch — iff:
  ///   - @p pool is non-null (someone owns a buffer_pool we can allocate
  ///     from), and
  ///   - @p io_ctx is non-null AND reports
  ///     @c supports_vector_host_read() (the backend can serve the
  ///     batched IO the prefetcher dispatches), and
  ///   - @p inflight_budget_chunks > 0.
  /// Otherwise the cache is unarmed: no threads start and @c insert
  /// returns an empty handle.  Per-file metadata caching is handled by
  /// the @c metadata_store owned by the ioctx and is independent of
  /// this class.
  ///
  /// Ownership:
  ///   - @p pool is non-owning; the caller (typically @c scan_manager)
  ///     guarantees it outlives this cache.
  ///   - @p io_ctx is the back-pointer to the owning ioctx; held as a
  ///     raw pointer because the ioctx now owns this cache, so it is
  ///     guaranteed to outlive it.
  prefetching_cache(buffer_pool* pool, sirius_ioctx* io_ctx, size_t inflight_budget_chunks);
  ~prefetching_cache();

  prefetching_cache(prefetching_cache const&)            = delete;
  prefetching_cache& operator=(prefetching_cache const&) = delete;

  /// @return whether prefetch dispatch is wired up (pool + supporting
  /// ioctx + non-zero budget).
  [[nodiscard]] bool is_armed() const noexcept { return _armed; }

  /// Non-blocking read of a single range.
  /// Returns an empty pinned_view if the range is not cached or the cached
  /// entry does not fully cover [offset, offset+size).  Updates hit / miss
  /// counters (see summary()).  @p stream, when non-null, defers the
  /// returned pinned_view's read release until the stream reaches the
  /// release point — see @c pinned_view's ctor for the full contract.
  ///
  /// @p out_buffer — when non-null and the entry is in the @c allocated
  /// state, @c read() attempts to flip the entry to @c loading via a CAS
  /// and, on success, constructs a @c cached_host_buffer that wraps the
  /// entry's pre-allocated chunks.  The caller owns this buffer and must
  /// pass it to @c sirius_ioctx::device_read_async_io_using() to issue
  /// file → bounce → device IO.  If the CAS fails (evictor raced), or if
  /// @p out_buffer is null, the entry is treated as a miss.
  [[nodiscard]] pinned_view read(const sirius_io_object& obj,
                                 size_t offset,
                                 size_t size,
                                 cudaStream_t stream            = nullptr,
                                 cached_host_buffer* out_buffer = nullptr);

  /// One-line human-readable state: hit / partial-miss / full-miss counts,
  /// pool utilisation, and pending chunks.
  [[nodiscard]] std::string summary() const;

 private:
  /// Register ranges for a file and trigger background prefetch.  Private
  /// entry point: callers reach this through @c sirius_datasource::fadvise,
  /// which is the only friend.  See @c prefetching_handle for the
  /// cancellation contract.
  ///
  /// Ranges must be sorted by offset.  The cache retains @p obj via
  /// @c shared_from_this() so the underlying file handles stay open
  /// until all pending prefetch work for this file has completed.
  /// @p obj must already be owned by a @c std::shared_ptr.
  ///
  /// Returns an empty handle when the cache is dormant or no new prefetch
  /// work was scheduled; the file_entry is still recorded either way.
  [[nodiscard]] prefetching_handle insert(
    sirius_io_object& obj, const std::vector<cudf::io::text::byte_range_info>& ranges);

  // ---- work items dispatched through the queue ------------------------------

  struct prefetch_req {
    std::string file_key;
    /// shared_ptr keeps the io_object (and thus its file handles) alive
    /// until the worker has issued IO for this request.
    std::shared_ptr<sirius_io_object> io_obj;
    std::vector<std::shared_ptr<cache_entry>> entries;
    /// Shared with the @c prefetching_handle returned to the caller.  The
    /// caller flips this to false when the consumer no longer wants the
    /// data; the worker checks it on dequeue (and again before dispatch)
    /// and drops the request when stale.  Never null — @c insert always
    /// constructs a flag when it enqueues work.
    std::shared_ptr<std::atomic<bool>> alive;
  };
  using work_item = prefetch_req;

  // ---- per-file state -------------------------------------------------------

  struct file_entry {
    mutable std::shared_mutex mtx;  ///< Level-1 lock.
    /// shared_ptr so the file stays open as long as this cache keeps
    /// entries for it — outliving the caller's datasource if necessary.
    std::shared_ptr<sirius_io_object> io_obj;
    size_t file_size{0};
    std::vector<std::shared_ptr<cache_entry>> entries;  ///< Sorted by offset.

    /// Per-file demand counter + insert stamp.  See @c file_demand for
    /// the lifecycle (insert registers, handle dtor unregisters,
    /// evictor unregisters as an aging tick).
    file_demand demand;
  };

  /// Per-file "this wrapper has been overtaken by K newer inserts on the
  /// same file" cutoff.  Bigger means slower aging by staleness; the
  /// demand counter aging still drives eviction in the steady state.
  static constexpr uint64_t REQUEST_STALE_THRESHOLD = 8;

  // ---- helpers --------------------------------------------------------------

  void allocator_loop(std::stop_token stop);
  void io_dispatch_loop(std::stop_token stop);
  void evictor_loop(std::stop_token stop);
  void enqueue_work(work_item item);
  void abort_pending_entries() noexcept;

  /// Called by @c prefetching_handle::cancel() when the caller signals it no
  /// longer needs the prefetched data.  Wakes the evictor so it can
  /// immediately reclaim any pre-allocated memory for the cancelled request.
  void notify_disposed() noexcept;

  /// Release all chunks held by an entry back to the pool.
  void release_chunks(cache_entry& entry);

  /// Binary search for an entry whose logical range fully covers
  /// [offset, offset+size).  Returns nullptr on miss.  Updates
  /// _miss_range / _full_miss_count based on the classification.
  /// The caller updates _hit_count on successful pin.
  std::shared_ptr<cache_entry> find_entry(const std::vector<std::shared_ptr<cache_entry>>& entries,
                                          size_t offset,
                                          size_t size);

  // ---- prefetch_request: per-insert wrapper carried in _eviction_queue ----
  //
  // One per insert() call.  Bundles the entries the request produced
  // plus the metadata the evictor uses for FIFO + per-file aging:
  //   - @c file_request_state: non-owning pointer to the source
  //     @c file_entry's packed (stamp:48 | n:16) atomic.  The cache
  //     keeps file_entries alive for its lifetime; wrappers never
  //     outlive the cache, so this raw pointer is safe.
  //   - @c stamp_at_insert: snapshot of the file's stamp at insert
  //     time.  The evictor compares against the file's current stamp
  //     to detect requests that have been overtaken by newer activity
  //     on the same file (per-file generation staleness).
  //   - @c alive: shared with the corresponding @c work_item and the
  //     caller-side @c prefetching_handle.  Flipped to false by
  //     @c handle::cancel() (= fadvise(disposable)).
  struct prefetch_request {
    std::vector<std::shared_ptr<cache_entry>> entries;
    file_demand* demand{nullptr};
    uint64_t stamp_at_insert{0};
    std::shared_ptr<std::atomic<bool>> alive;
  };

  // ---- members (destruction order matters: worker joined first) -------------

  /// Non-owning pool pointer baked in at construction.  Owner (scan_manager)
  /// guarantees the pool outlives this cache.  Null on the unarmed path.
  buffer_pool* const _pool;

  /// IO context back-pointer.  The ioctx owns this cache, so the
  /// pointer is guaranteed valid for the cache's full lifetime; no
  /// shared_ptr required.
  sirius_ioctx* const _io_ctx;

  /// True when the prefetch machinery is wired up: pool + supporting ioctx +
  /// non-zero budget.  Const after construction; threads, queues, and the
  /// admission_control are all only meaningful when this is true.
  bool const _armed;

  /// Set true by the dtor BEFORE request_stop on any thread.  The allocator
  /// checks this just before enqueueing an eviction_request, so during
  /// shutdown the request queue stays empty.  The evictor checks it inside
  /// evict_chunks so a late-pushed request (the microsecond race between
  /// the allocator's flag check and its enqueue) resolves with a fast
  /// set_exception instead of stalling shutdown by looping over live
  /// wrappers.  The dtor must keep the evictor alive across join(allocator)
  /// so this resolution actually happens.
  std::atomic<bool> _shutting_down{false};

  // Observability counters (updated on every read() entry).
  std::atomic<uint64_t> _hit_count{0};
  // Hits that had to wait on a loading entry (subset of _hit_count).
  std::atomic<uint64_t> _hit_after_wait{0};

  // Entries in the allocated state that read() successfully flipped to
  // loading for a direct device read (caller received a cached_host_buffer).
  // Distinct from the miss counts: the caller did get the entry's chunks.
  std::atomic<uint64_t> _allocated_steal_count{0};

  // ---- Miss breakdown ------------------------------------------------------
  // Total misses = _miss_range + _miss_state_allocated_no_steal +
  //                _miss_state_steal_cas_lost + _miss_state_other +
  //                _full_miss_count.
  //
  // _miss_range: find_entry() found an entry whose offset <= read_offset but
  //   whose end < read_offset + read_size — i.e. range mismatch.  Indicates
  //   the registered fadvise range doesn't fully cover what the caller
  //   actually reads (e.g. cudf reads past the registered column-chunk data
  //   extent, or a read spans two adjacent-but-not-coalesced entries).
  std::atomic<uint64_t> _miss_range{0};
  // _miss_state_allocated_no_steal: find_entry covered, state was
  //   `allocated`, but the caller did not pass an out_buffer (host read) so
  //   the steal path was skipped.  Read fell through to a miss instead of
  //   waiting for the imminent load.
  std::atomic<uint64_t> _miss_state_allocated_no_steal{0};
  // _miss_state_steal_cas_lost: state was `allocated` and out_buffer was
  //   non-null but the try_start_loading CAS lost (evictor or another
  //   reader beat us).
  std::atomic<uint64_t> _miss_state_steal_cas_lost{0};
  // _miss_state_evicting: find_entry covered, state was `evicting` — the
  //   evictor has the entry and is freeing its chunks now.  Definitive
  //   eviction signal.
  std::atomic<uint64_t> _miss_state_evicting{0};
  // _miss_state_empty_never_allocated: find_entry covered, state was
  //   `empty`, and the entry was NEVER allocated (allocator hasn't
  //   reached it yet).  Indicates allocation is falling behind read
  //   consumption.
  std::atomic<uint64_t> _miss_state_empty_never_allocated{0};
  // _miss_state_empty_post_evict: find_entry covered, state was `empty`,
  //   entry was allocated AND successfully cached at some point, then
  //   evicted.  Indicates eviction churn — entry was loaded then dropped
  //   while still being read.
  std::atomic<uint64_t> _miss_state_empty_post_evict{0};
  // _miss_state_empty_load_failed: find_entry covered, state was `empty`,
  //   entry was allocated but never reached `cached`.  IO presumably
  //   failed (mark_load_failed reset state to empty).
  std::atomic<uint64_t> _miss_state_empty_load_failed{0};
  // _full_miss_count: file not in cache map, or no entry overlaps the read
  //   range at all.
  std::atomic<uint64_t> _full_miss_count{0};

  // Cumulative evictions since construction: entries whose chunks were
  // released back to the pool (+ total chunks freed).
  std::atomic<uint64_t> _evicted_entries{0};
  std::atomic<uint64_t> _evicted_chunks{0};

  mutable std::shared_mutex _map_mtx;
  std::unordered_map<std::string, std::unique_ptr<file_entry>> _file_cache;

  /// FIFO of @c prefetch_request wrappers.  Each insert() enqueues one
  /// wrapper here; the evictor pops, ages (or evicts), and rotates back
  /// to the tail.  See @c evictor_loop for the policy.
  duckdb_moodycamel::ConcurrentQueue<prefetch_request> _eviction_queue;

  /// Chunk-request backpressure channel: worker enqueues here when the
  /// pool is short, evictor wakes via the semaphore.
  duckdb_moodycamel::ConcurrentQueue<eviction_request> _request_queue;
  std::counting_semaphore<> _request_sem{0};
  static constexpr auto EVICTOR_POLL_INTERVAL = std::chrono::milliseconds(50);

  /// IO in-flight budget: units == chunks.  Worker acquires a slot sized
  /// to the batch before dispatching IO; the slot is carried into the
  /// completion callback and releases on destruction.  Constructed
  /// when @c _armed (with the budget passed to the ctor) and null
  /// otherwise.  Const after construction; the cache is single-life.
  std::unique_ptr<admission_control> const _inflight_budget;

  duckdb_moodycamel::ConcurrentQueue<work_item> _work_queue;
  std::atomic<uint64_t> _work_seq{0};

  /// Queue of allocated work items waiting for IO dispatch.  Produced by
  /// allocator_loop after chunks are assigned; consumed by io_dispatch_loop.
  std::deque<work_item> _io_dispatch_queue;
  std::mutex _io_dispatch_mtx;
  std::condition_variable _io_dispatch_cv;

  std::jthread _evictor_thread;
  std::jthread _allocator_thread;
  std::jthread _io_dispatch_thread;
};

}  // namespace sirius::io
