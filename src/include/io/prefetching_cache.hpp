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
#include "io/types.hpp"

#include <cudf/io/datasource.hpp>

#include <cuda_runtime.h>

#include <concurrentqueue.h>
#include <cucascade/memory/fixed_size_host_memory_resource.hpp>

#include <atomic>
#include <condition_variable>
#include <deque>
#include <future>
#include <memory>
#include <mutex>
#include <semaphore>
#include <shared_mutex>
#include <span>
#include <stop_token>
#include <thread>
#include <unordered_map>
#include <vector>

namespace sirius::io {

class sirius_ioctx;

// ---------------------------------------------------------------------------
// file_demand — per-file packed (stamp:48 | n_pending:16) atomic
// ---------------------------------------------------------------------------
//
// The prefetching cache tracks "how many active prefetch_handles are
// outstanding for this file" (n_pending) and "how many prefetch requests
// have ever been registered for this file" (stamp).  Both live in a
// single 64-bit atomic so insert() can bump both with one CAS.
//
// Lifecycle:
//   - insert()                        → register_request()    (stamp+=1, n+=1)
//   - prefetching_handle dtor         → unregister_request()  (clamped n−=1)
//   - evictor pop of a live wrapper   → unregister_request()  (aging)
//
// The evictor reads load() to decide whether a queued prefetch_request
// is still wanted (n_pending > 0) and whether it's been overtaken by
// newer activity (stamp_now > stamp_at_insert + STALE_THRESHOLD).

class file_demand {
 public:
  /// Snapshot view returned by @c load.
  struct view {
    uint64_t stamp;
    uint16_t n_pending;
  };

  /// Bump stamp+=1 and n_pending+=1 (saturating at the 16-bit max).
  /// Returns the new stamp so the caller can snapshot it on the
  /// corresponding prefetch_request wrapper.
  uint64_t register_request() noexcept;

  /// Clamped CAS-decrement of n_pending.  No-op when n_pending is
  /// already 0.  Used by both handle dtor (consumer-done signal) and
  /// the evictor (aging signal).
  void unregister_request() noexcept;

  /// Atomically read both fields.
  [[nodiscard]] view load(std::memory_order ord = std::memory_order_acquire) const noexcept;

 private:
  static constexpr uint64_t N_BITS      = 16;
  static constexpr uint64_t N_MASK      = (1ULL << N_BITS) - 1;
  static constexpr uint64_t STAMP_SHIFT = N_BITS;

  static constexpr uint64_t pack(uint64_t stamp, uint16_t n) noexcept
  {
    return (stamp << STAMP_SHIFT) | n;
  }
  static constexpr view unpack(uint64_t v) noexcept
  {
    return {v >> STAMP_SHIFT, static_cast<uint16_t>(v & N_MASK)};
  }

  std::atomic<uint64_t> _packed{0};
};

// ---------------------------------------------------------------------------
// buffer_pool — growable pool of pinned chunks
// ---------------------------------------------------------------------------
//
// Backed by a @c cucascade::memory::fixed_size_host_memory_resource.  Each
// grow step requests CHUNKS_PER_SLAB blocks from the upstream resource and
// appends the raw pointers to an internal free list.  Blocks are never
// returned to the upstream resource until the pool is destroyed; allocate()
// pops from the free list and deallocate() pushes back.
//
// The chunk size is taken from @c mr.get_block_size() — all cache layout
// arithmetic that needs the chunk size reads it from @c chunk_bytes().

class buffer_pool {
 public:
  static constexpr uint32_t CHUNKS_PER_SLAB = 500;  // 500 chunks per slab

  /// @p initial_slabs slabs are allocated up-front from @p mr (clamped to
  /// @p max_slabs).  Default preserves the historical behaviour of warming
  /// the pool with up to 10 slabs at construction.
  buffer_pool(cucascade::memory::fixed_size_host_memory_resource& mr,
              uint32_t max_slabs,
              uint32_t initial_slabs = 10);
  ~buffer_pool();

  buffer_pool(buffer_pool const&)            = delete;
  buffer_pool& operator=(buffer_pool const&) = delete;

  /// Allocate a single chunk.  Returns nullptr when the pool is exhausted
  /// and the upstream resource cannot supply a fresh slab.
  std::byte* allocate();

  /// Bulk-allocate up to @p n chunks, appending pointers to @p out.
  /// Returns the number actually allocated (may be < n if the pool is
  /// exhausted and cannot grow).
  size_t allocate_bulk(size_t n, std::vector<std::byte*>& out);

  void deallocate_bulk(std::vector<std::byte*>& out);

  /// Return a chunk to the pool.
  void deallocate(std::byte* p);

  /// Restore the free list to "all chunks free".  Caller must guarantee
  /// every previously-handed-out chunk is no longer in use — the pool
  /// rebuilds its free list from the slabs it already holds without
  /// touching the upstream resource.  Slabs themselves are retained, so
  /// no allocation happens; subsequent allocate() calls reuse them.
  void reclaim_all();

  [[nodiscard]] size_t chunk_bytes() const noexcept { return _chunk_bytes; }
  [[nodiscard]] size_t slab_bytes() const noexcept
  {
    return static_cast<size_t>(CHUNKS_PER_SLAB) * _chunk_bytes;
  }
  [[nodiscard]] size_t capacity() const noexcept
  {
    return static_cast<size_t>(_total_chunks.load(std::memory_order_relaxed)) * _chunk_bytes;
  }
  [[nodiscard]] uint32_t free_count() const noexcept
  {
    return _total_free.load(std::memory_order_relaxed);
  }
  [[nodiscard]] uint32_t total_chunks() const noexcept
  {
    return _total_chunks.load(std::memory_order_relaxed);
  }
  /// Hard cap on chunks the pool can grow to (i.e. @c max_slabs ×
  /// CHUNKS_PER_SLAB).  Stable for the pool's lifetime.  Used by the
  /// prefetching_cache evictor to score pool pressure against the
  /// configured ceiling rather than the (lazily-grown) current size.
  [[nodiscard]] uint32_t max_chunks() const noexcept { return _max_slabs * CHUNKS_PER_SLAB; }

 private:
  /// Pull one slab worth of blocks from the upstream resource and append
  /// them to @c _free_list.  Caller must hold @c _mtx.
  bool grow_locked();

  cucascade::memory::fixed_size_host_memory_resource& _mr;
  size_t _chunk_bytes;
  uint32_t _max_slabs;

  // Protects _allocations and _free_list.
  std::mutex _mtx;
  // Held to keep upstream blocks alive for the lifetime of the pool —
  // the multiple_blocks_allocation destructor is what returns blocks to
  // the resource, so we never drop these until the pool is destroyed.
  std::vector<cucascade::memory::fixed_multiple_blocks_allocation> _allocations;
  std::vector<std::byte*> _free_list;

  std::atomic<uint32_t> _total_free{0};
  std::atomic<uint32_t> _total_chunks{0};
};

// ---------------------------------------------------------------------------
// entry_state — packed atomic state + pin_count
// ---------------------------------------------------------------------------
//
// Packs a 4-bit state enum and a 28-bit reader pin count into a single
// atomic uint32_t.  Every transition is a single CAS (or store), which
// eliminates the TOCTOU race between checking state and modifying pin_count.
//
// State machine (two-stage prefetch pipeline):
//
//   try_start_queueing()   try_allocate()    try_start_loading()
//   empty ───────────► queued ──────────► allocated ──────────► loading
//     ▲                  │                   │                     │
//     │                  │ try_cancel_       │ try_cancel_         │ try_mark_cached()
//     │                  │ queued()          │ allocated()         ▼
//     │                  ▼                   ▼              cached ◄────────────────
//     │                empty              empty               ▲    │
//     │                                                       │    │ try_start_evicting()
//     │          release_read() (last reader)                 │    ▼
//     │ mark_evicted() ◄─────────────────── in_use ◄──────── ─ evicting
//     │                                  (pin_count ≥ 1)
//     └── mark_load_failed() / try_mark_load_failed()  (loading → empty)
//
// Stage 1 (allocator_loop): queued → allocated  (chunks assigned)
// Stage 2 (io_dispatch_loop): allocated → loading → cached/empty

class entry_state {
 public:
  enum value : uint8_t {
    empty     = 0,
    queued    = 1,
    loading   = 2,
    cached    = 3,
    in_use    = 4,
    evicting  = 5,
    allocated = 6,  ///< chunks assigned, IO not yet dispatched
  };

  entry_state() noexcept = default;

  [[nodiscard]] value get_state() const noexcept
  {
    return unpack_state(_packed.load(std::memory_order_acquire));
  }

  [[nodiscard]] uint32_t get_pin_count() const noexcept
  {
    return unpack_pins(_packed.load(std::memory_order_acquire));
  }

  /// empty → queued.  Returns false if not empty.
  /// Called by insert() to claim responsibility for scheduling a load.
  bool try_start_queueing() noexcept
  {
    auto expected = pack(empty, 0);
    return _packed.compare_exchange_strong(expected, pack(queued, 0), std::memory_order_acq_rel);
  }

  /// queued → allocated.  Returns false if not queued.
  /// Called by allocator_loop after chunks have been assigned to the entry.
  bool try_allocate() noexcept
  {
    auto expected = pack(queued, 0);
    return _packed.compare_exchange_strong(expected, pack(allocated, 0), std::memory_order_acq_rel);
  }

  /// allocated → empty.  Returns false if not allocated.
  /// Caller must return the entry's chunks to the pool before or after this
  /// call (state=empty means the entry is invisible to new readers, so the
  /// chunk vector is exclusively owned by the caller at that point).
  /// Notifies any threads parked in wait_while_pending().
  bool try_cancel_allocated() noexcept
  {
    auto expected = pack(allocated, 0);
    bool ok = _packed.compare_exchange_strong(expected, pack(empty, 0), std::memory_order_acq_rel);
    if (ok) { _packed.notify_all(); }
    return ok;
  }

  /// allocated → loading.  Returns false if not in allocated state.
  /// Called by io_dispatch_loop when it takes ownership of an entry for IO.
  bool try_start_loading() noexcept
  {
    auto expected = pack(allocated, 0);
    return _packed.compare_exchange_strong(expected, pack(loading, 0), std::memory_order_acq_rel);
  }

  /// queued → empty.  Returns false if not queued.
  /// Used by refresh_cache to drop stale pending prefetches and by the
  /// worker's abort path to release orphaned entries.
  bool try_cancel_queued() noexcept
  {
    auto expected = pack(queued, 0);
    return _packed.compare_exchange_strong(expected, pack(empty, 0), std::memory_order_acq_rel);
  }

  /// loading → cached.  Caller must ensure state is loading.
  /// Wakes any readers parked in @c wait_while_pending().
  void mark_cached() noexcept
  {
    _packed.store(pack(cached, 0), std::memory_order_release);
    _packed.notify_all();
  }

  /// loading → cached using CAS.  Returns false if shutdown or another
  /// completion path already moved the entry out of loading.
  bool try_mark_cached() noexcept
  {
    auto expected = pack(loading, 0);
    bool ok = _packed.compare_exchange_strong(expected, pack(cached, 0), std::memory_order_acq_rel);
    if (ok) { _packed.notify_all(); }
    return ok;
  }

  /// loading → empty.  IO failed, chunks already freed by caller.
  /// Wakes any readers parked in @c wait_while_pending().
  void mark_load_failed() noexcept
  {
    _packed.store(pack(empty, 0), std::memory_order_release);
    _packed.notify_all();
  }

  /// loading → empty using CAS.  Returns false if the entry was already
  /// resolved or aborted.
  bool try_mark_load_failed() noexcept
  {
    auto expected = pack(loading, 0);
    bool ok = _packed.compare_exchange_strong(expected, pack(empty, 0), std::memory_order_acq_rel);
    if (ok) { _packed.notify_all(); }
    return ok;
  }

  /// (queued | allocated | loading) → empty.  Used during cache shutdown to
  /// wake readers parked on a load that may never complete.
  ///
  /// Returns the state the entry was in before the transition, or @c empty
  /// if no transition happened (state was not in the handled set).  The
  /// caller must check the return value: if it equals @c allocated, the
  /// caller is responsible for returning the entry's chunks to the pool
  /// (entry_state has no pool reference).
  value try_abort_pending() noexcept
  {
    uint32_t cur = _packed.load(std::memory_order_acquire);
    while (true) {
      auto st = unpack_state(cur);
      if (st != queued && st != allocated && st != loading) return empty;
      if (_packed.compare_exchange_weak(
            cur, pack(empty, 0), std::memory_order_acq_rel, std::memory_order_acquire)) {
        _packed.notify_all();
        return st;
      }
    }
  }

  /// Block while state is @c allocated or @c loading.  Returns when the
  /// state transitions to @c cached (success), @c empty (IO failed or
  /// cancelled), or any other terminal state.
  void wait_while_pending() noexcept
  {
    uint32_t cur = _packed.load(std::memory_order_acquire);
    while (true) {
      auto st = unpack_state(cur);
      if (st != allocated && st != loading) break;
      _packed.wait(cur, std::memory_order_relaxed);
      cur = _packed.load(std::memory_order_acquire);
    }
  }

  /// (cached | in_use) → in_use with pin_count+1.
  /// Returns false if the entry is not in a readable state.
  bool try_acquire_read() noexcept
  {
    uint32_t cur = _packed.load(std::memory_order_acquire);
    while (true) {
      auto st = unpack_state(cur);
      if (st != cached && st != in_use) return false;
      auto pins = unpack_pins(cur);
      auto next = pack(in_use, pins + 1);
      if (_packed.compare_exchange_weak(
            cur, next, std::memory_order_acq_rel, std::memory_order_acquire))
        return true;
    }
  }

  /// Decrement pin_count.  If it reaches 0, transition in_use → cached.
  /// Returns true if this was the last reader.
  bool release_read() noexcept
  {
    uint32_t cur = _packed.load(std::memory_order_acquire);
    assert(unpack_state(cur) == in_use && unpack_pins(cur) > 0);
    while (true) {
      auto pins      = unpack_pins(cur);
      auto new_pins  = pins - 1;
      auto new_state = new_pins == 0 ? cached : in_use;
      auto next      = pack(new_state, new_pins);
      if (_packed.compare_exchange_weak(
            cur, next, std::memory_order_acq_rel, std::memory_order_acquire))
        return new_pins == 0;
    }
  }

  /// cached (pin_count==0) → evicting.
  /// Returns false if state != cached or readers are present.
  bool try_start_evicting() noexcept
  {
    auto expected = pack(cached, 0);
    return _packed.compare_exchange_strong(expected, pack(evicting, 0), std::memory_order_acq_rel);
  }

  /// evicting → empty.  Caller must ensure state is evicting.
  void mark_evicted() noexcept { _packed.store(pack(empty, 0), std::memory_order_release); }

 private:
  static constexpr uint32_t STATE_BITS = 4;
  static constexpr uint32_t STATE_MASK = (1U << STATE_BITS) - 1;
  static constexpr uint32_t PIN_SHIFT  = STATE_BITS;

  static constexpr uint32_t pack(value s, uint32_t pins) noexcept
  {
    return static_cast<uint32_t>(s) | (pins << PIN_SHIFT);
  }
  static constexpr value unpack_state(uint32_t v) noexcept
  {
    return static_cast<value>(v & STATE_MASK);
  }
  static constexpr uint32_t unpack_pins(uint32_t v) noexcept { return v >> PIN_SHIFT; }

  std::atomic<uint32_t> _packed{pack(empty, 0)};
};

// ---------------------------------------------------------------------------
// cache_entry — per-range metadata
// ---------------------------------------------------------------------------
//
// State transitions are managed by the entry_state class above.
// See entry_state's state machine diagram for the full picture.

struct alignas(64) cache_entry {
  cudf::io::text::byte_range_info logical_range;
  cudf::io::text::byte_range_info physical_range;

  /// Size of each chunk in @c chunks (== buffer_pool::chunk_bytes() at the
  /// time of entry creation).  Stored on the entry so pinned_view doesn't
  /// need a pool reference to do chunk-index arithmetic.
  size_t chunk_bytes{0};

  /// Pointers to pinned chunks (each of size @c chunk_bytes) from buffer_pool
  /// backing this range.
  std::vector<std::byte*> chunks;

  /// Packed state + pin_count.  All state transitions go through this.
  entry_state state;

  cache_entry(cudf::io::text::byte_range_info logical,
              cudf::io::text::byte_range_info physical,
              size_t chunk_bytes)
    : logical_range(logical), physical_range(physical), chunk_bytes(chunk_bytes)
  {
  }

  cache_entry(cache_entry const&)            = delete;
  cache_entry& operator=(cache_entry const&) = delete;
};

// ---------------------------------------------------------------------------
// eviction request
// ---------------------------------------------------------------------------

struct eviction_request {
  /// Promise resolves when the evictor has freed at least @c n_chunks_needed
  /// back to the buffer pool.  The worker then retries @c pool.allocate_bulk
  /// to grab them.
  std::promise<void> promise;
  size_t n_chunks_needed;
};

// ---------------------------------------------------------------------------
// pinned_view — RAII read guard with per-chunk access
// ---------------------------------------------------------------------------
//
// Acquires a read pin on the cache_entry on construction (via
// entry_state::try_acquire_read — a single atomic CAS that transitions
// cached → in_use with pin_count+1).  Releases on destruction (via
// entry_state::release_read).
//
// Because chunks are scattered in memory there is NO single contiguous
// span.  Instead the view exposes individual chunk spans via operator[].

class pinned_view {
 public:
  pinned_view() = default;
  /// @p stream is the caller's CUDA stream.  When non-null, the read pin
  /// is released via a host callback enqueued on this stream, so the entry
  /// stays in_use (and therefore non-evictable) until any cudaMemcpyAsync
  /// the caller submitted earlier has finished consuming the chunks.  When
  /// null, the read is released synchronously on destruction (host-only
  /// reads have no async work to wait on).
  pinned_view(std::shared_ptr<cache_entry> entry, cudaStream_t stream);
  ~pinned_view();

  pinned_view(pinned_view&& o) noexcept;
  pinned_view& operator=(pinned_view&& o) noexcept;

  pinned_view(pinned_view const&)            = delete;
  pinned_view& operator=(pinned_view const&) = delete;

  /// Number of 1MB chunks backing this range.
  [[nodiscard]] size_t num_chunks() const noexcept;

  /// Access chunk @p i (physical data). Full CHUNK_BYTES except
  /// possibly the last chunk which may be shorter.
  [[nodiscard]] std::span<const std::byte> operator[](size_t i) const noexcept;

  /// Logical range this view covers.
  [[nodiscard]] cudf::io::text::byte_range_info logical_range() const noexcept;

  /// Physical (O_DIRECT aligned) range.
  [[nodiscard]] cudf::io::text::byte_range_info physical_range() const noexcept;

  /// Logical size (what the user actually requested).
  [[nodiscard]] size_t size() const noexcept;

  /// Slice the cached data at logical [offset, offset+size) into a vector of
  /// non_owning_buffers, one per chunk boundary crossed.  The caller must
  /// ensure [offset, offset+size) lies within the entry's logical range.
  [[nodiscard]] std::vector<cudf::io::datasource::non_owning_buffer> slice(size_t offset,
                                                                           size_t size) const;

  explicit operator bool() const noexcept { return _entry != nullptr; }

 private:
  void unpin();

  std::shared_ptr<cache_entry> _entry;
  cudaStream_t _stream{nullptr};
};

// ---------------------------------------------------------------------------
// cached_host_buffer — pre-allocated bounce buffers for device reads
// ---------------------------------------------------------------------------
//
// Vended by @c prefetching_cache::read() when the caller passes a non-null
// out-pointer and the target entry is in the @c allocated state.
// @c read() flips the entry to @c loading and transfers ownership to this
// object; the caller then passes it to
// @c sirius_ioctx::device_read_async_io_using() to issue file → bounce →
// device IO using the chunks that are already allocated.
//
// Lifecycle:
//   constructed → prepare_device_requests() called → IO dispatched
//       → mark_cached() on success  (loading → cached)
//       → mark_load_failed() on failure  (loading → empty, chunks freed)
//   If destroyed before either mark is called (e.g. IO never dispatched),
//   the destructor calls mark_load_failed() as a safety net.
//
// Thread safety: a single owner thread; mark_cached/mark_load_failed are
// called from the reactor completion callback (same logical sequence).

class cached_host_buffer {
 public:
  cached_host_buffer() = default;

  /// Construct from an entry that MUST already be in the @c loading state.
  /// @p pool is used to reclaim chunks on failure; must outlive this object.
  cached_host_buffer(std::shared_ptr<cache_entry> entry, buffer_pool* pool) noexcept
    : _entry(std::move(entry)), _pool(pool)
  {
  }

  ~cached_host_buffer()
  {
    if (_entry) mark_load_failed();
  }

  cached_host_buffer(cached_host_buffer const&)            = delete;
  cached_host_buffer& operator=(cached_host_buffer const&) = delete;

  cached_host_buffer(cached_host_buffer&& o) noexcept : _entry(std::move(o._entry)), _pool(o._pool)
  {
    o._pool = nullptr;
  }

  cached_host_buffer& operator=(cached_host_buffer&& o) noexcept
  {
    if (this != &o) {
      if (_entry) mark_load_failed();
      _entry  = std::move(o._entry);
      _pool   = o._pool;
      o._pool = nullptr;
    }
    return *this;
  }

  explicit operator bool() const noexcept { return _entry != nullptr; }

  /// Build one @c device_read_req per chunk in the entry, covering the full
  /// physical range.  Chunks overlapping the caller's logical
  /// [offset, offset+size) are configured to H2D-copy into @p dst; padding
  /// chunks at the edges carry @c data_size == 0 (IO-only, no copy).
  ///
  /// @p ctx must be pre-built by the caller with @c pending == chunks().size()
  /// so each chunk's @c chunk_done() / @c chunk_failed() decrements to zero
  /// exactly once — the caller's completion handler fires when the last chunk
  /// resolves.
  template <typename Handle>
  [[nodiscard]] std::vector<device_read_req<Handle>> prepare_device_requests(
    Handle handle,
    size_t offset,
    size_t size,
    uint8_t* dst,
    cudaStream_t stream,
    int device_id,
    std::shared_ptr<request_context> ctx) const
  {
    assert(_entry != nullptr);
    auto const phys_off    = static_cast<size_t>(_entry->physical_range.offset());
    auto const phys_size   = static_cast<size_t>(_entry->physical_range.size());
    auto const chunk_bytes = _entry->chunk_bytes;
    auto const n_chunks    = _entry->chunks.size();

    std::vector<device_read_req<Handle>> reqs;
    reqs.reserve(n_chunks);

    size_t produced = 0;
    for (size_t i = 0; i < n_chunks; ++i) {
      auto const chunk_file_off = phys_off + i * chunk_bytes;
      auto const chunk_io_size  = std::min(chunk_bytes, phys_size - i * chunk_bytes);
      auto const chunk_file_end = chunk_file_off + chunk_io_size;

      device_read_req<Handle> req;
      req.handle    = handle;
      req.file_off  = chunk_file_off;
      req.io_size   = chunk_io_size;
      req.bounce    = reinterpret_cast<uint8_t*>(_entry->chunks[i]);
      req.stream    = stream;
      req.device_id = device_id;
      req.ctx       = ctx;

      // Determine how much (if any) of this chunk falls within the user's
      // logical request window and should be H2D-copied to dst.
      auto const useful_start = std::max(chunk_file_off, offset);
      auto const useful_end   = std::min(chunk_file_end, offset + size);
      if (useful_start < useful_end) {
        req.data_off  = useful_start - chunk_file_off;
        req.data_size = useful_end - useful_start;
        req.dst       = dst + produced;
        produced += req.data_size;
      }
      // else: alignment-only chunk — bounce is filled, but data_size == 0
      // so no H2D copy is issued for this chunk.

      reqs.push_back(std::move(req));
    }

    return reqs;
  }

  /// Transition the entry loading → cached.  Wakes waiting readers.
  /// Safe to call only once; clears the internal entry pointer afterwards.
  void mark_cached() noexcept
  {
    if (!_entry) return;
    _entry->state.try_mark_cached();
    _entry = nullptr;
  }

  /// Transition the entry loading → empty; return chunks to the pool.
  /// Safe to call only once; clears the internal entry pointer afterwards.
  void mark_load_failed() noexcept
  {
    if (!_entry) return;
    if (_pool) { _pool->deallocate_bulk(_entry->chunks); }
    _entry->state.try_mark_load_failed();
    _entry = nullptr;
  }

 private:
  std::shared_ptr<cache_entry> _entry;
  buffer_pool* _pool{nullptr};
};

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
  [[nodiscard]] pinned_view read(const sirius_io_object& obj,
                                 size_t offset,
                                 size_t size,
                                 cudaStream_t stream = nullptr);

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
  /// _partial_miss_count / _full_miss_count based on the classification.
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

  // Observability counters (updated on every read() / read_ranges() entry).
  std::atomic<uint64_t> _hit_count{0};
  std::atomic<uint64_t> _partial_miss_count{0};
  std::atomic<uint64_t> _full_miss_count{0};

  // Hits that had to wait on a loading entry (subset of _hit_count).
  std::atomic<uint64_t> _hit_after_wait{0};

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
