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

// Shared cache entry types used by both prefetching_cache and the IO context
// virtual interface (device_read_async_io_using).  Extracted here to break the
// circular include between io_context.hpp and prefetching_cache.hpp.

#include "io/types.hpp"

#include <cudf/io/datasource.hpp>

#include <cuda_runtime.h>

#include <cucascade/memory/fixed_size_host_memory_resource.hpp>

#include <atomic>
#include <cassert>
#include <future>
#include <memory>
#include <mutex>
#include <span>
#include <vector>

namespace sirius::io {

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
// is still wanted (n_pending > 0).  When n_pending == 0 every handle
// for this file has been released and the wrapper's entries can be
// reclaimed.  The stamp field is no longer consulted for eviction
// decisions (kept for potential diagnostics / future use).

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

  void deallocate_bulk(std::vector<std::byte*>& out) noexcept;

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
// State machine:
//
//   try_allocate()                       try_start_loading()
//   empty ──────────────► allocated ◄─┬─────────────────► loading
//     ▲                   │     ▲     │                       │
//     │                   │     │     │                       │
//     │ mark_evicted()    │     └─────┴─────────┐             │ try_mark_cached()
//     │                   │       try_revert_   │             ▼
//     │            try_start_     loading_to_   │       cached ◄────────────────
//     │            evicting_     allocated()    │         ▲     │
//     │            from_         (io failure /  │         │     │ try_start_evicting()
//     │            allocated()   budget bail)   │         │     ▼
//     │                   ▼                     │  release_read()  evicting
//     └──────────────── evicting ◄──────────────┴────────┘  (last reader)
//                            ▲
//                            │
//                       in_use(pin≥1)
//                            ▲
//                  try_acquire_read() / release_read()
//
// Allocator (allocator_loop): empty → allocated.
// IO dispatch (io_dispatch_loop): allocated → loading on budget acquire.
//   - On successful IO completion: loading → cached.
//   - On IO failure / budget bail / mid-dispatch cancel: loading → allocated
//     (chunks stay attached for a future read-driven retry).
// Evictor: cached → evicting → empty (releases chunks).
//          allocated → evicting → empty (releases chunks).
// Readers: cached ↔ in_use(pin) via try_acquire_read / release_read.
//
// `loading` is the only state that doesn't have a direct path to `evicting` —
// the in-flight IO owns the chunks and decides whether to revert
// (loading → allocated) or commit (loading → cached).

class entry_state {
 public:
  enum value : uint8_t {
    empty     = 0,
    allocated = 1,  ///< chunks assigned, IO not yet dispatched
    loading   = 2,
    cached    = 3,
    in_use    = 4,
    evicting  = 5,
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

  /// empty → allocated.  Returns false if not empty.  Called by
  /// allocator_loop's Phase 4 to atomically claim an entry and assign
  /// chunks to it; the CAS is the exclusive claim, so concurrent
  /// allocator passes / inserts covering the same range see the entry
  /// in `allocated` (or later) and skip.
  bool try_allocate() noexcept
  {
    auto expected = pack(empty, 0);
    return _packed.compare_exchange_strong(expected, pack(allocated, 0), std::memory_order_acq_rel);
  }

  /// allocated → evicting.  Returns false if not allocated.  Unified
  /// with the cached → evicting path so all eviction goes through the
  /// `evicting` state — callers free chunks and then call
  /// mark_evicted() to land at `empty`.  Notifies any threads parked
  /// in wait_while_pending().
  bool try_start_evicting_from_allocated() noexcept
  {
    auto expected = pack(allocated, 0);
    bool ok =
      _packed.compare_exchange_strong(expected, pack(evicting, 0), std::memory_order_acq_rel);
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

  /// loading → in_use(pin_count = 1) using a single CAS.  Used by
  /// @c cached_host_buffer to finish a direct-device read while keeping the
  /// entry pinned across the caller's H2D copy: a subsequent stream callback
  /// drops the pin (in_use → cached) once the stream completes.  Closes the
  /// race window where the evictor could otherwise reclaim the entry's
  /// pinned chunks before the H2D drains them.
  ///
  /// Returns false if shutdown or another completion path already moved the
  /// entry out of loading.
  bool try_finish_loading_pinned() noexcept
  {
    auto expected = pack(loading, 0);
    bool ok = _packed.compare_exchange_strong(expected, pack(in_use, 1), std::memory_order_acq_rel);
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

  /// loading → allocated using CAS.  Returns false if the entry was
  /// already resolved or aborted.  Used by io_dispatch_loop's failure
  /// paths to revert an entry whose IO did not complete: the entry's
  /// chunks stay attached so a subsequent allocated-steal read can
  /// retry the load with a fresh request_context, instead of
  /// discarding the entry to `empty` and forcing the next reader
  /// through find_entry → miss_state_empty_post_drain.
  bool try_revert_loading_to_allocated() noexcept
  {
    auto expected = pack(loading, 0);
    bool ok =
      _packed.compare_exchange_strong(expected, pack(allocated, 0), std::memory_order_acq_rel);
    if (ok) { _packed.notify_all(); }
    return ok;
  }

  /// (allocated | loading) → empty.  Used during cache shutdown to wake
  /// readers parked on a load that may never complete.
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
      if (st != allocated && st != loading) return empty;
      if (_packed.compare_exchange_weak(
            cur, pack(empty, 0), std::memory_order_acq_rel, std::memory_order_acquire)) {
        _packed.notify_all();
        return st;
      }
    }
  }

  /// Block while state is @c loading.  Returns when the state transitions
  /// to @c cached (success), @c empty (IO failed or cancelled), or any
  /// other terminal state.  Does NOT wait on @c allocated — callers that
  /// receive an @c allocated entry either steal it for a direct device read
  /// (via @c cached_host_buffer) or treat it as a miss and fall through to
  /// the backend.
  void wait_while_pending() noexcept
  {
    uint32_t cur = _packed.load(std::memory_order_acquire);
    while (true) {
      auto st = unpack_state(cur);
      if (st != loading) break;
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

  /// One-shot diagnostic flag: set by the allocator on the first
  /// empty → allocated transition.  Lets read()'s miss-classifier
  /// distinguish "entry was never allocated yet" (allocation falling
  /// behind reads) from "entry was allocated but later evicted /
  /// load-failed" (eviction churn).  Never cleared after set.
  std::atomic<bool> ever_allocated{false};

  /// One-shot diagnostic flag: set when the entry first reaches the
  /// `cached` state (IO completed successfully).  Combined with
  /// ever_allocated, lets read()'s miss-classifier separate
  /// "allocated-then-evicted" (ever_cached=true) from "allocated-then-
  /// load-failed" (ever_cached=false).  Never cleared after set.
  std::atomic<bool> ever_cached{false};

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
// Acquires a read pin on one or more contiguous cache_entries on
// construction (via @c entry_state::try_acquire_read — a single atomic
// CAS per entry that transitions cached → in_use with pin_count+1).
// Releases on destruction (via @c entry_state::release_read on each
// entry; one stream callback for the whole batch when @p stream is
// non-null).
//
// Multi-entry views are produced when a read range spans two or more
// adjacent prefetched entries that are all in @c cached / @c in_use.
// Mixed-state coverage (any entry @c loading / @c allocated / @c
// evicting / @c empty) is treated as a miss by the cache and is not
// expressed through this type.
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

  /// Multi-entry overload.  Tries to acquire a read pin on every entry;
  /// on partial failure (any @c try_acquire_read returns false), the
  /// pins already acquired are released and the view becomes empty.
  /// Entries must be contiguous in logical-offset order — the multi-
  /// entry hit path in @c prefetching_cache::read() guarantees this.
  pinned_view(std::vector<std::shared_ptr<cache_entry>> entries, cudaStream_t stream);

  ~pinned_view();

  pinned_view(pinned_view&& o) noexcept;
  pinned_view& operator=(pinned_view&& o) noexcept;

  pinned_view(pinned_view const&)            = delete;
  pinned_view& operator=(pinned_view const&) = delete;

  /// Number of 1MB chunks backing this view — sum across all entries.
  [[nodiscard]] size_t num_chunks() const noexcept;

  /// Access chunk @p i (physical data). Full CHUNK_BYTES except
  /// possibly the last chunk in any entry, which may be shorter.
  /// Indexing walks entries in order: chunks of entry 0, then entry 1, …
  [[nodiscard]] std::span<const std::byte> operator[](size_t i) const noexcept;

  /// Logical range this view covers — the union extent of the
  /// underlying entries.  For multi-entry views this assumes no gaps
  /// between adjacent entries (enforced by the cache's hit path).
  [[nodiscard]] cudf::io::text::byte_range_info logical_range() const noexcept;

  /// Physical (O_DIRECT aligned) range.  Union extent of the
  /// underlying entries' physical ranges.  Only fully meaningful for
  /// single-entry views; for multi-entry views the per-entry physical
  /// ranges may overlap because each entry was independently aligned.
  [[nodiscard]] cudf::io::text::byte_range_info physical_range() const noexcept;

  /// Logical size (sum of constituent entries' logical sizes).
  [[nodiscard]] size_t size() const noexcept;

  /// Slice the cached data at logical [offset, offset+size) into a vector of
  /// non_owning_buffers, one per chunk boundary crossed.  Adjacent buffers
  /// that turn out to be virtually contiguous (same slab) are coalesced.
  /// The caller must ensure [offset, offset+size) lies within
  /// @c logical_range().
  [[nodiscard]] std::vector<cudf::io::datasource::non_owning_buffer> slice(size_t offset,
                                                                           size_t size) const;

  explicit operator bool() const noexcept { return !_entries.empty(); }

 private:
  void unpin();

  std::vector<std::shared_ptr<cache_entry>> _entries;
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
  /// @p pool must not be null when @p entry is non-null — the pool is needed
  /// to return the entry's chunks on failure.
  cached_host_buffer(std::shared_ptr<cache_entry> entry, buffer_pool* pool) noexcept
    : _entry(std::move(entry)), _pool(pool)
  {
    assert(!_entry || _pool);  // non-null entry requires a pool to reclaim chunks
  }

  ~cached_host_buffer()
  {
    if (_entry) mark_load_failed();
  }

  cached_host_buffer(cached_host_buffer const&)            = delete;
  cached_host_buffer& operator=(cached_host_buffer const&) = delete;

  cached_host_buffer(cached_host_buffer&& o) noexcept : _entry(std::move(o._entry)), _pool(o._pool)
  {
    assert(!o._entry);  // shared_ptr move guarantees the source is null
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
  /// physical range.  Chunks overlapping [offset, offset+size) are configured
  /// to H2D-copy into @p dst; alignment-padding chunks carry
  /// @c data_size == 0 (IO-only, no copy).
  ///
  /// @p offset and @p size are **file-absolute** byte positions (same coordinate
  /// space as @c physical_range.offset()).  The caller is responsible for
  /// translating any logical offset before calling.
  ///
  /// The returned requests have a null @c ctx field.  The caller must create
  /// a @c request_context with @c pending == reqs.size() and patch it onto
  /// every request before dispatching, following the pattern in
  /// @c host_read_ranges_async_io in @c templated_ioctx.
  template <typename Handle>
  [[nodiscard]] std::vector<device_read_req<Handle>> prepare_device_requests(Handle handle,
                                                                             size_t offset,
                                                                             size_t size,
                                                                             uint8_t* dst,
                                                                             cudaStream_t stream,
                                                                             int device_id) const
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
      // ctx is left null here; caller patches it after creating request_context.

      // Determine how much (if any) of this chunk falls within the user's
      // request window and should be H2D-copied to dst.
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
  ///
  /// Uses the CAS variant rather than the unconditional store so that a
  /// racing @c try_abort_pending() (cache shutdown) that already moved
  /// the entry to @c empty is tolerated silently — the shutdown will have
  /// already woken all waiters.
  ///
  /// Prefer @c mark_cached_with_stream() when an async H2D copy is in flight
  /// against the entry's chunks — otherwise the evictor can reclaim them
  /// before the copy drains.
  void mark_cached() noexcept
  {
    if (!_entry) return;
    _entry->state.try_mark_cached();
    _entry = nullptr;
  }

  /// Transition loading → in_use(1) atomically and defer the pin release via
  /// a host callback enqueued on @p stream.  The entry stays non-evictable
  /// (in_use) until the stream completes the H2D copy that was launched
  /// against its chunks, at which point pin_count drops to zero and the
  /// entry transitions back to cached.  Safe to call only once; clears the
  /// internal entry pointer afterwards.
  ///
  /// Out-of-line because the implementation reuses the host-callback
  /// machinery defined in @c prefetching_cache.cpp (same path as
  /// @c pinned_view's async release).
  void mark_cached_with_stream(cudaStream_t stream) noexcept;

  /// Transition the entry loading → empty; return chunks to the pool.
  /// Safe to call only once; clears the internal entry pointer afterwards.
  void mark_load_failed() noexcept
  {
    if (!_entry) return;
    // Revert loading → allocated so the entry's chunks stay attached
    // for a future read-driven retry (mirrors io_dispatch_loop's
    // failure path).  Only deallocate if the CAS lost — that means
    // shutdown's abort_pending_entries moved the entry out of
    // `loading`, and since abort doesn't free chunks for the
    // loading-state transition, we own them here.
    if (!_entry->state.try_revert_loading_to_allocated()) {
      if (_pool) { _pool->deallocate_bulk(_entry->chunks); }
    }
    _entry = nullptr;
  }

 private:
  std::shared_ptr<cache_entry> _entry;
  buffer_pool* _pool{nullptr};
};

}  // namespace sirius::io
