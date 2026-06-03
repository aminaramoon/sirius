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

#include "io/cache/prefetching_cache.hpp"

#include "cucascade/cuda/event.hpp"
#include "exec/semi_future.hpp"
#include "exec/try.hpp"
#include "io/io_context.hpp"
#include "io/types.hpp"

#include <rmm/cuda_stream_view.hpp>

#include <cuda_runtime.h>

#include <spdlog/spdlog.h>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <exception>
#include <memory>
#include <mutex>
#include <optional>
#include <shared_mutex>
#include <utility>

namespace sirius::io::cache {

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

void buffer_pool::deallocate_bulk(std::vector<std::byte*>& out) noexcept
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

std::vector<cached_chunk*> prefetching_cache::file_entry::update_and_get_chunks(
  std::span<size_t> incoming, uint32_t ticker)
{
  std::vector<cached_chunk*> result(incoming.size());

  // Phase 1: classify under shared lock — find which offsets already exist.
  // Track the indices of incoming items that need to be inserted.
  std::vector<size_t> missing_indices;  // indices into `incoming`/`result`
  {
    std::shared_lock lock(mtx);

    auto s           = chunks.begin();
    const auto s_end = chunks.end();

    for (size_t i = 0; i < incoming.size(); ++i) {
      const size_t off = incoming[i];
      s = std::lower_bound(s, s_end, off, [](const std::unique_ptr<cached_chunk>& c, size_t v) {
        return c->offset < v;
      });

      if (s != s_end && (*s)->offset == off) {
        s->get()->lifecycle.on_request(ticker);
        result[i] = s->get();  // existing
      } else {
        missing_indices.push_back(i);  // mark for insertion
      }
    }

    if (missing_indices.empty()) {
      return result;  // fast path: nothing to insert
    }
  }

  // Phase 2: upgrade to exclusive lock and insert missing chunks.
  // Another writer may have inserted some of our "missing" offsets in the
  // gap between unlocking and re-locking, so re-check each one.
  std::vector<std::unique_ptr<cached_chunk>> to_insert;
  to_insert.reserve(missing_indices.size());

  {
    std::unique_lock lock(mtx);

    auto s     = chunks.begin();
    auto s_end = chunks.end();

    for (size_t idx : missing_indices) {
      const size_t off = incoming[idx];
      s = std::lower_bound(s, s_end, off, [](const std::unique_ptr<cached_chunk>& c, size_t v) {
        return c->offset < v;
      });

      if (s != s_end && (*s)->offset == off) {
        result[idx] = s->get();  // someone else inserted it
      } else {
        auto chunk = std::make_unique<cached_chunk>(off);
        chunk->lifecycle.on_request(ticker);
        result[idx] = chunk.get();  // capture raw ptr before move
        to_insert.push_back(std::move(chunk));
      }
    }

    if (to_insert.empty()) {
      return result;  // all races lost, but result is filled
    }

    // Bulk merge: to_insert is sorted because missing_indices is in order
    // and incoming is sorted+unique.
    const auto mid = chunks.size();
    chunks.reserve(mid + to_insert.size());
    chunks.insert(chunks.end(),
                  std::make_move_iterator(to_insert.begin()),
                  std::make_move_iterator(to_insert.end()));
    std::inplace_merge(
      chunks.begin(),
      chunks.begin() + mid,
      chunks.end(),
      [](const std::unique_ptr<cached_chunk>& a, const std::unique_ptr<cached_chunk>& b) {
        return a->offset < b->offset;
      });
  }

  return result;
}

std::vector<cached_chunk*> prefetching_cache::file_entry::fetch_chunks(std::size_t offset,
                                                                       std::size_t size) const
{
  constexpr size_t chunk_size = 1 << 20;  // must match buffer_pool's chunk size
  if (size == 0) return {};

  std::shared_lock lock(mtx);

  // Align the request to chunk boundaries.
  const std::size_t first_chunk_off = (offset / chunk_size) * chunk_size;
  const std::size_t end_off         = offset + size;
  const std::size_t last_chunk_off  = ((end_off - 1) / chunk_size) * chunk_size;
  const std::size_t expected_count  = (last_chunk_off - first_chunk_off) / chunk_size + 1;

  if (chunks.size() < expected_count) return {};

  // Find the chunk containing `offset`.
  auto first_it = std::lower_bound(
    chunks.begin(),
    chunks.end(),
    first_chunk_off,
    [](const std::unique_ptr<cached_chunk>& c, std::size_t v) { return c->offset < v; });

  if (first_it == chunks.end() || (*first_it)->offset != first_chunk_off) {
    return {};  // first chunk missing
  }

  // Check the last chunk is at the expected position.
  const std::size_t first_idx = std::distance(chunks.begin(), first_it);
  const std::size_t last_idx  = first_idx + expected_count - 1;

  if (last_idx >= chunks.size()) return {};
  if (chunks[last_idx]->offset != last_chunk_off) return {};

  // Coverage confirmed by the invariant: sorted + non-overlapping + fixed-size
  // means consecutive chunks differ by exactly chunk_size. If the first is at
  // first_chunk_off and the (expected_count-1)th is at last_chunk_off, the
  // intermediates are forced.
  std::vector<cached_chunk*> result;
  result.reserve(expected_count);
  for (std::size_t i = 0; i < expected_count; ++i) {
    auto* chunk = chunks[first_idx + i].get();
    chunk->lifecycle.on_consume();
    result.push_back(chunk);
  }
  return result;
}

prefetching_cache::prefetching_cache(buffer_pool* pool,
                                     sirius_ioctx* io_ctx,
                                     size_t inflight_budget_chunks)
  : _pool(pool), _io_ctx(io_ctx), _armed(true), _rate_limiter(inflight_budget_chunks)
{
  _preparation_thread = std::jthread([this](std::stop_token st) { prepare_loop(std::move(st)); },
                                     _preparation_stop_source.get_token());
  _prefetch_thread    = std::jthread([this](std::stop_token st) { prefetch_loop(std::move(st)); },
                                  _prefetch_stop_source.get_token());
  _evictor_thread     = std::jthread([this](std::stop_token st) { evict_loop(std::move(st)); },
                                 _evictor_stop_source.get_token());
}

prefetching_cache::~prefetching_cache()
{
  _shutting_down.store(true, std::memory_order_release);
  _preparation_stop_source.request_stop();
  _preparation_queue.enqueue(nullptr);
  _prefetch_stop_source.request_stop();
  _prefetch_queue.enqueue(nullptr);
  _evictor_stop_source.request_stop();
}

// ===========================================================================
// insert
// ===========================================================================

prefetching_cache::file_entry& prefetching_cache::get_or_create_file_entry(
  const sirius_io_object& obj)
{
  const auto& key = obj.raw_file_cache_id();
  std::shared_lock lk(_map_mtx);
  auto it = _file_cache.find(key);
  if (it == _file_cache.end()) {
    lk.unlock();
    std::unique_lock ulk(_map_mtx);
    auto [new_it, inserted] = _file_cache.try_emplace(key, std::make_unique<file_entry>());
    it                      = new_it;
    if (inserted) {
      it->second->file_size = obj.size();
      it->second->io_obj    = obj.shared_from_this();
      it->second->chunks.reserve((obj.size() + _pool->chunk_bytes() - 1) / _pool->chunk_bytes());
    }
  }
  return *it->second;
}

prefetching_handle prefetching_cache::insert(const sirius_io_object& obj,
                                             std::span<const byte_range> ranges)
{
  const auto& key = obj.raw_file_cache_id();
  auto& file      = get_or_create_file_entry(obj);

  std::vector<size_t> chunk_offsets;  // sorted
  std::size_t last_offset = 0;
  std::ranges::for_each(ranges, [&](auto const& r) {
    auto const off         = static_cast<size_t>(r.offset());
    auto const sz          = static_cast<size_t>(r.size());
    auto const chunk_bytes = _pool->chunk_bytes();
    auto const aligned_off = (r.offset() / chunk_bytes) * chunk_bytes;
    auto const aligned_sz =
      ((off + sz + chunk_bytes - 1) / chunk_bytes) * chunk_bytes - aligned_off;
    for (size_t o = aligned_off; o < aligned_off + aligned_sz; o += chunk_bytes) {
      if (o < last_offset) continue;  // already covered by a previous range, overlapping ranges
      last_offset = o;
      chunk_offsets.push_back(o);
    }
  });

  auto chunks_to_fetch =
    file.update_and_get_chunks(chunk_offsets, _ticker.load(std::memory_order_relaxed));

  auto [work, handle] = preparation_work_item::create(file, std::move(chunks_to_fetch));
  _preparation_queue.enqueue(std::move(work));

  return std::move(handle);
}

bool prefetching_cache::host_read(const sirius_io_object& obj,
                                  size_t offset,
                                  size_t size,
                                  std::byte* dst)
{
  if (size == 0 || dst == nullptr) return true;

  file_entry* file = nullptr;
  {
    std::shared_lock lk(_map_mtx);
    auto it = _file_cache.find(obj.raw_file_cache_id());
    if (it == _file_cache.end()) { return {}; }
    file = it->second.get();
  }

  auto chunks = file->fetch_chunks(offset, size);
  if (chunks.empty()) { return false; }  // full-miss

  auto iter =
    std::ranges::find_if(chunks, [](cached_chunk* c) { return !c->state.acquire_read(); });

  if (iter != chunks.end()) {
    std::for_each(chunks.begin(), iter, [](cached_chunk* c) { c->state.release_read(); });
    return false;
  }

  auto const end_offset = offset + size;
  auto const chunk_size = _pool->chunk_bytes();

  for (auto* chunk : chunks) {
    auto const chunk_begin = std::max(offset, chunk->offset);
    auto const chunk_end   = std::min(end_offset, chunk->offset + chunk_size);
    auto const copy_size   = chunk_end - chunk_begin;
    auto const src_offset  = chunk_begin - chunk->offset;
    auto const dst_offset  = chunk_begin - offset;

    std::memcpy(dst + dst_offset, chunk->data + src_offset, copy_size);
    chunk->state.release_read();
  }

  return true;
}

exec::semi_future<bool> prefetching_cache::device_read_async(const sirius_io_object& obj,
                                                             size_t offset,
                                                             size_t size,
                                                             std::byte* dst,
                                                             rmm::cuda_stream_view stream)
{
  if (size == 0 || dst == nullptr) { return true; }

  file_entry* file = nullptr;
  {
    std::shared_lock lk(_map_mtx);
    auto it = _file_cache.find(obj.raw_file_cache_id());
    if (it != _file_cache.end()) { file = it->second.get(); }
  }

  if (file == nullptr) { return false; }

  auto chunks = file->fetch_chunks(offset, size);
  if (chunks.empty()) { return false; }  // full-miss

  // check marks that are cached, or in-use for reuse
  auto cached_pnt = std::stable_partition(
    chunks.begin(), chunks.end(), [](cached_chunk* c) { return c->state.acquire_read(); });

  if (!_io_ctx->supports_host_to_device_read() && cached_pnt != chunks.end()) {
    std::for_each(chunks.begin(), cached_pnt, [](cached_chunk* c) { c->state.release_read(); });
    return false;
  }

  // check if any chunks is allocated but not yet loaded, which means we can reuse it for the
  // current read and skip waiting for the prefetch to complete.
  auto last_loading_pnt = std::find_if(
    cached_pnt, chunks.end(), [](cached_chunk* c) { return !c->state.mark_loading(); });

  // we are not waiting for currently loading one
  if (last_loading_pnt != chunks.end()) {
    std::for_each(chunks.begin(), cached_pnt, [](cached_chunk* c) { c->state.release_read(); });
    std::for_each(cached_pnt, last_loading_pnt, [](cached_chunk* c) {
      static_cast<void>(c->state.mark_load_failed());
    });
    return false;
  }

  // copy cached chunks to destination buffer asynchronously
  std::optional<cucascade::cuda::cuda_event> copy_evnt;
  if (cached_pnt != chunks.begin()) {
    copy_evnt.emplace();
    size_t chunk_bytes = _pool->chunk_bytes();
    for (cached_chunk* c : std::span(chunks.begin(), cached_pnt)) {
      size_t chunk_start = c->offset;
      size_t copy_start  = std::max(chunk_start, offset);
      size_t copy_end    = std::min(chunk_start + chunk_bytes, offset + size);
      size_t copy_size   = copy_end - copy_start;
      size_t src_off     = copy_start - chunk_start;
      size_t dst_off     = copy_start - offset;
      cudaMemcpyAsync(dst + dst_off, c->data + src_off, copy_size, cudaMemcpyHostToDevice, stream);
    }
    copy_evnt->record(stream);
  }

  // populate the cache for the missing chunks through prefetching, and we can read from the cache
  // once the prefetching is done,
  if (cached_pnt != chunks.end()) {
    std::vector<io::io_object_segment> segments;
    segments.reserve(std::distance(cached_pnt, chunks.end()));
    for (cached_chunk* c : std::span(cached_pnt, chunks.end())) {
      assert(c->data != nullptr);
      segments.emplace_back(c->offset, _pool->chunk_bytes(), c->data, true);
    }
    auto io_fut = _io_ctx->host_to_device_read_async_io(
      obj, segments, offset, size, reinterpret_cast<uint8_t*>(dst), stream);

    // Number of leading chunks that were already cached and pinned (in_use)
    // for the H2D copy above; the remainder ([n_in_use, end)) are the chunks
    // we just flipped to `loading` and handed to the IO.
    auto const n_in_use = static_cast<size_t>(cached_pnt - chunks.begin());

    // Return a future that, when awaited, first waits for the prefetch IO to
    // complete (the IO future resolves once the H2D copies are *enqueued*),
    // then synchronizes the stream so every enqueued H2D copy — both the
    // already-cached chunks and the freshly-loaded ones — has actually
    // drained.  Only THEN do we mutate chunk state: releasing a read pin or
    // publishing loading -> cached makes the chunk evictable, so doing it
    // before the copy finishes would let the evictor reclaim a bounce buffer
    // mid-copy.  A failed IO is surfaced as an exception (not a false result):
    // reaching this path means we attempted the read, so callers must not
    // retry it through a different code path.
    return std::move(io_fut).defer(
      [stream, n_in_use, chunks = std::move(chunks)](exec::try_t<size_t>&& res) mutable -> bool {
        bool ok = !res.has_exception();

        std::exception_ptr cuda_exception = nullptr;
        try {
          stream.synchronize();
        } catch (...) {
          cuda_exception = std::current_exception();
          ok             = false;
        }

        // Already-cached chunks: drop the read pin we took for the copy.
        std::for_each_n(chunks.begin(), n_in_use, [](cached_chunk* c) { c->state.release_read(); });

        // Freshly-loaded chunks: on success publish loading -> cached; on
        // failure revert loading -> allocated so a later read can retry with
        // the chunks still attached.
        auto transition =
          ok ? [](cached_chunk* c) { static_cast<void>(c->state.mark_cached()); }
             : [](cached_chunk* c) { static_cast<void>(c->state.mark_load_failed()); };
        std::for_each(chunks.begin() + n_in_use, chunks.end(), transition);

        if (res.has_exception() || cuda_exception) {
          std::rethrow_exception(res.has_exception() ? std::move(res).exception() : cuda_exception);
        }
        return true;
      });
  }

  // All requested chunks were already cached: return a future that, when
  // awaited, synchronizes on the H2D copy event and only then drops the read
  // pins.  Releasing a pin makes the chunk evictable, so it must wait until
  // the copy that reads the chunk has actually completed.
  return exec::make_semi_future(true).defer(
    [copy_evnt = std::move(copy_evnt),
     chunks    = std::move(chunks)](exec::try_t<bool>&& status) mutable -> bool {
      std::exception_ptr copy_exception = nullptr;
      try {
        if (copy_evnt) { copy_evnt->synchronize(); }
      } catch (...) {
        copy_exception = std::current_exception();
      }
      std::ranges::for_each(chunks, [](cached_chunk* c) { c->state.release_read(); });
      if (copy_exception || status.has_exception()) {
        std::rethrow_exception(copy_exception ? copy_exception : status.exception());
      }
      return true;
    });
}

std::string prefetching_cache::summary() const { return ""; }

void prefetching_cache::prepare_for_query() noexcept
{
  _ticker.fetch_add(1, std::memory_order_relaxed);
}

// ===========================================================================

void prefetching_cache::prepare_loop(std::stop_token st)
{
  while (!_shutting_down && !st.stop_requested()) {
    preparation_request req = nullptr;
    _preparation_queue.wait_dequeue(req);
    if (req == nullptr) { continue; }  // spurious wakeup or shutdown

    if (req->is_cancelled()) { continue; }  // request was cancelled

    auto& chunks = req->chunks;

    // how many buffers we need to allocate from the pool to prepare this request?
    std::size_t n_chunks_needed = std::ranges::count_if(
      chunks, [](cached_chunk* c) { return c->state.get_state() == entry_state::empty; });

    std::vector<std::byte*> buffers;
    buffers.reserve(n_chunks_needed);
    auto n_allocated = _pool->allocate_bulk(n_chunks_needed, buffers);
    if (n_allocated != n_chunks_needed) {
      // Pool is exhausted and cannot grow.  Return the buffers we did get and
      // re-enqueue the work for a retry after the evictor frees some.
      // todo(amin): needs eviction and then backoff
      if (n_allocated > 0) _pool->deallocate_bulk(buffers);
      _preparation_queue.enqueue(std::move(req));
      continue;
    }

    for (size_t i = 0; i < n_chunks_needed; ++i) {
      if (chunks[i]->state.mark_queued()) {
        chunks[i]->data = buffers[i];  // revert the swap if we lost the
        if (!chunks[i]->state.mark_allocated()) {
          spdlog::error(
            "prefetching_cache: chunk at offset {} was marked queued but failed to mark "
            "allocated",
            chunks[i]->offset);
        }
      }
    }

    if (!_io_ctx->supports_vector_host_read() ||
        _io_ctx->preferred_prefetching_mode() == prefetching_mode::immediate) {
      // either the backend doesn't support scatter-gather reads or it prefers not to reuse
      // buffers for multiple reads.  In either case, we can skip the prefetching step and let the
      // read() path handle the IO directly into the caller's buffer.
      continue;
    }

    auto prefetch_req    = std::make_unique<prefetch_work_item>();
    prefetch_req->file   = req->file;
    prefetch_req->chunks = std::move(chunks);
    prefetch_req->alive  = req->state();

    if (!st.stop_requested()) { _prefetch_queue.enqueue(std::move(prefetch_req)); }
  }
}

void prefetching_cache::prefetch_loop(std::stop_token st)
{
  constexpr bool device_accessible = true;
  while (!_shutting_down && !st.stop_requested()) {
    prefetch_request req = nullptr;
    _prefetch_queue.wait_dequeue(req);
    if (req == nullptr || req->is_cancelled()) { continue; }

    auto& allocated_chunks = req->chunks;
    auto& io_obj           = req->file->io_obj;
    std::vector<io::io_object_segment> segments;

    segments.reserve(allocated_chunks.size());
    allocated_chunks.erase(
      std::remove_if(allocated_chunks.begin(),
                     allocated_chunks.end(),
                     [&](cached_chunk* c) {
                       if (c->state.mark_loading()) {
                         segments.emplace_back(
                           c->offset, _pool->chunk_bytes(), c->data, device_accessible);
                         return false;
                       }
                       return true;
                     }),
      allocated_chunks.end());

    // request was cancelled
    if (req->is_cancelled()) {
      std::ranges::for_each(
        allocated_chunks, [&](cached_chunk* c) { static_cast<void>(c->state.mark_load_failed()); });
      continue;
    }

    // todo(amin): mark the cache_entries
    _io_ctx->host_read_ranges_async_io(*io_obj, segments)
      .via(&_io_cb_dispatcher)
      .then_try([chunks = std::move(allocated_chunks)](exec::try_t<size_t>&& res) mutable {
        auto transition =
          res.has_value() ? [](cached_chunk* c) { static_cast<void>(c->state.mark_cached()); }
                          : [](cached_chunk* c) { static_cast<void>(c->state.mark_load_failed()); };
        std::ranges::for_each(chunks, transition);
      });
  }
}

void prefetching_cache::evict_loop(std::stop_token st)
{
  while (!_shutting_down && !st.stop_requested()) {}
}

// ===========================================================================
// prefetching_handle
// ===========================================================================

void prefetching_handle::cancel() noexcept
{
  if (!_alive) return;
  // Flip the shared alive flag to false.  The cache's preparation / prefetch
  // workers read this through work_item::is_cancelled() and drop still-pending
  // entries.  Idempotent and thread-safe (the flag is atomic).
  _alive->store(false, std::memory_order_release);
}

}  // namespace sirius::io::cache
