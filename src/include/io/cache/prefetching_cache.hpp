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

#include "blockingconcurrentqueue.h"
#include "exec/scoped_dispatcher.hpp"
#include "exec/semi_future.hpp"
#include "exec/thread_pool.hpp"
#include "io/cache/types.hpp"
#include "io/details/admission_control.hpp"

#include <concurrentqueue.h>

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <deque>
#include <list>
#include <memory>
#include <semaphore>
#include <shared_mutex>
#include <stop_token>
#include <thread>
#include <unordered_map>
#include <vector>

namespace sirius::io {
class sirius_ioctx;
class sirius_datasource;
}  // namespace sirius::io

namespace sirius::io::cache {

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
  friend class sirius::io::sirius_datasource;
  // prefetching_handle calls notify_disposed() on cancel — needs access to
  // the private method.
  friend class prefetching_handle;

 public:
  prefetching_cache(buffer_pool* pool, sirius_ioctx* io_ctx, size_t inflight_budget_chunks);
  ~prefetching_cache();

  prefetching_cache(prefetching_cache const&)            = delete;
  prefetching_cache& operator=(prefetching_cache const&) = delete;

  [[nodiscard]] bool is_armed() const noexcept { return _armed; }

  [[nodiscard]] bool host_read(const sirius_io_object& obj,
                               size_t offset,
                               size_t size,
                               std::byte* dst);

  [[nodiscard]] exec::semi_future<bool> device_read_async(const sirius_io_object& obj,
                                                          size_t offset,
                                                          size_t size,
                                                          std::byte* device_ptr,
                                                          rmm::cuda_stream_view stream);

  [[nodiscard]] std::string summary() const;

  void prepare_for_query() noexcept;

 private:
  [[nodiscard]] prefetching_handle insert(
    const sirius_io_object& obj, const std::vector<cudf::io::text::byte_range_info>& ranges);

  struct file_entry {
    std::vector<cached_chunk*> update_and_get_chunks(std::span<size_t> incoming, uint32_t ticker);

    std::vector<cached_chunk*> fetch_chunks(std::size_t offset, std::size_t size) const;

    mutable std::shared_mutex mtx;
    std::shared_ptr<const sirius_io_object> io_obj;
    std::vector<std::unique_ptr<cached_chunk>> chunks;
    size_t file_size{0};
  };

  void prepare_loop(std::stop_token st);
  void prefetch_loop(std::stop_token st);
  void evict_loop(std::stop_token st);

  file_entry& get_or_create_file_entry(const sirius_io_object& obj);

  buffer_pool* const _pool;

  sirius_ioctx* const _io_ctx;

  bool const _armed;

  std::atomic<bool> _shutting_down{false};

  std::atomic<uint32_t> _ticker{0};  // see prefetch_stats::snapshot for layout

  struct preparation_work_item {
    static std::pair<std::unique_ptr<preparation_work_item>, prefetching_handle> create(
      file_entry& file, std::vector<cached_chunk*> ranges)
    {
      auto handle = prefetching_handle();
      return std::make_pair(std::unique_ptr<preparation_work_item>(new preparation_work_item{
                              std::addressof(file), std::move(ranges), handle.state()}),
                            std::move(handle));
    }

    [[nodiscard]] bool is_cancelled() const noexcept { return !alive->load(); }

    std::shared_ptr<const std::atomic<bool>> state() const noexcept { return alive; }

    file_entry* file;
    std::vector<cached_chunk*> chunks;

   private:
    preparation_work_item(file_entry* file,
                          std::vector<cached_chunk*> ranges,
                          std::shared_ptr<const std::atomic<bool>> alive) noexcept
      : file(file), chunks(std::move(ranges)), alive(std::move(alive))
    {
    }

    std::shared_ptr<const std::atomic<bool>> alive;
  };

  using preparation_request = std::unique_ptr<preparation_work_item>;

  std::jthread _preparation_thread;
  duckdb_moodycamel::BlockingConcurrentQueue<preparation_request> _preparation_queue;
  std::stop_source _preparation_stop_source;

  struct prefetch_work_item {
    file_entry* file;
    std::vector<cached_chunk*> chunks;

    [[nodiscard]] bool is_cancelled() const noexcept { return !alive->load(); }

    std::shared_ptr<const std::atomic<bool>> alive;
  };
  using prefetch_request = std::unique_ptr<prefetch_work_item>;

  std::jthread _prefetch_thread;
  io::admission_control _rate_limiter;
  duckdb_moodycamel::BlockingConcurrentQueue<prefetch_request> _prefetch_queue;
  std::stop_source _prefetch_stop_source;

  struct eviction_work_item {
    file_entry* file;
    std::vector<cached_chunk*> chunks;
  };
  using eviction_request = std::unique_ptr<eviction_work_item>;
  std::jthread _evictor_thread;
  duckdb_moodycamel::BlockingConcurrentQueue<eviction_request> _eviction_queue;
  std::stop_source _evictor_stop_source;

  mutable std::shared_mutex _map_mtx;
  std::unordered_map<std::string, std::unique_ptr<file_entry>> _file_cache;

  exec::static_thread_pool _io_cb_thread_pool{
    2, "io_cb"};  // single-threaded pool for IO completion callbacks
  exec::scoped_dispatcher _io_cb_dispatcher{_io_cb_thread_pool, 2};
};

}  // namespace sirius::io::cache
