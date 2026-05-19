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

#include "io/sirius_datasource.hpp"

#include <concurrentqueue.h>
#include <cudf/io/text/byte_range_info.hpp>

#include <chrono>
#include <cstddef>
#include <memory>
#include <semaphore>
#include <stop_token>
#include <utility>
#include <vector>

namespace sirius::scan_manager {

/**
 * @brief Pipeline-ordered sequencer for @c fadvise(opportunistic) calls.
 *
 * Per-pipeline slots collect (datasource, ranges) pairs emitted by each
 * pipeline's split provider during metadata scan.  A single sequencer
 * task drains the slots in the order they were registered, calling
 * @c fadvise(opportunistic, ranges) on each entry's datasource, and
 * advances to the next slot only after the current slot signals closure
 * (a null-datasource sentinel).  This serialises the opportunistic
 * fadvise tier across pipelines so the prefetching cache receives ranges
 * in execution order rather than in metadata-scan-completion order —
 * giving the cache its longest possible lead time for the head-of-line
 * pipeline before later pipelines start competing for the buffer pool.
 *
 * Usage:
 *   - scan_manager calls @c add_pipeline_slot(pipeline_id) once per
 *     parquet pipeline that needs opportunistic prefetching.  The
 *     returned pointer is handed to that pipeline's split provider.
 *   - The provider pushes one @c fadvise_entry per file via the slot's
 *     queue + semaphore; it pushes a sentinel (null datasource) when
 *     all batches of its metadata scan are complete.
 *   - scan_manager calls @c register_ranges(stop, dispatcher) once, which
 *     enqueues the sequencer task on the dispatcher.  The task processes
 *     slots in insertion order until either all slots are drained or the
 *     stop_token fires.
 */
class pipeline_ordered_prefetching_manager {
 public:
  /// One unit of work pushed by a provider.  A null @c datasource is the
  /// closure sentinel: the sequencer treats it as "slot done, move on".
  struct fadvise_entry {
    std::shared_ptr<sirius::io::sirius_datasource> datasource;
    std::vector<cudf::io::text::byte_range_info> ranges;
  };

  /// Per-pipeline mailbox.  The provider produces, the sequencer task
  /// consumes.  Holds its own semaphore so the sequencer can block on
  /// an empty slot without spinning.
  struct pipeline_slot {
    std::size_t pipeline_id{0};
    duckdb_moodycamel::ConcurrentQueue<fadvise_entry> queue;
    std::counting_semaphore<> sem{0};

    /// Push one (datasource, ranges) pair.  Pass a default-constructed
    /// entry (or one with null datasource) to signal closure.
    void push(fadvise_entry entry)
    {
      queue.enqueue(std::move(entry));
      sem.release();
    }

    /// Closure sentinel — the provider calls this when all metadata
    /// scans for its pipeline have completed.
    void close() { push({}); }
  };

  pipeline_ordered_prefetching_manager()                                                 = default;
  pipeline_ordered_prefetching_manager(pipeline_ordered_prefetching_manager const&)      = delete;
  pipeline_ordered_prefetching_manager& operator=(pipeline_ordered_prefetching_manager const&) =
    delete;

  /// Register a slot for @p pipeline_id.  Slots are processed by the
  /// sequencer task in the order they were added — typically scan_manager
  /// adds them in pipeline-id order so the head-of-line pipeline drains
  /// first.  The returned pointer is valid for the manager's lifetime.
  pipeline_slot* add_pipeline_slot(std::size_t pipeline_id);

  /// Spawn the sequencer task on @p dispatcher.  The dispatcher must
  /// expose @c enqueue(callable) — typically a @c scoped_dispatcher or
  /// any thread-pool-like type with that surface.  Call once after all
  /// slots have been added (the task captures the slot list by
  /// reference; adding slots after this call has undefined ordering).
  template <class Dispatcher>
  void register_ranges(std::stop_token stop, Dispatcher& dispatcher)
  {
    dispatcher.enqueue([this, stop = std::move(stop)]() { run_sequencer(stop); });
  }

 private:
  /// The sequencer body.  Walks _slots in order; for each, drains
  /// entries (semaphore-blocked with a poll timeout so stop_token is
  /// observed promptly) until it hits a closure sentinel (null
  /// datasource) or stop is requested.
  void run_sequencer(std::stop_token const& stop);

  static constexpr auto SEQUENCER_POLL_INTERVAL = std::chrono::milliseconds(50);

  /// unique_ptr storage: the slot contains a semaphore and a moodycamel
  /// queue, both of which are non-movable, so we need stable addresses
  /// in the vector.
  std::vector<std::unique_ptr<pipeline_slot>> _slots;
};

}  // namespace sirius::scan_manager
