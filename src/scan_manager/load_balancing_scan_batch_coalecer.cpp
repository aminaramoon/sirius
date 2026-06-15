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

#include "scan_manager/load_balancing_scan_batch_coalecer.hpp"

#include "io/io_context.hpp"
#include "log/logging.hpp"

namespace sirius::scan_manager {

load_balancing_scan_batch_coalecer::metadata_processing_state*
load_balancing_scan_batch_coalecer::register_pipeline(std::size_t pipeline_id)
{
  auto slot         = std::make_unique<metadata_processing_state>();
  slot->pipeline_id = pipeline_id;
  auto* p           = slot.get();
  _slots.emplace(pipeline_id, std::move(slot));
  return p;
}

void load_balancing_scan_batch_coalecer::worker_loop(std::stop_token const& stop)
{
  for (auto& slot_ptr : _slots) {
    auto& slot = *slot_ptr;
    while (!stop.stop_requested()) {
      // Block on the semaphore, but with a poll timeout so the
      // stop_token is observed promptly even when a slot is silent
      // (provider hasn't pushed yet).
      if (!slot.sem.try_acquire_for(SEQUENCER_POLL_INTERVAL)) continue;

      fadvise_entry entry;
      if (!slot.queue.try_dequeue(entry)) continue;  // spurious wake — unlikely

      if (!entry.datasource) {
        // Closure sentinel — provider is done with this pipeline.
        // Drop any leftover entries (there shouldn't be any if the
        // provider pushes sentinel last, but be defensive).
        SIRIUS_LOG_DEBUG("[pipeline_ordered_prefetching_manager] pipeline_id={} closed",
                         slot.pipeline_id);
        break;
      }

      SIRIUS_LOG_INFO(
        "[fadvise opportunistic] pipeline_id={} ranges={}", slot.pipeline_id, entry.ranges.size());

      entry.datasource->fadvise(entry.ranges);
      entry.datasource->prefetch(sirius::io::cache::prefetching_stage::opportunistic);
    }
    if (stop.stop_requested()) break;
  }
}

}  // namespace sirius::scan_manager
