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
load_balancing_scan_batch_coalecer::register_pipeline(std::size_t pipeline_id,
                                                      split_connector& connector,
                                                      std::unique_ptr<op::scan::batch_coalecer> coalecer,
                                                      std::shared_ptr<balancing_strategy> balancer)
{
  // TODO(plumbing): the per-pipeline coalecing/balancing worker loop is not yet
  // implemented (see worker_loop). This just records the supplied pieces on the
  // slot so the new register_pipeline interface compiles and is consumable.
  auto slot         = std::make_unique<metadata_processing_state>();
  slot->pipeline_id = pipeline_id;
  slot->coalecer    = std::move(coalecer);
  slot->balancer    = std::move(balancer);
  // The slot holds a shared_ptr but callers pass a non-owning reference to a
  // connector owned by the scan operator; store a non-owning alias.
  slot->connector = std::shared_ptr<split_connector>(std::shared_ptr<void>{}, &connector);
  auto* p         = slot.get();
  _slots.emplace(pipeline_id, std::move(slot));
  return p;
}

void load_balancing_scan_batch_coalecer::worker_loop([[maybe_unused]] std::stop_token const& stop)
{
  // TODO(plumbing): the original fadvise-sequencer body referenced the old
  // per-slot semaphore/fadvise_entry model that no longer exists on
  // metadata_processing_state. Left as a no-op until the new coalecing/
  // balancing worker loop is implemented.
}

}  // namespace sirius::scan_manager
