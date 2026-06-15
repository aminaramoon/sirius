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

#include "op/scan/sirius_gpu_scan_operator_data.hpp"

#include <stop_token>

namespace sirius::scan_manager {

load_balancing_scan_batch_coalecer::metadata_processing_state*
load_balancing_scan_batch_coalecer::register_pipeline(op::scan::sirius_gpu_scan_operator* scan_op,
                                                      std::shared_ptr<balancing_strategy> balancer)
{
  if (!scan_op) return nullptr;

  auto connector   = scan_op->get_split_connector().shared_from_this();
  auto ingestible  = scan_op->get_ingestible().shared_from_this();
  auto coalecer    = ingestible->create_batch_coalecer();
  auto uid         = scan_op->get_operator_id();
  auto pipeline_id = scan_op->get_pipeline()->get_pipeline_id();
  auto state       = std::make_unique<metadata_processing_state>(
    uid,
    pipeline_id,
    std::move(coalecer),
    std::move(connector),
    std::move(balancer),
    ingestible->create_post_filter_and_projection_info());
  _pipeline_order.push_back(uid);
  auto state_ptr = state.get();
  _slots[uid]    = std::move(state);
  return state_ptr;
}

void load_balancing_scan_batch_coalecer::worker_loop([[maybe_unused]] std::stop_token const& stop)
{
  for (auto pipeline_id : _pipeline_order) {
    if (stop.stop_requested()) { break; }
    auto& state = *_slots[pipeline_id];
    process_entry(state, stop);
  }
}

void load_balancing_scan_batch_coalecer::process_entry(metadata_processing_state& state,
                                                       std::stop_token const& stop)
{
  std::stop_callback stop_cb(stop, [&state] { state.queue.enqueue(nullptr); });

  auto& batch_queue = state.queue;
  bool is_closed    = false;
  while (!is_closed && !stop.stop_requested()) {
    std::unique_ptr<op::scan::scan_info> entry;
    batch_queue.wait_dequeue(entry);
    is_closed    = entry == nullptr;
    auto batches = [&]() {
      if (is_closed) {
        return state.coalecer->flush();
      } else {
        return state.coalecer->push(std::move(entry));
      }
    }();
    for (auto& batch : batches) {
      auto op_data = std::make_unique<op::scan::scan_operator_input>(
        std::move(batch), state.filter_and_projection_info);
      auto dev_id = state.balancer->get_next_gpu(state.pipeline_id, op_data.get());
      if (dev_id >= 0) { op_data->set_preferred_device_id(dev_id); }

      auto fadvise_hints = op_data->get_fadvise_hints();
      if (!fadvise_hints.empty()) {
        for (auto& hint : fadvise_hints) {
          if (hint.datasource && !hint.ranges.empty()) {
            hint.datasource->fadvise(hint.ranges, dev_id);
          };
        }
      }
      op_data->prefetch(io::cache::prefetching_stage::opportunistic);
      state.connector->push_split(std::move(op_data));
    }
    if (is_closed) {
      state.connector->close();
      break;
    }
  }
}

}  // namespace sirius::scan_manager
