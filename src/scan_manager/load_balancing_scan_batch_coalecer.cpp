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
  auto state_ptr = state.get();
  _slots[uid]    = std::move(state);
  return state_ptr;
}

void load_balancing_scan_batch_coalecer::worker_loop([[maybe_unused]] std::stop_token const& stop)
{
  std::stop_callback stop_cb(
    stop, [this] { _ready_pipelines.enqueue(std::numeric_limits<std::size_t>::max()); });

  while (!stop.stop_requested()) {
    std::size_t pipeline_id;
    if (!_ready_pipelines.wait_dequeue_timed(pipeline_id, SEQUENCER_POLL_INTERVAL)) { continue; }
    if (pipeline_id == std::numeric_limits<std::size_t>::max()) { break; }

    auto it = _slots.find(pipeline_id);
    if (it == _slots.end()) {
      spdlog::error("Received ready signal for unknown pipeline_id {}", pipeline_id);
      continue;
    }
    while (!stop.stop_requested()) {
      process_entry(*it->second);
    }
  }
}

void load_balancing_scan_batch_coalecer::process_entry(metadata_processing_state& state)
{
  auto& batch_queue = state.queue;
  std::unique_ptr<op::scan::scan_info> entry;
  while (batch_queue.try_dequeue(entry)) {
    bool is_closed = entry == nullptr;
    auto batches   = [&]() {
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
      state.connector->push_split(std::move(op_data));
    }
    if (is_closed) {
      state.connector->close();
      break;
    }
  }
}

}  // namespace sirius::scan_manager
