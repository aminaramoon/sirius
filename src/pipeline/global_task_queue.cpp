/*
 * Copyright 2025, Sirius Contributors.
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

#include "pipeline/global_task_queue.hpp"

#include "op/scan/duckdb_scan_task.hpp"
#include "pipeline/gpu_pipeline_task.hpp"

namespace sirius {
namespace pipeline {

void global_task_queue::push(std::unique_ptr<sirius::parallel::itask> task)
{
  if (dynamic_cast<op::duckdb_scan_task*>(task.get()) != nullptr) {
    _scan_task_queue.push(std::move(task));
  } else {
    _task_queue.push(std::move(task));
  }
}

std::unique_ptr<sirius::parallel::itask> global_task_queue::try_pop(
  [[maybe_unused]] int32_t device_id)
{
  std::unique_ptr<sirius::parallel::itask> task;
  if (_task_queue.try_pop(task)) { return task; }
  return nullptr;
}

std::unique_ptr<sirius::parallel::itask> global_task_queue::try_pop_scan_task()
{
  std::unique_ptr<sirius::parallel::itask> task;
  if (_scan_task_queue.try_pop(task)) { return task; }
  return nullptr;
}

}  // namespace pipeline
}  // namespace sirius
