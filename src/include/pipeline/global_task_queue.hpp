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

#include "exec/interruptible_mpmc.hpp"
#include "parallel/task.hpp"

#include <memory>

namespace sirius {
namespace pipeline {

class global_task_queue : std::enable_shared_from_this<global_task_queue> {
 public:
  global_task_queue()  = default;
  ~global_task_queue() = default;

  void push(std::unique_ptr<sirius::parallel::itask> task);

  std::unique_ptr<sirius::parallel::itask> try_pop([[maybe_unused]] int32_t device_id);

  std::unique_ptr<sirius::parallel::itask> try_pop_scan_task();

 private:
  exec::interruptible_mpmc<std::unique_ptr<sirius::parallel::itask>> _task_queue;
  exec::interruptible_mpmc<std::unique_ptr<sirius::parallel::itask>> _scan_task_queue;
};

}  // namespace pipeline
}  // namespace sirius
