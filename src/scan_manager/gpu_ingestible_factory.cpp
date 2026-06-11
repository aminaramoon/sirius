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

#include "scan_manager/gpu_ingestible_factory.hpp"

#include "log/logging.hpp"
#include "op/scan/parquet_gpu_ingestible.hpp"
#include "op/scan/pinned_table_gpu_ingestible.hpp"
#include "op/scan/scan_plan.hpp"
#include "op/scan/scan_utils.hpp"
#include "scan_manager/sirius_scan_manager.hpp"

#include <algorithm>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace sirius::scan_manager {

std::shared_ptr<op::scan::gpu_ingestible> gpu_ingestible_factory::produce(
  std::unique_ptr<op::scan::ingestible_table_info> table_info, std::size_t op_id)
{
  if (!table_info) { return nullptr; }

  return op::scan::make_gpu_ingestible(std::move(table_info));
}

}  // namespace sirius::scan_manager
