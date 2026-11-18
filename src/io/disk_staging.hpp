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

#ifdef CUCASCADE_HAS_CUFILE
#include <cucascade/io/cufile/placement.hpp>

#include <cstddef>

namespace sirius::io {

// Both plans use the effective piece sizes selected by plan_cufile_pool. A flat
// pool's explicit count overrides each eligible device's planned reactor count.
// Distinct slot sizes cannot share a reservation; distinct devices never do.
inline std::size_t staging_slots_for_device(int device,
                                            cucascade::io::cufile::pool_plan const& plan,
                                            cucascade::io::cufile::config const& cfg,
                                            std::size_t flat_reactors)
{
  for (auto const& group : plan.groups) {
    if (group.device == device && group.staging_slots != 0) {
      return flat_reactors == 0 ? group.staging_slots
                                : flat_reactors * cfg.staging_slots_per_reactor;
    }
  }
  return 0;
}

inline std::size_t shared_scan_staging_slots(int device,
                                             cucascade::io::cufile::config const& disk_cfg,
                                             cucascade::io::cufile::pool_plan const* scan_plan,
                                             cucascade::io::cufile::config const& scan_cfg,
                                             std::size_t scan_flat_reactors)
{
  if (!scan_plan || disk_cfg.staging_slot_bytes() != scan_cfg.staging_slot_bytes()) { return 0; }
  return staging_slots_for_device(device, *scan_plan, scan_cfg, scan_flat_reactors);
}

}  // namespace sirius::io
#endif
