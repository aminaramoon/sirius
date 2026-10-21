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

#include "preload/hw_caps.hpp"

#include <cucascade/memory/memory_space.hpp>

#ifdef CUCASCADE_HAS_CUFILE
#include <cucascade/io/cufile/platform.hpp>
#endif

namespace sirius::preload {

namespace {

bool real_gds_available() noexcept
{
#ifdef CUCASCADE_HAS_CUFILE
  return cucascade::io::cufile::gds_available();
#else
  return false;
#endif
}

}  // namespace

bool gpu_has_c2c(memory::topology_index const& topology, int device) noexcept
{
  for (auto const& gpu : topology.get_topology().gpus) {
    if (static_cast<int>(gpu.id) == device) {
      return gpu.runtime_attributes.has_value() && gpu.runtime_attributes->c2c;
    }
  }
  return false;
}

bool disk_uses_gds(cucascade::memory::memory_space const& disk) noexcept
{
  if (disk.get_tier() != cucascade::memory::Tier::DISK) { return false; }
  // Probe the platform first: without real GDS the context does not matter.
  if (!real_gds_available()) { return false; }
  try {
    auto const context = disk.get_io_context();
    return gds_in_use(
      true,
      context ? std::optional{context->type()} : std::optional<cucascade::io::io_context_type>{});
  } catch (...) {
    return false;
  }
}

}  // namespace sirius::preload
