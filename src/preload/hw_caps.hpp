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

#include "memory/topology_index.hpp"

#include <cucascade/io/io_context.hpp>

#include <optional>

namespace cucascade::memory {
class memory_space;
}  // namespace cucascade::memory

namespace sirius::preload {

/// Hardware capabilities forced by tests. Production leaves both empty, so they are detected.
struct hw_caps_override {
  std::optional<bool> c2c;
  std::optional<bool> gds;
};

/**
 * @brief Whether GPU @p device reaches host memory over a cache-coherent chip-to-chip link.
 *
 * Read from the topology's runtime attributes. False when the device is not in the topology or
 * its runtime attributes were not discovered.
 */
[[nodiscard]] bool gpu_has_c2c(memory::topology_index const& topology, int device) noexcept;

/**
 * @brief Whether a DISK space's transfers use GPUDirect Storage.
 *
 * @param gds_available Whether real GDS is usable on this host (not cuFile compatibility mode).
 * @param context_type The type of the space's io context; empty when none is set yet, in which
 *        case the DISK converters adopt the default context, whose automatic backend picks cuFile
 *        exactly when real GDS is available.
 */
[[nodiscard]] constexpr bool gds_in_use(
  bool gds_available, std::optional<cucascade::io::io_context_type> context_type) noexcept
{
  if (!gds_available) { return false; }
  return !context_type || *context_type == cucascade::io::io_context_type::cufile;
}

/**
 * @brief Whether DISK space @p disk moves data with GPUDirect Storage.
 *
 * Reads the space's io context without installing one. False for a space that is not DISK.
 * Unless the context is explicitly a non-cuFile one, this probes the cuFile driver, which opens
 * it on first use (and may create `cufile.log` in the working directory): call it once, at
 * start-up, and only when preload runs and a DISK space exists.
 */
[[nodiscard]] bool disk_uses_gds(cucascade::memory::memory_space const& disk) noexcept;

}  // namespace sirius::preload
