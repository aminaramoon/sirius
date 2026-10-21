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

#include "catch.hpp"
#include "memory/topology_index.hpp"
#include "preload/hw_caps.hpp"

#include <cucascade/memory/config.hpp>
#include <cucascade/memory/memory_space.hpp>
#include <cucascade/memory/topology_discovery.hpp>

#ifdef CUCASCADE_HAS_CUFILE
#include <cucascade/io/cufile/platform.hpp>
#endif

#include <unistd.h>

#include <filesystem>
#include <optional>
#include <string>
#include <utility>
#include <vector>

using cucascade::io::io_context_type;
using cucascade::memory::gpu_runtime_attributes;
using cucascade::memory::gpu_topology_info;
using cucascade::memory::system_topology_info;
using sirius::memory::topology_index;
using namespace sirius::preload;

namespace {

/// A GPU entry; @p c2c empty means its runtime attributes were not discovered.
gpu_topology_info make_gpu(unsigned int id, std::optional<bool> c2c)
{
  gpu_topology_info gpu;
  gpu.id        = id;
  gpu.numa_node = 0;
  if (c2c) {
    gpu_runtime_attributes attrs;
    attrs.c2c              = *c2c;
    gpu.runtime_attributes = attrs;
  }
  return gpu;
}

}  // namespace

TEST_CASE("gpu_has_c2c reads the topology's runtime attributes", "[preload_policy][hw_caps]")
{
  system_topology_info topo;
  topo.gpus.push_back(make_gpu(0, std::nullopt));
  topo.gpus.push_back(make_gpu(1, true));
  topo.gpus.push_back(make_gpu(2, false));
  topo.num_gpus = static_cast<unsigned int>(topo.gpus.size());
  topology_index const index{topo, {0, 1, 2}};

  CHECK_FALSE(gpu_has_c2c(index, 0));  // attributes not discovered
  CHECK(gpu_has_c2c(index, 1));
  CHECK_FALSE(gpu_has_c2c(index, 2));
  CHECK_FALSE(gpu_has_c2c(index, 7));  // not in the topology
  CHECK_FALSE(gpu_has_c2c(index, -1));
}

TEST_CASE("gds_in_use needs real GDS and a cuFile (or default) context",
          "[preload_policy][hw_caps]")
{
  STATIC_REQUIRE_FALSE(gds_in_use(false, std::nullopt));
  STATIC_REQUIRE_FALSE(gds_in_use(false, io_context_type::cufile));
  STATIC_REQUIRE(gds_in_use(true, std::nullopt));  // the default context picks cuFile with GDS
  STATIC_REQUIRE(gds_in_use(true, io_context_type::cufile));
  STATIC_REQUIRE_FALSE(gds_in_use(true, io_context_type::uring));
}

TEST_CASE("disk_uses_gds on a DISK space without an io context follows the platform",
          "[preload_policy][hw_caps]")
{
  auto const dir = std::filesystem::temp_directory_path() /
                   ("sirius_preload_hw_caps_" + std::to_string(::getpid()));
  std::filesystem::create_directories(dir);
  {
    cucascade::memory::disk_memory_space_config cfg;
    cfg.disk_id         = 0;
    cfg.memory_capacity = std::size_t{1} << 30;
    cfg.mount_paths     = dir.string();
    cucascade::memory::memory_space const disk{cfg};
    REQUIRE(disk.get_io_context() == nullptr);

#ifdef CUCASCADE_HAS_CUFILE
    bool const gds = cucascade::io::cufile::gds_available();
#else
    bool const gds = false;
#endif
    CHECK(disk_uses_gds(disk) == gds_in_use(gds, std::nullopt));
    // Reading the context must not install one.
    CHECK(disk.get_io_context() == nullptr);
  }
  std::filesystem::remove_all(dir);
}
