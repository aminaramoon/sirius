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

// test
#include <catch.hpp>

// sirius
#include <memory/topology_index.hpp>

// cucascade
#include <cucascade/memory/topology_discovery.hpp>

// standard library
#include <utility>
#include <vector>

using cucascade::memory::gpu_topology_info;
using cucascade::memory::system_topology_info;
using sirius::memory::topology_index;

namespace {

/// Build a topology with the given (gpu id, numa node) pairs.
system_topology_info make_topology(std::vector<std::pair<unsigned int, int>> const& gpus)
{
  system_topology_info topo;
  topo.num_gpus = static_cast<unsigned int>(gpus.size());
  for (auto const& [id, node] : gpus) {
    gpu_topology_info gpu;
    gpu.id        = id;
    gpu.numa_node = node;
    topo.gpus.push_back(std::move(gpu));
  }
  return topo;
}

}  // namespace

TEST_CASE("topology_index maps each GPU to its NUMA node", "[topology_index]")
{
  topology_index index{make_topology({{0, 0}, {1, 0}, {2, 1}, {3, 1}})};

  CHECK(index.numa_node_of(0) == 0);
  CHECK(index.numa_node_of(1) == 0);
  CHECK(index.numa_node_of(2) == 1);
  CHECK(index.numa_node_of(3) == 1);
}

TEST_CASE("topology_index returns -1 for an unknown GPU", "[topology_index]")
{
  topology_index index{make_topology({{0, 0}})};

  CHECK(index.numa_node_of(7) == -1);
}

TEST_CASE("topology_index lists the GPUs on a NUMA node", "[topology_index]")
{
  topology_index index{make_topology({{0, 0}, {1, 0}, {2, 1}, {3, 1}, {4, 1}})};

  auto node0 = index.gpus_of(0);
  std::vector<int> node0_ids(node0.begin(), node0.end());
  CHECK(node0_ids == std::vector<int>{0, 1});

  auto node1 = index.gpus_of(1);
  std::vector<int> node1_ids(node1.begin(), node1.end());
  CHECK(node1_ids == std::vector<int>{2, 3, 4});
}

TEST_CASE("topology_index returns an empty span for a NUMA node with no GPUs", "[topology_index]")
{
  topology_index index{make_topology({{0, 0}})};

  CHECK(index.gpus_of(5).empty());
}

TEST_CASE("topology_index groups GPUs with an unknown (-1) NUMA node", "[topology_index]")
{
  topology_index index{make_topology({{0, -1}, {1, -1}, {2, 0}})};

  CHECK(index.numa_node_of(0) == -1);
  CHECK(index.numa_node_of(1) == -1);

  auto unknown = index.gpus_of(-1);
  std::vector<int> unknown_ids(unknown.begin(), unknown.end());
  CHECK(unknown_ids == std::vector<int>{0, 1});
}

TEST_CASE("topology_index retains the topology it was built from", "[topology_index]")
{
  topology_index index{make_topology({{0, 0}, {1, 1}})};

  auto const& topo = index.get_topology();
  CHECK(topo.num_gpus == 2u);
  REQUIRE(topo.gpus.size() == 2u);
  CHECK(topo.gpus[0].id == 0u);
  CHECK(topo.gpus[1].numa_node == 1);
}

TEST_CASE("topology_index handles an empty topology", "[topology_index]")
{
  topology_index index{system_topology_info{}};

  CHECK(index.numa_node_of(0) == -1);
  CHECK(index.gpus_of(0).empty());
}
