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
#include "io/disk_staging.hpp"
#include "sirius_config.hpp"

#include <yaml-cpp/yaml.h>  // IWYU pragma: keep

#include <fstream>
#include <string>
#include <variant>
#include <vector>

namespace {
cucascade::memory::system_topology_info single_gpu_topology()
{
  cucascade::memory::system_topology_info topology{};
  topology.num_gpus       = 1;
  topology.num_numa_nodes = 1;
  topology.gpus           = {{.id = 0, .numa_node = 0, .memory_total_bytes = 8ULL << 30}};
  topology.numa_nodes     = {{.id = 0, .memory_capacity = 8ULL << 30, .has_cpus = true}};
  return topology;
}

sirius::sirius_config disk_io_config(std::string const& io, bool low_level = false)
{
  auto root = YAML::Load(low_level ? R"(sirius:
  space:
    gpu: [{device_id: 0, memory_capacity: 1Gi}]
    disk:
      - {disk_id: 0, mount_path: /tmp/sirius_io_config, memory_capacity: 1Gi}
      - {disk_id: 1, mount_path: /tmp/sirius_io_config_1, memory_capacity: 1Gi}
)"
                                   : R"(sirius:
  memory:
    gpu: {usage_limit_bytes: 1Gi}
    disk: {downgrade_root_dirs: /tmp/sirius_io_config}
)");
  if (!io.empty()) {
    if (low_level) {
      root["sirius"]["space"]["disk"][0]["io"] = YAML::Load(io);
    } else {
      root["sirius"]["memory"]["disk"]["io"] = YAML::Load(io);
    }
  }
  return sirius::parsed_sirius_config::from_node(root).resolve(single_gpu_topology());
}

sirius::scan_manager::scan_manager_config scan_config(std::string const& body)
{
  return sirius::parsed_sirius_config::from_node(
           YAML::Load("sirius:\n  executor:\n    scan_manager:\n" + body))
    .get_scan_manager_config();
}
}  // namespace

TEST_CASE("DISK io absence retains lazy default and empty block uses cuCascade defaults",
          "[config][disk_io_config]")
{
  for (bool low_level : {false, true}) {
    CHECK_FALSE(disk_io_config("", low_level).get_disk_io_options());
    CHECK_FALSE(disk_io_config("null", low_level).get_disk_io_options());
    auto config = disk_io_config("{}", low_level);
    REQUIRE(config.get_disk_io_options());
    auto const& opt = *config.get_disk_io_options();
    cucascade::disk_io_context_options defaults;
    CHECK(opt.backend == defaults.backend);
    CHECK(opt.staging_numa_node == defaults.staging_numa_node);
    CHECK(opt.io.uring_n_reactors == defaults.io.uring_n_reactors);
#ifdef CUCASCADE_HAS_CUFILE
    CHECK(opt.io.cufile_n_reactors == defaults.io.cufile_n_reactors);
#endif
    CHECK(opt.io.uring.write_split_bytes == defaults.io.uring.write_split_bytes);
    CHECK(opt.io.uring.low_service_interval == defaults.io.uring.low_service_interval);
    CHECK(opt.io.uring.use_odirect == defaults.io.uring.use_odirect);
#ifdef CUCASCADE_HAS_CUFILE
    CHECK(opt.io.cufile.write_split_bytes == defaults.io.cufile.write_split_bytes);
    CHECK(opt.io.cufile.devices == std::vector<int>{0});
#endif
    for (auto const& space : config.get_memory_space_configs()) {
      if (auto disk = std::get_if<cucascade::memory::disk_memory_space_config>(&space)) {
        CHECK_FALSE(disk->io_context);  // resolution never starts a context
      }
    }
  }
}

TEST_CASE("DISK io options builder supports both YAML surfaces and every backend field",
          "[config][disk_io_config]")
{
  for (bool low_level : {false, true}) {
    auto config = disk_io_config(R"(
backend: automatic
uring_n_reactors: 7
staging_numa_node: 0
uring:
  n_max_concurrent_scans: 3
  slices_per_pass: 6
  range_batch_slices: 10
  use_odirect: false
  low_service_interval: 2
  write_split_bytes: 16Mi
)"
#ifdef CUCASCADE_HAS_CUFILE
                                 R"(
cufile_n_reactors: 9
cufile:
  max_inflight: 12
  max_op_bytes: 32Mi
  write_split_bytes: 128Mi
  gds_threshold: 64Ki
  use_odirect: false
  compat_mode: on
  staging_mode: never
  staging_piece_bytes: 4Mi
  staging_slots_per_reactor: 3
  fuse_gap_bytes: 1Mi
  max_staging_bytes_per_device: 512Mi
  reactors_per_device: 8
  pin_to_device_cpus: false
  devices: [0]
  storage_hint_path: /tmp/sirius-config-data.parquet
  poll_interval_us: 20
  low_service_interval: 5
)"
#endif
                                 ,
                                 low_level);
    REQUIRE(config.get_disk_io_options());
    auto const& opt = *config.get_disk_io_options();
    CHECK(opt.backend == cucascade::disk_io_backend::automatic);
    CHECK(opt.io.uring_n_reactors == 7);
#ifdef CUCASCADE_HAS_CUFILE
    CHECK(opt.io.cufile_n_reactors == 9);
#endif
    CHECK(opt.staging_numa_node == 0);
    CHECK(opt.io.uring.n_max_concurrent_scans == 3);
    CHECK(opt.io.uring.n_max_concurrent_scans_explicit);
    CHECK(opt.io.uring.slices_per_pass == 6);
    CHECK(opt.io.uring.range_batch_slices == 10);
    CHECK_FALSE(opt.io.uring.use_odirect);
    CHECK(opt.io.uring.low_service_interval == 2);
    CHECK(opt.io.uring.write_split_bytes == 16UL << 20);
#ifdef CUCASCADE_HAS_CUFILE
    auto const& cf = opt.io.cufile;
    CHECK(cf.max_inflight == 12);
    CHECK(cf.max_op_bytes == 32UL << 20);
    CHECK(cf.write_split_bytes == 128UL << 20);
    CHECK(cf.gds_threshold == 64UL << 10);
    CHECK_FALSE(cf.use_odirect);
    CHECK(cf.compat_mode == cucascade::io::cufile::compat::on);
    CHECK(cf.staging_mode == cucascade::io::cufile::staging::never);
    CHECK(cf.staging_piece_bytes == 4UL << 20);
    CHECK(cf.staging_slots_per_reactor == 3);
    CHECK(cf.fuse_gap_bytes == 1UL << 20);
    CHECK(cf.max_staging_bytes_per_device == 512UL << 20);
    CHECK(cf.reactors_per_device == 8);
    CHECK_FALSE(cf.pin_to_device_cpus);
    CHECK(cf.devices == std::vector<int>{0});
    CHECK(cf.storage_hint_path == "/tmp/sirius-config-data.parquet");
    CHECK(cf.poll_interval_us == 20);
    CHECK(cf.low_service_interval == 5);
#endif
  }
  CHECK(disk_io_config("{backend: uring}").get_disk_io_options()->backend ==
        cucascade::disk_io_backend::uring);
#ifdef CUCASCADE_HAS_CUFILE
  auto config = disk_io_config("{backend: cufile, uring_n_reactors: 0, cufile_n_reactors: 0}");
  CHECK(config.get_disk_io_options()->backend == cucascade::disk_io_backend::cufile);
  CHECK(config.get_disk_io_options()->io.cufile_n_reactors == 0);
  CHECK_NOTHROW(
    disk_io_config("{backend: cufile, staging_numa_node: 0, uring: {use_odirect: false}}"));
#else
  for (auto const* body : {"{backend: cufile}", "{cufile: {}}", "{cufile_n_reactors: 0}"}) {
    CHECK_THROWS_WITH(disk_io_config(body),
                      Catch::Matchers::ContainsSubstring("built without cuFile"));
  }
#endif
}

TEST_CASE("DISK io validates backend counts combinations and shared backend settings",
          "[config][disk_io_config]")
{
  auto invalid = GENERATE("{backend: native}",
                          "{backend: CUFILE}",
                          "{uring_n_reactors: -1}",
                          "{uring_n_reactors: 0}",
                          "{backend: uring, uring_n_reactors: 0}",
                          "{cufile_n_reactors: -1}",
                          "{staging_numa_node: -2}",
                          "{staging_numa_node: 4}",
                          "{unknown: 1}",
                          "{uring: {low_service_interval: -1}}",
                          "{uring: {write_split_bytes: 4095}}",
                          "{uring: {write_split_bytes: -1}}",
                          "{uring: {slices_per_pass: 65}}",
                          "{cufile: {write_split_bytes: 4095}}",
                          "{cufile: {max_inflight: 0}}",
                          "{cufile: {staging_piece_bytes: 6000}}",
                          "{cufile: {storage_hint_path: /}}",
                          "{cufile: {n_max_concurrent_scans: 0}}",
                          "{cufile: {devices: [0, 0]}}",
                          "{cufile: {devices: [1]}}",
                          "{cufile: {use_odirect: false, compat_mode: off}}");
  CAPTURE(invalid);
  CHECK_THROWS_WITH(disk_io_config(invalid), Catch::Matchers::ContainsSubstring("disk.io"));
  CHECK_THROWS_WITH(disk_io_config(invalid, true), Catch::Matchers::ContainsSubstring("disk.io"));
}

TEST_CASE("DISK io is specified only once across low-level spaces", "[config][disk_io_config]")
{
  auto root = YAML::Load(R"(sirius:
  space:
    disk:
      - {disk_id: 0, io: {backend: uring}}
      - {disk_id: 1, io: {backend: cufile}}
)");
  CHECK_THROWS_WITH(sirius::parsed_sirius_config::from_node(root),
                    Catch::Matchers::ContainsSubstring("specify once"));
}

TEST_CASE("Scan throughput fields parse and survive io_config conversion",
          "[config][scan_manager][io_throughput_config]")
{
  auto io =
    scan_config(
      "      cache: {mode: os}\n"
      "      uring: {use_odirect: false, low_service_interval: 7, write_split_bytes: 16Mi}\n"
      "      cufile: {write_split_bytes: 32Mi, fuse_gap_bytes: 1Mi}\n")
      .to_io_config();
  CHECK_FALSE(io.uring.use_odirect);
  CHECK(io.uring.low_service_interval == 7);
  CHECK(io.uring.write_split_bytes == 16UL << 20);
#ifdef CUCASCADE_HAS_CUFILE
  CHECK(io.cufile.write_split_bytes == 32UL << 20);
  CHECK(io.cufile.fuse_gap_bytes == 1UL << 20);
#endif
  for (auto const* cache : {"none", "cucs"}) {
    CHECK(scan_config(std::string("      cache: {mode: ") + cache +
                      "}\n      uring: {use_odirect: true}\n")
            .uring.use_odirect);
  }
  for (auto const* body : {"      uring: {low_service_interval: -1}\n",
                           "      uring: {write_split_bytes: 4095}\n",
                           "      cufile: {write_split_bytes: -1}\n",
                           "      cache: {mode: os}\n      uring: {use_odirect: true}\n",
                           "      cache: {mode: cucs}\n      uring: {use_odirect: false}\n"}) {
    CHECK_THROWS(scan_config(body));
  }
  auto zero = scan_config(
                "      uring: {low_service_interval: 0, write_split_bytes: 0}\n"
                "      cufile: {write_split_bytes: 0}\n")
                .to_io_config();
  CHECK(zero.uring.low_service_interval == 0);
  CHECK(zero.uring.write_split_bytes == 0);
#ifdef CUCASCADE_HAS_CUFILE
  CHECK(zero.cufile.write_split_bytes == 0);
#endif
  auto defaults = scan_config("      uring: {}\n").to_io_config();
  CHECK(defaults.uring.use_odirect);
  CHECK(defaults.uring.low_service_interval == cucascade::io::uring::config{}.low_service_interval);
  CHECK(defaults.uring.write_split_bytes == cucascade::io::uring::config{}.write_split_bytes);
#ifdef CUCASCADE_HAS_CUFILE
  CHECK(defaults.cufile.write_split_bytes == cucascade::io::cufile::config{}.write_split_bytes);
#endif
}

#ifdef CUCASCADE_HAS_CUFILE
TEST_CASE("DISK staging reservations sum only matching scan slots on the same GPU",
          "[config][disk_io_config][io_throughput_config]")
{
  namespace cf = cucascade::io::cufile;
  cf::config disk_cfg;
  cf::config scan_cfg;
  disk_cfg.staging_slots_per_reactor = 3;
  scan_cfg.staging_slots_per_reactor = 2;
  cf::pool_plan disk_plan;
  disk_plan.groups = {{.device = 0, .staging_slots = 24}, {.device = 1, .staging_slots = 12}};
  cf::pool_plan scan_plan;
  scan_plan.groups = {{.device = 1, .staging_slots = 64}, {.device = 0, .staging_slots = 32}};
  auto disk        = [&](int device, std::size_t flat = 0) {
    return sirius::io::staging_slots_for_device(device, disk_plan, disk_cfg, flat);
  };
  auto scan = [&](int device, std::size_t flat = 0) {
    return sirius::io::shared_scan_staging_slots(device, disk_cfg, &scan_plan, scan_cfg, flat);
  };
  CHECK(disk(0) + scan(0) == 56);
  CHECK(disk(1) + scan(1) == 76);
  CHECK(disk(0, 5) + scan(0, 7) == 29);
  CHECK(disk(1, 5) + scan(1) == 79);
  CHECK(disk(0) + scan(0, 7) == 38);
  CHECK(disk(2) == 0);
  CHECK(scan(2, 7) == 0);
  CHECK(sirius::io::shared_scan_staging_slots(0, disk_cfg, nullptr, scan_cfg, 0) == 0);
  scan_cfg.staging_piece_bytes *= 2;
  CHECK(scan(0) == 0);
  CHECK(scan(0, 7) == 0);
  scan_cfg                          = disk_cfg;
  scan_plan.groups[1].staging_slots = 0;
  CHECK(scan(0, 7) == 0);
  disk_plan.groups[0].staging_slots = 0;
  CHECK(disk(0, 5) == 0);
}
#endif

TEST_CASE("Maximum I/O throughput recipes are valid YAML configurations",
          "[config][disk_io_config][io_throughput_config]")
{
  std::ifstream doc(SIRIUS_PROJECT_ROOT "/docs/super-sirius/configuration.md");
  REQUIRE(doc.is_open());
  bool recipes = false;
  bool yaml    = false;
  std::string block;
  unsigned count = 0;
  for (std::string line; std::getline(doc, line);) {
    if (line == "<!-- io-throughput-recipes:start -->") {
      recipes = true;
      continue;
    }
    if (line == "<!-- io-throughput-recipes:end -->") { break; }
    if (!recipes) { continue; }
    if (line == "```yaml") {
      yaml = true;
      block.clear();
    } else if (yaml && line == "```") {
      CAPTURE(block);
#ifndef CUCASCADE_HAS_CUFILE
      if (block.find("backend: cufile") != std::string::npos) {
        yaml = false;
        continue;
      }
#endif
      CHECK_NOTHROW(
        sirius::parsed_sirius_config::from_node(YAML::Load(block)).resolve(single_gpu_topology()));
      ++count;
      yaml = false;
    } else if (yaml) {
      block += line + '\n';
    }
  }
  CHECK(count > 0);
}
