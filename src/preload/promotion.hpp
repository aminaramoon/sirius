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

/**
 * @file promotion.hpp
 * @brief One promotion of one batch to a higher tier: HOST->GPU, DISK->GPU or DISK->HOST.
 *
 * @ref promote_one runs on a preload lane thread without any executor lock. It reserves target
 * memory before it touches the batch, never waits for a batch lock, re-checks the ceiling and the
 * hold-offs right before the copy, and maps every failure to a @ref promotion_outcome.
 */

#include "memory/memory_wait_board.hpp"
#include "preload/preload_policy.hpp"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <span>
#include <vector>

namespace cucascade {
class idata_representation;
namespace memory {
class exclusive_stream_pool;
class memory_space;
}  // namespace memory
}  // namespace cucascade

namespace sirius::exec {
class query_lifecycle_registry;
}  // namespace sirius::exec

namespace sirius::preload {

/// The representations a promotion distinguishes, by exact type.
enum class repr_kind : std::uint8_t {
  host_data,  ///< cucascade::host_data_representation
  disk_data,  ///< cucascade::disk_data_representation
  encoded,    ///< cucascade::encoded_data_representation
  other       ///< anything else: never promoted
};

/// The kind of @p data, matched on its exact dynamic type.
[[nodiscard]] repr_kind classify(cucascade::idata_representation const& data) noexcept;

/**
 * @brief Peak GPU bytes needed to materialize @p data on a GPU.
 *
 * `peak_materialization_bytes(data)`, plus, for host and disk tables, the INT64 copy of every
 * STRING column's INT32 offsets (`8 x (rows + 1)`, at any nesting depth): the upload widens those
 * offsets while the INT32 column is still allocated. Saturates instead of overflowing.
 */
[[nodiscard]] std::size_t promotion_peak_bytes(cucascade::idata_representation const& data);

/// A GPU target as a promotion sees it.
struct promotion_gpu {
  cucascade::memory::memory_space* space{nullptr};
  /// A promotion whose exact need exceeds this is refused with @c too_big.
  std::size_t max_batch_bytes{0};
  /// The preload stream pool of this GPU. Must be non-null.
  cucascade::memory::exclusive_stream_pool* streams{nullptr};
};

/// A HOST staging target as a promotion sees it.
struct promotion_host {
  cucascade::memory::memory_space* space{nullptr};
  /// Indices into @ref promotion_env::gpus of the GPUs on this host's NUMA node. Staging yields
  /// while any of them is held off or at its downgrade trigger: their spills need this host.
  std::vector<std::size_t> local_gpus;
  /// Pool that orders the staging copy (DISK->HOST copies do not run on it). Null uses the
  /// calling thread's default stream.
  cucascade::memory::exclusive_stream_pool* ordering_streams{nullptr};
};

/// Everything @ref promote_one reads besides the job. Borrowed; must outlive the call.
struct promotion_env {
  exec::query_lifecycle_registry& lifecycle;
  memory::memory_wait_board const& board;
  /// GPU targets, by @ref target::index.
  std::span<promotion_gpu const> gpus;
  /// HOST targets, by @ref target::index.
  std::span<promotion_host const> hosts;
  /// Tests only. Runs on the promoting thread once the target reservation is made and admitted,
  /// before the batch is locked.
  std::function<void(promotion_job const&)> test_hook{};
};

/**
 * @brief Promote one batch to the job's target. Never throws and never blocks on a batch lock.
 *
 * Order of checks (each refusal returns at once and releases what it holds):
 * 1. A work lease on the job's query, held for the whole call (@c retired otherwise).
 * 2. GPU target: a hold-off on that GPU; HOST target: a hold-off or trigger on a NUMA-local GPU
 *    (@c held_off).
 * 3. A non-blocking reservation of @c est_reserve in the target space (@c no_memory), then the
 *    ceiling: consumed bytes, counting that reservation, at or below `job.ceiling` and the space
 *    below its downgrade trigger (@c gated).
 * 4. The batch: alive (@c expired), idle and lockable without waiting (@c contended), not already
 *    promoted (@c already_resident), where the job expects it (@c stale), and for class B still
 *    unclaimed by a task (@c stale with the unchanged location).
 * 5. The representation: GPUs only ever receive a decoded @c gpu_table_representation, so an
 *    encoded batch targets a GPU only through a registered decoder; anything else is
 *    @c unsupported.
 * 6. The exact need (@ref promotion_peak_bytes with slack, times the OOM multiplier, for a GPU;
 *    the target's allocation size for HOST): @c too_big above the GPU's per-batch limit, else
 *    the reservation grows to it (@c no_memory).
 * 7. Hold-offs (@c held_off) and the ceiling (@c gated) once more, right before the copy.
 *
 * A GPU copy runs on a stream of the GPU's preload pool with the reservation attached to the
 * calling thread's allocation tracker under a fail-on-overrun policy, so a misestimate throws
 * @c rmm::out_of_memory (@c oom) instead of exceeding the ceiling. A HOST copy draws from the
 * reservation. An encoded batch is staged with @c relocate_to, any other with @c convert_to.
 * Any other exception gives @c failed. On @c promoted, the result carries the exact need in bytes,
 * the copy's @ref promotion_kind and a conversion sequence taken under the batch lock.
 */
[[nodiscard]] promotion_result promote_one(promotion_job const& job,
                                           promotion_env const& env) noexcept;

}  // namespace sirius::preload
