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

#include "log/logging.hpp"

#include <cuda/stream>

#include <cucascade/cudf/gpu_data_representation.hpp>
#include <cucascade/cudf/host_data_representation.hpp>
#include <cucascade/data/data_batch.hpp>
#include <cucascade/data/encoded_data_representation.hpp>
#include <cucascade/memory/memory_space.hpp>
#include <data/data_batch_utils.hpp>
#include <data/sirius_converter_registry.hpp>

#include <cstdint>
#include <format>
#include <memory>
#include <optional>
#include <stdexcept>
#include <tuple>
#include <typeinfo>
#include <variant>

namespace sirius {
namespace pipeline {

struct lock_to_existing_batch {
  cucascade::read_only_data_batch ro_lock;
};

// A new batch was created to appropriately prepare the batch.
// The ro_lock is a lock on this new batch which may be stored for
// future reference.
struct lock_to_new_batch {
  std::shared_ptr<cucascade::data_batch> new_batch;
  cucascade::read_only_data_batch ro_lock;
};

using lock_and_prepare_batch_result = std::variant<lock_to_existing_batch, lock_to_new_batch>;

namespace detail {

/**
 * @brief Whether @p data is exactly a cucascade::encoded_data_representation.
 *
 * Exact type match on purpose: other device-resident encodings, such as
 * compressed_device_representation (decompressed in place by the scan operators), keep the
 * same-space pass-through of lock_and_prepare_batch.
 */
[[nodiscard]] inline bool is_encoded_representation(
  const cucascade::idata_representation* data) noexcept
{
  return data != nullptr && typeid(*data) == typeid(cucascade::encoded_data_representation);
}

/**
 * @brief Throw a descriptive error when @p data is encoded and no decoder can turn it into a
 *        gpu_table_representation; a no-op for every other representation.
 *
 * Run before any conversion or clone of the batch, so a missing decoder surfaces with the
 * encoding name instead of the converter registry's type-name error, and the batch is untouched.
 *
 * @throws std::runtime_error if @p data is an encoded_data_representation and no
 *         encoded_data_representation -> gpu_table_representation converter is registered.
 */
inline void require_decoder_if_encoded(const cucascade::idata_representation& data,
                                       const cucascade::representation_converter_registry& registry,
                                       const cucascade::memory::memory_space& target_space,
                                       uint64_t batch_id)
{
  if (!is_encoded_representation(&data) ||
      registry.has_converter_for<cucascade::gpu_table_representation>(data)) {
    return;
  }
  const auto& encoded = data.cast<cucascade::encoded_data_representation>();
  throw std::runtime_error(
    std::format("lock_and_prepare_batch: batch {} holds encoded data (encoding '{}', version {}) "
                "on GPU {} but no encoded_data_representation -> gpu_table_representation "
                "decoder is registered to prepare it for GPU {}",
                batch_id,
                encoded.get_metadata().encoding,
                encoded.get_metadata().version,
                encoded.get_memory_space().get_device_id(),
                target_space.get_device_id()));
}

/**
 * @brief Decode the encoded representation held under @p mut in place into a
 *        gpu_table_representation in @p target_space; the batch keeps its id.
 *
 * @pre is_encoded_representation(mut.get_data()); checked, a violation throws std::bad_cast
 *      before the batch is touched.
 * @throws std::runtime_error if no encoded_data_representation -> gpu_table_representation
 *         converter is registered. The batch is left unchanged.
 * @throws rmm::out_of_memory (or whatever the decoder throws) if decoding fails. The encoded
 *         representation stays installed.
 */
inline void decode_encoded_in_place(cucascade::mutable_data_batch& mut,
                                    cucascade::representation_converter_registry& registry,
                                    const cucascade::memory::memory_space& target_space,
                                    ::cuda::stream_ref stream)
{
  const auto& data = *mut.get_data();
  static_cast<void>(data.cast<cucascade::encoded_data_representation>());  // checks the @pre
  require_decoder_if_encoded(data, registry, target_space, mut.get_batch_id());
  mut.convert_to<cucascade::gpu_table_representation>(registry, &target_space, stream);
}

/**
 * @brief GPU-target preparation under an exclusive lock, for a batch whose caller has already
 *        released its shared lock.
 *
 * The upgrade to the exclusive lock is not atomic (shared released, then exclusive acquired): a
 * concurrent consumer may have converted the batch in the gap, so this dispatches on the state
 * observed under the exclusive lock:
 * - already in the target space: encoded bytes are decoded in place, anything else is used as is;
 * - on a different GPU: cloned (a GPU-resident batch is never moved cross-device);
 * - on HOST or DISK: converted in place (move), freeing the spilled copy.
 *
 * Encoded data on a different GPU is cloned through the same decoder, so a missing decoder is
 * reported by require_decoder_if_encoded before the clone. The returned shared lock is acquired
 * after the exclusive lock is released (again not atomic): a concurrent downgrade may move the
 * batch out of @p target_space in between, so the caller can end up holding a shared lock on a
 * batch outside @p target_space.
 */
[[nodiscard]] inline lock_and_prepare_batch_result prepare_for_gpu_under_exclusive_lock(
  const std::shared_ptr<cucascade::data_batch>& batch,
  const cucascade::memory::memory_space& target_space,
  cucascade::representation_converter_registry& registry,
  ::cuda::stream_ref stream)
{
  {
    cucascade::mutable_data_batch mut_accessor = batch->to_mutable();

    const auto* current_space = mut_accessor.get_memory_space();
    if (current_space != nullptr && current_space->get_id() == target_space.get_id()) {
      // Already in the target space. Encoded bytes must still be decoded before an operator reads
      // them; checked before the GPU-tier clone test below, which would otherwise copy them into
      // a new batch. Anything else skips the wasteful same-space deep copy.
      if (is_encoded_representation(mut_accessor.get_data())) {
        decode_encoded_in_place(mut_accessor, registry, target_space, stream);
      }
    } else if (mut_accessor.get_current_tier() == cucascade::memory::Tier::GPU) {
      // Upgraded to a DIFFERENT GPU in the gap: never move a GPU-resident batch cross-device
      // (that would steal it from the device that just materialized it) — clone instead. The
      // clone happens under the exclusive lock we already hold; pinning the state here avoids
      // yet another non-atomic lock transition.
      require_decoder_if_encoded(
        *mut_accessor.get_data(), registry, target_space, mut_accessor.get_batch_id());
      // TODO(dhruv9vats): attach a DataBatch probe so compute IDs resolve to a traced clone.
      auto clone = mut_accessor.clone_to<cucascade::gpu_table_representation>(
        registry, sirius::get_next_batch_id(), &target_space, stream);
      return lock_to_new_batch{
        .new_batch = clone,
        .ro_lock   = clone->to_read_only(),
      };
      // mut_accessor to the original batch dropped.
    } else {
      // HOST/DISK -> GPU upgrade/readback: convert in place (move), freeing the spilled copy.
      mut_accessor.convert_to<cucascade::gpu_table_representation>(registry, &target_space, stream);
    }
    // exclusive mutable access dropped.
  }

  return lock_to_existing_batch{
    .ro_lock = batch->to_read_only(),
  };
}

}  // namespace detail

/**
 * @brief Lock or prepare a single data batch for processing in the requested memory space.
 *
 * Acquires a read-only (shared) lock on the batch. On a memory-space mismatch the behavior
 * depends on where the data currently lives:
 *
 * - GPU -> GPU (cross-device): the batch is cloned into the target space via
 *   read_only_data_batch::clone_to while only the shared lock is held. The source is never
 *   exclusively locked or mutated — concurrent readers (e.g. probe tasks on another GPU sharing
 *   a build batch) proceed during the transfer — and it stays resident in its original space for
 *   consumers local to that device. The returned accessor references the NEW (cloned) batch.
 *   Source lifetime is ownership-driven: whoever still needs the original (repositories, other
 *   tasks) holds a reference to it; once the last reference drops it is freed. The source is
 *   back in the idle state as soon as this function returns, so it is downgrade-eligible.
 *
 * - HOST/DISK -> GPU (upgrade/readback): the batch is upgraded to a mutable (exclusive) lock and
 *   converted in-place, freeing the spilled copy — intentional move semantics for the common
 *   single-consumer case. The returned accessor references the same batch. If a concurrent
 *   consumer converted the batch during the (non-atomic) lock upgrade, the state observed under
 *   the exclusive lock is re-dispatched: same space -> no-op; a different GPU -> clone (a
 *   GPU-resident batch is never moved cross-device); still host/disk -> move as planned.
 *
 * - Same space, GPU target, and the data is exactly a cucascade::encoded_data_representation:
 *   the batch is upgraded to an exclusive lock and decoded in place through the registered
 *   encoded -> gpu_table converter, so operators never see undecoded bytes. The returned
 *   accessor references the same batch. Other same-space representations (including
 *   compressed_device_representation) are returned as is under the shared lock.
 *
 * @param batch                   The batch to lock/prepare.
 * @param requested_memory_space  Target memory space; may be nullptr to use the batch's current
 *                                space.
 * @param stream                  CUDA stream used for any data-movement kernels.
 * @return A read_only_data_batch holding the shared lock (on the clone for cross-GPU inputs),
 *         or std::nullopt on failure.
 * @throws rmm::out_of_memory  If a GPU memory allocation fails during the conversion or clone.
 * @throws std::runtime_error  If encoded data bound for a GPU target has no registered decoder.
 */
inline std::optional<lock_and_prepare_batch_result> lock_and_prepare_batch(
  const std::shared_ptr<cucascade::data_batch>& batch,
  const cucascade::memory::memory_space* requested_memory_space,
  ::cuda::stream_ref stream)
{
  if (!batch) { return std::nullopt; }

  // Opportunistic stream rebind (best-effort, non-blocking): if the batch already resides in
  // the target GPU memory space, rebind its deallocation stream to this task's stream so that
  // when the task later frees the data the free lands on the active stream's free list rather
  // than on the stream the data was produced on. We use try_to_mutable() so we never block --
  // if another reader holds the batch (e.g. a probe task on another GPU sharing a build batch)
  // or it is otherwise busy, we simply skip the rebind; correctness is unaffected. Gating on a
  // space match guarantees the stream and the data live on the same device (important for
  // multi-GPU). The mismatch case below converts via `stream`, which already allocates the new
  // table on it, so no rebind is needed there.
  if (std::optional<cucascade::mutable_data_batch> mut = batch->try_to_mutable()) {
    const auto* current_space = mut->get_memory_space();
    const auto* rebind_target =
      requested_memory_space != nullptr ? requested_memory_space : current_space;
    if (current_space != nullptr && rebind_target != nullptr &&
        current_space->get_id() == rebind_target->get_id() &&
        rebind_target->get_tier() == cucascade::memory::Tier::GPU) {
      mut->rebind_stream(stream);
    }
  }

  // Acquire a read-only lock
  cucascade::read_only_data_batch read_accessor = batch->to_read_only();

  // Determine the target memory space
  const auto* target_space =
    requested_memory_space != nullptr ? requested_memory_space : read_accessor.get_memory_space();
  if (target_space == nullptr) { return std::nullopt; }

  // Memory space matches — return the read-only accessor directly, unless it holds encoded bytes
  // on the target GPU, which must be decoded first. Those are re-checked and decoded under the
  // exclusive lock; the batch keeps its id.
  if (read_accessor.get_memory_space() != nullptr &&
      read_accessor.get_memory_space()->get_id() == target_space->get_id()) {
    if (target_space->get_tier() != cucascade::memory::Tier::GPU ||
        !detail::is_encoded_representation(read_accessor.get_data())) {
      return lock_to_existing_batch{.ro_lock = std::move(read_accessor)};
    }
    std::ignore =
      cucascade::data_batch::to_idle(std::move(read_accessor));  // release the read lock
    return detail::prepare_for_gpu_under_exclusive_lock(
      batch, *target_space, sirius::converter_registry::get(), stream);
  }

  // Memory space mismatch — clone or move depending on where the data lives.
  auto& registry = sirius::converter_registry::get();

  switch (target_space->get_tier()) {
    case cucascade::memory::Tier::GPU: {
      if (read_accessor.get_current_tier() == cucascade::memory::Tier::GPU) {
        // Cross-GPU: clone into the target space under the shared lock. The source is never
        // exclusively locked or mutated. clone_to routes through convert_gpu_to_gpu, which waits
        // on the source's writer event and synchronizes its copy stream before returning, so
        // holding read_accessor across the call is sufficient ordering and the copy is complete
        // when it returns. No normalize_gpu_list_offsets here: that fixup only reverses the
        // INT64 offsets promotion of host->GPU reconstruction, and a GPU->GPU copy preserves the
        // source's column types (a GPU-resident source is already normalized — the same-space
        // fast path above depends on that invariant).
        detail::require_decoder_if_encoded(
          *read_accessor.get_data(), registry, *target_space, read_accessor.get_batch_id());
        // TODO(dhruv9vats): attach a DataBatch probe so compute IDs resolve to a traced clone.
        auto clone = read_accessor.clone_to<cucascade::gpu_table_representation>(
          registry, get_next_batch_id(), target_space, stream);
        return lock_to_new_batch{
          .new_batch = clone,
          .ro_lock   = clone->to_read_only(),
        };
        // read_accessor to the original batch dropped.
      }

      // HOST/DISK -> GPU upgrade/readback, re-dispatched under the exclusive lock.
      std::ignore =
        cucascade::data_batch::to_idle(std::move(read_accessor));  // release the read lock
      return detail::prepare_for_gpu_under_exclusive_lock(batch, *target_space, registry, stream);
    }
    case cucascade::memory::Tier::HOST: {
      {
        // release the read lock to get mutable access
        std::ignore = cucascade::data_batch::to_idle(std::move(read_accessor));
        cucascade::mutable_data_batch mut_accessor = batch->to_mutable();
        // Same non-atomic-upgrade race as the GPU arm: skip if already in the target space.
        if (mut_accessor.get_memory_space() == nullptr ||
            mut_accessor.get_memory_space()->get_id() != target_space->get_id()) {
          auto host_reservation =
            const_cast<cucascade::memory::memory_space*>(target_space)
              ->make_reservation_or_null(mut_accessor.get_data()->get_size_in_bytes());
          if (host_reservation != nullptr) {
            mut_accessor.convert_to<cucascade::host_data_representation>(
              registry, *host_reservation, stream);
          } else {
            SIRIUS_LOG_WARN(
              "lock_or_prepare_batch: host reservation failed for batch {} ({} bytes) — "
              "proceeding without reservation, converter may OOM",
              mut_accessor.get_batch_id(),
              mut_accessor.get_data()->get_size_in_bytes());
            mut_accessor.convert_to<cucascade::host_data_representation>(
              registry, target_space, stream);
          }
        }
        // exclusive mutable access dropped.
      }
      return lock_to_existing_batch{
        .ro_lock = batch->to_read_only(),
      };
    }
    default:
      SIRIUS_LOG_ERROR("lock_or_prepare_batch: unsupported target tier for batch {}",
                       read_accessor.get_batch_id());
      return std::nullopt;
  }
}

}  // namespace pipeline
}  // namespace sirius
