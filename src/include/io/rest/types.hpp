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

#include "io/io_request.hpp"
#include "io/s3/s3_object_ref.hpp"
#include "io/types.hpp"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

namespace sirius::io::rest {

/// Write-callback target for one in-flight transfer.  libcurl hands the
/// response body to the reactor in arbitrarily-sized pieces; @c buf_sink copies
/// them into @c dst at a running @c written cursor and ALWAYS reports the full
/// incoming size back to curl (never a short count, which would abort the
/// transfer with CURLE_WRITE_ERROR).  Bytes beyond @c capacity are counted in
/// @c total_received but not stored, so the reactor can detect a server that
/// ignored the Range header (e.g. returned the whole object) after the fact.
struct buf_sink {
  std::uint8_t* dst{nullptr};
  std::size_t capacity{0};
  std::size_t written{0};
  std::size_t total_received{0};

  void reset() noexcept
  {
    written        = 0;
    total_received = 0;
  }
};

/// Response headers the reactor inspects on completion.  @c content_range
/// validates that a 206 honored the requested byte range; @c retry_after drives
/// the retry delay when the server asks the client to back off.
struct header_capture {
  std::string content_range;
  std::string retry_after;

  void reset() noexcept
  {
    content_range.clear();
    retry_after.clear();
  }
};

/// One ranged HTTP GET.  The unit the reactor submits as a single easy handle.
///
/// @c object identifies the bucket/key so the reactor can re-authorize on every
/// attempt (presigned URLs expire).  @c chunk carries the file range
/// [offset, offset+size) and its destination buffer; a null destination
/// (@c chunk.data() == nullptr) means the reactor stages the read through one
/// of its own pinned bounce slots before the host->device copy.  @c cpy_req is
/// non-null only for device-bound reads.  @c attempt persists across retries
/// (the chunk is re-enqueued by the retry engine) so backoff can grow.
struct rest_chunked_rx_request {
  s3::s3_object_ref object;
  io_object_segment chunk;
  std::size_t file_size{0};
  std::size_t attempt{0};
  std::unique_ptr<device_cpy_request> cpy_req;
  std::shared_ptr<request_manager> manager;

  /// True iff this read's bytes must be host->device copied after landing.
  [[nodiscard]] bool is_device() const noexcept { return cpy_req != nullptr; }

  /// Issue the host->device copy (batch + optional event) for this chunk.  The
  /// host source is @c chunk.data() (the bounce slot or caller buffer).
  cudaError_t copy_h2d_async(cudaEvent_t event = nullptr) noexcept
  {
    if (cpy_req) { return cpy_req->copy_async(chunk.data(), chunk.size, event); }
    return cudaSuccess;
  }

  /// True iff the H2D copy must be synchronized through a CUDA event: it stages
  /// through a reactor-owned bounce slot (null destination) that can only be
  /// recycled once the copy off it has completed.
  [[nodiscard]] bool needs_event_for_synchronization() const noexcept
  {
    return !chunk.is_buffer_allocated() && cpy_req != nullptr;
  }
};

/// The per-reactor request container for the REST backend: the shared
/// rx_request_t template instantiated for the REST chunk type.
using rest_rx_request = rx_request_t<rest_chunked_rx_request>;

}  // namespace sirius::io::rest
