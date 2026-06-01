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

#include "exec/semi_future.hpp"
#include "io/types.hpp"

#include <cudf/io/datasource.hpp>
#include <cudf/io/text/byte_range_info.hpp>

#include <rmm/cuda_device.hpp>
#include <rmm/cuda_stream_view.hpp>

#include <cuda_runtime.h>

#include <atomic>
#include <cstddef>
#include <exception>
#include <memory>
#include <source_location>
#include <system_error>
#include <utility>
#include <variant>

namespace sirius::io::uring {

class request_manager {
 public:
  using error_type = std::variant<std::exception_ptr, cudaError_t, std::error_code>;
  explicit request_manager(std::size_t total_bytes) : total_bytes(total_bytes) {}
  ~request_manager()
  {
    if (has_error()) {
      promise.set_exception(first_exception);
    } else {
      promise.set_value(bytes_read.load(std::memory_order_acquire));
    }
  }

  void chunk_complete(std::size_t byted_read)
  {
    auto completed = bytes_read.fetch_add(byted_read, std::memory_order_acq_rel) + byted_read;
  }

  void report_error(const error_type& e, std::source_location loc = std::source_location::current())
  {
    if (!error_reported.exchange(true, std::memory_order_acq_rel)) {
      first_exception = to_exception_ptr(e, loc);
    }
  }

  [[nodiscard]] bool has_error() const noexcept
  {
    return error_reported.load(std::memory_order_acquire);
  }

  [[nodiscard]] exec::semi_future<size_t> get_future() noexcept
  {
    return promise.get_semi_future();
  }

  const std::size_t total_bytes;

 private:
  [[nodiscard]] std::exception_ptr to_exception_ptr(const error_type& e,
                                                    std::source_location loc) const noexcept
  {
    if (std::holds_alternative<std::exception_ptr>(e)) {
      return std::get<std::exception_ptr>(e);
    } else if (std::holds_alternative<cudaError_t>(e)) {
      auto err = std::get<cudaError_t>(e);
      return std::make_exception_ptr(
        std::runtime_error("CUDA error: " + std::string(cudaGetErrorString(err)) + " at " +
                           loc.file_name() + ":" + std::to_string(loc.line())));
    } else if (std::holds_alternative<std::error_code>(e)) {
      auto err = std::get<std::error_code>(e);
      return std::make_exception_ptr(std::system_error(
        err, "System error at " + std::string(loc.file_name()) + ":" + std::to_string(loc.line())));
    }
    return nullptr;  // Should never reach here
  }

  std::atomic<std::size_t> bytes_read{0};
  std::atomic<bool> error_reported{false};
  std::exception_ptr first_exception{nullptr};
  exec::promise<size_t> promise;
};

struct device_cpy_request {
  cudaError_t copy_async(uint8_t* host_buffer,
                         [[maybe_unused]] size_t bytes,
                         cudaEvent_t event = nullptr) noexcept
  {
    assert(dst != nullptr && "Caller must provide a valid device destination buffer for the copy.");
    assert(host_buffer != nullptr && "Caller must provide a valid host buffer for the copy.");
    assert(bytes > offset + size && "Caller must ensure the copy fits in the host buffer.");
    rmm::cuda_set_device_raii device_guard(rmm::cuda_device_id{device_id});
    auto* src_ptr   = host_buffer + offset;
    cudaError_t err = cudaMemcpyAsync(dst, src_ptr, size, cudaMemcpyHostToDevice, stream);
    if (err == cudaSuccess && event != nullptr) { err = cudaEventRecord(event, stream); }
    return err;
  }

  uint8_t* dst{nullptr};
  size_t offset{0};
  size_t size{0};
  rmm::cuda_stream_view stream;
  int device_id{-1};
};

struct chunked_rx_request {
  int fd;
  io_object_segment chunk;

  [[nodiscard]] io_object_segment get_remaining_chunk(size_t offset) const noexcept
  {
    if (offset >= chunk.size) return io_object_segment{0, 0};
    return io_object_segment{chunk.offset + offset, chunk.size - offset, chunk.data() + offset};
  }

  cudaError_t copy_h2d_async(cudaEvent_t event = nullptr) noexcept
  {
    if (cpy_req) [[likely]] {
      return cpy_req->copy_async(chunk.data(), chunk.size, event);
    } else {
      return cudaSuccess;
    }
  }

  [[nodiscard]] bool needs_event_for_synchronization() const noexcept
  {
    return !chunk.is_buffer_allocated() && cpy_req != nullptr;
  }

  std::unique_ptr<device_cpy_request> cpy_req;
  std::shared_ptr<request_manager> manager;
};

struct rx_request {
  static std::unique_ptr<rx_request> create(
    std::vector<std::unique_ptr<chunked_rx_request>> reqs) noexcept
  {
    return std::unique_ptr<rx_request>(new rx_request(std::move(reqs)));
  }

  [[nodiscard]] std::size_t size() const noexcept { return requests.size(); }

  static std::vector<std::unique_ptr<rx_request>> splits(std::unique_ptr<rx_request> req,
                                                         std::size_t n_splits) noexcept
  {
    std::vector<std::unique_ptr<rx_request>> result;
    auto chunks = req->get_all_chunks();
    req.reset(nullptr);
    if (n_splits == 0 || chunks.empty()) return result;

    std::size_t chunks_per_split = (chunks.size() + n_splits - 1) / n_splits;
    std::vector<std::unique_ptr<chunked_rx_request>> current_batch;

    for (auto& chunk : chunks) {
      current_batch.push_back(std::move(chunk));
      if (current_batch.size() >= chunks_per_split) {
        result.push_back(create(std::move(current_batch)));
        current_batch.clear();
      }
    }

    if (!current_batch.empty()) { result.push_back(create(std::move(current_batch))); }
    return result;
  }

  std::unique_ptr<chunked_rx_request> get_next_chunk() noexcept
  {
    if (requests.empty()) return nullptr;
    auto next_req = std::move(requests.back());
    requests.pop_back();
    return next_req;
  }

  std::vector<std::unique_ptr<chunked_rx_request>> get_all_chunks() noexcept
  {
    return std::move(requests);
  }

  exec::semi_future<size_t> get_future() noexcept
  {
    if (requests.empty()) {
      exec::promise<size_t> promise;
      promise.set_value(0);
      return promise.get_semi_future();
    }
    return requests.front()->manager->get_future();
  }

 private:
  std::vector<std::unique_ptr<chunked_rx_request>> requests;

  explicit rx_request(std::vector<std::unique_ptr<chunked_rx_request>> reqs)
    : requests(std::move(reqs))
  {
  }
};

}  // namespace sirius::io::uring
