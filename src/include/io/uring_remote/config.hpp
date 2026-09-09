/*
 * Copyright 2026, Sirius Contributors.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#pragma once

#include <cstddef>
#include <stdexcept>

namespace sirius::io::uring_remote {

/// Transport-only settings. Logical reads use scan_manager.rest verbatim.
struct config {
  bool enabled{false};
  std::size_t max_connections{256};
  std::size_t queue_depth{1024};
  std::size_t receive_buffer_bytes{256UL << 10};
  std::size_t max_header_bytes{64UL << 10};
  /// Explicit opt-in for local HTTP test endpoints; HTTPS always requires kTLS.
  bool allow_plaintext{false};

  void validate() const
  {
    if (max_connections == 0 || max_connections > 4096) {
      throw std::invalid_argument("uring_remote.max_connections must be in [1, 4096]");
    }
    if (queue_depth < 2 * max_connections || queue_depth > 32768) {
      throw std::invalid_argument(
        "uring_remote.queue_depth must be >= 2 * max_connections and <= 32768");
    }
    if (receive_buffer_bytes < 16384 || receive_buffer_bytes > (4UL << 20)) {
      throw std::invalid_argument("uring_remote.receive_buffer_bytes must be in [16 KiB, 4 MiB]");
    }
    if (max_header_bytes < 1024 || max_header_bytes > (1UL << 20)) {
      throw std::invalid_argument("uring_remote.max_header_bytes must be in [1 KiB, 1 MiB]");
    }
  }
};

}  // namespace sirius::io::uring_remote
