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

#include "io/rest/types.hpp"

#include <curl/curl.h>

#include <functional>
#include <memory>
#include <stdexcept>
#include <vector>

namespace sirius::io::rest {

/// A ring failure is fatal to the worker: every outstanding operation must be
/// drained before any caller buffer can be released.
class remote_transport_error : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};

/// Worker-confined adapter. The REST scheduler owns logical reads, segmentation,
/// retries, staging, and CUDA completion for either network transport.
class remote_transport {
 public:
  struct completion {
    std::size_t slot;
    CURLcode error{CURLE_OK};  // Preserve the existing retry classification.
    long status{0};
    std::string detail;
  };

  virtual ~remote_transport()                     = default;
  virtual int completion_fd() const noexcept      = 0;
  virtual int timeout_ms() const noexcept         = 0;
  virtual void submit(std::size_t slot,
                      authorized_request request,
                      range bytes,
                      buf_sink& sink,
                      header_capture& headers)    = 0;
  virtual void warmup(authorized_request request) = 0;
  virtual void flush()                            = 0;
  virtual std::vector<completion> poll()          = 0;
  /// Cancel and drain kernel operations before releasing caller buffers.
  virtual void shutdown() noexcept = 0;
};

using remote_transport_factory = std::function<std::unique_ptr<remote_transport>()>;

}  // namespace sirius::io::rest
