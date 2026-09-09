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

#include "io/rest/rest_ioctx.hpp"
#include "io/uring_remote/uring_remote_reactor.hpp"

namespace sirius::io::uring_remote {

/// Preserves S3 LIST, known-size opens, footer probes, and load balancing.
/// It remains a restful io_context so S3 globbing and eager readahead agree.
class uring_remote_ioctx final : public rest::rest_ioctx {
 public:
  uring_remote_ioctx(std::size_t n_reactors,
                     std::shared_ptr<rest::rest_reactor::reactor_context> ctx,
                     config transport_config);
};

}  // namespace sirius::io::uring_remote
