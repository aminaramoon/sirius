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

#include "io/rest/rest_reactor.hpp"
#include "io/uring_remote/config.hpp"

namespace sirius::io::uring_remote {

/// The REST read scheduler with an io_uring + kernel TLS data transport.
/// Inheriting the scheduler preserves host/device/cache-fragment behavior.
class uring_remote_reactor final : public rest::rest_reactor {
 public:
  uring_remote_reactor(std::shared_ptr<reactor_context> ctx,
                       config transport_config,
                       std::string_view tname = "uring_remote");
};

std::unique_ptr<rest::remote_transport> make_transport(rest::config const& reads,
                                                       config const& transport);

}  // namespace sirius::io::uring_remote
