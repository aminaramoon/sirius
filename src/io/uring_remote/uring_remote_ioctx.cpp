/*
 * Copyright 2026, Sirius Contributors.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#include "io/uring_remote/uring_remote_ioctx.hpp"

#include <format>

namespace sirius::io::uring_remote {

uring_remote_reactor::uring_remote_reactor(std::shared_ptr<reactor_context> ctx,
                                           config transport_config,
                                           std::string_view tname)
  : rest_reactor(
      [&] {
        transport_config.validate();
        if (!ctx) throw std::invalid_argument("uring_remote: null reactor context");
        auto reads            = ctx->cfg();
        reads.max_connections = transport_config.max_connections;
        return std::make_shared<reactor_context>(
          reads, ctx->authorizer(), ctx->host_memory_resource());
      }(),
      tname,
      [ctx, transport_config] { return make_transport(ctx->cfg(), transport_config); })
{
}

uring_remote_ioctx::uring_remote_ioctx(std::size_t n_reactors,
                                       std::shared_ptr<rest::rest_reactor::reactor_context> ctx,
                                       config transport_config)
  : rest_ioctx([&] {
      if (n_reactors == 0)
        throw std::invalid_argument("uring_remote: reactor count must be positive");
      std::vector<std::unique_ptr<rest::rest_reactor>> reactors;
      reactors.reserve(n_reactors);
      for (std::size_t i = 0; i < n_reactors; ++i) {
        reactors.push_back(std::make_unique<uring_remote_reactor>(
          ctx, transport_config, std::format("uring-s3-{}", i)));
      }
      return reactors;
    }())
{
}

}  // namespace sirius::io::uring_remote
