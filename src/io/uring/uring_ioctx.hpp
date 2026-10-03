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

#include "io/templated_ioctx.hpp"
#include "io/uring/uring_reactor.hpp"

#include <condition_variable>
#include <mutex>
#include <stop_token>
#include <thread>
#include <vector>

namespace sirius::io::uring {

// ---------------------------------------------------------------------------
// uring_ioctx
// ---------------------------------------------------------------------------

/**
 * @brief io_uring-backed ioctx. Thin specialisation of
 *        @c templated_ioctx<uring_reactor>.
 */
class uring_ioctx : public templated_ioctx<uring_reactor> {
 public:
  /// Build a pool of @p n_reactors reactors, all sharing @p ctx (one context
  /// per pool: it carries the per-reactor @c config and the pinned bounce-staging
  /// resource, which must outlive this ioctx).  The ioctx config is sourced from
  /// the reactors themselves — see @c templated_ioctx.
  uring_ioctx(size_t n_reactors, std::shared_ptr<uring_reactor::reactor_context> ctx);

  [[nodiscard]] io_context_type type() const noexcept override { return io_context_type::uring; }

  /// Start the reactors, then the gauge sampler (see @ref reactor_gauges).
  void start() override;

  /// Stop the gauge sampler, then the reactors.
  void shutdown() noexcept override;

  /// One gauge snapshot per reactor, in reactor order.  Each call restarts the
  /// reactors' max-inflight windows.
  [[nodiscard]] std::vector<uring_reactor::gauges> reactor_gauges() noexcept;

 private:
  /// Logs one DEBUG line per busy reactor every sampling period.  The thread
  /// only checks the log level while the sink is above DEBUG, so it costs a
  /// timed wakeup and nothing else in normal runs.
  void sample_gauges(std::stop_token const& stop_token);

  std::mutex _sampler_mutex;
  std::condition_variable_any _sampler_cv;
  /// Declared last: its destructor stops and joins the sampler before the
  /// members it reads go away.
  std::jthread _gauge_sampler;
};

}  // namespace sirius::io::uring
