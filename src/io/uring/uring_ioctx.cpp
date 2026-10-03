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

#include "io/uring/uring_ioctx.hpp"

#include "exec/thread_util.hpp"
#include "io/uring/uring_reactor.hpp"
#include "log/logging.hpp"

#include <chrono>
#include <cstdint>
#include <exception>
#include <format>
#include <memory>
#include <tuple>

namespace sirius::io::uring {

namespace {

constexpr std::chrono::milliseconds GAUGE_SAMPLE_PERIOD{250};

[[nodiscard]] bool gauges_enabled() noexcept
{
  try {
    return sirius::log::get_sink()->should_log(sirius::log::level::debug);
  } catch (...) {
    return false;
  }
}

}  // namespace

uring_ioctx::uring_ioctx(size_t n_reactors, std::shared_ptr<uring_reactor::reactor_context> ctx)
  : templated_ioctx<uring_reactor>(n_reactors, [ctx = std::move(ctx), i = 0]() mutable {
      return std::make_unique<uring_reactor>(ctx, std::format("reactor-{}", i++));
    })
{
}

void uring_ioctx::start()
{
  templated_ioctx<uring_reactor>::start();
  if (_gauge_sampler.joinable()) { return; }
  _gauge_sampler = std::jthread([this](std::stop_token const& st) { sample_gauges(st); });
  std::ignore    = sirius::exec::thread_util::set_thread_name(_gauge_sampler, "uring_gauges");
}

void uring_ioctx::shutdown() noexcept
{
  if (_gauge_sampler.joinable()) {
    _gauge_sampler.request_stop();
    try {
      _gauge_sampler.join();
    } catch (...) {  // NOLINT(bugprone-empty-catch)
      // Joining a live sampler cannot fail in practice; never let teardown throw.
    }
  }
  templated_ioctx<uring_reactor>::shutdown();
}

std::vector<uring_reactor::gauges> uring_ioctx::reactor_gauges() noexcept
{
  std::vector<uring_reactor::gauges> out;
  try {
    out.reserve(_reactors.size());
    for (auto& reactor : _reactors) {
      out.push_back(reactor->take_gauges());
    }
  } catch (...) {
    out.clear();
  }
  return out;
}

void uring_ioctx::sample_gauges(std::stop_token const& stop_token)
{
  using clock = std::chrono::steady_clock;
  std::vector<uring_reactor::gauges> previous;
  auto previous_at = clock::now();

  while (!stop_token.stop_requested()) {
    {
      std::unique_lock lock(_sampler_mutex);
      // Returns early on stop; the predicate never fires otherwise.
      _sampler_cv.wait_for(lock, stop_token, GAUGE_SAMPLE_PERIOD, [] { return false; });
    }
    if (stop_token.stop_requested()) { break; }
    if (!gauges_enabled()) {
      previous.clear();
      continue;
    }

    auto const now     = clock::now();
    auto const current = reactor_gauges();
    auto const seconds = std::chrono::duration<double>(now - previous_at).count();
    try {
      for (std::size_t i = 0; i < current.size(); ++i) {
        auto const& g      = current[i];
        auto const started = i < previous.size() ? g.requests_started - previous[i].requests_started
                                                 : std::uint64_t{0};
        auto const bytes =
          i < previous.size() ? g.bytes_submitted - previous[i].bytes_submitted : std::uint64_t{0};
        bool const idle = g.max_inflight_ops == 0 && g.pending_ops == 0 &&
                          g.active_remaining_slices == 0 && g.queued_requests == 0 &&
                          started == 0 && bytes == 0;
        if (idle) { continue; }
        SIRIUS_LOG_DEBUG(
          "[uring_gauges] reactor={} inflight={} max_inflight={} pending_ops={} "
          "active_slices={} queued_requests={} queued_MiB={} started={} MiB_s={:.0f}",
          i,
          g.inflight_ops,
          g.max_inflight_ops,
          g.pending_ops,
          g.active_remaining_slices,
          g.queued_requests,
          g.queued_bytes >> 20,
          started,
          seconds > 0 ? static_cast<double>(bytes) / static_cast<double>(1 << 20) / seconds : 0.0);
      }
    } catch (...) {  // NOLINT(bugprone-empty-catch)
      // Diagnostics only: a failed format must not take the sampler down.
    }
    previous    = current;
    previous_at = now;
  }
}

}  // namespace sirius::io::uring
