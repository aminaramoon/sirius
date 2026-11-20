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

#include <cucascade/cudf/datasource.hpp>
#include <cucascade/io/io_context.hpp>

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <utility>
#include <vector>

namespace sirius::io {

/**
 * @file bulk_read.hpp
 * @brief Sirius's choice of the bulk (vectored) Parquet read route per io context.
 *
 * The bulk route of @c op::scan::materialize_parquet reads every column chunk of a split's
 * file in ONE vectored device request.  cuCascade lets a backend ask for it with the
 * compile-time @c ioctx::prefers_bulk_io() (REST does: every request is a round trip).  The
 * cuFile backend says no, yet with fused lanes on it wants the whole split in one request too,
 * because it only fuses the slices of one request.  Sirius therefore asks for the route on a
 * context it owns by registering it here (see @c scan_manager.cufile_bulk_read); every reader
 * of the preference goes through @ref prefers_bulk_read, so the route and the decisions keyed
 * off it agree.
 */

namespace detail {

/// The io contexts a Sirius choice is registered on (see @ref context_registration).
struct context_registry {
  std::mutex mutex;
  std::vector<cucascade::io::ioctx const*> contexts;
  /// Fast path: nothing is registered (every backend but a configured cuFile one).
  std::atomic<std::size_t> count{0};

  [[nodiscard]] bool contains(cucascade::io::ioctx const& ctx) noexcept
  {
    if (count.load(std::memory_order_acquire) == 0) { return false; }
    std::lock_guard lock(mutex);
    return std::ranges::find(contexts, &ctx) != contexts.end();
  }
};

/// Contexts whose Parquet scans take the bulk route (@c scan_manager.cufile_bulk_read).
inline context_registry& bulk_read_registry() noexcept
{
  static context_registry registry;
  return registry;
}

/// Contexts whose Parquet device reads wait for their stream first
/// (@c scan_manager.cufile_bulk_read_sync_before_submit).
inline context_registry& sync_before_submit_registry() noexcept
{
  static context_registry registry;
  return registry;
}

}  // namespace detail

/// Whether Sirius asked for the bulk route on @p ctx (see @ref bulk_read_registration).
[[nodiscard]] inline bool bulk_read_registered(cucascade::io::ioctx const& ctx) noexcept
{
  return detail::bulk_read_registry().contains(ctx);
}

/// Whether Parquet column chunks served by @p ctx are read on the bulk route: the backend's
/// own preference (REST), or a Sirius registration (a cuFile context with
/// @c cufile_bulk_read on).
[[nodiscard]] inline bool prefers_bulk_read(cucascade::io::ioctx const& ctx) noexcept
{
  return ctx.prefers_bulk_io() || bulk_read_registered(ctx);
}

/// @ref prefers_bulk_read for the context serving @p source.
[[nodiscard]] inline bool prefers_bulk_read(cucascade::io::datasource const& source) noexcept
{
  if (source.prefers_bulk_io()) { return true; }
  auto const ctx = source.io_ctx();
  return ctx && bulk_read_registered(*ctx);
}

/// Whether Parquet device reads served by @p source wait for their stream before they are
/// submitted: its context is registered with @c cufile_bulk_read_sync_before_submit on.
[[nodiscard]] inline bool syncs_before_submit(cucascade::io::datasource const& source) noexcept
{
  auto const ctx = source.io_ctx();
  return ctx && detail::sync_before_submit_registry().contains(*ctx);
}

/// Process-wide counters of the waits @c cufile_bulk_read_sync_before_submit adds, for the
/// scan manager's shutdown stats.
struct sync_before_submit_stats {
  std::atomic<std::uint64_t> syncs{0};    ///< Stream synchronizes done before a submit.
  std::atomic<std::uint64_t> sync_ns{0};  ///< Time spent in them.
};

[[nodiscard]] inline sync_before_submit_stats& sync_before_submit_counters() noexcept
{
  static sync_before_submit_stats stats;
  return stats;
}

/**
 * @brief Registers one io context in one of the registries above for as long as it lives.
 *
 * The owner of the context (the scan manager, for its default cuFile context) holds one and
 * must destroy it before it releases the context, so the address is never confused with a
 * later context's.  Datasources that outlive the registration fall back to the default
 * behaviour.  Move-only.
 */
template <detail::context_registry& (*Registry)() noexcept>
class context_registration {
 public:
  context_registration() noexcept = default;

  explicit context_registration(cucascade::io::ioctx const& ctx) : _ctx(&ctx)
  {
    auto& registry = Registry();
    std::lock_guard lock(registry.mutex);
    registry.contexts.push_back(_ctx);
    registry.count.fetch_add(1, std::memory_order_release);
  }

  context_registration(context_registration&& other) noexcept
    : _ctx(std::exchange(other._ctx, nullptr))
  {
  }

  context_registration& operator=(context_registration&& other) noexcept
  {
    if (this != &other) {
      reset();
      _ctx = std::exchange(other._ctx, nullptr);
    }
    return *this;
  }

  context_registration(context_registration const&)            = delete;
  context_registration& operator=(context_registration const&) = delete;

  ~context_registration() { reset(); }

  /// Withdraw the registration (no-op when empty).
  void reset() noexcept
  {
    if (_ctx == nullptr) { return; }
    auto& registry = Registry();
    std::lock_guard lock(registry.mutex);
    if (auto const it = std::ranges::find(registry.contexts, _ctx); it != registry.contexts.end()) {
      registry.contexts.erase(it);
      registry.count.fetch_sub(1, std::memory_order_release);
    }
    _ctx = nullptr;
  }

  [[nodiscard]] bool active() const noexcept { return _ctx != nullptr; }

 private:
  cucascade::io::ioctx const* _ctx{nullptr};
};

/// Asks for the bulk route on one io context (@c scan_manager.cufile_bulk_read).
using bulk_read_registration = context_registration<&detail::bulk_read_registry>;

/// Makes one io context's Parquet device reads wait for their stream before they are submitted
/// (@c scan_manager.cufile_bulk_read_sync_before_submit): the bulk route once per vectored
/// request, the general route once per cuDF device read.
using sync_before_submit_registration = context_registration<&detail::sync_before_submit_registry>;

}  // namespace sirius::io
