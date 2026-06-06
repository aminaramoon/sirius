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

#include "io/io_context.hpp"

#include "exec/semi_future.hpp"
#include "exec/try.hpp"
#include "io/cache/prefetching_cache.hpp"

#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <memory>
#include <utility>

namespace sirius::io {

sirius_ioctx::sirius_ioctx()  = default;
sirius_ioctx::~sirius_ioctx() = default;

void sirius_ioctx::initialize_cache(cache::buffer_pool* pool,
                                    size_t inflight_budget_chunks) noexcept
{
  // One-shot.  Repeated calls are silent no-ops so callers can be
  // robust to multiple wiring sites.
  if (_cache) return;
  try {
    _cache = std::make_unique<cache::prefetching_cache>(pool, this, inflight_budget_chunks);
  } catch (const std::exception& e) {
    SIRIUS_LOG_ERROR("prefetching_cache construction failed: {}", e.what());
    _cache.reset();
  } catch (...) {
    SIRIUS_LOG_ERROR("prefetching_cache construction failed: unknown error");
    _cache.reset();
  }
}

void sirius_ioctx::shutdown_cache() noexcept { _cache.reset(); }

}  // namespace sirius::io
