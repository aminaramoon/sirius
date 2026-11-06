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

#include <cudf/types.hpp>

#include <cstddef>

namespace sirius::op::scan {

/// Experimental batch-local clustering hint, in the FINISHED scan output's coordinates.
/// Does not establish global ordering and must never replace a query's ORDER BY.
struct scan_sort_key {
  std::size_t output_ordinal;
  cudf::order direction          = cudf::order::ASCENDING;
  cudf::null_order null_ordering = cudf::null_order::BEFORE;
};

}  // namespace sirius::op::scan
