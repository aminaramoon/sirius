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

namespace sirius::op {
class sirius_physical_operator;
}

namespace sirius::planner {

/// Reads SIRIUS_EXP_SCAN_BATCH_SORT at planning time; only the exact value "1" enables it.
[[nodiscard]] bool scan_batch_sort_enabled();

/// Attach conservative batch-local scan sorting hints before GPU pipeline expansion.
/// The nearest direct-reference HASH_JOIN / ORDER_BY / TOP_N consumer wins. Trace only
/// projection references and explicit filter output mappings; all other semantics are barriers.
/// No operator is removed or otherwise rewritten. Returns the number of admitted scans.
[[nodiscard]] unsigned admit_scan_batch_sort(op::sirius_physical_operator& root);

}  // namespace sirius::planner
