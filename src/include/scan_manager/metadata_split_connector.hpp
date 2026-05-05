/*
 * Copyright 2025, Sirius Contributors.
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

#include "io/gpu_ingestible.hpp"
#include "op/scan/sirius_gpu_scan_operator.hpp"
#include "scan_manager/split_connector.hpp"

#include <cudf/column/column.hpp>

#include <cucascade/memory/memory_space.hpp>
#include <duckdb/planner/expression.hpp>

#include <memory>

namespace sirius::scan_manager {

class metadata_split_connector : public split_connector {
 public:
  metadata_split_connector(io::gpu_ingestible& ingestible, op::scan::sirius_gpu_scan_operator& op);

  ~metadata_split_connector() override = default;

  std::optional<std::unique_ptr<op::operator_data>> get_next_split() override;

  [[nodiscard]] bool is_closed() const override;

  [[nodiscard]] bool has_more_splits() const override;

 private:
  io::gpu_ingestible* _ingestible{nullptr};
};

}  // namespace sirius::scan_manager
