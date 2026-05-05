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

#include "scan_manager/metadata_split_connector.hpp"

#include "data/data_batch_utils.hpp"
#include "op/sirius_physical_operator.hpp"
#include "scan_manager/split_connector.hpp"

#include <cudf/column/column.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/table/table_view.hpp>

#include <cucascade/data/data_batch.hpp>
#include <cucascade/data/gpu_data_representation.hpp>

#include <optional>

namespace sirius::scan_manager {

metadata_split_connector::metadata_split_connector(io::gpu_ingestible& ingestible,
                                                   op::scan::sirius_gpu_scan_operator& op)
{
}

std::optional<std::unique_ptr<op::operator_data>> metadata_split_connector::get_next_split()
{
  return std::nullopt;
}

bool metadata_split_connector::is_closed() const { return true; }

bool metadata_split_connector::has_more_splits() const { return !is_closed(); }

}  // namespace sirius::scan_manager
