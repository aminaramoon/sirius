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

#include <io/gpu_ingestible.hpp>

#include <stdexcept>

namespace sirius::io {

std::unique_ptr<gpu_ingestible> make_gpu_ingestible(
  std::unique_ptr<ingestible_table_info> table_info)
{
  if (!table_info) {
    throw std::runtime_error("[make_gpu_ingestible] table_info is null.");
  }
  switch (table_info->type()) {
    case ingestible_type::PARQUET:
      throw std::runtime_error(
        "[make_gpu_ingestible] PARQUET ingestible is not yet implemented.");
    case ingestible_type::DUCKDB:
      throw std::runtime_error(
        "[make_gpu_ingestible] DUCKDB ingestible is not yet implemented.");
  }
  throw std::runtime_error("[make_gpu_ingestible] unknown ingestible_type.");
}

}  // namespace sirius::io
