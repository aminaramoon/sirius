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

#include "planner/scan_sort_plan_pass.hpp"

#include <duckdb/planner/expression/bound_reference_expression.hpp>
#include <log/logging.hpp>
#include <op/cudf_sort_order.hpp>
#include <op/scan/scan_batch_sort.hpp>
#include <op/sirius_physical_filter.hpp>
#include <op/sirius_physical_grouped_aggregate.hpp>
#include <op/sirius_physical_hash_join.hpp>
#include <op/sirius_physical_order.hpp>
#include <op/sirius_physical_projection.hpp>
#include <op/sirius_physical_table_scan.hpp>
#include <op/sirius_physical_top_n.hpp>
#include <op/sirius_physical_ungrouped_aggregate.hpp>

#include <cstdlib>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

namespace sirius::planner {
namespace {
using op::SiriusPhysicalOperatorType;
using op::scan::scan_sort_key;
using keys = std::vector<scan_sort_key>;

bool safe_key_type(logical_type const& type)
{
  switch (type.id()) {
    case type_id::BOOLEAN:
    case type_id::TINYINT:
    case type_id::SMALLINT:
    case type_id::INTEGER:
    case type_id::BIGINT:
    case type_id::UTINYINT:
    case type_id::USMALLINT:
    case type_id::UINTEGER:
    case type_id::UBIGINT:
    case type_id::DATE:
    case type_id::TIMESTAMP_SEC:
    case type_id::TIMESTAMP_MS:
    case type_id::TIMESTAMP:
    case type_id::TIMESTAMP_NS:
    case type_id::DECIMAL: return true;
    default: return false;  // No floats (NaNs), strings, nested or unsupported carriers in v1.
  }
}

// A row permutation below FIRST, LIMIT, WINDOW, or an unfamiliar operator can change values,
// even when the immediate consumer is a hash join. Fail closed for the whole experimental plan.
bool order_insensitive_plan(op::sirius_physical_operator const& node)
{
  switch (node.type) {
    case SiriusPhysicalOperatorType::TABLE_SCAN:
    case SiriusPhysicalOperatorType::PROJECTION:
    case SiriusPhysicalOperatorType::FILTER:
    case SiriusPhysicalOperatorType::DYNAMIC_FILTER:
    case SiriusPhysicalOperatorType::ORDER_BY:
    case SiriusPhysicalOperatorType::TOP_N: break;
    case SiriusPhysicalOperatorType::HASH_JOIN:
      switch (node.Cast<op::sirius_physical_hash_join>().join_type) {
        case duckdb::JoinType::INNER:
        case duckdb::JoinType::LEFT:
        case duckdb::JoinType::RIGHT:
        case duckdb::JoinType::OUTER:
        case duckdb::JoinType::SEMI:
        case duckdb::JoinType::ANTI:
        case duckdb::JoinType::MARK: break;
        default: return false;  // SINGLE can choose KEEP_ANY, depending on input order.
      }
      break;
    case SiriusPhysicalOperatorType::HASH_GROUP_BY:
      if (node.Cast<op::sirius_physical_grouped_aggregate>().has_first()) { return false; }
      break;
    case SiriusPhysicalOperatorType::UNGROUPED_AGGREGATE:
      for (auto const& expression :
           node.Cast<op::sirius_physical_ungrouped_aggregate>().aggregates) {
        auto const* aggregate = expression ? std::get_if<ast::aggregate>(&expression->v) : nullptr;
        if (!aggregate || aggregate->function() == aggregate_id::first) { return false; }
      }
      break;
    default: return false;
  }
  for (auto const& child : node.children) {
    if (!child || !order_insensitive_plan(*child)) { return false; }
  }
  return true;
}

unsigned attach(op::sirius_physical_operator& node, keys candidate)
{
  if (candidate.empty()) { return 0; }
  for (auto const& key : candidate) {
    if (key.output_ordinal >= node.types.size()) { return 0; }
  }
  if (node.type == SiriusPhysicalOperatorType::TABLE_SCAN) {
    auto& scan = node.Cast<op::sirius_physical_table_scan>();
    for (auto const& key : candidate) {
      if (!safe_key_type(scan.types[key.output_ordinal])) { return 0; }
    }
    if (!scan.batch_sort_keys.empty()) { return 0; }
    scan.batch_sort_keys = std::move(candidate);
    SIRIUS_LOG_INFO("[scan_batch_sort_plan] scan_node_id={} admitted=1 keys={}",
                    scan.scan_node_id,
                    scan.batch_sort_keys.size());
    return 1;
  }
  if (node.children.size() != 1 || !node.children[0]) { return 0; }
  switch (node.type) {
    case SiriusPhysicalOperatorType::PROJECTION: {
      auto const& projection = node.Cast<op::sirius_physical_projection>();
      for (auto& key : candidate) {
        if (key.output_ordinal >= projection.select_list.size()) { return 0; }
        auto const& expression = projection.select_list[key.output_ordinal];
        if (!expression || !expression->is_reference()) { return 0; }
        key.output_ordinal = expression->as_reference().column_index;
      }
      break;
    }
    case SiriusPhysicalOperatorType::FILTER: {
      auto const& filter = node.Cast<op::sirius_physical_filter>();
      if (auto const* columns = std::get_if<std::vector<cudf::size_type>>(&filter.output_columns)) {
        for (auto& key : candidate) {
          if (key.output_ordinal >= columns->size() || (*columns)[key.output_ordinal] < 0) {
            return 0;
          }
          key.output_ordinal = static_cast<std::size_t>((*columns)[key.output_ordinal]);
        }
      }
      break;
    }
    case SiriusPhysicalOperatorType::DYNAMIC_FILTER: break;
    default: return 0;
  }
  return attach(*node.children[0], std::move(candidate));
}

keys order_keys(duckdb::vector<duckdb::BoundOrderByNode> const& orders)
{
  keys result;
  for (auto const& order : orders) {
    if (!order.expression ||
        order.expression->GetExpressionClass() != duckdb::ExpressionClass::BOUND_REF ||
        (order.type != duckdb::OrderType::ASCENDING &&
         order.type != duckdb::OrderType::DESCENDING) ||
        (order.null_order != duckdb::OrderByNullType::NULLS_FIRST &&
         order.null_order != duckdb::OrderByNullType::NULLS_LAST)) {
      return {};
    }
    result.push_back({order.expression->Cast<duckdb::BoundReferenceExpression>().Index(),
                      op::to_cudf_order(order.type),
                      op::to_cudf_null_order(order.type, order.null_order)});
  }
  return result;
}

unsigned visit(op::sirius_physical_operator& node)
{
  unsigned admitted = 0;
  // Bottom-up, and attach never crosses another consumer: the nearest consumer owns the hint.
  for (auto& child : node.children) {
    if (child) { admitted += visit(*child); }
  }
  if (node.type == SiriusPhysicalOperatorType::HASH_JOIN && node.children.size() == 2) {
    auto const& join = node.Cast<op::sirius_physical_hash_join>();
    keys left;
    keys right;
    for (auto const& condition : join.conditions) {
      if (condition.comparison != comparison_type::equal || !condition.left || !condition.right ||
          !condition.left->is_reference() || !condition.right->is_reference()) {
        return admitted;
      }
      left.push_back({condition.left->as_reference().column_index});
      right.push_back({condition.right->as_reference().column_index});
    }
    admitted += attach(*node.children[0], std::move(left));
    admitted += attach(*node.children[1], std::move(right));
  } else if (node.children.size() == 1) {
    if (node.type == SiriusPhysicalOperatorType::ORDER_BY) {
      admitted +=
        attach(*node.children[0], order_keys(node.Cast<op::sirius_physical_order>().orders));
    } else if (node.type == SiriusPhysicalOperatorType::TOP_N) {
      admitted +=
        attach(*node.children[0], order_keys(node.Cast<op::sirius_physical_top_n>().orders));
    }
  }
  return admitted;
}
}  // namespace

bool scan_batch_sort_enabled()
{
  auto const* value = std::getenv("SIRIUS_EXP_SCAN_BATCH_SORT");
  return value && std::string_view(value) == "1";
}

unsigned admit_scan_batch_sort(op::sirius_physical_operator& root)
{
  if (!order_insensitive_plan(root)) { return 0; }
  return visit(root);
}

}  // namespace sirius::planner
