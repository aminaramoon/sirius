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

#include <catch.hpp>
#include <duckdb/planner/expression/bound_reference_expression.hpp>
#include <duckdb/planner/operator/logical_dummy_scan.hpp>
#include <expression/ast/node.hpp>
#include <helper/type_conversions.hpp>
#include <op/sirius_physical_filter.hpp>
#include <op/sirius_physical_grouped_aggregate.hpp>
#include <op/sirius_physical_hash_join.hpp>
#include <op/sirius_physical_order.hpp>
#include <op/sirius_physical_projection.hpp>
#include <op/sirius_physical_table_scan.hpp>
#include <op/sirius_physical_ungrouped_aggregate.hpp>
#include <planner/scan_sort_plan_pass.hpp>

#include <cstdlib>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace {
using namespace sirius::op;
using sirius::planner::admit_scan_batch_sort;

duckdb::vector<sirius::logical_type> ints(std::size_t width)
{
  return duckdb::vector<sirius::logical_type>(width,
                                              sirius::logical_type::make(sirius::type_id::INTEGER));
}

std::unique_ptr<sirius::ast::node> ref(uint32_t index)
{
  return std::make_unique<sirius::ast::node>(
    sirius::ast::reference{index, sirius::logical_type::make(sirius::type_id::INTEGER)});
}

duckdb::unique_ptr<sirius_physical_table_scan> make_scan_leaf()
{
  // Storage ids and scan projection deliberately differ from output ordinals.
  return duckdb::make_uniq<sirius_physical_table_scan>(
    ints(3),
    duckdb::BoundTableFunction(duckdb::TableFunction("stub", {}, nullptr, nullptr)),
    nullptr,
    ints(3),
    duckdb::vector<duckdb::ColumnIndex>{
      duckdb::ColumnIndex(91), duckdb::ColumnIndex(8), duckdb::ColumnIndex(42)},
    duckdb::vector<std::size_t>{2, 0, 1},
    duckdb::vector<std::string>{"a", "b", "c"},
    nullptr,
    10,
    duckdb::ExtraOperatorInfo{},
    duckdb::vector<duckdb::Value>{},
    duckdb::virtual_column_map_t{},
    3);
}

duckdb::unique_ptr<sirius_physical_projection> projection(
  duckdb::unique_ptr<sirius_physical_operator> child, std::vector<uint32_t> inputs)
{
  duckdb::vector<std::unique_ptr<sirius::ast::node>> expressions;
  for (auto input : inputs) {
    expressions.push_back(ref(input));
  }
  auto result =
    duckdb::make_uniq<sirius_physical_projection>(ints(inputs.size()), std::move(expressions), 10);
  result->children.push_back(std::move(child));
  return result;
}

duckdb::unique_ptr<sirius_physical_order> order(
  duckdb::unique_ptr<sirius_physical_operator> child,
  std::size_t ordinal,
  duckdb::OrderType direction   = duckdb::OrderType::ASCENDING,
  duckdb::OrderByNullType nulls = duckdb::OrderByNullType::NULLS_LAST)
{
  duckdb::vector<duckdb::BoundOrderByNode> orders;
  orders.emplace_back(
    direction,
    nulls,
    duckdb::make_uniq<duckdb::BoundReferenceExpression>(duckdb::LogicalType::INTEGER, ordinal));
  // ORDER BY's output projections must not be used to resolve its INPUT expression.
  auto result = duckdb::make_uniq<sirius_physical_order>(
    child->types, std::move(orders), duckdb::vector<std::size_t>{2, 0, 1}, 10);
  result->children.push_back(std::move(child));
  return result;
}

duckdb::unique_ptr<sirius_physical_hash_join> join(
  duckdb::unique_ptr<sirius_physical_operator> left,
  duckdb::unique_ptr<sirius_physical_operator> right)
{
  duckdb::LogicalDummyScan logical(duckdb::TableIndex(0));
  logical.types = duckdb::vector<duckdb::LogicalType>(left->types.size() + right->types.size(),
                                                      duckdb::LogicalType::INTEGER);
  duckdb::vector<sirius::join_condition> conditions;
  sirius::join_condition condition;
  condition.left  = ref(1);
  condition.right = ref(2);
  conditions.push_back(std::move(condition));
  return duckdb::make_uniq<sirius_physical_hash_join>(logical,
                                                      std::move(left),
                                                      std::move(right),
                                                      std::move(conditions),
                                                      duckdb::JoinType::INNER,
                                                      duckdb::vector<std::size_t>{},
                                                      duckdb::vector<std::size_t>{},
                                                      duckdb::vector<sirius::logical_type>{},
                                                      10);
}
}  // namespace

TEST_CASE("scan batch sort uses child-local direct join references", "[planner][scan_batch_sort]")
{
  auto left  = make_scan_leaf();
  auto right = make_scan_leaf();
  auto* lhs  = left.get();
  auto* rhs  = right.get();
  auto plan  = join(std::move(left), std::move(right));
  REQUIRE(admit_scan_batch_sort(*plan) == 2);
  REQUIRE(lhs->batch_sort_keys.size() == 1);
  REQUIRE(rhs->batch_sort_keys.size() == 1);
  CHECK(lhs->batch_sort_keys[0].output_ordinal == 1);
  CHECK(rhs->batch_sort_keys[0].output_ordinal == 2);

  SECTION("SINGLE joins can choose arbitrary input rows and are refused")
  {
    lhs->batch_sort_keys.clear();
    rhs->batch_sort_keys.clear();
    plan->join_type = duckdb::JoinType::SINGLE;
    CHECK(admit_scan_batch_sort(*plan) == 0);
    CHECK(lhs->batch_sort_keys.empty());
    CHECK(rhs->batch_sort_keys.empty());
  }
  SECTION("non-equality join fails closed")
  {
    lhs->batch_sort_keys.clear();
    rhs->batch_sort_keys.clear();
    plan->conditions[0].comparison = sirius::comparison_type::not_distinct_from;
    CHECK(admit_scan_batch_sort(*plan) == 0);
    CHECK(lhs->batch_sort_keys.empty());
    CHECK(rhs->batch_sort_keys.empty());
  }
}

TEST_CASE("scan batch sort traces only explicit projection and filter mappings",
          "[planner][scan_batch_sort]")
{
  auto leaf   = make_scan_leaf();
  auto* raw   = leaf.get();
  auto filter = duckdb::make_uniq<sirius_physical_filter>(
    ints(2), ref(0), 10, std::vector<cudf::size_type>{2, 0});
  filter->children.push_back(std::move(leaf));
  auto project    = projection(std::move(filter), {1, 0});
  auto* projected = project.get();
  auto plan       = order(
    std::move(project), 0, duckdb::OrderType::DESCENDING, duckdb::OrderByNullType::NULLS_LAST);

  SECTION("direct mapping uses output coordinates")
  {
    REQUIRE(admit_scan_batch_sort(*plan) == 1);
    REQUIRE(raw->batch_sort_keys.size() == 1);
    CHECK(raw->batch_sort_keys[0].output_ordinal == 0);
    CHECK(raw->batch_sort_keys[0].direction == cudf::order::DESCENDING);
    CHECK(raw->batch_sort_keys[0].null_ordering == cudf::null_order::BEFORE);
  }
  SECTION("computed key is not unwrapped")
  {
    projected->select_list[0] = std::make_unique<sirius::ast::node>(
      sirius::ast::cast{ref(1), sirius::logical_type::make(sirius::type_id::INTEGER), false});
    CHECK(admit_scan_batch_sort(*plan) == 0);
    CHECK(raw->batch_sort_keys.empty());
  }
  SECTION("negative explicit filter mapping fails closed")
  {
    auto& child_filter = projected->children[0]->Cast<sirius_physical_filter>();
    std::get<std::vector<cudf::size_type>>(child_filter.output_columns)[1] = -1;
    CHECK(admit_scan_batch_sort(*plan) == 0);
    CHECK(raw->batch_sort_keys.empty());
  }
  SECTION("out of range key fails closed")
  {
    projected->select_list[0] = ref(99);
    CHECK(admit_scan_batch_sort(*plan) == 0);
    CHECK(raw->batch_sort_keys.empty());
  }
}

TEST_CASE("scan batch sort nearest consumer wins and unsafe ancestors veto",
          "[planner][scan_batch_sort]")
{
  auto leaf    = make_scan_leaf();
  auto* raw    = leaf.get();
  auto nearest = order(std::move(leaf), 2);
  auto plan    = order(std::move(nearest), 0);
  SECTION("outer ordering cannot cross the nearer ordering")
  {
    CHECK(admit_scan_batch_sort(*plan) == 1);
    REQUIRE(raw->batch_sort_keys.size() == 1);
    CHECK(raw->batch_sort_keys[0].output_ordinal == 2);
  }
  SECTION("a LIMIT above the consumer vetoes all hints")
  {
    struct limit_stub : sirius_physical_operator {
      explicit limit_stub(SiriusPhysicalOperatorType type)
        : sirius_physical_operator(type, ints(3), 10)
      {
      }
      sirius::OrderPreservationType output_order_preservation() const override
      { return sirius::OrderPreservationType::INSERTION_ORDER; }
    } limit(GENERATE(SiriusPhysicalOperatorType::LIMIT,
                     SiriusPhysicalOperatorType::STREAMING_LIMIT,
                     SiriusPhysicalOperatorType::UNORDERED_STREAMING_LIMIT));
    limit.children.push_back(std::move(plan));
    CHECK(admit_scan_batch_sort(limit) == 0);
    CHECK(raw->batch_sort_keys.empty());
  }
  SECTION("grouped FIRST above the consumer vetoes all hints")
  {
    duckdb::vector<std::unique_ptr<sirius::ast::node>> aggregates;
    std::vector<std::unique_ptr<sirius::ast::node>> args;
    args.push_back(ref(1));
    aggregates.push_back(std::make_unique<sirius::ast::node>(
      sirius::ast::aggregate{sirius::aggregate_id::first,
                             std::move(args),
                             sirius::logical_type::make(sirius::type_id::INTEGER),
                             false}));
    duckdb::vector<std::unique_ptr<sirius::ast::node>> groups;
    groups.push_back(ref(0));
    sirius_physical_grouped_aggregate first(ints(2), std::move(aggregates), std::move(groups), 10);
    REQUIRE(first.has_first());
    first.children.push_back(std::move(plan));
    CHECK(admit_scan_batch_sort(first) == 0);
    CHECK(raw->batch_sort_keys.empty());
  }
  SECTION("FIRST above the consumer vetoes all hints")
  {
    duckdb::vector<std::unique_ptr<sirius::ast::node>> aggregates;
    std::vector<std::unique_ptr<sirius::ast::node>> args;
    args.push_back(ref(0));
    aggregates.push_back(std::make_unique<sirius::ast::node>(
      sirius::ast::aggregate{sirius::aggregate_id::first,
                             std::move(args),
                             sirius::logical_type::make(sirius::type_id::INTEGER),
                             false}));
    sirius_physical_ungrouped_aggregate first(
      ints(1), std::move(aggregates), 10, duckdb::TupleDataValidityType::CAN_HAVE_NULL_VALUES);
    first.children.push_back(std::move(plan));
    CHECK(admit_scan_batch_sort(first) == 0);
    CHECK(raw->batch_sort_keys.empty());
  }
}

TEST_CASE("scan batch sort unsafe sibling vetoes the whole query", "[planner][scan_batch_sort]")
{
  auto left   = make_scan_leaf();
  auto* lhs   = left.get();
  auto right  = make_scan_leaf();
  auto* rhs   = right.get();
  auto unsafe = projection(std::move(right), {0, 1, 2});
  // An unknown/window-like sibling is a barrier even though the left input maps directly.
  unsafe->type = SiriusPhysicalOperatorType::WINDOW;
  auto plan    = join(std::move(left), std::move(unsafe));
  CHECK(admit_scan_batch_sort(*plan) == 0);
  CHECK(lhs->batch_sort_keys.empty());
  CHECK(rhs->batch_sort_keys.empty());
}

TEST_CASE("scan batch sort refuses unsupported key types", "[planner][scan_batch_sort]")
{
  auto leaf     = make_scan_leaf();
  auto* raw     = leaf.get();
  raw->types[0] = sirius::logical_type::make(GENERATE(sirius::type_id::FLOAT,
                                                      sirius::type_id::DOUBLE,
                                                      sirius::type_id::VARCHAR,
                                                      sirius::type_id::SQLNULL));
  auto plan     = order(std::move(leaf), 0);
  CHECK(admit_scan_batch_sort(*plan) == 0);
  CHECK(raw->batch_sort_keys.empty());
}

TEST_CASE("scan batch sort planning environment is opt in", "[planner][scan_batch_sort]")
{
  constexpr char name[]            = "SIRIUS_EXP_SCAN_BATCH_SORT";
  auto const* old                  = std::getenv(name);
  std::optional<std::string> saved = old ? std::optional<std::string>{old} : std::nullopt;
  struct restore {
    char const* name;
    std::optional<std::string>& value;
    ~restore()
    {
      if (value)
        setenv(name, value->c_str(), 1);
      else
        unsetenv(name);
    }
  } restore_env{name, saved};
  unsetenv(name);
  CHECK_FALSE(sirius::planner::scan_batch_sort_enabled());
  for (auto const* value : {"", "0", "true", "01"}) {
    setenv(name, value, 1);
    CHECK_FALSE(sirius::planner::scan_batch_sort_enabled());
  }
  setenv(name, "1", 1);
  CHECK(sirius::planner::scan_batch_sort_enabled());
}
