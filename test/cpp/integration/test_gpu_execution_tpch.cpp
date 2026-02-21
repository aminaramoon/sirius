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

#include "op/sirius_physical_partition.hpp"

#include <cudf/utilities/default_stream.hpp>

#include <catch.hpp>
#include <duckdb.hpp>

#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <optional>
#include <set>
#include <string>

namespace fs = std::filesystem;

static fs::path get_project_root()
{
#ifdef SIRIUS_PROJECT_ROOT
  return fs::path(SIRIUS_PROJECT_ROOT);
#else
  return fs::path(__FILE__).parent_path().parent_path().parent_path().parent_path();
#endif
}

static fs::path get_tpch_parquet_dir()
{
  return get_project_root() / "test" / "cpp" / "integration" / "tpch" / "parquet";
}

static fs::path get_tpch_db_path()
{
  const char* env = std::getenv("SIRIUS_INTEGRATION_TEST_DB_PATH");
  auto db_path =
    env ? fs::path(env) : fs::path(__FILE__).parent_path() / "tpch/duckdb/integration.duckdb";
  REQUIRE(fs::exists(db_path));
  return db_path;
}

/**
 * @brief Catch2 test fixture for GPU execution tests.
 *
 * Initializes a DuckDB instance with the integration.cfg config and provides
 * a compare_gpu_vs_cpu method for validating GPU execution against CPU results.
 */
class GPUExecutionFixture {
 protected:
  struct SkipLoadDatabase {};

  // Used by derived classes — skips load_database so they can call their own override
  GPUExecutionFixture(SkipLoadDatabase)
  {
    auto cfg_path = fs::path(__FILE__).parent_path() / "integration.cfg";
    REQUIRE(fs::exists(cfg_path));
    setenv("SIRIUS_CONFIG_FILE", cfg_path.string().c_str(), 1);
    db  = std::make_unique<duckdb::DuckDB>(nullptr);
    con = std::make_unique<duckdb::Connection>(*db);
  }

 public:
  GPUExecutionFixture() : GPUExecutionFixture(SkipLoadDatabase{}) { load_database(); }

  ~GPUExecutionFixture() { unsetenv("SIRIUS_CONFIG_FILE"); }

  virtual void load_database()
  {
    auto db_path = get_tpch_db_path();
    con->Query("ATTACH DATABASE '" + db_path.string() + "' AS tpch;");
    con->Query("USE tpch;");
    // con->Query("SET enable_fallback_check = true;");
  }

  /**
   * @brief Run a query through gpu_execution and through DuckDB CPU, then compare results.
   *
   * Values are compared as strings via Value::ToString() which normalizes type differences
   * (e.g., HUGEINT vs BIGINT both render "50"). Row order is ignored by collecting rows
   * as sorted sets of string tuples.
   */
  static bool is_floating_point(duckdb::LogicalTypeId id)
  {
    return id == duckdb::LogicalTypeId::FLOAT || id == duckdb::LogicalTypeId::DOUBLE;
  }

  void compare_gpu_vs_cpu(const std::string& query,
                          std::optional<float> float_tolerance = std::nullopt)
  {
    // Run on GPU
    auto gpu_sql    = "CALL gpu_execution(\"" + query + "\")";
    auto gpu_result = con->Query(gpu_sql);
    REQUIRE(gpu_result);
    if (gpu_result->HasError()) {
      UNSCOPED_INFO("gpu_execution error: " << gpu_result->GetError());
    }
    REQUIRE_FALSE(gpu_result->HasError());

    // Run on CPU (plain DuckDB)
    auto cpu_result = con->Query(query);
    REQUIRE(cpu_result);
    REQUIRE_FALSE(cpu_result->HasError());

    // Compare dimensions
    REQUIRE(gpu_result->ColumnCount() == cpu_result->ColumnCount());
    REQUIRE(gpu_result->RowCount() == cpu_result->RowCount());

    // Use DuckDB to sort both result sets by all columns for deterministic comparison.
    // This avoids lexicographic vs numeric sort issues.
    auto ncols               = gpu_result->ColumnCount();
    std::string order_clause = " ORDER BY ";
    for (duckdb::idx_t c = 0; c < ncols; c++) {
      if (c > 0) order_clause += ", ";
      order_clause += std::to_string(c + 1);
    }

    // Strip trailing semicolons from query for subquery wrapping
    auto clean_query = query;
    while (!clean_query.empty() && (clean_query.back() == ';' || clean_query.back() == ' '))
      clean_query.pop_back();

    auto gpu_sorted =
      con->Query("SELECT * FROM gpu_execution(\"" + clean_query + "\")" + order_clause);
    auto cpu_sorted = con->Query("SELECT * FROM (" + clean_query + ") t" + order_clause);
    REQUIRE(gpu_sorted);
    if (gpu_sorted->HasError()) { UNSCOPED_INFO("gpu sorted error: " << gpu_sorted->GetError()); }
    REQUIRE_FALSE(gpu_sorted->HasError());
    REQUIRE(cpu_sorted);
    if (cpu_sorted->HasError()) { UNSCOPED_INFO("cpu sorted error: " << cpu_sorted->GetError()); }
    REQUIRE_FALSE(cpu_sorted->HasError());

    for (duckdb::idx_t r = 0; r < gpu_sorted->RowCount(); r++) {
      for (duckdb::idx_t c = 0; c < gpu_sorted->ColumnCount(); c++) {
        auto gpu_value = gpu_sorted->GetValue(c, r);
        auto cpu_value = cpu_sorted->GetValue(c, r);

        if (float_tolerance.has_value() && is_floating_point(gpu_value.type().id())) {
          double gpu_d = gpu_value.GetValue<double>();
          double cpu_d = cpu_value.GetValue<double>();
          double diff  = std::fabs(gpu_d - cpu_d);
          if (diff > static_cast<double>(float_tolerance.value())) {
            UNSCOPED_INFO("Row " << r << " Col " << c << " float mismatch: GPU=[" << gpu_d
                                 << "] CPU=[" << cpu_d << "] diff=" << diff
                                 << " tolerance=" << float_tolerance.value());
          }
          REQUIRE(diff <= static_cast<double>(float_tolerance.value()));
        } else {
          auto gpu_str = gpu_value.ToString();
          auto cpu_str = cpu_value.ToString();
          if (gpu_str != cpu_str) {
            UNSCOPED_INFO("Row " << r << " Col " << c << " mismatch: GPU=[" << gpu_str << "] CPU=["
                                 << cpu_str << "]");
          }
          REQUIRE(gpu_str == cpu_str);
        }
      }
    }
  }

  std::unique_ptr<duckdb::DuckDB> db;
  std::unique_ptr<duckdb::Connection> con;
};

class GPUExecutionParquetFixture : public GPUExecutionFixture {
 public:
  GPUExecutionParquetFixture() : GPUExecutionFixture(SkipLoadDatabase{}) { load_database(); }

  void load_database() override
  {
    auto parquet_dir = fs::path(__FILE__).parent_path() / "tpch/parquet";
    REQUIRE(fs::exists(parquet_dir));

    static const std::array<std::string, 8> tables = {
      "customer", "lineitem", "nation", "orders", "part", "partsupp", "region", "supplier"};

    for (const auto& table : tables) {
      auto parquet_path = (parquet_dir / (table + ".parquet")).string();
      auto result       = con->Query("CREATE VIEW " + table + " AS SELECT * FROM read_parquet('" +
                               parquet_path + "');");
      if (result->HasError()) {
        UNSCOPED_INFO("Failed to create view for " << table << ": " << result->GetError());
      }
      REQUIRE_FALSE(result->HasError());
    }
  }
};

//===----------------------------------------------------------------------===//
// Scan tests
//===----------------------------------------------------------------------===//

TEST_CASE_METHOD(GPUExecutionParquetFixture,
                 "run all integration tests",
                 "[integration][gpu_execution][duckdb]")
{
  auto queries_dir = fs::path(__FILE__).parent_path() / "queries";
  REQUIRE(fs::exists(queries_dir));

  std::vector<fs::path> sql_files;
  for (const auto& entry : fs::directory_iterator(queries_dir)) {
    if (entry.path().extension() == ".sql") { sql_files.push_back(entry.path()); }
  }
  std::sort(sql_files.begin(), sql_files.end());

  for (const auto& sql_file : sql_files) {
    std::ifstream file(sql_file);
    REQUIRE(file.is_open());
    std::string query((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    auto last_semi = query.rfind(';');
    if (last_semi != std::string::npos) { query = query.substr(0, last_semi + 1); }
    std::cerr << "Running query from: " << query << std::endl;
    compare_gpu_vs_cpu(query);
  }
}
