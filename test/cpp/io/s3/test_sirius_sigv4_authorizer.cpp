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

#include <io/io_errors.hpp>
#include <io/s3/sirius_sigv4_authorizer.hpp>
#include <io/s3/static_credentials.hpp>

#include <chrono>
#include <string>
#include <string_view>

using sirius::io::credential_error;
using sirius::io::s3::s3_authorized_request;
using sirius::io::s3::s3_object_ref;
using sirius::io::s3::s3_request_method;
using sirius::io::s3::sirius_sigv4_header_authorizer;
using sirius::io::s3::sirius_sigv4_presigned_authorizer;
using sirius::io::s3::static_credentials;

namespace {

static_credentials test_creds()
{
  static_credentials c;
  c.access_key_id     = "AKIAIOSFODNN7EXAMPLE";
  c.secret_access_key = "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY";
  return c;
}

bool contains(std::string const& haystack, std::string_view needle)
{
  return haystack.find(needle) != std::string::npos;
}

}  // namespace

TEST_CASE("presigned authorizer builds a path-style query-signed URL", "[s3][authorizer]")
{
  sirius_sigv4_presigned_authorizer auth(
    test_creds(), "us-east-1", "https://s3.us-east-1.amazonaws.com", std::chrono::minutes{5});

  s3_authorized_request const req =
    auth.authorize(s3_object_ref{"my-bucket", "path/to/object.parquet"},
                   s3_request_method::GET,
                   std::chrono::seconds{120});

  // Auth lives entirely in the query string, so no headers are returned.
  CHECK(req.headers.empty());

  CHECK(contains(req.url, "https://s3.us-east-1.amazonaws.com/my-bucket/path/to/object.parquet?"));
  CHECK(contains(req.url, "X-Amz-Algorithm=AWS4-HMAC-SHA256"));
  CHECK(contains(req.url, "X-Amz-Credential="));
  CHECK(contains(req.url, "X-Amz-Date="));
  CHECK(contains(req.url, "X-Amz-Expires=120"));  // per-call timeout drives expiry
  CHECK(contains(req.url, "X-Amz-SignedHeaders=host"));
  CHECK(contains(req.url, "X-Amz-Signature="));
}

TEST_CASE("presigned authorizer falls back to the default TTL", "[s3][authorizer]")
{
  sirius_sigv4_presigned_authorizer auth(
    test_creds(), "us-east-1", "https://s3.us-east-1.amazonaws.com", std::chrono::seconds{900});

  // Non-positive per-call timeout => construction-time default TTL.
  s3_authorized_request const req = auth.authorize(
    s3_object_ref{"b", "k"}, s3_request_method::HEAD, std::chrono::seconds{0});
  CHECK(contains(req.url, "X-Amz-Expires=900"));
}

TEST_CASE("header authorizer signs into the Authorization header", "[s3][authorizer]")
{
  sirius_sigv4_header_authorizer auth(
    test_creds(), "us-west-2", "https://s3.us-west-2.amazonaws.com");

  s3_authorized_request const req =
    auth.authorize(s3_object_ref{"bucket", "key"}, s3_request_method::GET, std::chrono::seconds{60});

  // Plain URL (no presign query) + signed headers.
  CHECK(req.url == "https://s3.us-west-2.amazonaws.com/bucket/key");
  CHECK(contains(req.url, "?") == false);

  bool has_auth = false;
  for (auto const& [k, v] : req.headers) {
    if (k == "Authorization") {
      has_auth = true;
      CHECK(contains(v, "AWS4-HMAC-SHA256"));
      CHECK(contains(v, "Credential="));
      CHECK(contains(v, "Signature="));
    }
  }
  CHECK(has_auth);
}

TEST_CASE("authorizer construction validates its inputs", "[s3][authorizer]")
{
  SECTION("empty credentials")
  {
    static_credentials empty;
    CHECK_THROWS_AS(
      sirius_sigv4_presigned_authorizer(empty, "us-east-1", "https://s3.amazonaws.com"),
      credential_error);
  }
  SECTION("empty region")
  {
    CHECK_THROWS_AS(
      sirius_sigv4_presigned_authorizer(test_creds(), "", "https://s3.amazonaws.com"),
      credential_error);
  }
  SECTION("endpoint with a path")
  {
    CHECK_THROWS_AS(
      sirius_sigv4_presigned_authorizer(test_creds(), "us-east-1", "https://s3.amazonaws.com/x"),
      credential_error);
  }
  SECTION("endpoint without a scheme")
  {
    CHECK_THROWS_AS(sirius_sigv4_presigned_authorizer(test_creds(), "us-east-1", "s3.amazonaws.com"),
                    credential_error);
  }
  SECTION("non-positive default TTL")
  {
    CHECK_THROWS_AS(sirius_sigv4_presigned_authorizer(
                      test_creds(), "us-east-1", "https://s3.amazonaws.com", std::chrono::seconds{0}),
                    credential_error);
  }
}
