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

#include "io/s3/s3_object_ref.hpp"

#include <chrono>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace sirius::io::s3 {

/// HTTP method an authorization is bound to. AWS presigned URLs (and SigV4
/// header signatures) are method-specific — a GET authorization is rejected
/// with SignatureDoesNotMatch if replayed as a HEAD. Sirius issues only
/// read-only S3 requests, so GET / HEAD are the only methods.
enum class s3_request_method : std::uint8_t { GET, HEAD };

/**
 * @brief A request authorized for a single S3 object access.
 *
 * @c url is fully qualified (@c "scheme://host/bucket/key[?...]"). @c headers
 * carries the authorization material when the scheme is header-based (SigV4
 * @c Authorization header); it is EMPTY for presigned-URL authorizers, where
 * all auth lives in the URL query string. The caller attaches every header
 * verbatim and may freely add an unsigned @c Range header in either mode.
 */
struct s3_authorized_request {
  std::string url;
  std::vector<std::pair<std::string, std::string>> headers;
};

/**
 * @brief Pluggable authorization seam for read-only S3 object access.
 *
 * Generalizes the legacy presign-only @c credential_provider: an authorizer
 * may return either a presigned URL (empty @c headers) or a plain URL plus
 * SigV4 @c Authorization headers, and it accepts a per-request validity
 * window. Lets downstream projects plug in their own signer (AWS SDK,
 * IMDS/STS chain, SSO, broker-issued URLs) without Sirius depending on
 * @c aws-sdk-cpp.
 *
 * @par Lifetime
 *   Implementations should be safe to share across threads via @c shared_ptr.
 *   Backends call @c authorize() once per request, inline at the call site
 *   that issues the underlying HTTP request — never at scan-task creation
 *   time, since the authorization carries an expiration and may go stale
 *   before a deferred task runs.
 *
 * @par Errors
 *   Implementations throw @c sirius::io::credential_error on credential /
 *   signing failure. Backends translate into the broader IO error path.
 */
class s3_request_authorizer {
 public:
  virtual ~s3_request_authorizer() = default;

  s3_request_authorizer()                                        = default;
  s3_request_authorizer(s3_request_authorizer const&)            = delete;
  s3_request_authorizer& operator=(s3_request_authorizer const&) = delete;

  /**
   * @brief Authorize @p method access to @p obj, valid for @p timeout.
   *
   * @param obj     Bucket + key to access.
   * @param method  GET or HEAD.
   * @param timeout Validity window. For presigned URLs this drives
   *                @c X-Amz-Expires; a non-positive value falls back to the
   *                authorizer's construction-time default. Header-based
   *                authorizers ignore it (the signature has its own clock skew
   *                allowance).
   *
   * @throw sirius::io::credential_error on credential / signing failure.
   */
  [[nodiscard]] virtual s3_authorized_request authorize(s3_object_ref const& obj,
                                                        s3_request_method method,
                                                        std::chrono::seconds timeout) = 0;
};

}  // namespace sirius::io::s3
