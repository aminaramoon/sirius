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

#include "io/s3/s3_request_authorizer.hpp"
#include "io/s3/static_credentials.hpp"

#include <chrono>
#include <string>

namespace sirius::io::s3 {

/**
 * @brief Common base for the hand-rolled SigV4 authorizers.
 *
 * Validates and stores static credentials + signing region, and parses the
 * service @p endpoint into (scheme, host[:port]). The endpoint must be
 * @c scheme://host[:port] with no path / query / fragment — it identifies the
 * S3 service, not a bucket or object.
 *
 * @throw sirius::io::credential_error on empty access key / secret / region,
 *        or a malformed endpoint.
 */
class sirius_sigv4_authorizer_base : public s3_request_authorizer {
 public:
  sirius_sigv4_authorizer_base(static_credentials creds,
                               std::string region,
                               std::string endpoint);

 protected:
  static_credentials _creds;
  std::string _region;
  std::string _scheme;  // "https" / "http"
  std::string _host;    // host[:port]
};

/**
 * @brief SigV4 presigned-URL authorizer (auth entirely in the URL query).
 *
 * @c authorize() returns @c {presigned_url, {}} (empty headers). The per-call
 * @c timeout becomes @c X-Amz-Expires; a non-positive value falls back to
 * @p default_ttl. Because the presigned signature covers only @c host (with an
 * @c UNSIGNED-PAYLOAD body hash), callers may add an unsigned @c Range header
 * to the resulting GET / HEAD without breaking the signature.
 *
 * Thread-safe: the underlying SigV4 routines are pure and members are
 * immutable after construction.
 */
class sirius_sigv4_presigned_authorizer final : public sirius_sigv4_authorizer_base {
 public:
  sirius_sigv4_presigned_authorizer(static_credentials creds,
                                    std::string region,
                                    std::string endpoint,
                                    std::chrono::seconds default_ttl = std::chrono::minutes{5});

  s3_authorized_request authorize(s3_object_ref const& obj,
                                  s3_request_method method,
                                  std::chrono::seconds timeout) override;

 private:
  std::chrono::seconds _ttl;
};

/**
 * @brief SigV4 header authorizer (auth in the @c Authorization header).
 *
 * @c authorize() returns @c {plain_url, signed_headers}. GET / HEAD have no
 * body and any @c Range is added unsigned by the caller, so the canonical
 * query and extra signed headers are empty and the payload hash is
 * @c sha256(""). The @c timeout argument is ignored.
 */
class sirius_sigv4_header_authorizer final : public sirius_sigv4_authorizer_base {
 public:
  sirius_sigv4_header_authorizer(static_credentials creds,
                                 std::string region,
                                 std::string endpoint);

  s3_authorized_request authorize(s3_object_ref const& obj,
                                  s3_request_method method,
                                  std::chrono::seconds timeout) override;
};

}  // namespace sirius::io::s3
