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

#include "io/rest/curl_handle.hpp"

#include <stdexcept>

namespace sirius::io::rest {

namespace {

// Receive buffer handed to libcurl for each transfer; larger than the default
// 16 KiB to cut write-callback round trips on multi-MiB ranged GETs.
constexpr long kRecvBufferSize     = 128L * 1024L;
constexpr long kConnectTimeoutMs   = 5'000L;
constexpr long kTransferTimeoutMs  = 30'000L;
constexpr long kMaxConnAgeSec      = 20L;
constexpr long kDnsCacheTimeoutSec = 600L;

// curl_global_init must run exactly once per process, before any handle is
// created.  curl_global_cleanup is intentionally never paired with it (see the
// class doc): it races with late static destructors in third-party libraries.
std::once_flag g_global_init_once;

void global_init_once()
{
  std::call_once(g_global_init_once, [] {
    if (curl_global_init(CURL_GLOBAL_DEFAULT) != 0) {
      throw std::runtime_error("rest: curl_global_init failed");
    }
  });
}

}  // namespace

global_curl_context& global_curl_context::instance()
{
  static global_curl_context ctx;
  return ctx;
}

global_curl_context::global_curl_context()
{
  global_init_once();

  CURLSH* sh = curl_share_init();
  if (sh == nullptr) { throw std::runtime_error("rest: curl_share_init failed"); }
  _share.reset(sh);

  // Share DNS, TLS sessions, and cookies across every easy handle in the
  // process.  Each setopt can fail (e.g. unsupported build), so check.
  auto share_set = [sh](CURLSHoption opt, auto value) {
    if (curl_share_setopt(sh, opt, value) != CURLSHE_OK) {
      throw std::runtime_error("rest: curl_share_setopt failed");
    }
  };
  share_set(CURLSHOPT_LOCKFUNC, &global_curl_context::lock_cb);
  share_set(CURLSHOPT_UNLOCKFUNC, &global_curl_context::unlock_cb);
  share_set(CURLSHOPT_USERDATA, this);
  share_set(CURLSHOPT_SHARE, CURL_LOCK_DATA_DNS);
  share_set(CURLSHOPT_SHARE, CURL_LOCK_DATA_SSL_SESSION);
  share_set(CURLSHOPT_SHARE, CURL_LOCK_DATA_COOKIE);
}

void global_curl_context::lock_cb(CURL* /*handle*/,
                                  curl_lock_data /*data*/,
                                  curl_lock_access /*access*/,
                                  void* userp)
{
  // Coarse single-mutex serialization across all shared data classes.
  static_cast<global_curl_context*>(userp)->_share_mtx.lock();
}

void global_curl_context::unlock_cb(CURL* /*handle*/, curl_lock_data /*data*/, void* userp)
{
  static_cast<global_curl_context*>(userp)->_share_mtx.unlock();
}

void configure_easy_handle(CURL* handle, CURLSH* share_handle)
{
  if (handle == nullptr) { throw std::runtime_error("rest: configure_easy_handle: null handle"); }

  // Connection reuse / sharing.
  if (share_handle != nullptr) {
    SIRIUS_CURL_CHECK(curl_easy_setopt(handle, CURLOPT_SHARE, share_handle));
  }
  SIRIUS_CURL_CHECK(curl_easy_setopt(handle, CURLOPT_MAXAGE_CONN, kMaxConnAgeSec));
  SIRIUS_CURL_CHECK(curl_easy_setopt(handle, CURLOPT_DNS_CACHE_TIMEOUT, kDnsCacheTimeoutSec));

  // TCP tuning.
  SIRIUS_CURL_CHECK(curl_easy_setopt(handle, CURLOPT_TCP_KEEPALIVE, 1L));
  SIRIUS_CURL_CHECK(curl_easy_setopt(handle, CURLOPT_TCP_NODELAY, 1L));

  // Multithreaded safety: no SIGALRM-based DNS timeouts.
  SIRIUS_CURL_CHECK(curl_easy_setopt(handle, CURLOPT_NOSIGNAL, 1L));

  // HTTP behavior.
  SIRIUS_CURL_CHECK(curl_easy_setopt(handle, CURLOPT_HTTP_VERSION, CURL_HTTP_VERSION_2_0));
  SIRIUS_CURL_CHECK(curl_easy_setopt(handle, CURLOPT_FOLLOWLOCATION, 1L));
  SIRIUS_CURL_CHECK(curl_easy_setopt(handle, CURLOPT_BUFFERSIZE, kRecvBufferSize));

  // Default timeouts; the reactor may override the whole-transfer timeout per
  // request based on its configuration.
  SIRIUS_CURL_CHECK(curl_easy_setopt(handle, CURLOPT_CONNECTTIMEOUT_MS, kConnectTimeoutMs));
  SIRIUS_CURL_CHECK(curl_easy_setopt(handle, CURLOPT_TIMEOUT_MS, kTransferTimeoutMs));
}

}  // namespace sirius::io::rest
