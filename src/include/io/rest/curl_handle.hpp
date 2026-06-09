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

#include "io/types.hpp"  // file_descriptor

#include <curl/curl.h>
#include <curl/multi.h>
#include <sys/epoll.h>
#include <sys/eventfd.h>
#include <sys/timerfd.h>

#include <cstring>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>

namespace sirius::io::rest {

// ---------------------------------------------------------------------------
// Error-checking macros
// ---------------------------------------------------------------------------

/// Evaluate a libcurl easy-interface call and throw on a non-OK code.
#define SIRIUS_CURL_CHECK(call)                                                         \
  do {                                                                                  \
    CURLcode _ec = (call);                                                              \
    if (_ec != CURLE_OK) {                                                              \
      throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + \
                               " libcurl error: " + curl_easy_strerror(_ec));           \
    }                                                                                   \
  } while (false)

/// Evaluate a libcurl multi-interface call and throw on a non-OK code.
#define SIRIUS_CURLM_CHECK(call)                                                        \
  do {                                                                                  \
    CURLMcode _mc = (call);                                                             \
    if (_mc != CURLM_OK) {                                                              \
      throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + \
                               " libcurl-multi error: " + curl_multi_strerror(_mc));    \
    }                                                                                   \
  } while (false)

// ---------------------------------------------------------------------------
// RAII handle wrappers
// ---------------------------------------------------------------------------

struct curl_easy_deleter {
  void operator()(CURL* h) const noexcept
  {
    if (h != nullptr) curl_easy_cleanup(h);
  }
};
/// Owns a @c CURL* easy handle; cleans up on destruction.
using curl_easy_ptr = std::unique_ptr<CURL, curl_easy_deleter>;

struct curl_multi_deleter {
  void operator()(CURLM* m) const noexcept
  {
    if (m != nullptr) curl_multi_cleanup(m);
  }
};
/// Owns a @c CURLM* multi handle; cleans up on destruction.  All easy handles
/// must be removed (@c curl_multi_remove_handle) before the multi is destroyed.
using curl_multi_ptr = std::unique_ptr<CURLM, curl_multi_deleter>;

struct curl_slist_deleter {
  void operator()(curl_slist* l) const noexcept
  {
    if (l != nullptr) curl_slist_free_all(l);
  }
};
/// Owns a @c curl_slist* header list; frees the whole list on destruction.
using curl_slist_ptr = std::unique_ptr<curl_slist, curl_slist_deleter>;

struct curl_share_deleter {
  void operator()(CURLSH* s) const noexcept
  {
    if (s != nullptr) curl_share_cleanup(s);
  }
};
/// Owns a @c CURLSH* share handle; cleans up on destruction.
using curl_share_ptr = std::unique_ptr<CURLSH, curl_share_deleter>;

// ---------------------------------------------------------------------------
// epoll / timerfd / eventfd factories (RAII via file_descriptor)
// ---------------------------------------------------------------------------
//
// All three are ordinary file descriptors closed with close(), so the existing
// file_descriptor RAII wrapper owns them directly.  These factories create the
// fd with the flags the reactor needs and throw on failure.

/// Create an epoll instance (@c EPOLL_CLOEXEC).
[[nodiscard]] inline file_descriptor make_epoll_fd()
{
  int fd = ::epoll_create1(EPOLL_CLOEXEC);
  if (fd < 0) {
    throw std::runtime_error(std::string("rest: epoll_create1 failed: ") + std::strerror(errno));
  }
  return file_descriptor{fd};
}

/// Create a monotonic, non-blocking timerfd (@c TFD_NONBLOCK | @c TFD_CLOEXEC).
[[nodiscard]] inline file_descriptor make_timer_fd()
{
  int fd = ::timerfd_create(CLOCK_MONOTONIC, TFD_NONBLOCK | TFD_CLOEXEC);
  if (fd < 0) {
    throw std::runtime_error(std::string("rest: timerfd_create failed: ") + std::strerror(errno));
  }
  return file_descriptor{fd};
}

/// Create a non-blocking eventfd (@c EFD_NONBLOCK | @c EFD_CLOEXEC), initial 0.
/// Used as the cross-thread wakeup that bridges the lock-free request queue and
/// CUDA copy-completion callbacks into the epoll loop.
[[nodiscard]] inline file_descriptor make_event_fd()
{
  int fd = ::eventfd(0, EFD_NONBLOCK | EFD_CLOEXEC);
  if (fd < 0) {
    throw std::runtime_error(std::string("rest: eventfd failed: ") + std::strerror(errno));
  }
  return file_descriptor{fd};
}

// ---------------------------------------------------------------------------
// global_curl_context
// ---------------------------------------------------------------------------

/**
 * @brief Process-wide libcurl initialization + shared cache handle.
 *
 * Performs @c curl_global_init exactly once (and deliberately never calls
 * @c curl_global_cleanup — it races with late static destructors in
 * third-party libraries; the bounded one-time leak is the accepted trade-off)
 * and owns a @c CURLSH share handle that pools DNS resolutions, TLS session
 * tickets, and cookies across every easy handle in the process.  Sharing these
 * avoids repeated DNS lookups and full TLS handshakes — a major throughput win
 * for many short ranged GETs against the same endpoint.
 *
 * The share handle's lock/unlock callbacks serialize access through a single
 * coarse mutex held here; that is sufficient for correctness and matches the
 * reference design.
 */
class global_curl_context {
 public:
  /// Lazily-constructed process singleton (thread-safe first-use init).
  static global_curl_context& instance();

  [[nodiscard]] CURLSH* share_handle() const noexcept { return _share.get(); }

  global_curl_context(global_curl_context const&)            = delete;
  global_curl_context& operator=(global_curl_context const&) = delete;

 private:
  global_curl_context();

  static void lock_cb(CURL* handle, curl_lock_data data, curl_lock_access access, void* userp);
  static void unlock_cb(CURL* handle, curl_lock_data data, void* userp);

  std::mutex _share_mtx;
  curl_share_ptr _share;
};

// ---------------------------------------------------------------------------
// Easy-handle configuration
// ---------------------------------------------------------------------------

/**
 * @brief Apply the standard high-performance options to a fresh easy handle.
 *
 * Sets HTTP/2, the shared DNS/TLS/cookie cache, TCP tuning (NODELAY,
 * keepalive), @c NOSIGNAL (required for multithreaded use), receive buffer
 * size, connect/transfer timeouts, redirect following, connection max-age, and
 * DNS cache timeout.  Per-request options (URL, Range, write/header callbacks,
 * private pointer, TLS verification) are set by the reactor at submit time.
 *
 * @param handle        the easy handle to configure.
 * @param share_handle  the process-wide @c CURLSH (DNS/TLS/cookie sharing).
 */
void configure_easy_handle(CURL* handle, CURLSH* share_handle);

}  // namespace sirius::io::rest
