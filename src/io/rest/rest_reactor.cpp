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

#include "io/rest/rest_reactor.hpp"

#include "io/rest/curl_handle.hpp"
#include "io/uri_parser.hpp"

#include <spdlog/spdlog.h>
#include <sys/epoll.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <cctype>
#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <random>
#include <stdexcept>
#include <string>
#include <system_error>
#include <thread>
#include <unordered_map>
#include <vector>

namespace sirius::io::rest {

namespace {

// Number of pinned bounce slots for reactor-staged device reads (and the
// upper bound on concurrent device transfers).  Easy-handle concurrency is
// bounded separately by config.max_connections.
constexpr std::size_t NUM_BOUNCE_SLOTS = 64;

// ---- libcurl callbacks -----------------------------------------------------

/// Write callback: copy curl's bytes into the sink's destination buffer at the
/// running cursor, and ALWAYS report the full incoming size so curl never
/// aborts the transfer with CURLE_WRITE_ERROR.  Overflow past capacity is
/// counted (total_received) but not stored, so a server that ignored the Range
/// header can be detected after the fact.
size_t write_to_sink(char* ptr, size_t size, size_t nmemb, void* userdata)
{
  auto* sink         = static_cast<buf_sink*>(userdata);
  size_t const bytes = size * nmemb;
  sink->total_received += bytes;
  if (sink->dst != nullptr && sink->written < sink->capacity) {
    size_t const room = sink->capacity - sink->written;
    size_t const n    = std::min(room, bytes);
    std::memcpy(sink->dst + sink->written, ptr, n);
    sink->written += n;
  }
  return bytes;
}

/// Discard callback for HEAD requests (no body expected, but be defensive).
size_t write_discard(char* /*ptr*/, size_t size, size_t nmemb, void* /*userdata*/)
{
  return size * nmemb;
}

/// Lowercase a byte.
char ascii_lower(char c) { return static_cast<char>(std::tolower(static_cast<unsigned char>(c))); }

/// Case-insensitively match @p line against "<name>:" and, on a hit, return the
/// trimmed value; otherwise return empty.
std::string match_header(std::string_view line, std::string_view name)
{
  if (line.size() < name.size() + 1) { return {}; }
  for (size_t i = 0; i < name.size(); ++i) {
    if (ascii_lower(line[i]) != ascii_lower(name[i])) { return {}; }
  }
  if (line[name.size()] != ':') { return {}; }
  std::string_view val = line.substr(name.size() + 1);
  // Trim surrounding whitespace and trailing CRLF.
  while (!val.empty() && (val.front() == ' ' || val.front() == '\t')) {
    val.remove_prefix(1);
  }
  while (!val.empty() &&
         (val.back() == '\r' || val.back() == '\n' || val.back() == ' ' || val.back() == '\t')) {
    val.remove_suffix(1);
  }
  return std::string(val);
}

/// Header callback: capture Content-Range and Retry-After.
size_t capture_header(char* buffer, size_t size, size_t nitems, void* userdata)
{
  auto* hc           = static_cast<header_capture*>(userdata);
  size_t const bytes = size * nitems;
  std::string_view line(buffer, bytes);
  if (auto v = match_header(line, "content-range"); !v.empty()) {
    hc->content_range = std::move(v);
  }
  if (auto v = match_header(line, "retry-after"); !v.empty()) { hc->retry_after = std::move(v); }
  return bytes;
}

// ---- retry classification --------------------------------------------------

/// HTTP status codes worth retrying (transient server / throttling).
bool is_retriable_status(long status) noexcept
{
  return status == 408 || status == 429 || (status >= 500 && status < 600);
}

/// libcurl error codes worth retrying (transient transport failures).
bool is_retriable_curl(CURLcode rc) noexcept
{
  switch (rc) {
    case CURLE_COULDNT_CONNECT:
    case CURLE_COULDNT_RESOLVE_HOST:
    case CURLE_OPERATION_TIMEDOUT:
    case CURLE_GOT_NOTHING:
    case CURLE_RECV_ERROR:
    case CURLE_SEND_ERROR:
    case CURLE_PARTIAL_FILE:
    case CURLE_SSL_CONNECT_ERROR:
    case CURLE_HTTP2_STREAM: return true;
    default: return false;
  }
}

// ---- per-request helpers ---------------------------------------------------

/// Presigned-URL TTL for a request: the whole-request timeout plus clock-skew
/// headroom, with a sane floor so very short timeouts still leave a usable
/// window.
std::chrono::seconds presign_ttl(const rest_reactor::config& cfg) noexcept
{
  long const base = cfg.request_timeout_s > 0 ? cfg.request_timeout_s + 60 : 300;
  return std::chrono::seconds{base};
}

/// Apply per-request TLS + timeout options on top of configure_easy_handle.
void apply_request_opts(CURL* h, const rest_reactor::config& cfg)
{
  if (!cfg.ca_bundle_path.empty()) {
    SIRIUS_CURL_CHECK(curl_easy_setopt(h, CURLOPT_CAINFO, cfg.ca_bundle_path.c_str()));
  }
  if (!cfg.tls_verify) {
    SIRIUS_CURL_CHECK(curl_easy_setopt(h, CURLOPT_SSL_VERIFYPEER, 0L));
    SIRIUS_CURL_CHECK(curl_easy_setopt(h, CURLOPT_SSL_VERIFYHOST, 0L));
  }
  if (cfg.request_timeout_s > 0) {
    SIRIUS_CURL_CHECK(curl_easy_setopt(h, CURLOPT_TIMEOUT, cfg.request_timeout_s));
  }
}

/// Build a header list from the authorizer's headers (empty in presigned mode)
/// plus an optional Range header.
curl_slist_ptr build_header_list(std::vector<std::pair<std::string, std::string>> const& headers,
                                 std::string const* range)
{
  curl_slist* list = nullptr;
  for (auto const& [k, v] : headers) {
    std::string const h = k + ": " + v;
    list                = curl_slist_append(list, h.c_str());
  }
  if (range != nullptr) { list = curl_slist_append(list, range->c_str()); }
  return curl_slist_ptr{list};
}

/// "Range: bytes=<lo>-<hi>" (inclusive end) for [offset, offset+size).
std::string range_header(size_t offset, size_t size)
{
  return "Range: bytes=" + std::to_string(offset) + "-" + std::to_string(offset + size - 1);
}

/// Backoff before the next attempt: honor a numeric Retry-After (seconds,
/// capped at 30 s) when present and enabled, else exponential base<<attempt
/// plus uniform jitter.
std::chrono::milliseconds compute_backoff(std::size_t attempt,
                                          std::string const& retry_after,
                                          const rest_reactor::config& cfg)
{
  if (cfg.honor_retry_after && !retry_after.empty()) {
    try {
      long const secs = std::stol(retry_after);
      if (secs >= 0) {
        return std::min(std::chrono::milliseconds{secs * 1000}, std::chrono::milliseconds{30'000});
      }
    } catch (...) {
      // Non-numeric (HTTP-date) Retry-After: fall through to exponential.
    }
  }
  std::size_t const shift = std::min<std::size_t>(attempt, 16);
  auto const base         = cfg.retry_backoff_base * (std::size_t{1} << shift);
  std::chrono::milliseconds jitter{0};
  if (cfg.retry_jitter.count() > 0) {
    thread_local std::mt19937 rng{std::random_device{}()};
    std::uniform_int_distribution<long> dist(0, cfg.retry_jitter.count());
    jitter = std::chrono::milliseconds{dist(rng)};
  }
  return base + jitter;
}

}  // namespace

// ---------------------------------------------------------------------------
// construction / lifecycle
// ---------------------------------------------------------------------------

rest_reactor::rest_reactor(config cfg, std::string_view tname) : _config(std::move(cfg))
{
  if (!_config.authorizer) {
    throw std::invalid_argument("rest_reactor: config.authorizer must be non-null");
  }
  if (_config.max_connections == 0) {
    throw std::invalid_argument("rest_reactor: max_connections must be > 0");
  }
  if (_config.max_retry_attempts == 0) { _config.max_retry_attempts = 1; }

  // Touch the process-wide curl context so global init + the shared cache are
  // ready before any handle is created (here or on the worker).
  (void)global_curl_context::instance();

  if (_config.host_memory_resource != nullptr) {
    _bounce_slot_size = _config.host_memory_resource->get_block_size();
    _bounce_storage.emplace(
      _config.host_memory_resource->allocate_multiple_blocks(NUM_BOUNCE_SLOTS * _bounce_slot_size));
  }

  _wakeup_fd = make_event_fd();

  _worker =
    std::jthread([this](const std::stop_token& st) { worker_loop(st); }, _stop_source.get_token());
  if (!tname.empty()) {
    std::string const full_name = std::string(tname) + "_worker";
    pthread_setname_np(_worker.native_handle(), full_name.c_str());
  }
}

rest_reactor::~rest_reactor() { shutdown(); }

void rest_reactor::interrupt()
{
  // Break the worker out of epoll_wait.
  uint64_t one = 1;
  ssize_t rc   = ::write(_wakeup_fd.get(), &one, sizeof(one));
  (void)rc;  // EAGAIN on a saturated counter is fine — a wakeup is already pending.
}

void rest_reactor::shutdown()
{
  if (_worker.joinable()) {
    _stop_source.request_stop();
    interrupt();
    _worker.join();
  }
}

void rest_reactor::enqueue(request_type_ptr req)
{
  auto chunks = req->get_all_chunks();
  enqueue_chunks(chunks);
}

void rest_reactor::enqueue_chunks(std::span<std::unique_ptr<rest_chunked_rx_request>> batch)
{
  if (batch.empty()) { return; }
  bool const ok = _requests.enqueue_bulk(std::make_move_iterator(batch.data()), batch.size());
  if (!ok) { throw std::runtime_error("rest_reactor::enqueue_chunks: enqueue_bulk failed"); }
  interrupt();
}

// ---------------------------------------------------------------------------
// request preparation (host paths)
// ---------------------------------------------------------------------------

rest_reactor::request_type_ptr rest_reactor::prep_host_rx_request(
  const reactor_config_type& /*cfg*/, const io_object_type& file, const io_object_segment& segment)
{
  if (segment.size == 0) { return rest_rx_request::create({}); }

  auto manager   = std::make_shared<request_manager>(segment.size, 1);
  auto req       = std::make_unique<rest_chunked_rx_request>();
  req->object    = file.object_ref();
  req->chunk     = segment;
  req->file_size = file.size();
  req->manager   = std::move(manager);

  std::vector<std::unique_ptr<rest_chunked_rx_request>> chunks;
  chunks.push_back(std::move(req));
  return rest_rx_request::create(std::move(chunks));
}

rest_reactor::request_type_ptr rest_reactor::prep_host_rxv_request(
  const reactor_config_type& /*cfg*/,
  const io_object_type& file,
  std::span<io_object_segment> segments)
{
  if (segments.empty()) { return rest_rx_request::create({}); }

  size_t const fsize = file.size();

  // First pass: clamp each segment to the file end and total the requested
  // bytes / count the non-empty segments (each becomes one ranged GET).
  size_t bytes_requested = 0;
  size_t n_chunks        = 0;
  for (auto const& s : segments) {
    size_t const clamped = s.offset < fsize ? std::min(s.size, fsize - s.offset) : 0;
    if (clamped == 0) { continue; }
    bytes_requested += clamped;
    ++n_chunks;
  }
  if (n_chunks == 0) { return rest_rx_request::create({}); }

  auto manager = std::make_shared<request_manager>(bytes_requested, n_chunks);

  std::vector<std::unique_ptr<rest_chunked_rx_request>> chunks;
  chunks.reserve(n_chunks);
  for (auto const& s : segments) {
    size_t const clamped = s.offset < fsize ? std::min(s.size, fsize - s.offset) : 0;
    if (clamped == 0) { continue; }
    auto req       = std::make_unique<rest_chunked_rx_request>();
    req->object    = file.object_ref();
    req->chunk     = io_object_segment{s.offset, clamped, s.data()};
    req->file_size = fsize;
    req->manager   = manager;
    chunks.push_back(std::move(req));
  }
  return rest_rx_request::create(std::move(chunks));
}

rest_reactor::request_type_ptr rest_reactor::prep_device_rx_request(
  const reactor_config_type& /*cfg*/,
  const io_object_type& /*file*/,
  uint8_t* /*dst*/,
  size_t /*offset*/,
  size_t /*size*/,
  rmm::cuda_stream_view /*stream*/,
  int /*device_id*/)
{
  // Implemented in Step 8 (device staging through bounce slots).
  throw std::runtime_error("rest_reactor::prep_device_rx_request: not yet implemented");
}

rest_reactor::request_type_ptr rest_reactor::prep_host_to_device_rx_request(
  const reactor_config_type& /*cfg*/,
  const io_object_type& /*file*/,
  std::span<io_object_segment> /*bounce*/,
  uint8_t* /*dst*/,
  size_t /*offset*/,
  size_t /*size*/,
  rmm::cuda_stream_view /*stream*/,
  int /*device_id*/)
{
  // Implemented in Step 8 (caller-supplied pinned bounce staging).
  throw std::runtime_error("rest_reactor::prep_host_to_device_rx_request: not yet implemented");
}

// ---------------------------------------------------------------------------
// synchronous paths
// ---------------------------------------------------------------------------

size_t rest_reactor::host_read(const io_object_type& file, size_t offset, size_t size, uint8_t* dst)
{
  if (size == 0) { return 0; }
  size = std::min(size, file.size() > offset ? file.size() - offset : size_t{0});
  if (size == 0) { return 0; }

  auto const obj = file.object_ref();
  std::string last_error;
  for (std::size_t attempt = 0; attempt < _config.max_retry_attempts; ++attempt) {
    buf_sink sink{dst, size, 0, 0};
    header_capture hc;

    auto const authd =
      _config.authorizer->authorize(obj, s3::s3_request_method::GET, presign_ttl(_config));

    curl_easy_ptr h{curl_easy_init()};
    if (!h) { throw std::runtime_error("rest_reactor::host_read: curl_easy_init failed"); }
    configure_easy_handle(h.get(), global_curl_context::instance().share_handle());
    apply_request_opts(h.get(), _config);

    std::string const range = range_header(offset, size);
    curl_slist_ptr hdrs     = build_header_list(authd.headers, &range);
    SIRIUS_CURL_CHECK(curl_easy_setopt(h.get(), CURLOPT_URL, authd.url.c_str()));
    SIRIUS_CURL_CHECK(curl_easy_setopt(h.get(), CURLOPT_HTTPHEADER, hdrs.get()));
    SIRIUS_CURL_CHECK(curl_easy_setopt(h.get(), CURLOPT_WRITEFUNCTION, &write_to_sink));
    SIRIUS_CURL_CHECK(curl_easy_setopt(h.get(), CURLOPT_WRITEDATA, &sink));
    SIRIUS_CURL_CHECK(curl_easy_setopt(h.get(), CURLOPT_HEADERFUNCTION, &capture_header));
    SIRIUS_CURL_CHECK(curl_easy_setopt(h.get(), CURLOPT_HEADERDATA, &hc));

    CURLcode const rc = curl_easy_perform(h.get());
    long status       = 0;
    curl_easy_getinfo(h.get(), CURLINFO_RESPONSE_CODE, &status);

    if (rc == CURLE_OK && (status == 206 || (status == 200 && offset == 0))) {
      // A 206 must have delivered exactly the requested range; a 200 is only
      // safe when we asked from offset 0 (the server ignored Range and sent the
      // whole object, which still starts at our offset).  Either way reject a
      // short delivery — feeding truncated bytes downstream would corrupt data.
      if (sink.written < size) {
        last_error =
          "short read (" + std::to_string(sink.written) + "/" + std::to_string(size) + " bytes)";
        // A truncated transfer is transient — retry.
      } else {
        return sink.written;
      }
    } else if (rc == CURLE_OK && status == 200 && offset != 0) {
      // Server ignored the Range header and returned the whole object; the
      // bytes we captured are from offset 0, not `offset` — non-retriable
      // misconfiguration (would loop forever).
      throw std::runtime_error("rest_reactor::host_read: server ignored Range (HTTP 200) for " +
                               obj.bucket + "/" + obj.key);
    } else {
      last_error =
        rc != CURLE_OK ? std::string(curl_easy_strerror(rc)) : ("HTTP " + std::to_string(status));
      bool const retriable = (rc != CURLE_OK && is_retriable_curl(rc)) ||
                             (rc == CURLE_OK && is_retriable_status(status));
      if (!retriable) {
        throw std::runtime_error("rest_reactor::host_read: " + last_error + " for " + obj.bucket +
                                 "/" + obj.key);
      }
    }

    if (attempt + 1 < _config.max_retry_attempts) {
      std::this_thread::sleep_for(compute_backoff(attempt, hc.retry_after, _config));
    }
  }
  throw std::runtime_error("rest_reactor::host_read: exhausted retries (" + last_error + ") for " +
                           obj.bucket + "/" + obj.key);
}

size_t rest_reactor::head_object_size(std::string_view bucket, std::string_view key)
{
  s3::s3_object_ref const obj{std::string(bucket), std::string(key)};
  std::string last_error;
  for (std::size_t attempt = 0; attempt < _config.max_retry_attempts; ++attempt) {
    header_capture hc;
    auto const authd =
      _config.authorizer->authorize(obj, s3::s3_request_method::HEAD, presign_ttl(_config));

    curl_easy_ptr h{curl_easy_init()};
    if (!h) { throw std::runtime_error("rest_reactor::head_object_size: curl_easy_init failed"); }
    configure_easy_handle(h.get(), global_curl_context::instance().share_handle());
    apply_request_opts(h.get(), _config);

    curl_slist_ptr hdrs = build_header_list(authd.headers, nullptr);
    SIRIUS_CURL_CHECK(curl_easy_setopt(h.get(), CURLOPT_URL, authd.url.c_str()));
    SIRIUS_CURL_CHECK(curl_easy_setopt(h.get(), CURLOPT_NOBODY, 1L));
    SIRIUS_CURL_CHECK(curl_easy_setopt(h.get(), CURLOPT_HTTPHEADER, hdrs.get()));
    SIRIUS_CURL_CHECK(curl_easy_setopt(h.get(), CURLOPT_WRITEFUNCTION, &write_discard));
    SIRIUS_CURL_CHECK(curl_easy_setopt(h.get(), CURLOPT_HEADERFUNCTION, &capture_header));
    SIRIUS_CURL_CHECK(curl_easy_setopt(h.get(), CURLOPT_HEADERDATA, &hc));

    CURLcode const rc = curl_easy_perform(h.get());
    long status       = 0;
    curl_easy_getinfo(h.get(), CURLINFO_RESPONSE_CODE, &status);

    if (rc == CURLE_OK && status == 200) {
      curl_off_t cl = -1;
      curl_easy_getinfo(h.get(), CURLINFO_CONTENT_LENGTH_DOWNLOAD_T, &cl);
      if (cl < 0) {
        throw std::runtime_error("rest_reactor::head_object_size: missing Content-Length for " +
                                 obj.bucket + "/" + obj.key);
      }
      return static_cast<size_t>(cl);
    }

    last_error =
      rc != CURLE_OK ? std::string(curl_easy_strerror(rc)) : ("HTTP " + std::to_string(status));
    bool const retriable =
      (rc != CURLE_OK && is_retriable_curl(rc)) || (rc == CURLE_OK && is_retriable_status(status));
    if (!retriable) {
      throw std::runtime_error("rest_reactor::head_object_size: " + last_error + " for " +
                               obj.bucket + "/" + obj.key);
    }
    if (attempt + 1 < _config.max_retry_attempts) {
      std::this_thread::sleep_for(compute_backoff(attempt, hc.retry_after, _config));
    }
  }
  throw std::runtime_error("rest_reactor::head_object_size: exhausted retries (" + last_error +
                           ") for " + obj.bucket + "/" + obj.key);
}

// ---------------------------------------------------------------------------
// capabilities / factory
// ---------------------------------------------------------------------------

bool rest_reactor::supports(std::string_view path)
{
  try {
    auto const parsed = sirius::io::parse(path);
    return parsed.scheme == "s3";
  } catch (...) {
    return false;
  }
}

std::unique_ptr<rest_reactor::io_object_type> rest_reactor::create_io_object(std::string /*path*/)
{
  // The object size requires a HEAD round-trip and the authorizer, both of
  // which live on the reactor instance — see rest_ioctx::create_io_object.
  throw std::logic_error(
    "rest_reactor::create_io_object: use rest_ioctx::create_io_object (needs HEAD + authorizer)");
}

std::vector<cudf::io::text::byte_range_info> rest_reactor::align_and_coalesce(
  std::span<const cudf::io::text::byte_range_info> ranges, std::optional<size_t> alignment)
{
  // No physical block alignment for REST: honor a caller alignment >= 1 as a
  // lower bound, otherwise treat alignment as 1 (byte) — i.e. pure coalescing.
  size_t const align = std::max<size_t>(alignment.value_or(1), 1);

  std::vector<cudf::io::text::byte_range_info> aligned;
  aligned.reserve(ranges.size());
  for (auto const& r : ranges) {
    if (r.size() <= 0) { continue; }
    auto const offset  = static_cast<size_t>(r.offset());
    auto const end     = offset + static_cast<size_t>(r.size());
    size_t const start = (offset / align) * align;
    size_t const stop  = ((end + align - 1) / align) * align;
    aligned.emplace_back(static_cast<int64_t>(start), static_cast<int64_t>(stop - start));
  }
  if (aligned.empty()) { return aligned; }

  std::sort(aligned.begin(), aligned.end(), [](auto const& a, auto const& b) {
    return a.offset() < b.offset();
  });

  std::vector<cudf::io::text::byte_range_info> coalesced;
  coalesced.reserve(aligned.size());
  coalesced.push_back(aligned.front());
  for (size_t i = 1; i < aligned.size(); ++i) {
    auto& last            = coalesced.back();
    auto const last_start = static_cast<size_t>(last.offset());
    auto const last_end   = last_start + static_cast<size_t>(last.size());
    auto const cur_start  = static_cast<size_t>(aligned[i].offset());
    auto const cur_end    = cur_start + static_cast<size_t>(aligned[i].size());
    if (cur_start <= last_end) {  // overlap or adjacency
      size_t const new_end = std::max(last_end, cur_end);
      last                 = {last.offset(), static_cast<int64_t>(new_end - last_start)};
    } else {
      coalesced.push_back(aligned[i]);
    }
  }
  return coalesced;
}

// ---------------------------------------------------------------------------
// worker loop (epoll + curl_multi_socket_action engine)
// ---------------------------------------------------------------------------

namespace {

/// One in-flight ranged GET on a borrowed easy handle.  Owns everything whose
/// address curl references for the duration of the transfer (the URL string,
/// the header list, the write/header targets), so the transfer object must
/// outlive its handle's membership in the multi.
struct transfer {
  std::unique_ptr<rest_chunked_rx_request> req;
  CURL* easy{nullptr};
  std::string url;  // backs CURLOPT_URL
  curl_slist_ptr headers;
  buf_sink sink;
  header_capture hc;
};

/// Worker-loop state reachable from curl's C socket/timer callbacks.
struct worker_state {
  CURLM* multi{nullptr};
  int epoll_fd{-1};
  int curl_timer_fd{-1};
};

/// CURLMOPT_SOCKETFUNCTION: mirror curl's interest in @p s into epoll.
int rest_socket_cb(CURL* /*easy*/, curl_socket_t s, int what, void* userp, void* socketp)
{
  auto* ws = static_cast<worker_state*>(userp);
  if (what == CURL_POLL_REMOVE) {
    // Best-effort delete; the socket may already be gone (ENOENT/EBADF).
    ::epoll_ctl(ws->epoll_fd, EPOLL_CTL_DEL, s, nullptr);
    return 0;
  }
  uint32_t events = 0;
  if (what == CURL_POLL_IN || what == CURL_POLL_INOUT) { events |= EPOLLIN; }
  if (what == CURL_POLL_OUT || what == CURL_POLL_INOUT) { events |= EPOLLOUT; }
  epoll_event ev{};
  ev.events  = events;
  ev.data.fd = s;
  // socketp is curl's per-socket user pointer: null on first sighting (ADD),
  // non-null thereafter (MOD).  We only use it as an "already added" marker.
  int const op = (socketp == nullptr) ? EPOLL_CTL_ADD : EPOLL_CTL_MOD;
  if (socketp == nullptr) { curl_multi_assign(ws->multi, s, ws); }
  ::epoll_ctl(ws->epoll_fd, op, s, &ev);
  return 0;
}

/// CURLMOPT_TIMERFUNCTION: arm/disarm the curl timerfd.
int rest_timer_cb(CURLM* /*multi*/, long timeout_ms, void* userp)
{
  auto* ws = static_cast<worker_state*>(userp);
  itimerspec its{};  // all-zero => disarm
  if (timeout_ms == 0) {
    its.it_value.tv_nsec = 1;  // fire essentially immediately
  } else if (timeout_ms > 0) {
    its.it_value.tv_sec  = timeout_ms / 1000;
    its.it_value.tv_nsec = (timeout_ms % 1000) * 1'000'000L;
  }
  ::timerfd_settime(ws->curl_timer_fd, 0, &its, nullptr);
  return 0;
}

/// Drain (and discard) all pending reads from a non-blocking fd.
void drain_fd(int fd) noexcept
{
  uint64_t v = 0;
  while (::read(fd, &v, sizeof(v)) > 0) {}
}

}  // namespace

void rest_reactor::worker_loop(const std::stop_token& stop_token)
{
  constexpr int MAX_EVENTS = 64;

  // Wake the loop out of epoll_wait when shutdown is requested.
  std::stop_callback const stop_cb(stop_token, [this] { interrupt(); });

  try {
    curl_multi_ptr multi{curl_multi_init()};
    if (!multi) { throw std::runtime_error("rest_reactor: curl_multi_init failed"); }

    file_descriptor epoll_fd      = make_epoll_fd();
    file_descriptor curl_timer_fd = make_timer_fd();
    worker_state ws{multi.get(), epoll_fd.get(), curl_timer_fd.get()};

    SIRIUS_CURLM_CHECK(curl_multi_setopt(multi.get(), CURLMOPT_SOCKETFUNCTION, &rest_socket_cb));
    SIRIUS_CURLM_CHECK(curl_multi_setopt(multi.get(), CURLMOPT_SOCKETDATA, &ws));
    SIRIUS_CURLM_CHECK(curl_multi_setopt(multi.get(), CURLMOPT_TIMERFUNCTION, &rest_timer_cb));
    SIRIUS_CURLM_CHECK(curl_multi_setopt(multi.get(), CURLMOPT_TIMERDATA, &ws));
    SIRIUS_CURLM_CHECK(
      curl_multi_setopt(multi.get(), CURLMOPT_PIPELINING, static_cast<long>(CURLPIPE_MULTIPLEX)));
    SIRIUS_CURLM_CHECK(curl_multi_setopt(
      multi.get(), CURLMOPT_MAX_HOST_CONNECTIONS, static_cast<long>(_config.max_connections)));
    SIRIUS_CURLM_CHECK(curl_multi_setopt(
      multi.get(), CURLMOPT_MAXCONNECTS, static_cast<long>(_config.max_connections)));

    auto epoll_add = [&](int fd, uint32_t events) {
      epoll_event ev{};
      ev.events  = events;
      ev.data.fd = fd;
      if (::epoll_ctl(epoll_fd.get(), EPOLL_CTL_ADD, fd, &ev) != 0) {
        throw std::runtime_error(std::string("rest_reactor: epoll_ctl ADD failed: ") +
                                 std::strerror(errno));
      }
    };
    epoll_add(_wakeup_fd.get(), EPOLLIN);
    epoll_add(curl_timer_fd.get(), EPOLLIN);

    // Easy-handle pool: max_connections handles configured once with the
    // static performance + TLS/timeout options; per-request options are set at
    // submit time.  Worker-thread-local, so no locking.
    CURLSH* const share = global_curl_context::instance().share_handle();
    std::vector<curl_easy_ptr> handle_storage;
    std::vector<CURL*> free_handles;
    handle_storage.reserve(_config.max_connections);
    free_handles.reserve(_config.max_connections);
    for (std::size_t i = 0; i < _config.max_connections; ++i) {
      curl_easy_ptr h{curl_easy_init()};
      if (!h) { throw std::runtime_error("rest_reactor: curl_easy_init failed"); }
      configure_easy_handle(h.get(), share);
      apply_request_opts(h.get(), _config);
      free_handles.push_back(h.get());
      handle_storage.push_back(std::move(h));
    }

    std::unordered_map<CURL*, std::unique_ptr<transfer>> inflight_map;
    int running = 0;

    auto setup_easy = [&](transfer& t) {
      auto authd = _config.authorizer->authorize(
        t.req->object, s3::s3_request_method::GET, presign_ttl(_config));
      t.url  = std::move(authd.url);
      t.sink = buf_sink{t.req->chunk.data(), t.req->chunk.size, 0, 0};
      t.hc.reset();
      std::string const range = range_header(t.req->chunk.offset, t.req->chunk.size);
      t.headers               = build_header_list(authd.headers, &range);
      CURL* const h           = t.easy;
      SIRIUS_CURL_CHECK(curl_easy_setopt(h, CURLOPT_URL, t.url.c_str()));
      SIRIUS_CURL_CHECK(curl_easy_setopt(h, CURLOPT_HTTPHEADER, t.headers.get()));
      SIRIUS_CURL_CHECK(curl_easy_setopt(h, CURLOPT_WRITEFUNCTION, &write_to_sink));
      SIRIUS_CURL_CHECK(curl_easy_setopt(h, CURLOPT_WRITEDATA, &t.sink));
      SIRIUS_CURL_CHECK(curl_easy_setopt(h, CURLOPT_HEADERFUNCTION, &capture_header));
      SIRIUS_CURL_CHECK(curl_easy_setopt(h, CURLOPT_HEADERDATA, &t.hc));
    };

    auto submit = [&]() {
      while (static_cast<std::size_t>(inflight_map.size()) < _config.max_connections &&
             !free_handles.empty()) {
        std::unique_ptr<rest_chunked_rx_request> dr;
        if (!_requests.try_dequeue(dr)) { break; }
        if (!dr) { continue; }
        if (dr->manager->has_error()) {
          dr.reset();
          continue;
        }
        // Device chunks with a null destination need a bounce slot (Step 8);
        // host chunks carry their own buffer.
        CURL* const h = free_handles.back();
        free_handles.pop_back();
        auto t  = std::make_unique<transfer>();
        t->req  = std::move(dr);
        t->easy = h;
        setup_easy(*t);
        curl_multi_add_handle(multi.get(), h);
        inflight_map.emplace(h, std::move(t));
      }
    };

    auto finish = [&](std::unique_ptr<transfer> t, CURLcode rc, long status) {
      auto& req           = *t->req;
      bool const ok_range = (status == 206) || (status == 200 && req.chunk.offset == 0);
      if (rc == CURLE_OK && ok_range && t->sink.written >= req.chunk.size) {
        // Host success.  (Device H2D copy is wired up in Step 8.)
        req.manager->chunk_complete(t->sink.written);
        return;
      }
      if (rc == CURLE_OK && status == 200 && req.chunk.offset != 0) {
        // Server ignored Range and returned the whole object: the bytes start
        // at offset 0, not req.offset — non-retriable, would loop forever.
        req.manager->report_error(std::make_exception_ptr(
          std::runtime_error("rest_reactor: server ignored Range (HTTP 200) for " +
                             req.object.bucket + "/" + req.object.key)));
        return;
      }
      // Error or truncated transfer.  Retry scheduling lands in Step 7; for now
      // a failure is terminal.
      std::string const msg = rc != CURLE_OK
                                ? std::string(curl_easy_strerror(rc))
                                : (ok_range ? "short read" : "HTTP " + std::to_string(status));
      req.manager->report_error(std::make_exception_ptr(std::runtime_error(
        "rest_reactor: " + msg + " for " + req.object.bucket + "/" + req.object.key)));
    };

    auto process_completions = [&]() {
      CURLMsg* msg = nullptr;
      int in_queue = 0;
      while ((msg = curl_multi_info_read(multi.get(), &in_queue)) != nullptr) {
        if (msg->msg != CURLMSG_DONE) { continue; }
        CURL* const h     = msg->easy_handle;
        CURLcode const rc = msg->data.result;
        long status       = 0;
        curl_easy_getinfo(h, CURLINFO_RESPONSE_CODE, &status);
        curl_multi_remove_handle(multi.get(), h);
        auto it = inflight_map.find(h);
        if (it == inflight_map.end()) { continue; }
        std::unique_ptr<transfer> t = std::move(it->second);
        inflight_map.erase(it);
        free_handles.push_back(h);
        finish(std::move(t), rc, status);
      }
    };

    std::array<epoll_event, MAX_EVENTS> events{};
    submit();  // kickstart anything already queued
    while (!stop_token.stop_requested()) {
      int const n = ::epoll_wait(epoll_fd.get(), events.data(), MAX_EVENTS, -1);
      if (n < 0) {
        if (errno == EINTR) { continue; }
        throw std::runtime_error(std::string("rest_reactor: epoll_wait failed: ") +
                                 std::strerror(errno));
      }
      for (int i = 0; i < n; ++i) {
        int const fd = events[i].data.fd;
        if (fd == _wakeup_fd.get()) {
          drain_fd(_wakeup_fd.get());
        } else if (fd == curl_timer_fd.get()) {
          drain_fd(curl_timer_fd.get());
          curl_multi_socket_action(multi.get(), CURL_SOCKET_TIMEOUT, 0, &running);
        } else {
          int ev_bitmask = 0;
          if (events[i].events & EPOLLIN) { ev_bitmask |= CURL_CSELECT_IN; }
          if (events[i].events & EPOLLOUT) { ev_bitmask |= CURL_CSELECT_OUT; }
          if (events[i].events & (EPOLLERR | EPOLLHUP)) { ev_bitmask |= CURL_CSELECT_ERR; }
          curl_multi_socket_action(multi.get(), fd, ev_bitmask, &running);
        }
      }
      process_completions();
      submit();
    }

    // Drain on shutdown: detach in-flight handles and cancel their requests so
    // no future is left unfulfilled, then cancel anything still queued.
    for (auto& [h, t] : inflight_map) {
      curl_multi_remove_handle(multi.get(), h);
      if (t) { t->req->manager->report_error(std::make_error_code(std::errc::operation_canceled)); }
    }
    inflight_map.clear();
  } catch (const std::exception& e) {
    spdlog::error("rest_reactor worker_loop: {}", e.what());
  }

  std::unique_ptr<rest_chunked_rx_request> dr;
  while (_requests.try_dequeue(dr)) {
    if (dr) { dr->manager->report_error(std::make_error_code(std::errc::operation_canceled)); }
    dr.reset();
  }
}

}  // namespace sirius::io::rest
