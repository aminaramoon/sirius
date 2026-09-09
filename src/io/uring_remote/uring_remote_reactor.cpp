/*
 * Copyright 2026, Sirius Contributors.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#include "io/uring_remote/uring_remote_reactor.hpp"

#include "io/uring_remote/http_response.hpp"
#include "log/logging.hpp"

#include <arpa/inet.h>
#include <liburing.h>
#include <linux/tls.h>
#include <netdb.h>
#include <netinet/tcp.h>
#include <openssl/err.h>
#include <openssl/ssl.h>
#include <openssl/x509v3.h>
#include <poll.h>

#include <array>
#include <chrono>
#include <climits>
#include <cstring>
#include <optional>
#include <unordered_map>

namespace sirius::io::uring_remote {
namespace {

using clock_type                   = std::chrono::steady_clock;
constexpr std::uint64_t cancel_bit = std::uint64_t{1} << 63;

struct endpoint {
  std::string host;
  std::string port;
  std::string authority;
  std::string target;
  bool tls{true};
  std::string origin() const { return (tls ? "https://" : "http://") + authority; }
};

endpoint parse_endpoint(std::string const& url)
{
  std::unique_ptr<CURLU, decltype(&curl_url_cleanup)> parsed{curl_url(), curl_url_cleanup};
  if (!parsed ||
      curl_url_set(parsed.get(), CURLUPART_URL, url.c_str(), CURLU_PATH_AS_IS) != CURLUE_OK) {
    throw std::invalid_argument("uring_remote: invalid authorized URL");
  }
  auto get = [&](CURLUPart part, unsigned flags = 0) {
    char* raw     = nullptr;
    auto const rc = curl_url_get(parsed.get(), part, &raw, flags);
    std::unique_ptr<char, decltype(&curl_free)> value{raw, curl_free};
    return rc == CURLUE_OK ? std::string{raw} : std::string{};
  };
  auto const scheme = get(CURLUPART_SCHEME);
  if (scheme != "http" && scheme != "https") {
    throw std::invalid_argument("uring_remote: only HTTP and HTTPS endpoints are supported");
  }
  if (!get(CURLUPART_USER).empty() || !get(CURLUPART_PASSWORD).empty() ||
      !get(CURLUPART_FRAGMENT).empty()) {
    throw std::invalid_argument("uring_remote: URL userinfo and fragments are unsupported");
  }
  endpoint result;
  result.tls       = scheme == "https";
  result.host      = get(CURLUPART_HOST);
  result.port      = get(CURLUPART_PORT, CURLU_DEFAULT_PORT);
  result.authority = result.host;
  if (result.port != (result.tls ? "443" : "80")) result.authority += ':' + result.port;
  if (result.host.starts_with('[') && result.host.ends_with(']')) {
    result.host = result.host.substr(1, result.host.size() - 2);
  }
  result.target = get(CURLUPART_PATH);
  if (result.target.empty()) result.target = "/";
  if (auto query = get(CURLUPART_QUERY); !query.empty()) result.target += '?' + query;
  if (result.host.empty() || result.port.empty() ||
      result.target.find_first_of("\r\n ") != std::string::npos) {
    throw std::invalid_argument("uring_remote: invalid HTTP authority or target");
  }
  return result;
}

std::string request_text(endpoint const& remote, rest::authorized_request const& auth, range bytes)
{
  std::string result = "GET " + remote.target + " HTTP/1.1\r\n";
  bool has_host      = false;
  for (auto const& [name, value] : auth.headers) {
    if (!httplib::detail::fields::is_field_valid(name, value)) {
      throw std::invalid_argument("uring_remote: invalid authorization header");
    }
    if (httplib::detail::case_ignore::equal(name, "Host")) has_host = true;
    if (httplib::detail::case_ignore::equal(name, "Range") ||
        httplib::detail::case_ignore::equal(name, "Connection") ||
        httplib::detail::case_ignore::equal(name, "Accept-Encoding")) {
      throw std::invalid_argument("uring_remote: authorizer supplied a transport-owned header");
    }
    result += name + ": " + value + "\r\n";
  }
  if (!has_host) result += "Host: " + remote.authority + "\r\n";
  result += "Range: bytes=" + std::to_string(bytes.offset) + '-' + std::to_string(bytes.end() - 1) +
            "\r\nAccept-Encoding: identity\r\nConnection: keep-alive\r\n\r\n";
  return result;
}

/// One outstanding primary SQE per slot. Cancellation CQEs are tracked
/// separately: a slot is never reused until BOTH CQEs have been consumed.
class uring_transport final : public rest::remote_transport {
 public:
  uring_transport(rest::config reads, config network)
    : _reads(std::move(reads)), _config(network), _slots(network.max_connections)
  {
    _config.validate();
    if (_reads.request_timeout_s < 0) throw std::invalid_argument("uring_remote: negative timeout");
#ifdef OPENSSL_NO_KTLS
    if (!_config.allow_plaintext) {
      throw std::runtime_error(
        "uring_remote: this OpenSSL was built without kTLS; run pixi run build-ktls-openssl "
        "and configure with -DSIRIUS_KTLS_OPENSSL_ROOT=$PWD/build/ktls-openssl");
    }
#endif
    _ssl_ctx.reset(SSL_CTX_new(TLS_client_method()));
    if (!_ssl_ctx) throw std::runtime_error("uring_remote: SSL_CTX_new failed");
    // TLS 1.2 avoids post-handshake TLS 1.3 KeyUpdate/session-ticket records
    // when application I/O bypasses SSL_read/SSL_write. Restrict to kTLS AEAD.
    if (SSL_CTX_set_min_proto_version(_ssl_ctx.get(), TLS1_2_VERSION) != 1 ||
        SSL_CTX_set_max_proto_version(_ssl_ctx.get(), TLS1_2_VERSION) != 1 ||
        SSL_CTX_set_cipher_list(_ssl_ctx.get(),
                                "ECDHE-ECDSA-AES128-GCM-SHA256:ECDHE-RSA-AES128-GCM-SHA256:"
                                "ECDHE-ECDSA-AES256-GCM-SHA384:ECDHE-RSA-AES256-GCM-SHA384") != 1) {
      throw std::runtime_error("uring_remote: cannot configure TLS 1.2 AES-GCM");
    }
    SSL_CTX_set_options(_ssl_ctx.get(), SSL_OP_ENABLE_KTLS | SSL_OP_NO_RENEGOTIATION);
    SSL_CTX_set_read_ahead(_ssl_ctx.get(), 0);
    SSL_CTX_set_verify(
      _ssl_ctx.get(), _reads.tls_verify ? SSL_VERIFY_PEER : SSL_VERIFY_NONE, nullptr);
    if (_reads.tls_verify) {
      auto const ok =
        _reads.ca_bundle_path.empty()
          ? SSL_CTX_set_default_verify_paths(_ssl_ctx.get())
          : SSL_CTX_load_verify_locations(_ssl_ctx.get(), _reads.ca_bundle_path.c_str(), nullptr);
      if (ok != 1) throw std::runtime_error("uring_remote: cannot load TLS CA bundle");
    }
    auto const rc = io_uring_queue_init(static_cast<unsigned>(_config.queue_depth), &_ring, 0);
    if (rc < 0)
      throw std::system_error(-rc, std::generic_category(), "uring_remote: io_uring_queue_init");
    _ring_live = true;
    SIRIUS_LOG_INFO("uring_remote: {} connections, {} SQ entries; HTTPS requires kTLS RX and TX",
                    _config.max_connections,
                    _config.queue_depth);
  }

  ~uring_transport() override { shutdown(); }

  int completion_fd() const noexcept override { return _ring.ring_fd; }

  int timeout_ms() const noexcept override
  {
    if (!_completed.empty()) return 0;
    int timeout    = -1;
    auto const now = clock_type::now();
    for (auto const& slot : _slots) {
      if (!slot.pending || !slot.deadline || slot.canceling) continue;
      auto const remaining =
        std::chrono::duration_cast<std::chrono::milliseconds>(*slot.deadline - now).count();
      auto const ms = static_cast<int>(std::clamp<std::int64_t>(remaining, 0, INT_MAX));
      timeout       = timeout < 0 ? ms : std::min(timeout, ms);
    }
    return timeout;
  }

  void submit(std::size_t index,
              rest::authorized_request auth,
              range bytes,
              rest::buf_sink& sink,
              rest::header_capture& headers) override
  {
    auto remote = parse_endpoint(auth.url);
    check_scheme(remote);
    auto text  = request_text(remote, auth, bytes);
    auto& slot = _slots.at(index);
    if (slot.sink != nullptr) throw std::logic_error("uring_remote: occupied slot");
    // A warm-up may still be connecting. Preserve its pending SQE and attach
    // the first real request; after completion reconnect if the origin changed.
    slot.next_remote = std::move(remote);
    slot.request     = std::move(text);
    slot.sink        = &sink;
    slot.headers     = &headers;
    slot.response.emplace(_config.max_header_bytes);
    slot.sent             = 0;
    slot.headers_captured = false;
    slot.deadline         = deadline();
    if (slot.pending || slot.cancel_pending) return;
    try {
      begin(index);
    } catch (rest::remote_transport_error const&) {
      throw;
    } catch (std::exception const& error) {
      finish(index, CURLE_FAILED_INIT, error.what());
    }
  }

  void warmup(rest::authorized_request auth) override
  {
    auto remote = parse_endpoint(auth.url);
    check_scheme(remote);
    for (std::size_t i = 0; i < _slots.size(); ++i) {
      auto& slot = _slots[i];
      if (slot.pending || slot.cancel_pending || slot.sink != nullptr) continue;
      slot.next_remote = remote;
      slot.deadline    = clock_type::now() + std::chrono::seconds{30};
      try {
        begin(i);
      } catch (rest::remote_transport_error const&) {
        throw;
      } catch (std::exception const& error) {
        finish(i, CURLE_COULDNT_CONNECT, error.what());
      }
    }
    flush();
  }

  std::vector<completion> poll() override
  {
    auto const now = clock_type::now();
    for (std::size_t i = 0; i < _slots.size(); ++i) {
      auto& slot = _slots[i];
      if (slot.pending && slot.deadline && now >= *slot.deadline && !slot.canceling) cancel(i);
    }
    io_uring_cqe* cqe = nullptr;
    // Bound one pass: a hot connection must not starve retries or CUDA polling.
    for (std::size_t count = 0; count < _config.queue_depth && io_uring_peek_cqe(&_ring, &cqe) == 0;
         ++count) {
      auto const id     = io_uring_cqe_get_data64(cqe);
      auto const result = cqe->res;
      io_uring_cqe_seen(&_ring, cqe);
      auto const index = static_cast<std::size_t>((id & ~cancel_bit) - 1);
      auto& slot       = _slots.at(index);
      if (id & cancel_bit)
        slot.cancel_pending = false;
      else
        slot.pending = false;
      if (slot.canceling) {
        if (!slot.pending && !slot.cancel_pending)
          finish(index, CURLE_OPERATION_TIMEDOUT, "request timed out");
        continue;
      }
      try {
        advance(index, result);
      } catch (rest::remote_transport_error const&) {
        throw;
      } catch (std::exception const& error) {
        finish(index, CURLE_WEIRD_SERVER_REPLY, error.what());
      }
    }
    flush();
    auto results = std::move(_completed);
    _completed.clear();
    return results;
  }

  void shutdown() noexcept override
  {
    if (!_ring_live) return;
    // Do not release sockets, msghdr/iovecs, or caller memory while the kernel
    // can reference them. Submit cancellation and reap every outstanding CQE.
    for (std::size_t i = 0; i < _slots.size(); ++i) {
      auto& slot = _slots[i];
      if (slot.pending && !slot.canceling) {
        try {
          cancel(i);
        } catch (...) {
          if (slot.fd) ::shutdown(slot.fd.get(), SHUT_RDWR);
        }
      }
    }
    while (std::any_of(
      _slots.begin(), _slots.end(), [](auto const& s) { return s.pending || s.cancel_pending; })) {
      auto const submitted = io_uring_submit(&_ring);
      if (submitted < 0 && submitted != -EINTR) {
        for (auto& slot : _slots)
          if (slot.fd) ::shutdown(slot.fd.get(), SHUT_RDWR);
      }
      io_uring_cqe* cqe = nullptr;
      auto const rc     = io_uring_wait_cqe(&_ring, &cqe);
      if (rc < 0) continue;
      auto const id = io_uring_cqe_get_data64(cqe);
      auto& slot    = _slots[static_cast<std::size_t>((id & ~cancel_bit) - 1)];
      if (id & cancel_bit)
        slot.cancel_pending = false;
      else
        slot.pending = false;
      io_uring_cqe_seen(&_ring, cqe);
    }
    io_uring_queue_exit(&_ring);
    _ring_live = false;
    for (auto& slot : _slots)
      close(slot);
  }

 private:
  enum class operation { connect, handshake, send, receive };
  struct socket_address {
    sockaddr_storage addr;
    socklen_t size;
  };
  struct connection {
    file_descriptor fd;
    std::unique_ptr<SSL, decltype(&SSL_free)> ssl{nullptr, SSL_free};
    endpoint remote;
    endpoint next_remote;
    bool established{false};
    bool pending{false};
    bool cancel_pending{false};
    bool canceling{false};
    operation op{operation::connect};
    clock_type::time_point idle_since{};
    std::optional<clock_type::time_point> deadline;
    sockaddr_storage address{};
    socklen_t address_size{0};
    std::vector<socket_address> connect_addresses;
    std::size_t next_address{0};
    std::string request;
    std::size_t sent{0};
    rest::buf_sink* sink{nullptr};
    rest::header_capture* headers{nullptr};
    std::optional<http_response> response;
    std::vector<char> buffer;
    std::vector<iovec> iovecs;
    msghdr message{};
    alignas(cmsghdr) std::array<char, CMSG_SPACE(1)> control{};
    bool direct{false};
    bool headers_captured{false};
  };

  struct dns_entry {
    std::vector<socket_address> addresses;
    std::size_t next{0};
    clock_type::time_point expires;
  };

  std::optional<clock_type::time_point> deadline() const
  {
    if (_reads.request_timeout_s == 0) return std::nullopt;
    return clock_type::now() + std::chrono::seconds{_reads.request_timeout_s};
  }

  void check_scheme(endpoint const& remote) const
  {
    if (!remote.tls && !_config.allow_plaintext) {
      throw std::invalid_argument(
        "uring_remote: HTTP requires allow_plaintext: true; use HTTPS for S3");
    }
#ifdef OPENSSL_NO_KTLS
    if (remote.tls) throw std::runtime_error("uring_remote: OpenSSL was built without kTLS");
#endif
  }

  static void close(connection& slot) noexcept
  {
    slot.ssl.reset();
    slot.fd          = file_descriptor{};
    slot.established = false;
  }

  io_uring_sqe* sqe(std::uint64_t id)
  {
    auto* entry = io_uring_get_sqe(&_ring);
    if (!entry) {
      flush();
      entry = io_uring_get_sqe(&_ring);
    }
    if (!entry) throw rest::remote_transport_error("uring_remote: submission queue exhausted");
    io_uring_sqe_set_data64(entry, id);
    return entry;
  }

  void flush() override
  {
    while (io_uring_sq_ready(&_ring) != 0) {
      auto const rc = io_uring_submit(&_ring);
      if (rc == -EINTR) continue;
      if (rc < 0)
        throw rest::remote_transport_error(std::string("uring_remote: submit: ") +
                                           std::strerror(-rc));
      if (rc == 0) throw rest::remote_transport_error("uring_remote: submit made no progress");
    }
  }

  void cancel(std::size_t index)
  {
    auto& slot  = _slots[index];
    auto* entry = sqe(cancel_bit | (index + 1));
    io_uring_prep_cancel64(entry, index + 1, 0);
    slot.canceling      = true;
    slot.cancel_pending = true;
  }

  void resolve(connection& slot)
  {
    auto& cache = _dns[slot.remote.origin()];
    if (cache.addresses.empty() || clock_type::now() >= cache.expires) {
      addrinfo hints{};
      hints.ai_socktype = SOCK_STREAM;
      hints.ai_family   = AF_UNSPEC;
      addrinfo* raw     = nullptr;
      auto const rc = getaddrinfo(slot.remote.host.c_str(), slot.remote.port.c_str(), &hints, &raw);
      std::unique_ptr<addrinfo, decltype(&freeaddrinfo)> addresses{raw, freeaddrinfo};
      if (rc != 0) throw std::runtime_error("uring_remote: DNS lookup failed");
      cache.addresses.clear();
      for (auto* address = raw; address != nullptr; address = address->ai_next) {
        if (address->ai_addrlen > sizeof(sockaddr_storage)) continue;
        socket_address item{};
        item.size = static_cast<socklen_t>(address->ai_addrlen);
        std::memcpy(&item.addr, address->ai_addr, item.size);
        cache.addresses.push_back(item);
      }
      cache.expires = clock_type::now() + std::chrono::seconds{60};
    }
    if (cache.addresses.empty())
      throw std::runtime_error("uring_remote: DNS returned no TCP addresses");
    slot.connect_addresses = cache.addresses;
    auto const first       = cache.next++ % cache.addresses.size();
    std::rotate(slot.connect_addresses.begin(),
                slot.connect_addresses.begin() + first,
                slot.connect_addresses.end());
    slot.next_address = 0;
  }

  void begin(std::size_t index)
  {
    auto& slot = _slots[index];
    auto const stale =
      _reads.conn_max_age.count() > 0 && clock_type::now() - slot.idle_since >= _reads.conn_max_age;
    if (slot.established && (slot.remote.origin() != slot.next_remote.origin() || stale))
      close(slot);
    if (slot.established) {
      if (slot.sink)
        send(index);
      else
        slot.deadline.reset();
      return;
    }
    slot.remote = slot.next_remote;
    try {
      resolve(slot);
    } catch (...) {
      finish(index, CURLE_COULDNT_RESOLVE_HOST, "DNS lookup failed");
      return;
    }
    connect_next(index);
  }

  void connect_next(std::size_t index)
  {
    auto& slot = _slots[index];
    close(slot);
    // Try every DNS answer within this attempt. A shared round-robin alone can
    // otherwise repeatedly assign an unreachable address to the same request.
    while (slot.next_address < slot.connect_addresses.size()) {
      auto const& address = slot.connect_addresses[slot.next_address++];
      slot.address        = address.addr;
      slot.address_size   = address.size;
      slot.fd             = file_descriptor{
        ::socket(slot.address.ss_family, SOCK_STREAM | SOCK_NONBLOCK | SOCK_CLOEXEC, 0)};
      if (slot.fd) break;
    }
    if (!slot.fd) {
      finish(index, CURLE_COULDNT_CONNECT, "could not connect to any resolved address");
      return;
    }
    int one = 1;
    ::setsockopt(slot.fd.get(), IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));
    ::setsockopt(slot.fd.get(), SOL_SOCKET, SO_KEEPALIVE, &one, sizeof(one));
    auto* entry = sqe(index + 1);
    io_uring_prep_connect(
      entry, slot.fd.get(), reinterpret_cast<sockaddr*>(&slot.address), slot.address_size);
    slot.op      = operation::connect;
    slot.pending = true;
  }

  void handshake(std::size_t index)
  {
    auto& slot = _slots[index];
    ERR_clear_error();
    auto const rc = SSL_connect(slot.ssl.get());
    if (rc == 1) {
      if (BIO_get_ktls_recv(SSL_get_rbio(slot.ssl.get())) != 1 ||
          BIO_get_ktls_send(SSL_get_wbio(slot.ssl.get())) != 1) {
        finish(index,
               CURLE_NOT_BUILT_IN,
               "kTLS RX/TX unavailable: enable CONFIG_TLS and the tls kernel module, "
               "and build OpenSSL with enable-ktls (see docs/super-sirius/uring-remote.md)");
        return;
      }
      if (SSL_pending(slot.ssl.get()) || SSL_has_pending(slot.ssl.get())) {
        finish(index, CURLE_SSL_CONNECT_ERROR, "unexpected buffered data after TLS handshake");
        return;
      }
      if (!_reported_ktls) {
        SIRIUS_LOG_INFO("uring_remote: verified kTLS RX=1 TX=1, TLS 1.2, cipher {}",
                        SSL_get_cipher_name(slot.ssl.get()));
        _reported_ktls = true;
      }
      connected(index);
      return;
    }
    auto const error = SSL_get_error(slot.ssl.get(), rc);
    if (error != SSL_ERROR_WANT_READ && error != SSL_ERROR_WANT_WRITE) {
      auto const cert_error = SSL_get_verify_result(slot.ssl.get()) != X509_V_OK;
      finish(index,
             cert_error ? CURLE_PEER_FAILED_VERIFICATION : CURLE_SSL_CONNECT_ERROR,
             cert_error ? "TLS certificate/hostname verification failed" : "TLS handshake failed");
      return;
    }
    auto* entry = sqe(index + 1);
    io_uring_prep_poll_add(entry, slot.fd.get(), error == SSL_ERROR_WANT_READ ? POLLIN : POLLOUT);
    slot.op      = operation::handshake;
    slot.pending = true;
  }

  void connected(std::size_t index)
  {
    auto& slot       = _slots[index];
    slot.established = true;
    slot.idle_since  = clock_type::now();
    if (slot.sink)
      begin(index);
    else
      slot.deadline.reset();
  }

  void send(std::size_t index)
  {
    auto& slot  = _slots[index];
    auto* entry = sqe(index + 1);
    io_uring_prep_send(entry,
                       slot.fd.get(),
                       slot.request.data() + slot.sent,
                       slot.request.size() - slot.sent,
                       MSG_NOSIGNAL);
    slot.op      = operation::send;
    slot.pending = true;
  }

  void receive(std::size_t index)
  {
    auto& slot = _slots[index];
    slot.iovecs.clear();
    auto const direct = slot.response->direct_remaining();
    slot.direct       = direct > 0 && direct <= slot.sink->capacity - slot.sink->written;
    if (slot.direct) {
      auto remaining = std::min(direct, _config.receive_buffer_bytes);
      auto cursor    = slot.sink->cursor;
      for (auto i = slot.sink->active;
           i < slot.sink->buffers.size() && remaining > 0 && slot.iovecs.size() < 64;
           ++i) {
        auto const& buffer = slot.sink->buffers[i];
        if (!buffer.iov_base) {
          slot.direct = false;
          break;
        }
        auto const n = std::min(remaining, buffer.iov_len - cursor);
        if (n > 0) slot.iovecs.push_back(iovec{static_cast<char*>(buffer.iov_base) + cursor, n});
        remaining -= n;
        cursor = 0;
      }
      if (slot.iovecs.empty()) slot.direct = false;
    }
    if (!slot.direct) {
      if (slot.buffer.empty()) slot.buffer.resize(_config.receive_buffer_bytes);
      slot.iovecs.assign(1, iovec{slot.buffer.data(), slot.buffer.size()});
    }
    slot.message            = {};
    slot.message.msg_iov    = slot.iovecs.data();
    slot.message.msg_iovlen = slot.iovecs.size();
    if (slot.remote.tls) {
      slot.message.msg_control    = slot.control.data();
      slot.message.msg_controllen = slot.control.size();
    }
    auto* entry = sqe(index + 1);
    io_uring_prep_recvmsg(entry, slot.fd.get(), &slot.message, 0);
    // Software kTLS decrypts in the receiving task. Force io-wq execution so
    // ready sockets cannot move that CPU work back onto the submitting reactor.
    if (slot.remote.tls) entry->flags |= IOSQE_ASYNC;
    slot.op      = operation::receive;
    slot.pending = true;
  }

  static void scatter(rest::buf_sink& sink, char const* source, std::size_t bytes)
  {
    if (bytes > sink.capacity - sink.written)
      throw std::runtime_error("HTTP response exceeds requested range");
    sink.total_received += bytes;
    while (bytes > 0 && sink.active < sink.buffers.size()) {
      auto const& buffer = sink.buffers[sink.active];
      auto const n       = std::min(bytes, buffer.iov_len - sink.cursor);
      if (source && buffer.iov_base)
        std::memcpy(static_cast<char*>(buffer.iov_base) + sink.cursor, source, n);
      if (source) source += n;
      sink.cursor += n;
      sink.written += n;
      bytes -= n;
      if (sink.cursor == buffer.iov_len) {
        ++sink.active;
        sink.cursor = 0;
      }
    }
    if (bytes != 0) throw std::runtime_error("HTTP body has no destination");
  }

  void advance(std::size_t index, int result)
  {
    auto& slot = _slots[index];
    if (result < 0) {
      if (slot.op == operation::connect) {
        connect_next(index);
        return;
      }
      if (result == -EAGAIN || result == -EINTR) {
        if (slot.op == operation::send)
          send(index);
        else if (slot.op == operation::receive)
          receive(index);
        else if (slot.op == operation::handshake)
          handshake(index);
      } else {
        finish(
          index, CURLE_RECV_ERROR, std::string("socket I/O failed: ") + std::strerror(-result));
      }
      return;
    }
    switch (slot.op) {
      case operation::connect:
        if (!slot.remote.tls) {
          connected(index);
          return;
        }
        slot.ssl.reset(SSL_new(_ssl_ctx.get()));
        if (!slot.ssl || SSL_set_fd(slot.ssl.get(), slot.fd.get()) != 1) {
          finish(index, CURLE_SSL_CONNECT_ERROR, "cannot initialize TLS connection");
          return;
        }
        {
          std::array<unsigned char, 16> ip{};
          bool const is_ip = inet_pton(AF_INET, slot.remote.host.c_str(), ip.data()) == 1 ||
                             inet_pton(AF_INET6, slot.remote.host.c_str(), ip.data()) == 1;
          if (!is_ip && SSL_set_tlsext_host_name(slot.ssl.get(), slot.remote.host.c_str()) != 1) {
            finish(index, CURLE_SSL_CONNECT_ERROR, "cannot set TLS SNI");
            return;
          }
          if (_reads.tls_verify) {
            auto* param = SSL_get0_param(slot.ssl.get());
            X509_VERIFY_PARAM_set_hostflags(param, X509_CHECK_FLAG_NO_PARTIAL_WILDCARDS);
            auto const ok = is_ip ? X509_VERIFY_PARAM_set1_ip_asc(param, slot.remote.host.c_str())
                                  : SSL_set1_host(slot.ssl.get(), slot.remote.host.c_str());
            if (ok != 1) {
              finish(
                index, CURLE_PEER_FAILED_VERIFICATION, "cannot configure hostname verification");
              return;
            }
          }
        }
        handshake(index);
        return;
      case operation::handshake: handshake(index); return;
      case operation::send:
        if (result == 0) {
          finish(index, CURLE_SEND_ERROR, "zero-byte send");
          return;
        }
        slot.sent += static_cast<std::size_t>(result);
        if (slot.sent < slot.request.size())
          send(index);
        else
          receive(index);
        return;
      case operation::receive: break;
    }

    if (slot.message.msg_flags & (MSG_CTRUNC | MSG_TRUNC)) {
      finish(index, CURLE_RECV_ERROR, "truncated kTLS ancillary data");
      return;
    }
    for (auto* control = CMSG_FIRSTHDR(&slot.message); control != nullptr;
         control       = CMSG_NXTHDR(&slot.message, control)) {
      if (control->cmsg_level == SOL_TLS && control->cmsg_type == TLS_GET_RECORD_TYPE &&
          (control->cmsg_len < CMSG_LEN(1) || *CMSG_DATA(control) != 23)) {
        finish(index, CURLE_RECV_ERROR, "TLS control record interrupted response");
        return;
      }
    }
    if (result == 0) {
      try {
        slot.response->eof();
      } catch (...) {
        finish(index, CURLE_PARTIAL_FILE, "peer closed before response completed");
        return;
      }
    } else if (slot.direct) {
      scatter(*slot.sink, nullptr, static_cast<std::size_t>(result));
      slot.response->consume_direct(static_cast<std::size_t>(result));
    } else {
      slot.response->consume(std::string_view{slot.buffer.data(), static_cast<std::size_t>(result)},
                             [&](std::string_view bytes) {
                               if (slot.response->status == 200 || slot.response->status == 206) {
                                 scatter(*slot.sink, bytes.data(), bytes.size());
                               }
                             });
    }
    if (slot.response->headers_complete() && !slot.headers_captured) {
      slot.headers_captured = true;
      auto const& headers   = slot.response->headers;
      auto const range      = headers.find("Content-Range");
      if (range != headers.end()) slot.headers->content_range = range->second;
      auto const retry = headers.find("Retry-After");
      if (retry != headers.end()) slot.headers->retry_after = retry->second;
      auto const status = slot.response->status;
      if (status != 200 && status != 206) {
        // Error XML does not belong in the caller's buffer. Closing the socket
        // lets the shared retry policy act without draining an unbounded body.
        slot.response->keep_alive = false;
        finish(index, CURLE_OK, {});
        return;
      }
    }
    if (slot.response->complete())
      finish(index, CURLE_OK, {});
    else
      receive(index);
  }

  void finish(std::size_t index, CURLcode error, std::string detail)
  {
    auto& slot = _slots[index];
    if (slot.pending || slot.cancel_pending)
      throw std::logic_error("uring_remote: completing pending I/O");
    auto const status = slot.response ? slot.response->status : 0;
    if (error != CURLE_OK || !slot.response || !slot.response->complete() ||
        !slot.response->keep_alive)
      close(slot);
    slot.idle_since = clock_type::now();
    slot.deadline.reset();
    slot.canceling = false;
    if (slot.sink)
      _completed.push_back(completion{index, error, status, std::move(detail)});
    else if (error != CURLE_OK)
      SIRIUS_LOG_DEBUG("uring_remote: warmup failed: {}", detail);
    slot.sink    = nullptr;
    slot.headers = nullptr;
    slot.response.reset();
    slot.request.clear();
  }

  rest::config _reads;
  config _config;
  std::unique_ptr<SSL_CTX, decltype(&SSL_CTX_free)> _ssl_ctx{nullptr, SSL_CTX_free};
  io_uring _ring{};
  bool _ring_live{false};
  bool _reported_ktls{false};
  std::vector<connection> _slots;
  std::unordered_map<std::string, dns_entry> _dns;
  std::vector<completion> _completed;
};

}  // namespace

std::unique_ptr<rest::remote_transport> make_transport(rest::config const& reads,
                                                       config const& network)
{
  return std::make_unique<uring_transport>(reads, network);
}

}  // namespace sirius::io::uring_remote
