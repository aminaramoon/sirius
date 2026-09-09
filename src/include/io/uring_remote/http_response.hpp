/*
 * Copyright 2026, Sirius Contributors.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#pragma once

#include <httplib.h>

#include <charconv>
#include <limits>
#include <stdexcept>
#include <string_view>

namespace sirius::io::uring_remote {

/// Incremental response framing around cpp-httplib's header parser. Only the
/// bounded headers/chunk framing are buffered; body spans go straight to a sink.
/// Normal S3 Content-Length bodies can bypass this parser through recvmsg iovecs.
class http_response {
 public:
  explicit http_response(std::size_t header_limit) : _limit(header_limit) {}

  long status{0};
  httplib::Headers headers;
  bool keep_alive{false};

  bool complete() const noexcept { return _state == state::done; }
  bool headers_complete() const noexcept { return _state != state::headers; }
  std::size_t direct_remaining() const noexcept { return _state == state::fixed ? _remaining : 0; }

  void consume_direct(std::size_t n)
  {
    if (_state != state::fixed || n > _remaining) fail("invalid direct body length");
    _remaining -= n;
    if (_remaining == 0) _state = state::done;
  }

  void eof()
  {
    if (_state == state::until_close) _state = state::done;
    if (!complete()) fail("truncated HTTP response");
  }

  template <class Sink>
  void consume(std::string_view input, Sink&& sink)
  {
    while (!input.empty()) {
      if (_state == state::fixed || _state == state::chunk_body || _state == state::until_close) {
        auto const n =
          _state == state::until_close ? input.size() : std::min(_remaining, input.size());
        sink(input.substr(0, n));
        input.remove_prefix(n);
        if (_state != state::until_close) {
          _remaining -= n;
          if (_remaining == 0) _state = _state == state::fixed ? state::done : state::chunk_crlf;
        }
        continue;
      }
      if (_state == state::done) fail("bytes after HTTP response");

      // Consume a line without retaining any body bytes from the same recv.
      auto const lf = input.find('\n');
      auto const n  = lf == std::string_view::npos ? input.size() : lf + 1;
      if (n > _limit - std::min(_line.size(), _limit)) fail("HTTP line exceeds limit");
      _line.append(input.substr(0, n));
      input.remove_prefix(n);
      if (lf == std::string_view::npos) return;
      if (_line.size() < 2 || !_line.ends_with("\r\n")) fail("HTTP line lacks CRLF");
      std::string_view line{_line.data(), _line.size() - 2};

      switch (_state) {
        case state::headers:
          _header_bytes += _line.size();
          if (_header_bytes > _limit) fail("HTTP headers exceed limit");
          if (status == 0) {
            if (line.size() < 12 ||
                (line.substr(0, 9) != "HTTP/1.1 " && line.substr(0, 9) != "HTTP/1.0 ") ||
                (line.size() > 12 && line[12] != ' '))
              fail("invalid HTTP status line");
            status = static_cast<long>(number(line.substr(9, 3)));
            if (status < 100 || status > 599) fail("invalid HTTP status");
            keep_alive = line[7] == '1';
          } else if (line.empty()) {
            finish_headers();
          } else {
            _header_block += _line;
          }
          break;
        case state::chunk_size: {
          auto const semicolon = line.find(';');
          _remaining           = number(line.substr(0, semicolon), 16);
          _state               = _remaining == 0 ? state::trailers : state::chunk_body;
          break;
        }
        case state::chunk_crlf:
          if (!line.empty()) fail("missing CRLF after chunk");
          _state = state::chunk_size;
          break;
        case state::trailers:
          _header_bytes += _line.size();
          if (_header_bytes > _limit) fail("HTTP trailers exceed limit");
          if (line.empty()) {
            _state = state::done;
          } else if (!httplib::detail::parse_header(
                       line.data(),
                       line.data() + line.size(),
                       [](std::string const& key, std::string const&) {
                         if (httplib::detail::case_ignore::equal(key, "Content-Length") ||
                             httplib::detail::case_ignore::equal(key, "Transfer-Encoding")) {
                           fail("framing header in HTTP trailer");
                         }
                       })) {
            fail("invalid HTTP trailer");
          }
          break;
        default: fail("invalid HTTP parser state");
      }
      _line.clear();
    }
  }

 private:
  enum class state {
    headers,
    fixed,
    until_close,
    chunk_size,
    chunk_body,
    chunk_crlf,
    trailers,
    done
  };

  [[noreturn]] static void fail(char const* message) { throw std::runtime_error(message); }

  static std::size_t number(std::string_view value, int base = 10)
  {
    std::size_t result   = 0;
    auto const [end, ec] = std::from_chars(value.data(), value.data() + value.size(), result, base);
    if (value.empty() || ec != std::errc{} || end != value.data() + value.size()) {
      fail("invalid HTTP numeric field");
    }
    return result;
  }

  void finish_headers()
  {
    httplib::detail::BufferStream stream;
    _header_block += "\r\n";
    stream.write(_header_block.data(), _header_block.size());
    if (!httplib::detail::read_headers(stream, headers)) fail("invalid HTTP headers");
    _header_block.clear();
    if (status < 200) {
      if (status == 101 || ++_interim > 8)
        fail("unsupported HTTP upgrade or interim response count");
      status = 0;
      headers.clear();
      return;
    }
    if (httplib::detail::has_header_token(headers, "Connection", "close")) keep_alive = false;
    auto const encoding = headers.find("Content-Encoding");
    if (encoding != headers.end() &&
        !httplib::detail::case_ignore::equal(encoding->second, "identity")) {
      fail("encoded HTTP body cannot satisfy an S3 byte range");
    }
    auto const length   = headers.find("Content-Length");
    auto const transfer = headers.find("Transfer-Encoding");
    if (length != headers.end() && transfer != headers.end()) fail("ambiguous HTTP framing");
    if (status == 204 || status == 304) {
      _state = state::done;
    } else if (transfer != headers.end()) {
      if (headers.count("Transfer-Encoding") != 1 ||
          !httplib::detail::case_ignore::equal(transfer->second, "chunked")) {
        fail("unsupported HTTP transfer encoding");
      }
      _state = state::chunk_size;
    } else if (length != headers.end()) {
      _remaining = number(length->second);
      _state     = _remaining == 0 ? state::done : state::fixed;
    } else {
      keep_alive = false;
      _state     = state::until_close;
    }
  }

  std::size_t _limit;
  state _state{state::headers};
  std::size_t _remaining{0};
  std::size_t _header_bytes{0};
  std::size_t _interim{0};
  std::string _line;
  std::string _header_block;
};

}  // namespace sirius::io::uring_remote
