/*
 * Copyright 2026, Sirius Contributors.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#define CPPHTTPLIB_OPENSSL_SUPPORT
#include "io/rest/mock_authorizer.hpp"
#include "io/sirius_datasource.hpp"
#include "io/uring_remote/http_response.hpp"
#include "io/uring_remote/uring_remote_ioctx.hpp"

#include <rmm/cuda_stream.hpp>
#include <rmm/device_buffer.hpp>

#include <catch.hpp>
#include <cucascade/memory/numa_region_pinned_host_allocator.hpp>

#include <atomic>
#include <cstdlib>
#include <thread>

using namespace std::chrono_literals;
using namespace sirius::io;
using sirius::io::uring_remote::http_response;

namespace {

class server_fixture {
 public:
  explicit server_fixture(std::size_t size        = (33UL << 20) + 17,
                          char const* certificate = nullptr,
                          char const* key         = nullptr)
    : payload(size, '\0')
  {
    for (std::size_t i = 0; i < size; ++i)
      payload[i] = static_cast<char>((i * 17 + 3) % 251);
    if (certificate)
      server = std::make_unique<httplib::SSLServer>(certificate, key);
    else
      server = std::make_unique<httplib::Server>();
    server->new_task_queue = [] { return new httplib::ThreadPool(96); };
    server->set_keep_alive_max_count(1000);
    server->set_read_timeout(2, 0);
    server->set_write_timeout(2, 0);
    server->Get("/object", [&](httplib::Request const& request, httplib::Response& response) {
      auto const attempt = ++requests;
      int const count    = ++active;
      auto old           = peak.load();
      while (count > old && !peak.compare_exchange_weak(old, count)) {}
      struct active_guard {
        std::atomic<int>& active;
        ~active_guard() { --active; }
      } guard{active};
      if (delay.count() > 0) std::this_thread::sleep_for(delay);
      if (attempt <= fail_first || fail_all) {
        response.status = failure_status;
        response.set_header("Retry-After", "0");
        response.set_content("retry", "text/plain");
        return;
      }
      // cpp-httplib implements the incoming Range over the content provider.
      for (auto const& [lo, hi] : request.ranges) {
        auto const bytes = static_cast<std::size_t>(hi - lo + 1);
        auto previous    = largest.load();
        while (bytes > previous && !largest.compare_exchange_weak(previous, bytes)) {}
      }
      response.set_content_provider(
        payload.size(),
        "application/octet-stream",
        [&, attempt](std::size_t offset, std::size_t length, httplib::DataSink& sink) {
          if (truncate_first && attempt == 1) {
            sink.write(payload.data() + offset, std::min<std::size_t>(length, 17));
            return false;
          }
          return sink.write(payload.data() + offset, length);
        });
    });
    port = server->bind_to_any_port("127.0.0.1");
    if (port <= 0) throw std::runtime_error("test server bind failed");
    worker = std::jthread([this] { server->listen_after_bind(); });
    server->wait_until_ready();
  }

  ~server_fixture() { server->stop(); }

  std::string url(bool tls = false, std::string const& host = "localhost") const
  {
    return std::string{tls ? "https://" : "http://"} + host + ':' + std::to_string(port) +
           "/object";
  }

  std::string payload;
  std::atomic<int> requests{0}, active{0}, peak{0};
  std::atomic<std::size_t> largest{0};
  std::chrono::milliseconds delay{0};
  int fail_first{0};
  bool fail_all{false};
  bool truncate_first{false};
  int failure_status{503};

 private:
  std::unique_ptr<httplib::Server> server;
  int port{0};
  std::jthread worker;
};

std::shared_ptr<rest::rest_ioctx> context(
  std::string url,
  bool uring                                                   = true,
  std::size_t connections                                      = 4,
  cucascade::memory::fixed_size_host_memory_resource* resource = nullptr,
  std::string ca                                               = {})
{
  rest::config reads;
  reads.request_timeout_s       = 2;
  reads.max_connections         = connections;
  reads.max_retry_attempts      = 3;
  reads.max_auth_retry_attempts = 2;
  reads.retry_backoff_base      = 1ms;
  reads.retry_jitter            = 0ms;
  reads.ca_bundle_path          = std::move(ca);
  auto authorizer =
    std::make_shared<rest::mock_authorizer>(rest::authorized_request{std::move(url), {}});
  auto shared = std::make_shared<rest::rest_reactor::reactor_context>(reads, authorizer, resource);
  if (!uring) {
    auto result = std::make_shared<rest::rest_ioctx>(1, shared);
    result->start();
    return result;
  }
  uring_remote::config network;
  network.allow_plaintext      = true;
  network.max_connections      = connections;
  network.receive_buffer_bytes = 16384;  // Exercise many partial completions.
  auto result = std::make_shared<uring_remote::uring_remote_ioctx>(1, shared, network);
  result->start();
  return result;
}

std::shared_ptr<sirius_datasource> open(rest::rest_ioctx& ctx, std::size_t size)
{
  return ctx.open_datasource("s3://bucket/object", static_cast<std::uint64_t>(size));
}

}  // namespace

TEST_CASE("uring HTTP framing handles every byte boundary", "[uring_remote][http]")
{
  auto const wire = std::string{
    "HTTP/1.1 100 Continue\r\n\r\nHTTP/1.1 206 Partial Content\r\n"
    "cOnTeNt-RaNgE: bytes 7-11/99\r\nTransfer-Encoding: chunked\r\n\r\n"
    "2;test=yes\r\nab\r\n3\r\ncde\r\n0\r\nETag: trailer\r\n\r\n"};
  for (std::size_t split = 0; split <= wire.size(); ++split) {
    http_response response{4096};
    std::string body;
    auto sink = [&](std::string_view bytes) { body += bytes; };
    response.consume(std::string_view{wire}.substr(0, split), sink);
    response.consume(std::string_view{wire}.substr(split), sink);
    REQUIRE(response.complete());
    REQUIRE(body == "abcde");
    REQUIRE(response.status == 206);
    REQUIRE(response.headers.find("Content-Range")->second == "bytes 7-11/99");
  }
  http_response response{4096};
  std::string body;
  for (auto const c : wire)
    response.consume(std::string_view{&c, 1}, [&](auto bytes) { body += bytes; });
  REQUIRE(response.complete());
  REQUIRE(body == "abcde");
}

TEST_CASE("uring HTTP framing rejects ambiguous truncated and oversized responses",
          "[uring_remote][http]")
{
  auto discard = [](auto) {};
  for (auto const* headers : {"Content-Length: 3\r\nContent-Length: 4\r\n",
                              "Content-Length: 3\r\nTransfer-Encoding: chunked\r\n",
                              "Content-Length: 18446744073709551616\r\n",
                              "Content-Length: -1\r\n",
                              "Transfer-Encoding: gzip\r\n",
                              "Content-Encoding: gzip\r\n"}) {
    http_response response{4096};
    CHECK_THROWS(
      response.consume(std::string{"HTTP/1.1 206 Partial\r\n"} + headers + "\r\n", discard));
  }
  http_response truncated{4096};
  truncated.consume("HTTP/1.1 200 OK\r\nContent-Length: 8\r\n\r\nshort", discard);
  CHECK_THROWS(truncated.eof());
  http_response oversized{32};
  CHECK_THROWS(
    oversized.consume("HTTP/1.1 200 OK\r\nHeader: this header exceeds the limit\r\n\r\n", discard));
  http_response direct{4096};
  direct.consume("HTTP/1.1 200 OK\r\nContent-Length: 3\r\n\r\n", discard);
  REQUIRE(direct.direct_remaining() == 3);
  direct.consume_direct(1);
  CHECK_FALSE(direct.complete());
  direct.consume_direct(2);
  CHECK(direct.complete());
  CHECK_THROWS(direct.consume("extra", discard));
}

TEST_CASE("uring remote preserves REST host reads EOF and physical chunk bounds", "[uring_remote]")
{
  bool const uring = GENERATE(false, true);
  server_fixture server;
  auto ctx    = context(server.url(), uring);
  auto object = open(*ctx, server.payload.size());
  std::vector<std::uint8_t> output(server.payload.size());
  std::vector<slice> slices{slice{0, output.size(), output.data()}};
  auto future = ctx->host_readv_async_io(object->get_io_object(), slices);
  REQUIRE(std::move(future).get(10s) == output.size());
  REQUIRE(std::memcmp(output.data(), server.payload.data(), output.size()) == 0);
  CHECK(server.requests > 1);
  CHECK(server.largest <= (16UL << 20));
  CHECK(object->host_read(server.payload.size() - 17, 100, output.data()) == 17);
  CHECK(std::memcmp(output.data(), server.payload.data() + server.payload.size() - 17, 17) == 0);
  CHECK(object->host_read(server.payload.size(), 1, output.data()) == 0);
}

TEST_CASE("uring remote services more than 64 simultaneous connections on one reactor",
          "[uring_remote]")
{
  server_fixture server{4096};
  server.delay = 300ms;
  auto ctx     = context(server.url(false, "127.0.0.1"), true, 80);
  auto object  = open(*ctx, server.payload.size());
  std::vector<std::array<std::uint8_t, 4096>> outputs(80);
  std::vector<sirius::exec::semi_future<std::size_t>> futures;
  for (auto& output : outputs) {
    std::vector<slice> slices{slice{0, output.size(), output.data()}};
    futures.push_back(ctx->host_readv_async_io(object->get_io_object(), slices));
  }
  for (auto& future : futures)
    REQUIRE(std::move(future).get(10s) == 4096);
  CHECK(server.peak > 64);
  CHECK(server.peak <= 80);
  for (auto const& output : outputs)
    REQUIRE(std::memcmp(output.data(), server.payload.data(), output.size()) == 0);
}

TEST_CASE("uring remote retries HTTP failures and drains cancellation", "[uring_remote]")
{
  server_fixture server{32768};
  auto ctx    = context(server.url(false, "127.0.0.1"));
  auto object = open(*ctx, server.payload.size());
  std::array<std::uint8_t, 4096> output{};
  SECTION("503 retry") { server.fail_first = 2; }
  SECTION("short read retry") { server.truncate_first = true; }
  SECTION("403 bounded retry")
  {
    server.fail_first     = 1;
    server.failure_status = 403;
  }
  SECTION("terminal error")
  {
    server.fail_all       = true;
    server.failure_status = 404;
  }
  SECTION("cancel stalled receive") { server.delay = 500ms; }
  std::vector<slice> slices{slice{7, output.size(), output.data()}};
  auto future = ctx->host_readv_async_io(object->get_io_object(), slices);
  if (server.delay.count() > 0) {
    auto const until = std::chrono::steady_clock::now() + 2s;
    while (server.requests == 0 && std::chrono::steady_clock::now() < until)
      std::this_thread::yield();
    ctx->shutdown();
    CHECK_THROWS(std::move(future).get(2s));
    output.fill(0xa5);
    std::this_thread::sleep_for(600ms);
    CHECK(std::all_of(output.begin(), output.end(), [](auto b) { return b == 0xa5; }));
  } else if (server.fail_all) {
    CHECK_THROWS(std::move(future).get(5s));
    CHECK(server.requests == 1);
  } else {
    REQUIRE(std::move(future).get(5s) == output.size());
    CHECK(server.requests == server.fail_first + (server.truncate_first ? 2 : 1));
    CHECK(std::memcmp(output.data(), server.payload.data() + 7, output.size()) == 0);
  }
}

TEST_CASE("uring remote preserves staged and caller-host device reads", "[uring_remote][gpu]")
{
  int count = 0;
  if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0) {
    WARN("No CUDA device");
    return;
  }
  bool const uring = GENERATE(false, true);
  cucascade::memory::numa_region_pinned_host_memory_resource upstream{0, true};
  cucascade::memory::fixed_size_host_memory_resource resource{
    0, upstream, 128UL << 20, 128UL << 20, 1UL << 20, 64, 1};
  server_fixture server{(17UL << 20) + 31};
  auto ctx    = context(server.url(false, "127.0.0.1"), uring, 2, &resource);
  auto object = open(*ctx, server.payload.size());
  rmm::cuda_stream stream;
  rmm::device_buffer device(server.payload.size(), stream);
  std::vector<std::uint8_t> output(server.payload.size());
  SECTION("staged device read")
  {
    auto future = ctx->device_read_async_io(
      object->get_io_object(), 0, output.size(), static_cast<std::uint8_t*>(device.data()), stream);
    REQUIRE(std::move(future).get(10s) == output.size());
  }
  SECTION("host to device read")
  {
    void* raw = nullptr;
    REQUIRE(cudaMallocHost(&raw, output.size()) == cudaSuccess);
    std::unique_ptr<void, decltype(&cudaFreeHost)> host{raw, cudaFreeHost};
    std::vector<prepared_io_slice> slices;
    slices.emplace_back(range{0, output.size()},
                        host_buffer{static_cast<std::uint8_t*>(raw)},
                        device_buffer{static_cast<std::uint8_t*>(device.data()), stream});
    auto future = ctx->host_device_readv_async_io(object->get_io_object(), std::move(slices));
    REQUIRE(std::move(future).get(10s) == output.size());
    std::memset(raw, 0, output.size());
  }
  REQUIRE(cudaMemcpyAsync(
            output.data(), device.data(), output.size(), cudaMemcpyDeviceToHost, stream.value()) ==
          cudaSuccess);
  stream.synchronize();
  CHECK(std::memcmp(output.data(), server.payload.data(), output.size()) == 0);
}

TEST_CASE("uring remote requires verified kernel TLS on HTTPS", "[uring_remote][ktls]")
{
  auto const* certificate = std::getenv("SIRIUS_TEST_KTLS_CERT");
  auto const* key         = std::getenv("SIRIUS_TEST_KTLS_KEY");
  if (!certificate || !key) {
    WARN("Run pixi run test-uring-remote to exercise kTLS HTTPS");
    return;
  }
  server_fixture server{(1UL << 20) + 3, certificate, key};
  auto ctx    = context(server.url(true), true, 4, nullptr, certificate);
  auto object = open(*ctx, server.payload.size());
  std::vector<std::uint8_t> output(server.payload.size());
  REQUIRE(object->host_read(0, output.size(), output.data()) == output.size());
  CHECK(std::memcmp(output.data(), server.payload.data(), output.size()) == 0);
  // Reuse the verified TLS socket for a partial GET.
  REQUIRE(object->host_read(11, 100, output.data()) == 100);
  CHECK(std::memcmp(output.data(), server.payload.data() + 11, 100) == 0);
  auto wrong_host = context(server.url(true, "127.0.0.1"), true, 1, nullptr, certificate);
  auto untrusted  = open(*wrong_host, server.payload.size());
  CHECK_THROWS(untrusted->host_read(0, 100, output.data()));

  // localhost may resolve to both ::1 and 127.0.0.1; only IPv4 is listening.
  // This also exercises per-connection address fallback under concurrency.
  server.delay      = 300ms;
  auto parallel_ctx = context(server.url(true), true, 80, nullptr, certificate);
  auto parallel_obj = open(*parallel_ctx, server.payload.size());
  std::vector<std::array<std::uint8_t, 4096>> outputs(80);
  std::vector<sirius::exec::semi_future<std::size_t>> futures;
  for (auto& destination : outputs) {
    std::vector<slice> slices{slice{7, destination.size(), destination.data()}};
    futures.push_back(parallel_ctx->host_readv_async_io(parallel_obj->get_io_object(), slices));
  }
  for (auto& future : futures)
    REQUIRE(std::move(future).get(10s) == 4096);
  CHECK(server.peak > 64);
  CHECK(server.peak <= 80);
  for (auto const& destination : outputs)
    REQUIRE(std::memcmp(destination.data(), server.payload.data() + 7, destination.size()) == 0);
}

TEST_CASE("uring remote serves grouped footer stash reads without a GET", "[uring_remote]")
{
  server_fixture server{4096};
  auto ctx    = context(server.url(false, "127.0.0.1"));
  auto stash  = rest::make_shared_byte_span(std::vector<std::uint8_t>(64, 0x3a));
  auto object = std::make_shared<rest::rest_io_object>(
    "s3://bucket/object", "bucket", "object", 4096, 4032, stash);
  std::array<std::uint8_t, 16> output{};
  std::vector<slice> slices{slice{4080, output.size(), output.data()}};
  auto future = ctx->host_readv_async_io(*object, slices);
  REQUIRE(std::move(future).get(5s) == output.size());
  CHECK(std::all_of(output.begin(), output.end(), [](auto b) { return b == 0x3a; }));
  CHECK(server.requests == 0);
}

TEST_CASE("uring remote preserves fragmented cache fill publication and EOF clipping",
          "[uring_remote][gpu]")
{
  int count = 0;
  if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0) {
    WARN("No CUDA device");
    return;
  }
  bool const uring                 = GENERATE(false, true);
  constexpr std::size_t block_size = 1UL << 20;
  cucascade::memory::numa_region_pinned_host_memory_resource upstream{0, true};
  cucascade::memory::fixed_size_host_memory_resource resource{
    0, upstream, 64UL << 20, 64UL << 20, block_size, 32, 1};
  server_fixture server{17 * block_size + 31};
  auto ctx    = context(server.url(false, "127.0.0.1"), uring, 2, &resource);
  auto object = open(*ctx, server.payload.size());
  std::vector<cache::cached_chunk> chunks(18);
  std::vector<std::vector<std::uint8_t>> buffers(18, std::vector<std::uint8_t>(block_size, 0xa5));
  std::vector<cache::cached_chunk*> pointers;
  for (std::size_t i = 0; i < chunks.size(); ++i) {
    chunks[i].offset = i * block_size;
    chunks[i].data   = buffers[i].data();
    pointers.push_back(&chunks[i]);
  }
  std::atomic<std::size_t> published{0};
  std::atomic<bool> failed{false};
  std::vector<prepared_io_slice> slices;
  slices.emplace_back(range{37, server.payload.size() - 37}, host_buffer{std::move(pointers)});
  slices.back().on_complete = std::make_shared<prepared_io_completion>(
    [&](std::span<cache::cached_chunk* const> completed, bool success) noexcept {
      if (!success) failed = true;
      published += completed.size();
    });
  auto future = ctx->host_device_readv_async_io(object->get_io_object(), std::move(slices));
  REQUIRE(std::move(future).get(10s) == server.payload.size() - 37);
  CHECK_FALSE(failed);
  CHECK(published == chunks.size());
  for (std::size_t i = 0; i < chunks.size(); ++i) {
    auto const bytes = std::min(block_size, server.payload.size() - i * block_size);
    REQUIRE(std::memcmp(buffers[i].data(), server.payload.data() + i * block_size, bytes) == 0);
  }
  CHECK(buffers.back()[31] == 0xa5);
  CHECK(server.largest <= (16UL << 20));
}
