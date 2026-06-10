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

#include <cudf/io/text/byte_range_info.hpp>

#include <catch.hpp>
#include <io/rest/rest_reactor.hpp>
#include <io/rest/types.hpp>
#include <io/types.hpp>

#include <cstdint>
#include <optional>
#include <span>
#include <vector>

using cudf::io::text::byte_range_info;
using sirius::io::io_object_segment;
using sirius::io::rest::rest_chunked_rx_request;
using sirius::io::rest::rest_io_object;
using sirius::io::rest::rest_reactor;

namespace {

// Non-null buffer base for segments; the pure prep/coalesce logic never
// dereferences it.
uint8_t* fake_ptr(uintptr_t v) { return reinterpret_cast<uint8_t*>(v); }

std::vector<byte_range_info> coalesce(std::vector<byte_range_info> ranges,
                                      std::optional<size_t> alignment = std::nullopt)
{
  return rest_reactor::align_and_coalesce(
    std::span<const byte_range_info>(ranges.data(), ranges.size()), alignment);
}

}  // namespace

TEST_CASE("rest_reactor::supports only accepts s3 URLs", "[rest]")
{
  CHECK(rest_reactor::supports("s3://bucket/key"));
  CHECK(rest_reactor::supports("s3://bucket/path/to/obj.parquet"));
  CHECK_FALSE(rest_reactor::supports("file:///tmp/x"));
  CHECK_FALSE(rest_reactor::supports("https://host/obj"));
  CHECK_FALSE(rest_reactor::supports("/local/abs/path"));
  CHECK_FALSE(rest_reactor::supports("not a uri"));
}

TEST_CASE("align_and_coalesce coalesces without alignment by default", "[rest]")
{
  SECTION("empty input") { CHECK(coalesce({}).empty()); }
  SECTION("zero-size ranges dropped")
  {
    auto out = coalesce({byte_range_info{100, 0}});
    CHECK(out.empty());
  }
  SECTION("disjoint ranges stay separate and sorted")
  {
    auto out = coalesce({byte_range_info{200, 50}, byte_range_info{0, 50}});
    REQUIRE(out.size() == 2);
    CHECK(out[0].offset() == 0);
    CHECK(out[0].size() == 50);
    CHECK(out[1].offset() == 200);
    CHECK(out[1].size() == 50);
  }
  SECTION("overlapping ranges merge")
  {
    auto out = coalesce({byte_range_info{0, 100}, byte_range_info{50, 100}});
    REQUIRE(out.size() == 1);
    CHECK(out[0].offset() == 0);
    CHECK(out[0].size() == 150);
  }
  SECTION("adjacent ranges merge")
  {
    auto out = coalesce({byte_range_info{0, 100}, byte_range_info{100, 100}});
    REQUIRE(out.size() == 1);
    CHECK(out[0].offset() == 0);
    CHECK(out[0].size() == 200);
  }
}

TEST_CASE("align_and_coalesce honors a caller alignment as a lower bound", "[rest]")
{
  // align=4096: [100,200) -> [0,4096); [9000,9100) -> [8192,12288).  The two
  // rounded ranges leave a gap (4096..8192), so they stay separate.
  auto out = coalesce({byte_range_info{100, 100}, byte_range_info{9000, 100}}, 4096);
  REQUIRE(out.size() == 2);
  CHECK(out[0].offset() == 0);
  CHECK(out[0].size() == 4096);
  CHECK(out[1].offset() == 8192);
  CHECK(out[1].size() == 4096);

  // After rounding, [100,200) and [3000,3100) both land in [0,4096) and merge.
  auto merged = coalesce({byte_range_info{100, 100}, byte_range_info{3000, 100}}, 4096);
  REQUIRE(merged.size() == 1);
  CHECK(merged[0].offset() == 0);
  CHECK(merged[0].size() == 4096);
}

TEST_CASE("prep_host_rx_request builds a single chunk for the segment", "[rest]")
{
  rest_reactor::config cfg;  // authorizer unused by prep
  rest_io_object const file("s3://bkt/key", "bkt", "key", /*size=*/1 << 20);

  SECTION("non-empty segment")
  {
    auto req = rest_reactor::prep_host_rx_request(
      cfg, file, io_object_segment{4096, 8192, fake_ptr(0x1000)});
    REQUIRE(req->size() == 1);
    auto chunks = req->get_all_chunks();
    REQUIRE(chunks.size() == 1);
    CHECK(chunks[0]->object.bucket == "bkt");
    CHECK(chunks[0]->object.key == "key");
    CHECK(chunks[0]->chunk.offset == 4096);
    CHECK(chunks[0]->chunk.size == 8192);
    CHECK_FALSE(chunks[0]->is_device());
  }
  SECTION("zero-size segment yields no chunks")
  {
    auto req =
      rest_reactor::prep_host_rx_request(cfg, file, io_object_segment{0, 0, fake_ptr(0x1)});
    CHECK(req->size() == 0);
  }
}

TEST_CASE("prep_host_rxv_request builds one chunk per non-empty segment", "[rest]")
{
  rest_reactor::config cfg;
  rest_io_object const file("s3://bkt/key", "bkt", "key", /*size=*/10000);

  SECTION("three in-range segments")
  {
    std::vector<io_object_segment> segs{io_object_segment{0, 100, fake_ptr(0x1)},
                                        io_object_segment{500, 100, fake_ptr(0x2)},
                                        io_object_segment{9000, 100, fake_ptr(0x3)}};
    auto req = rest_reactor::prep_host_rxv_request(cfg, file, segs);
    REQUIRE(req->size() == 3);
    auto chunks = req->get_all_chunks();
    REQUIRE(chunks.size() == 3);
    for (auto const& c : chunks) {
      CHECK(c->object.bucket == "bkt");
      CHECK_FALSE(c->is_device());
    }
  }
  SECTION("segment past EOF is clamped away")
  {
    std::vector<io_object_segment> segs{io_object_segment{0, 100, fake_ptr(0x1)},
                                        io_object_segment{20000, 100, fake_ptr(0x2)}};
    auto req = rest_reactor::prep_host_rxv_request(cfg, file, segs);
    REQUIRE(req->size() == 1);  // the past-EOF segment contributes nothing
  }
  SECTION("segment straddling EOF is clamped to the file end")
  {
    std::vector<io_object_segment> segs{io_object_segment{9900, 1000, fake_ptr(0x1)}};
    auto req = rest_reactor::prep_host_rxv_request(cfg, file, segs);
    REQUIRE(req->size() == 1);
    auto chunks = req->get_all_chunks();
    REQUIRE(chunks.size() == 1);
    CHECK(chunks[0]->chunk.offset == 9900);
    CHECK(chunks[0]->chunk.size == 100);  // clamped from 1000 to 10000-9900
  }
  SECTION("empty segment list yields no chunks")
  {
    std::vector<io_object_segment> segs;
    auto req = rest_reactor::prep_host_rxv_request(cfg, file, segs);
    CHECK(req->size() == 0);
  }
}

TEST_CASE("prep_host_rx_request splits a large range into chunk_size pieces", "[rest]")
{
  rest_reactor::config cfg;
  cfg.chunk_size   = 4096;
  cfg.max_n_chunks = 16;
  rest_io_object const file("s3://bkt/key", "bkt", "key", /*size=*/1 << 20);

  // 8 * chunk_size laid out contiguously -> 8 single-buffer chunks.
  constexpr uintptr_t kBase = 0x10000;
  auto req =
    rest_reactor::prep_host_rx_request(cfg, file, io_object_segment{0, 8 * 4096, fake_ptr(kBase)});
  REQUIRE(req->size() == 8);

  auto chunks = req->get_all_chunks();
  REQUIRE(chunks.size() == 8);
  size_t total = 0;
  for (size_t i = 0; i < chunks.size(); ++i) {
    auto const& c = chunks[i]->chunk;
    CHECK(c.offset == i * 4096);
    CHECK(c.size == 4096);
    CHECK(c.n_chunks() == 1);  // a split piece is a plain single-buffer GET
    CHECK(reinterpret_cast<uintptr_t>(c.data()) == kBase + i * 4096);
    total += c.size;
  }
  CHECK(total == 8 * 4096);  // pieces cover the whole range, contiguously
}

TEST_CASE("prep_host_rxv_request fuses file-adjacent segments into a scatter GET", "[rest]")
{
  rest_reactor::config cfg;
  rest_io_object const file("s3://bkt/key", "bkt", "key", /*size=*/1 << 20);

  SECTION("three contiguous segments, separate buffers -> one multi-buffer chunk")
  {
    cfg.chunk_size   = 1 << 20;  // big enough to fit all three
    cfg.max_n_chunks = 16;
    std::vector<io_object_segment> segs{io_object_segment{0, 100, fake_ptr(0xA00)},
                                        io_object_segment{100, 100, fake_ptr(0xB00)},
                                        io_object_segment{200, 100, fake_ptr(0xC00)}};
    auto req = rest_reactor::prep_host_rxv_request(cfg, file, segs);
    REQUIRE(req->size() == 1);
    auto chunks = req->get_all_chunks();
    REQUIRE(chunks.size() == 1);
    CHECK(chunks[0]->chunk.offset == 0);
    CHECK(chunks[0]->chunk.size == 300);      // one contiguous range
    CHECK(chunks[0]->chunk.n_chunks() == 3);  // scattered across 3 buffers
    CHECK(chunks[0]->chunk.is_vectored());
  }
  SECTION("max_n_chunks caps the fusion")
  {
    cfg.chunk_size   = 1 << 20;
    cfg.max_n_chunks = 2;  // at most 2 buffers per request
    std::vector<io_object_segment> segs{io_object_segment{0, 100, fake_ptr(0xA00)},
                                        io_object_segment{100, 100, fake_ptr(0xB00)},
                                        io_object_segment{200, 100, fake_ptr(0xC00)}};
    auto req = rest_reactor::prep_host_rxv_request(cfg, file, segs);
    REQUIRE(req->size() == 2);  // [0,200) over 2 buffers, then [200,300)
  }
  SECTION("chunk_size caps the fused span")
  {
    cfg.chunk_size   = 250;  // two 100B segments fit, the third spills over
    cfg.max_n_chunks = 16;
    std::vector<io_object_segment> segs{io_object_segment{0, 100, fake_ptr(0xA00)},
                                        io_object_segment{100, 100, fake_ptr(0xB00)},
                                        io_object_segment{200, 100, fake_ptr(0xC00)}};
    auto req = rest_reactor::prep_host_rxv_request(cfg, file, segs);
    REQUIRE(req->size() == 2);  // [0,200) then [200,300)
  }
  SECTION("non-contiguous segments stay separate")
  {
    cfg.chunk_size   = 1 << 20;
    cfg.max_n_chunks = 16;
    std::vector<io_object_segment> segs{io_object_segment{0, 100, fake_ptr(0xA00)},
                                        io_object_segment{500, 100, fake_ptr(0xB00)}};
    auto req = rest_reactor::prep_host_rxv_request(cfg, file, segs);
    REQUIRE(req->size() == 2);
  }
  SECTION("a large segment in the vector is split")
  {
    cfg.chunk_size   = 4096;
    cfg.max_n_chunks = 16;
    std::vector<io_object_segment> segs{io_object_segment{0, 3 * 4096, fake_ptr(0xA00)}};
    auto req = rest_reactor::prep_host_rxv_request(cfg, file, segs);
    REQUIRE(req->size() == 3);
  }
}
