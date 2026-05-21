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

// Lock-free fixed-size slot allocator using atomic bitmasks.
//
// Design:
//   * 1 bit per slot: 1 = free, 0 = in use.
//   * Backed by std::array<std::atomic<uint64_t>, num_words>, one word per 64
//     slots, each word on its own cache line.
//   * Acquire = TZCNT + CAS on one 64-bit word — no linear scan over slots.
//   * Release = single fetch_or (distinct bits never race, no CAS loop needed).
//   * "Any free?" falls out of the same load the acquire path uses.
//   * Blocking acquire parks on a single shared release-generation counter
//     (`_release_seq`), NOT on the per-word bits.  Per-word parking has a
//     lost-wakeup hazard: a waiter blocked on word 0 is not woken by
//     `notify_one()` on word 1, so releases on word 1 can leave the waiter
//     stuck even though slots are free elsewhere.  The seq counter is bumped
//     on every release, so any release wakes any waiter.

#pragma once

#include <array>
#include <atomic>
#include <bit>
#include <cassert>
#include <cstddef>
#include <cstdint>

namespace sirius::io {

template <std::size_t N>
class slot_pool {
  static_assert(N >= 1, "slot_pool needs at least one slot");

  using word_t                               = std::uint64_t;
  static constexpr std::size_t bits_per_word = 64;
  static constexpr std::size_t num_words     = (N + bits_per_word - 1) / bits_per_word;

  // Mask of valid bits within word w.  All but possibly the last word are
  // fully populated; the tail word masks off bits beyond N.
  static constexpr word_t full_mask(std::size_t w) noexcept
  {
    const std::size_t n = (w + 1 == num_words) ? (N - w * bits_per_word) : bits_per_word;
    return (n == 64) ? ~word_t{0} : ((word_t{1} << n) - 1);
  }

  // Each word on its own cache line to avoid CAS ping-pong on concurrent
  // acquires from different words.
  struct alignas(64) word_storage {
    std::atomic<word_t> bits;
  };

 public:
  static constexpr std::size_t capacity = N;
  static constexpr int no_slot          = -1;

  slot_pool() noexcept
  {
    for (std::size_t w = 0; w < num_words; ++w) {
      _words[w].bits.store(full_mask(w), std::memory_order_relaxed);
      assert(_words[w].bits.is_lock_free());
    }
  }

  slot_pool(const slot_pool&)            = delete;
  slot_pool& operator=(const slot_pool&) = delete;

  // Returns a free slot index in [0, N), or no_slot if the pool is exhausted.
  // `hint` biases which word is probed first to spread CAS contention across
  // cache lines; correctness is independent of the hint.
  int try_acquire(unsigned hint = 0) noexcept
  {
    const std::size_t start = (num_words == 1) ? 0 : (hint % num_words);
    for (std::size_t i = 0; i < num_words; ++i) {
      const int idx = acquire_from_word((start + i) % num_words);
      if (idx != no_slot) return idx;
    }
    return no_slot;
  }

  // Blocks until a slot becomes available, then acquires it.
  //
  // Uses the seq-counter wait pattern: snapshot _release_seq before the
  // final try_acquire so that any release racing between our last failed
  // try_acquire and our wait() bumps the counter past our snapshot and
  // makes wait() return immediately (no lost wakeup).  Any release from
  // any word wakes any waiter — no cross-word lost-wakeup hazard.
  int acquire(unsigned hint = 0) noexcept
  {
    for (;;) {
      if (int idx = try_acquire(hint); idx != no_slot) return idx;
      const uint64_t seq = _release_seq.load(std::memory_order_acquire);
      if (int idx = try_acquire(hint); idx != no_slot) return idx;
      _release_seq.wait(seq, std::memory_order_relaxed);
    }
  }

  // Returns the slot to the pool.  The caller must have previously acquired
  // this slot via try_acquire() or acquire().
  void release(int idx) noexcept
  {
    assert(idx >= 0 && static_cast<std::size_t>(idx) < N);
    const std::size_t w        = static_cast<std::size_t>(idx) / bits_per_word;
    const std::size_t b        = static_cast<std::size_t>(idx) % bits_per_word;
    const word_t bit           = word_t{1} << b;
    [[maybe_unused]] auto prev = _words[w].bits.fetch_or(bit, std::memory_order_release);
    assert((prev & bit) == 0 && "double-release of slot");
    // Bump the shared release-generation counter and wake one waiter on
    // it.  notify_one is sufficient even with multiple waiters: every
    // waiter re-runs try_acquire on wake and the next release wakes the
    // next waiter.  Bumping the counter unconditionally (even when no
    // waiter is parked) is essentially free — notify_one on an empty
    // wait list is a no-op fast path in glibc/libc++.
    _release_seq.fetch_add(1, std::memory_order_release);
    _release_seq.notify_one();
  }

  // Approximate (relaxed) count of free slots — suitable for heuristics only.
  std::size_t approx_free() const noexcept
  {
    std::size_t n = 0;
    for (std::size_t w = 0; w < num_words; ++w)
      n += static_cast<std::size_t>(std::popcount(_words[w].bits.load(std::memory_order_relaxed)));
    return n;
  }

  bool any_free() const noexcept
  {
    for (std::size_t w = 0; w < num_words; ++w)
      if (_words[w].bits.load(std::memory_order_relaxed) != 0) return true;
    return false;
  }

 private:
  int acquire_from_word(std::size_t w) noexcept
  {
    word_t mask = _words[w].bits.load(std::memory_order_relaxed);
    while (mask != 0) {
      const int b      = std::countr_zero(mask);
      const word_t bit = word_t{1} << b;
      if (_words[w].bits.compare_exchange_weak(
            mask, mask & ~bit, std::memory_order_acquire, std::memory_order_relaxed))
        return static_cast<int>(w * bits_per_word + b);
    }
    return no_slot;
  }

  std::array<word_storage, num_words> _words{};

  // Shared release-generation counter for the seq-wait pattern in
  // acquire().  Kept on its own cache line so release()'s fetch_add
  // doesn't false-share with the per-word CAS traffic.
  alignas(64) std::atomic<uint64_t> _release_seq{0};
};

}  // namespace sirius::io
