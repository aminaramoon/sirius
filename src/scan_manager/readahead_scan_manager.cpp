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

#include "scan_manager/readahead_scan_manager.hpp"

#include "log/logging.hpp"
#include "op/scan/gpu_ingestible_types.hpp"
#include "op/scan/parquet_gpu_ingestible.hpp"
#include "op/scan/prefetched_ranges.hpp"
#include "op/scan/table_scan/scan_contract.hpp"
#include "op/sirius_physical_operator.hpp"
#include "planner/query.hpp"
#include "planner/query_index.hpp"

#include <cucascade/cudf/datasource.hpp>
#include <cucascade/io/cache/fs_cache.hpp>
#include <cucascade/io/io_context.hpp>
#include <cucascade/memory/memory_reservation_manager.hpp>
#include <cucascade/memory/memory_space.hpp>

#include <algorithm>
#include <array>
#include <atomic>
#include <cassert>
#include <chrono>
#include <exception>
#include <format>
#include <memory>
#include <optional>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace sirius::scan_manager {

readahead_scan_manager::readahead_scan_manager(
  event::query_event_publisher& publisher,
  std::size_t budget,
  std::shared_ptr<op::scan::physical_check_counters> physical_counters,
  device_prefetch_options device,
  shared_budget_options shared)
  : event::query_event_subscriber(publisher,
                                  {event::event_type::task_deployed,
                                   event::event_type::task_queue_empty,
                                   event::event_type::memory_downgrade_for_task,
                                   event::event_type::wait_for_memory_for_task}),
    _device(device),
    _physical_counters(std::move(physical_counters)),
    _budget(budget),
    _gatekeeper(static_cast<int>(budget)),
    _shared_budget(std::move(shared.budget)),
    _budget_override(shared.limit_override),
    _zero_backend_limit(shared.zero_backend_limit)
{
  if (!_device.enabled()) { return; }
  using cucascade::memory::Tier;
  auto const spaces   = _device.memory->get_memory_spaces_for_tier(Tier::GPU);
  auto const n_spaces = std::max<std::size_t>(spaces.size(), 1);
  _max_parked_splits  = _device.max_parked_splits > 0
                          ? _device.max_parked_splits
                          : std::max<std::size_t>(2, 2 * _device.pipeline_width / n_spaces);
  for (auto const* listed : spaces) {
    if (listed == nullptr || listed->get_device_id() < 0) { continue; }
    auto const id = listed->get_device_id();
    auto* space   = _device.memory->get_memory_space(Tier::GPU, id);
    if (space == nullptr) { continue; }
    // A GPU whose stream cannot be created keeps its splits on the host tier.
    try {
      auto tier    = std::make_unique<device_tier>();
      tier->space  = space;
      tier->stream = op::scan::device_stream::create(id);
      tier->window = std::make_shared<device_window>();
      tier->issuer = std::make_unique<device_prefetch_issuer>(
        *space, tier->stream, tier->window, _device.fraction, _max_parked_splits);
      auto const index = static_cast<std::size_t>(id);
      if (_tiers.size() <= index) { _tiers.resize(index + 1); }
      _tiers[index] = std::move(tier);
    } catch (std::exception const& e) {
      SIRIUS_LOG_WARN("[readahead] device tier on GPU {} disabled: {}", id, e.what());
    }
  }
}

readahead_scan_manager::~readahead_scan_manager() { stop(); }

void readahead_scan_manager::prepare_for_query(const sirius::planner::query& query)
{
  _query_token     = sirius::value_of(query.query_id());
  _query_id        = query.query_id();
  auto scan_orders = query.get_scan_operators();

  size_t index = 0;
  for (auto* scan_op : scan_orders) {
    if (scan_op->type != op::SiriusPhysicalOperatorType::GPU_SCAN) { continue; }
    _operator_id_to_queue_index[scan_op->get_operator_id()] = index++;
    _ordered_work_queues.emplace_back(std::make_unique<prefetch_work_queue>());
    // Without this every queue reports operator 0, so every completion is
    // attributed to the wrong operator.
    _ordered_work_queues.back()->operator_id = scan_op->get_operator_id();
  }
}

void readahead_scan_manager::on_task_deployed(event::event_id_t,
                                              event::timestamp_t,
                                              query_id_t query_id,
                                              std::size_t,
                                              op::SiriusPhysicalOperatorType operator_type,
                                              int) noexcept
{
  if (query_id != _query_id) return;
  // A scan deployment means a pipeline thread is about to read for itself, and
  // prefetching alongside it only reorders the device queue.  Anything else is a
  // thread that went to compute instead, which is capacity the executor is not
  // about to use.
  if (operator_type == op::SiriusPhysicalOperatorType::GPU_SCAN) { return; }
  arm_prefetching();
}

void readahead_scan_manager::on_memory_downgrade_for_task(event::event_id_t,
                                                          event::timestamp_t,
                                                          query_id_t query_id,
                                                          std::size_t,
                                                          int,
                                                          std::size_t) noexcept
{
  if (query_id != _query_id) return;
  // The executor is spilling to make room, so the GPU does no work for the
  // duration and the device's IO path is unambiguously free.
  arm_prefetching();
}

void readahead_scan_manager::on_wait_for_memory_for_task(event::event_id_t,
                                                         event::timestamp_t,
                                                         query_id_t query_id,
                                                         std::size_t,
                                                         int,
                                                         std::size_t) noexcept
{
  if (query_id != _query_id) return;
  // Parked waiting on memory somebody else holds: same idle GPU as a downgrade,
  // arrived at differently.
  arm_prefetching();
}

void readahead_scan_manager::on_task_queue_empty(event::event_id_t, event::timestamp_t) noexcept
{
  // Nothing is waiting to be dispatched, so whatever the executor is doing it is
  // not about to read.  The strongest idle signal there is.
  arm_prefetching();
}

void readahead_scan_manager::start(prefetch_strategy strategy)
{
  // A backend that publishes a zero budget has opted out; running a worker that
  // can never issue anything would just be a thread parked on a condvar.
  if (_budget == 0) { return; }
  // Start-once.  A second call while the worker runs would move a fresh jthread
  // over the live one, and that move joins a worker whose stop token nothing
  // ever requests -- the worker watches @c _stop_source, not the jthread's own
  // source.  After @ref stop the gate is shut for good and the event subscriber
  // is torn down, so a second worker could only spin without issuing anything.
  if (is_running() || _stop_source.stop_requested()) { return; }
  _strategy = strategy;
  if (_device.enabled()) {
    std::string gpus;
    for (std::size_t id = 0; id < _tiers.size(); ++id) {
      if (!_tiers[id]) { continue; }
      gpus += std::format("{}{}", gpus.empty() ? "" : ",", id);
    }
    SIRIUS_LOG_INFO(
      "[readahead] device tier: fraction={:.2f} gpus=[{}] max_parked={} release_host_staging={}",
      _device.fraction,
      gpus,
      _max_parked_splits,
      _device.release_host_staging);
  }
  _prefetch_worker =
    std::jthread([this](const std::stop_token& st) { worker_loop(st); }, _stop_source.get_token());
  // Only now start draining the event-publisher mailbox: the hooks below arm
  // prefetching, and there must be a worker for them to arm.
  event::query_event_subscriber::start();

  // Eager does not wait to be invited: it reads ahead as far as the budget
  // allows from the moment there is anything to read.  Opportunistic stays
  // parked until one of the executor-idle signals below arms it.
  if (_strategy == prefetch_strategy::eager) { arm_prefetching(); }
}

void readahead_scan_manager::stop() noexcept
{
  // Joining the worker from the worker would deadlock: see the class comment.
  assert(_prefetch_worker.get_id() != std::this_thread::get_id());
  // First, so no hook can be dispatched into this object -- or re-arm the
  // prefetching we are about to tear down -- while the rest of teardown runs.
  event::query_event_subscriber::stop();
  _stop_source.request_stop();
  // Cuts short the worker's wait for a ticket, which is otherwise the one place
  // teardown can sit for a full timeout -- and a ticket handed out now would
  // only buy a prefetch that is about to be abandoned.
  _gatekeeper.stop();
  if (_prefetch_worker.joinable()) { _prefetch_worker.join(); }
  // Give in-flight device prefetches a moment to return their tickets, so the summary
  // below describes a settled query (the stopped gatekeeper gives up every wait at once,
  // so it cannot drain for us).  Bounded: an IO that never completes must not hold up
  // teardown.
  // Skipped once the summary is out or when only the destructor is stopping us: completions can
  // no longer lock the manager then, so the count would never fall.
  if (!_summary_logged.load(std::memory_order_acquire) && !weak_from_this().expired()) {
    constexpr auto k_device_drain = std::chrono::milliseconds{200};
    auto const deadline           = std::chrono::steady_clock::now() + k_device_drain;
    while (_device_inflight.load(std::memory_order_acquire) > 0 &&
           std::chrono::steady_clock::now() < deadline) {
      std::this_thread::sleep_for(std::chrono::milliseconds{1});
    }
  }
  // One line per query, on the way down: the manager is per-query, so this is
  // the last moment its counters describe a whole query and nothing else.  Once:
  // the query's drain stops the manager and its destructor stops it again.
  if (!_summary_logged.exchange(true)) { SIRIUS_LOG_INFO("[readahead] {}", summary()); }
}

std::string readahead_scan_manager::summary() const
{
  using kind         = prefetch_outcome_kind;
  auto const load    = [](auto const& c) { return c.load(std::memory_order_relaxed); };
  auto const outcome = [this](kind k) { return _counters.outcome(k); };

  auto const prefetched = outcome(kind::prefetched);
  auto const waited     = outcome(kind::wait_for_prefetch);
  auto const mem        = outcome(kind::skipped_memory_pressure);
  auto const behind     = outcome(kind::skipped_fell_behind);
  auto const nothing    = outcome(kind::nothing_to_issue);

  std::uint64_t consumed  = 0;
  std::uint64_t wrong_gpu = 0;
  std::uint64_t wait_ns   = 0;
  std::uint64_t parked    = 0;
  for (auto const& tier : _tiers) {
    if (!tier) { continue; }
    consumed += load(tier->window->consumed);
    wrong_gpu += load(tier->window->wrong_device);
    wait_ns += load(tier->window->wait_ns);
    parked += load(tier->window->parked_splits);
  }

  return std::format(
    "issued={}[prefetched={} wait_for_prefetch={}] "
    "skipped={}[memory_pressure={} fell_behind={} nothing_to_issue={}] "
    "candidates={}[dropped_expired={} dropped_fell_behind={}] "
    "operators_drained={} pacing[memory_retries={} idle_polls={} gate_timeouts={}] "
    "executor_reads={}[borrowed={}] "
    "device[prefetched={} wait_for_prefetch={} wasted={} failed={} consumed={} wrong_gpu={} "
    "skipped[memory={} backoff={} unsupported={} too_large={} window={}] "
    "MiB={} wait_ms={} parked={}]",
    prefetched + waited,
    prefetched,
    waited,
    mem + behind + nothing,
    mem,
    behind,
    nothing,
    load(_counters.candidates_taken),
    load(_counters.dropped_expired),
    load(_counters.dropped_fell_behind),
    load(_counters.operators_drained),
    load(_counters.memory_retries),
    load(_counters.idle_polls),
    load(_counters.gate_timeouts),
    load(_counters.cold_read_tickets),
    load(_counters.borrowed),
    outcome(kind::device_prefetched),
    outcome(kind::device_wait_for_prefetch),
    outcome(kind::device_wasted),
    outcome(kind::device_failed),
    consumed,
    wrong_gpu,
    load(_counters.device_skipped_memory),
    load(_counters.device_skipped_backoff),
    load(_counters.device_skipped_unsupported),
    load(_counters.device_skipped_too_large),
    load(_counters.device_skipped_window),
    load(_counters.device_bytes) >> 20,
    wait_ns / 1'000'000,
    parked);
}

bool readahead_scan_manager::is_running() const noexcept
{
  // The worker itself, not the budget: the budget is settled at construction
  // now, so it says only that a worker COULD run, not that one does.
  return _prefetch_worker.joinable() && !_stop_source.stop_requested();
}

void readahead_scan_manager::register_scan_task(std::shared_ptr<op::scan::scan_info> const& task,
                                                std::size_t operator_id)
{
  if (!task) { return; }
  auto op_index        = _operator_id_to_queue_index.at(operator_id);
  auto& prefetch_queue = _ordered_work_queues.at(op_index);
  prefetch_queue->push(task);
  if (_physical_counters && _physical_counters->track_units) {
    try {
      if (auto const* parquet = dynamic_cast<op::scan::parquet_split_info const*>(task.get())) {
        std::vector<std::string_view> seen;
        for (auto const& slice : parquet->rg_slices) {
          if (std::ranges::find(seen, slice.file_path) != seen.end()) continue;
          seen.emplace_back(slice.file_path);
          _physical_counters->readahead_registration(_query_token, slice.file_path);
        }
      }
    } catch (...) {  // Test observation must not affect a published split.
    }
  }
}

void readahead_scan_manager::mark_operator_closed(std::size_t operator_id)
{
  auto op_index        = _operator_id_to_queue_index.at(operator_id);
  auto& prefetch_queue = _ordered_work_queues.at(op_index);
  prefetch_queue->close();
}

void readahead_scan_manager::update_scan_state(std::size_t,
                                               const op::scan::scan_info* task,
                                               cucascade::io::cache::scan_stage stage)
{
  // A zero budget means the backend opted out, so there is no readahead for this
  // read to compete with and no budget to charge it against.
  if (task == nullptr || _budget == 0) { return; }
  if (stage != cucascade::io::cache::scan_stage::reading &&
      stage != cucascade::io::cache::scan_stage::disposed) {
    return;
  }
  // A disposal frees memory on the split's GPU, which ends that GPU's device back-off.  Counted
  // for every disposal, with or without a ticket entry, and before the bookkeeping lock: it needs
  // only the atomic.
  if (stage == cucascade::io::cache::scan_stage::disposed && !_tiers.empty()) {
    if (auto const device = task->preferred_device()) {
      if (auto* tier = tier_for(*device)) {
        tier->disposals.fetch_add(1, std::memory_order_relaxed);
      }
    }
  }
  auto key = task->weak_from_this();
  std::lock_guard lock(_tickets_mutex);
  auto it = _tickets.find(key);

  if (stage == cucascade::io::cache::scan_stage::reading) {
    // Stage publication precedes this call. Do not recreate an entry if a
    // disposal overtook this report while it was waiting for the mutex.
    if (task->get_scan_stage() == cucascade::io::cache::scan_stage::disposed) return;
    if (it != _tickets.end()) {
      it->second->demand_started = true;
      return;
    }
    auto ticket            = std::make_shared<scan_ticket>();
    ticket->owner          = scan_ticket::phase::demand;
    ticket->demand_started = true;
    // Demand acquisition never waits for budget capacity. Serialize it with
    // ownership publication so prefetch cannot install a second charge.
    if (_shared_budget) ticket->shared = acquire_shared_ticket(*task, true);
    _tickets.emplace(key, std::move(ticket));
    _counters.cold_read_tickets.fetch_add(1, std::memory_order_relaxed);
    if (_gatekeeper.acquire_or_borrow()) {
      _counters.borrowed.fetch_add(1, std::memory_order_relaxed);
    }
    return;
  }

  if (stage == cucascade::io::cache::scan_stage::disposed && it != _tickets.end()) {
    // An in-flight prefetch still owns both credits until its completion, even
    // if its consumer is disposed or the manager is destroyed in the meantime.
    if (it->second->owner == scan_ticket::phase::demand) {
      it->second->shared.reset();
      _gatekeeper.release();
    }
    _tickets.erase(it);
  }
}

prefetch_outcome_kind readahead_scan_manager::classify_prefetch(
  bool allocation_failed,
  bool split_alive,
  bool issued_io,
  cucascade::io::cache::scan_stage stage) noexcept
{
  // A hard failure outranks anything the consumer was doing: with no buffers
  // there was never an attempt to be early or late for.
  if (allocation_failed) { return prefetch_outcome_kind::skipped_memory_pressure; }
  // The split is gone, so whatever it was going to be read for already happened.
  if (!split_alive) { return prefetch_outcome_kind::skipped_fell_behind; }
  if (!issued_io) {
    return stage >= cucascade::io::cache::scan_stage::preparing
             ? prefetch_outcome_kind::skipped_fell_behind
             : prefetch_outcome_kind::nothing_to_issue;
  }
  // IO landed.  `reading` is carved out of "preparing or higher" because it is
  // the more specific case: the consumer is on this split right now and is
  // waiting on the very prefetch that just settled.
  if (stage == cucascade::io::cache::scan_stage::reading) {
    return prefetch_outcome_kind::wait_for_prefetch;
  }
  if (stage >= cucascade::io::cache::scan_stage::preparing) {
    return prefetch_outcome_kind::skipped_fell_behind;
  }
  return prefetch_outcome_kind::prefetched;
}

readahead_scan_manager::prefetch_candidate readahead_scan_manager::get_next_prefetching_candidate()
{
  // Never blocks.  There is nothing to wait on here that the worker cannot wait
  // on better: it holds a ticket while it is in here, and an empty queue is a
  // reason to give that ticket back, not to sit on it.
  while (!_stop_source.stop_requested()) {
    auto const index = _cursor.load(std::memory_order_relaxed);
    // Every operator has drained.  The queues are in execution order, so there
    // is nothing behind the cursor to go back for.
    if (index >= _ordered_work_queues.size()) { return {}; }

    auto& prefetch_queue = _ordered_work_queues.at(index);
    auto next            = prefetch_queue->get_next_candidate(_counters);
    if (!next.has_value()) { return {}; }  // empty, but this operator is still open
    if (*next) { return {std::move(*next), prefetch_queue->operator_id}; }

    // Finished: drained AND closed, so move past it for good and try the next
    // operator straight away -- it may have splits queued already.
    _counters.operators_drained.fetch_add(1, std::memory_order_relaxed);
    _cursor.fetch_add(1, std::memory_order_relaxed);
  }
  return {};
}

void readahead_scan_manager::settle_prefetch(split_key const& split,
                                             std::shared_ptr<scan_ticket> const& ticket,
                                             bool covered)
{
  std::lock_guard lock(_tickets_mutex);
  auto it             = _tickets.find(split);
  auto task           = split.lock();
  bool const retained = it != _tickets.end() && it->second == ticket && task &&
                        task->get_scan_stage() != cucascade::io::cache::scan_stage::disposed;
  if (retained && !covered && ticket->demand_started) {
    // Failure or partial coverage leaves demand IO outstanding. Transfer the
    // existing charge; disposal will release it, without a gap or double count.
    ticket->owner = scan_ticket::phase::demand;
    return;
  }
  ticket->shared.reset();
  _gatekeeper.release();
  if (retained && covered) {
    ticket->owner = scan_ticket::phase::covered;
  } else if (it != _tickets.end() && it->second == ticket) {
    _tickets.erase(it);
  }
}

void readahead_scan_manager::on_prefetch_complete(split_key const& split,
                                                  std::shared_ptr<scan_ticket> const& ticket,
                                                  op::scan::scan_info::prefetch_outcome out)
{
  auto const task = split.lock();
  // Only complete, successful coverage eliminates future demand IO. Merely
  // issuing one datasource says nothing about the rest of this split.
  settle_prefetch(split, ticket, out.ok && out.covered && out.declined == 0);
  _counters.record(
    classify_prefetch(out.declined_memory_pressure > 0,
                      task != nullptr,
                      out.issued > 0,
                      task ? task->get_scan_stage() : cucascade::io::cache::scan_stage::disposed));
}

readahead_scan_manager::device_tier* readahead_scan_manager::tier_for(int device) const noexcept
{
  if (device < 0 || static_cast<std::size_t>(device) >= _tiers.size()) { return nullptr; }
  return _tiers[static_cast<std::size_t>(device)].get();
}

void readahead_scan_manager::enter_backoff(device_tier& tier) noexcept
{
  tier.in_backoff        = true;
  tier.backoff_disposals = tier.disposals.load(std::memory_order_relaxed);
  tier.backoff_until     = std::chrono::steady_clock::now() + _device.backoff;
}

bool readahead_scan_manager::in_backoff(device_tier& tier) noexcept
{
  if (!tier.in_backoff) { return false; }
  if (tier.disposals.load(std::memory_order_relaxed) != tier.backoff_disposals ||
      std::chrono::steady_clock::now() >= tier.backoff_until) {
    tier.in_backoff = false;
    return false;
  }
  return true;
}

bool readahead_scan_manager::try_issue_device(prefetch_candidate const& candidate,
                                              split_key const& split,
                                              std::shared_ptr<scan_ticket> const& ticket)
{
  auto const& task  = candidate.task;
  auto const device = task->preferred_device();
  auto* tier        = device ? tier_for(*device) : nullptr;
  if (tier == nullptr || !task->supports_device_prefetch()) {
    _counters.device_skipped_unsupported.fetch_add(1, std::memory_order_relaxed);
    return false;
  }
  if (in_backoff(*tier)) {
    _counters.device_skipped_backoff.fetch_add(1, std::memory_order_relaxed);
    return false;
  }
  _counters.device_attempts.fetch_add(1, std::memory_order_relaxed);

  // Before issue(): the completion may run, and lower it, before issue() returns.
  _device_inflight.fetch_add(1, std::memory_order_relaxed);
  auto outcome = device_prefetch_issue::error;
  try {
    outcome = tier->issuer->issue(
      task,
      // `ticket` by value, like the host attempt's completion: its shared-budget token must come
      // back even when the completion fires after the manager is gone.  `weak`, not shared: a
      // completion firing after the query tore the manager down must not resurrect it.
      [ticket, weak = weak_from_this(), split](device_prefetch_result const& result) noexcept {
        if (auto self = weak.lock()) { self->on_device_prefetch_complete(split, ticket, result); }
      });
  } catch (...) {
    // Building the callback may allocate; issue() itself reports failures as outcomes.
    _device_inflight.fetch_sub(1, std::memory_order_release);
    SIRIUS_LOG_WARN("[readahead] device prefetch attempt abandoned: callback setup failed");
    return false;
  }

  if (outcome != device_prefetch_issue::issued) {
    _device_inflight.fetch_sub(1, std::memory_order_release);
  }
  switch (outcome) {
    case device_prefetch_issue::issued: return true;
    case device_prefetch_issue::memory:
      _counters.device_skipped_memory.fetch_add(1, std::memory_order_relaxed);
      enter_backoff(*tier);
      return false;
    case device_prefetch_issue::too_large:
      _counters.device_skipped_too_large.fetch_add(1, std::memory_order_relaxed);
      return false;
    case device_prefetch_issue::window_full:
      _counters.device_skipped_window.fetch_add(1, std::memory_order_relaxed);
      return false;
    case device_prefetch_issue::unsupported:
      _counters.device_skipped_unsupported.fetch_add(1, std::memory_order_relaxed);
      return false;
    // The host tier's first check records an overtaken split itself.
    case device_prefetch_issue::fell_behind:
    case device_prefetch_issue::no_ranges:
    case device_prefetch_issue::error: return false;
  }
  return false;
}

void readahead_scan_manager::on_device_prefetch_complete(
  split_key const& split,
  std::shared_ptr<scan_ticket> const& ticket,
  device_prefetch_result const& result) noexcept
{
  using cucascade::io::cache::scan_stage;
  using kind      = prefetch_outcome_kind;
  auto const task = split.lock();
  // First, like on_prefetch_complete: both credits are free once the IO settles.  A copy that did
  // not land hands them to a demand read that already started (the consumer's own read is then
  // the one charged) or releases them; a landed copy releases them and leaves the zero-capacity
  // `covered` marker, so a later `reading` report charges nothing.  Replaces the old
  // prefetched -> attempted correction and the direct gatekeeper release.
  settle_prefetch(split, ticket, /*covered=*/result.ok);
  if (!result.ok) {
    _counters.record(kind::device_failed);
  } else {
    _counters.device_bytes.fetch_add(result.bytes, std::memory_order_relaxed);
    auto const stage = task ? task->get_scan_stage() : scan_stage::disposed;
    _counters.record(!task || stage == scan_stage::disposed ? kind::device_wasted
                     : stage == scan_stage::reading         ? kind::device_wait_for_prefetch
                                                            : kind::device_prefetched);
  }

  // The split's bytes are on the device, so its host staging will not be read again: dispose
  // the cache requests now (the handles' consumer stage only; the split's own stage belongs to
  // its consumer) and kick each evictor once, without waiting for it.  The cache behind
  // io_ctx()->cache() is alive here: it is shut down only with the scan manager, after every
  // query drained, and the split being alive means its query is still running.
  if (result.ok && _device.release_host_staging && task) {
    std::array<cucascade::io::cache::fs_cache*, 4> kicked{};
    std::size_t n_kicked = 0;
    auto const kick      = [&](cucascade::io::cache::fs_cache* cache) {
      if (std::find(kicked.begin(), kicked.begin() + n_kicked, cache) !=
          kicked.begin() + n_kicked) {
        return;
      }
      if (n_kicked < kicked.size()) { kicked[n_kicked++] = cache; }
      cache->evict(std::max<std::size_t>(result.bytes, 1));
    };
    for (auto const& ds : task->datasources()) {
      // Releasing staging early is an optimisation: a failure here must not escape.
      try {
        if (!ds || !ds->uses_fs_cache()) { continue; }
        ds->update(scan_stage::disposed);
        if (auto* cache = ds->io_ctx()->cache()) { kick(cache); }
      } catch (...) {
      }
    }
  }

  // After the settle: stop() waits on this for the ticket.
  _device_inflight.fetch_sub(1, std::memory_order_release);
}

void readahead_scan_manager::worker_loop(const std::stop_token& st)
{
  // Bounds how long a stop can go unnoticed if it races the wait; a stop that
  // lands while parked here wakes the gate directly.
  constexpr auto k_slot_wait = std::chrono::milliseconds{100};
  // Back-off when the order has nothing to prefetch right now.
  constexpr auto k_idle_wait = std::chrono::milliseconds{10};

  enum class issue_result { issued, finished, memory_pressure };
  auto try_issue =
    [&](prefetch_candidate const& candidate, bool delayed_retry, bool& slot_transferred) {
      if (candidate.task->has_fallen_behind()) {
        _counters.record(prefetch_outcome_kind::skipped_fell_behind);
        return issue_result::finished;
      }
      std::shared_ptr<scan_ticket> ticket;
      split_key split = candidate.task;
      try {
        ticket = std::make_shared<scan_ticket>();
        if (_shared_budget) {
          ticket->shared = acquire_shared_ticket(*candidate.task, false);
          if (!ticket->shared) {
            _counters.record(prefetch_outcome_kind::skipped_memory_pressure);
            return issue_result::finished;
          }
        }
        // The shared-budget wait above must not hold the bookkeeping mutex:
        // demand can win while we wait, and must never wait for speculative IO.
        std::lock_guard lock(_tickets_mutex);
        if (candidate.task->has_fallen_behind() || !_tickets.emplace(split, ticket).second) {
          _counters.record(prefetch_outcome_kind::skipped_fell_behind);
          return issue_result::finished;
        }
        slot_transferred = true;
      } catch (...) {
        _counters.record(prefetch_outcome_kind::skipped_memory_pressure);
        return issue_result::finished;
      }
      auto attempt = [&] {
        // Phase 1, the device tier: once per candidate, never once per host retry (a delayed retry
        // exists only because phase 2 met memory pressure; a phase-1 refusal is not retried).  The
        // ticket and its shared-budget token are already published above, so a refusal leaves them
        // with this attempt: the host phase below uses the same ticket, and a non-issued result
        // settles it in the caller.
        //
        // The device tier may run this manager although no backend of the query has an fs_cache.
        // The host loop then finds nothing to stage (prepare_for_prefetching returns {0, 0, 0})
        // and records nothing_to_issue without entering the memory-retry path: it must never
        // assume a cache exists.
        if (!delayed_retry && _device.enabled() && try_issue_device(candidate, split, ticket)) {
          return issue_result::issued;
        }
        bool evict_on_failure = delayed_retry;
        while (!st.stop_requested()) {
          if (candidate.task->has_fallen_behind()) {
            _counters.record(prefetch_outcome_kind::skipped_fell_behind);
            return issue_result::finished;
          }

          if (evict_on_failure) _counters.memory_retries.fetch_add(1, std::memory_order_relaxed);
          op::scan::scan_info::prepare_outcome prep;
          // This runs on a jthread, so an escaping exception is std::terminate for
          // the whole process -- far too much for a best-effort prefetcher.  Both
          // calls allocate (the fan-in completion here, the evictor's latch inside
          // prepare), so bad_alloc is reachable exactly when the readahead is
          // earning its keep.  A throw is treated as "not issued": prefetch can only
          // throw before its prefetch_completion exists, because everything from
          // there on is noexcept and the report is what hands the slot over, so the
          // accounting is settled below on every non-issued path.
          try {
            prep = candidate.task->prepare_for_prefetching(evict_on_failure);
            if (prep.ready()) {
              candidate.task->prefetch([ticket, weak = weak_from_this(), split](
                                         op::scan::scan_info::prefetch_outcome out) noexcept {
                // weak, not shared: a completion firing after the query tore the
                // manager down must not resurrect it.
                if (auto self = weak.lock()) { self->on_prefetch_complete(split, ticket, out); }
              });
              return issue_result::issued;
            }
          } catch (std::exception const& e) {
            SIRIUS_LOG_WARN("[readahead] prefetch attempt abandoned: {}", e.what());
            _counters.record(prefetch_outcome_kind::skipped_memory_pressure);
            return issue_result::finished;
          } catch (...) {
            SIRIUS_LOG_WARN("[readahead] prefetch attempt abandoned: unknown exception");
            _counters.record(prefetch_outcome_kind::skipped_memory_pressure);
            return issue_result::finished;
          }
          if (prep.fell_behind > 0 || candidate.task->has_fallen_behind()) {
            _counters.record(prefetch_outcome_kind::skipped_fell_behind);
            return issue_result::finished;
          }
          // Nothing was refused for want of memory, so there is nothing to wait for:
          // this split has no request to prepare at all.
          if (prep.failed == 0) {
            _counters.record(prefetch_outcome_kind::nothing_to_issue);
            return issue_result::finished;
          }
          // First retry with synchronous eviction. A further attempt must yield
          // capacity and back off, and is allowed only once by the outer loop.
          if (evict_on_failure) return issue_result::memory_pressure;
          evict_on_failure = true;
        }
        // Stopped mid-preparation: the pool never satisfied it.
        _counters.record(prefetch_outcome_kind::skipped_memory_pressure);
        return issue_result::finished;
      };
      auto result = attempt();
      if (result != issue_result::issued) settle_prefetch(split, ticket, false);
      return result;
    };

  prefetch_candidate retry;
  while (!st.stop_requested()) {
    // Every operator drained: nothing will ever be worth prefetching again.
    if (!retry && _cursor.load(std::memory_order_relaxed) >= _ordered_work_queues.size()) { break; }

    // One slot per in-flight prefetch.  Past this point the slot is held and has
    // to be released exactly once: by the completion when the IO was issued, or
    // by one of the give-up paths below.
    if (!_gatekeeper.acquire_for(k_slot_wait)) {
      _counters.gate_timeouts.fetch_add(1, std::memory_order_relaxed);
      continue;
    }
    if (st.stop_requested()) {
      _gatekeeper.release();
      break;
    }

    bool const delayed_retry = bool(retry);
    auto candidate = delayed_retry ? std::exchange(retry, {}) : get_next_prefetching_candidate();
    if (!candidate) {
      // Nothing to read ahead of yet.  Hand the slot straight back rather than
      // sit on it -- budget held by an idle worker is budget the rest of the
      // order cannot use.
      _counters.idle_polls.fetch_add(1, std::memory_order_relaxed);
      _gatekeeper.release();
      std::this_thread::sleep_for(k_idle_wait);
      continue;
    }

    if (!delayed_retry) _counters.candidates_taken.fetch_add(1, std::memory_order_relaxed);
    bool slot_transferred = false;
    auto const result     = try_issue(candidate, delayed_retry, slot_transferred);
    if (result == issue_result::issued) continue;
    if (!slot_transferred) _gatekeeper.release();
    if (result == issue_result::memory_pressure) {
      if (delayed_retry) {
        _counters.record(prefetch_outcome_kind::skipped_memory_pressure);
      } else {
        retry = std::move(candidate);
        // No IO capacity is held during backoff. Like idle polling, stop latency
        // is bounded by 10 ms; cancellation is checked before acquiring again.
        std::this_thread::sleep_for(k_idle_wait);
      }
    }
  }
  if (retry) _counters.record(prefetch_outcome_kind::skipped_memory_pressure);
}

shared_scan_budget::token readahead_scan_manager::acquire_shared_ticket(
  const op::scan::scan_info& task, bool demand)
{
  std::vector<shared_scan_budget::request> requests;
  for (auto const& source : task.datasources()) {
    auto backend = source->io_ctx();
    if (!backend || !backend->can_use_fs_cache()) continue;
    auto limit = _budget_override.value_or(backend->n_max_concurrent_scans());
    // A backend that publishes no budget (local uring by default) opts out of speculation, except
    // for the device tier: scan_manager_config::resolve_readahead runs the readahead on the
    // pipeline width there.  Config-only, so the backend's limit is the same for every query.
    if (!limit) limit = _zero_backend_limit;
    if (!limit) {
      if (!demand) return {};
      continue;
    }
    requests.push_back({reinterpret_cast<std::uintptr_t>(backend.get()), limit});
  }
  return _shared_budget->acquire(std::move(requests), _query_id, _stop_source.get_token(), demand);
}

void readahead_scan_manager::arm_prefetching()
{
  // Once: reload() adds the budget to the count, so a second arming would
  // double it.  Adding rather than assigning is what keeps the tickets the
  // executor borrowed before this point on the books -- those reads are still
  // in flight and will return them, so the readahead must not be handed a full
  // allowance on top.
  //
  // This is also the only arming there is -- the gatekeeper starts with no
  // tickets, so the worker's acquire simply times out until this runs.
  if (_prefetching_started.exchange(true)) { return; }
  _gatekeeper.reload();
}

}  // namespace sirius::scan_manager
