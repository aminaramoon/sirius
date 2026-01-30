/*
 * Copyright 2025, Sirius Contributors.
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

#include "exec/interruptible_mpmc.hpp"
#include "exec/thread_pool.hpp"

#include <concepts>
#include <functional>
#include <optional>

namespace sirius::exec {

template <typename T>
requires std::move_constructible<T>
struct i_publisher {
  virtual ~i_publisher() = default;

  virtual bool push(T item) noexcept = 0;
};


template <typename T>
requires std::default_initializable<T> && std::move_constructible<T>
struct i_subscriber {
  virtual ~i_subscriber() = default;

  virtual std::optional<T> pop() noexcept = 0;
};

template <typename T>
using subscriber = std::unique_ptr<i_subscriber<T>>;

template <typename T>
using publisher = std::unique_ptr<i_publisher<T>>;


template <typename T>
class channel {
 public:
  channel()                          = default;
  virtual ~channel()                 = default;
  channel(const channel&)            = delete;
  channel& operator=(const channel&) = delete;

  virtual publisher<T> get_publisher() = 0;

  virtual subscriber<T> get_subscriber() = 0;

  virtual void close() = 0;
};

template <typename T>
class shared_mpmc_channel : public channel<T> {
  std::shared_ptr<interruptible_mpmc<T>> _queue =
    std::make_shared<interruptible_mpmc<T>>();

  // Producer implementation for move-only types
  struct producer_impl : public i_publisher<T> {
   public:
    explicit producer_impl(std::shared_ptr<interruptible_mpmc<T>> queue) : _queue(std::move(queue))
    {
    }

    bool push(T&& item) noexcept override
    {
      try {
        return _queue->push(std::move(item));
      } catch (...) {
        return false;
      }
    }

   private:
    std::shared_ptr<interruptible_mpmc<T>> _queue;
  };

  struct consumer_impl : public i_subscriber<T> {
   public:
    explicit consumer_impl(std::shared_ptr<interruptible_mpmc<T>> queue) : _queue(std::move(queue))
    {
    }

    std::optional<T> pop() noexcept override
    {
      try {
        return _queue->pop();
      } catch (...) {
      }
      return std::nullopt;
    }

   private:
    std::shared_ptr<interruptible_mpmc<T>> _queue;
  };

 public:
  publisher<T> get_publisher() override { return std::make_unique<producer_impl>(_queue); }

  subscriber<T> get_subscriber() override { return std::make_unique<consumer_impl>(_queue); }

  void close() override
  {
    _queue->interrupt();
  }
};

}  // namespace sirius::exec
