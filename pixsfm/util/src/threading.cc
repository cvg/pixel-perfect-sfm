// Copyright (c) 2018, ETH Zurich and UNC Chapel Hill.
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
//     * Redistributions of source code must retain the above copyright
//       notice, this list of conditions and the following disclaimer.
//
//     * Redistributions in binary form must reproduce the above copyright
//       notice, this list of conditions and the following disclaimer in the
//       documentation and/or other materials provided with the distribution.
//
//     * Neither the name of ETH Zurich and UNC Chapel Hill nor the names of
//       its contributors may be used to endorse or promote products derived
//       from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//
// Author: Johannes L. Schoenberger (jsch-at-demuc-dot-de)

// Inspired from COLMAP, but allows for update check

#include "util/src/threading.h"

namespace pixsfm {

ThreadPool::ThreadPool(const int num_threads)
    : stopped_(false), num_active_workers_(0) {
  const int num_effective_threads = colmap::GetEffectiveNumThreads(num_threads);
  for (int index = 0; index < num_effective_threads; ++index) {
    std::function<void(void)> worker =
        std::bind(&ThreadPool::WorkerFunc, this, index);
    workers_.emplace_back(worker);
  }
}

ThreadPool::~ThreadPool() { Stop(); }

void ThreadPool::Stop() {
  {
    std::unique_lock<std::mutex> lock(mutex_);

    if (stopped_) {
      return;
    }

    stopped_ = true;

    std::queue<std::function<void()>> empty_tasks;
    std::swap(tasks_, empty_tasks);
  }

  task_condition_.notify_all();

  for (auto& worker : workers_) {
    worker.join();
  }

  finished_condition_.notify_all();
}

void ThreadPool::Wait(std::function<bool()> callback) {
  std::unique_lock<std::mutex> lock(mutex_);
  if (!tasks_.empty() || num_active_workers_ > 0) {
    finished_condition_.wait(lock, [this, callback]() {
      return (callback() || tasks_.empty() && num_active_workers_ == 0);
    });
    // Cleanup
    std::queue<std::function<void()>> empty;
    std::swap(tasks_, empty);
    finished_condition_.wait(lock, [this] { return num_active_workers_ == 0; });
  }
}

void ThreadPool::WorkerFunc(const int index) {
  {
    std::lock_guard<std::mutex> lock(mutex_);
    thread_id_to_index_.emplace(GetThreadId(), index);
  }

  while (true) {
    std::function<void()> task;
    {
      std::unique_lock<std::mutex> lock(mutex_);
      task_condition_.wait(lock,
                           [this] { return stopped_ || !tasks_.empty(); });
      if (stopped_ && tasks_.empty()) {
        return;
      }
      task = std::move(tasks_.front());
      tasks_.pop();
      num_active_workers_ += 1;
    }

    task();

    {
      std::unique_lock<std::mutex> lock(mutex_);
      num_active_workers_ -= 1;
    }

    finished_condition_.notify_all();
  }
}

std::thread::id ThreadPool::GetThreadId() const {
  return std::this_thread::get_id();
}

int ThreadPool::GetThreadIndex() {
  std::unique_lock<std::mutex> lock(mutex_);
  return thread_id_to_index_.at(GetThreadId());
}

}  // namespace pixsfm