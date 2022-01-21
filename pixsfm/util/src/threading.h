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
#pragma once

#include <atomic>
#include <climits>
#include <functional>
#include <future>
#include <list>
#include <queue>
#include <unordered_map>

#include <colmap/util/threading.h>

namespace pixsfm {

class ThreadPool {
 public:
  static const int kMaxNumThreads = -1;

  explicit ThreadPool(const int num_threads = kMaxNumThreads);

  ~ThreadPool();

  inline size_t NumThreads() const;

  // Add new task to the thread pool.
  template <class func_t, class... args_t>
  auto AddTask(func_t&& f, args_t&&... args)
      -> std::future<typename std::result_of<func_t(args_t...)>::type>;

  // Stop the execution of all workers.
  void Stop();

  // Wait until tasks are finished.
  void Wait(std::function<bool()>);

  // Get the unique identifier of the current thread.
  std::thread::id GetThreadId() const;

  // Get the index of the current thread. In a thread pool of size N,
  // the thread index defines the 0-based index of the thread in the pool.
  // In other words, there are the thread indices 0, ..., N-1.
  int GetThreadIndex();

 private:
  void WorkerFunc(const int index);

  std::vector<std::thread> workers_;
  std::queue<std::function<void()>> tasks_;

  std::mutex mutex_;
  std::condition_variable task_condition_;
  std::condition_variable finished_condition_;

  bool stopped_;
  int num_active_workers_;

  std::unordered_map<std::thread::id, int> thread_id_to_index_;
};

size_t ThreadPool::NumThreads() const { return workers_.size(); }

template <class func_t, class... args_t>
auto ThreadPool::AddTask(func_t&& f, args_t&&... args)
    -> std::future<typename std::result_of<func_t(args_t...)>::type> {
  typedef typename std::result_of<func_t(args_t...)>::type return_t;

  auto task = std::make_shared<std::packaged_task<return_t()>>(
      std::bind(std::forward<func_t>(f), std::forward<args_t>(args)...));

  std::future<return_t> result = task->get_future();

  {
    std::unique_lock<std::mutex> lock(mutex_);
    if (stopped_) {
      throw std::runtime_error("Cannot add task to stopped thread pool.");
    }
    tasks_.emplace([task]() { (*task)(); });
  }

  task_condition_.notify_one();

  return result;
}

}  // namespace pixsfm