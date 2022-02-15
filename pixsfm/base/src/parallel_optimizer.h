#pragma once
#include <chrono>
#include <map>
#include <type_traits>
#include <utility>
#include <vector>

#include <pybind11/pybind11.h>
namespace py = pybind11;
#include <colmap/util/threading.h>
#include <colmap/util/timer.h>

#include "util/src/py_interrupt.h"
#include "util/src/simple_logger.h"
#include "util/src/threading.h"

namespace pixsfm {

/*
This is a wrapper for parallel computations. It assumes the following interface
of a child class:

class MyOptimizer : public ParallelOptimizer<MyOptimizer> {
public:
    // On construction, set the number of threads you want to use. -1 is all.
    MyOptimizer(int n_threads) : ParallelOptimizer<MyOptimizer>(n_threads) {}

    // Given a parent object of the same class, create a subclass with similar
    // options (usually just copy variables like options or config)
    static MyOptimizer Create(MyOptimizer* other);

    // The method which performs computations on the subset "nodes_in_problem".
    // It is assumed that all computations performed in this
    // subset are independent of all other subsets!
    // Supported interface:
    template<int Ns..., typename Params...>
    bool RunSubset(const std::unordered_set<size_t>& nodes_in_problem,
            Params&... parameters) {
        // Perform computations on parameters here
    }

    // optional:
    template<int Ns..., typename Params...>
    bool Run(const std::vector<int>&  problem_labels, Params&... parameters) {
        ParallelOptimizer<MyOptimizer>::RunParallel<Ns...>(
            problem_labels, parameters...);
        return true;
    }
};

// Problem labels: Elements with same problem label are optimized together on
// same thread.

// Example: problem_labels = {0,1,2,1,-1,2}
// Subsets: {{0},{1,3}, {2,5}} -> Element at index 4 is ignored (label=-1)

MyOptimizer optim(-1);
bool success = optim.Run<>(problem_labels, <your parameters>);

// Will internally call:
//    MyOptimizer suboptim(&optim);
//    suboptim.RunSubset<>({0}, <your parameters>);

// On another thread:
//    MyOptimizer suboptim(&optim);
//    suboptim.RunSubset<>({1,3}, <your parameters>);

*/

template <typename Optimizer, typename idx_t = size_t>
class ParallelOptimizer {
 private:
  std::unordered_set<idx_t> dummy;  // We need to declare this first
 public:
  ParallelOptimizer(int n_threads) : n_threads_(n_threads) {}

  template <int... Ns, typename... Param>
  auto RunParallel(std::vector<int> problem_labels, Param&... parameters) {
  std::map<size_t, std::unordered_set<idx_t>> problem_map;

  STDLOG(DEBUG) << "Creating problem sets..." << std::endl;
  size_t actual_nodes = 0;

  for (idx_t node_idx = 0; node_idx < problem_labels.size(); ++node_idx) {
    if (problem_labels[node_idx] == -1) {
      continue;
    }
    actual_nodes++;
    problem_map[static_cast<size_t>(problem_labels[node_idx])].insert(node_idx);
  }

  STDLOG(DEBUG) << "Found " << problem_map.size()
                << " independent problem sets..." << std::endl;

  int effective_threads = colmap::GetEffectiveNumThreads(n_threads_);
  STDLOG(DEBUG) << "Solve using " << effective_threads << " threads."
                << std::endl;
  bool parallel = effective_threads > 1;
  SyncLogProgressbar bar(actual_nodes, true, false);

  if (parallel) {
    LOGCFG.silence_normal = true;
    bar.update();
  }

  colmap::Timer timer;

  std::mutex mutex_;

  ThreadPool thread_pool(effective_threads);

  std::atomic<bool> child_process_threw_exception;
  child_process_threw_exception = false;

  PyInterrupt py_interrupt;

  std::exception_ptr child_exception;

  std::atomic<size_t> cpu_time;
  cpu_time = 0;

  std::unordered_map<
      size_t, decltype(static_cast<Optimizer*>(this)->template RunSubset<Ns...>(
                  dummy, parameters...))>
      results;

  timer.Start();
  for (auto& it : problem_map) {
    std::unordered_set<idx_t>& nodes_in_problem = it.second;

    if (parallel) {
      thread_pool.AddTask([&]() -> void {
        if (child_process_threw_exception) {
          return;
        }

        try {
          // Perform copy so that suboptimizer has similar configs
          auto t1_c = std::chrono::high_resolution_clock::now();
          Optimizer suboptimizer =
              Optimizer::Create(static_cast<Optimizer*>(this));
          // Solve independent subset using suboptimizer
          auto res = suboptimizer.template RunSubset<Ns...>(nodes_in_problem,
                                                            parameters...);
          {
            std::lock_guard<std::mutex> lock(mutex_);
            results[it.first] = res;
          }
          auto t2_c = std::chrono::high_resolution_clock::now();
          cpu_time += static_cast<size_t>(
              std::chrono::duration_cast<std::chrono::microseconds>(t2_c - t1_c)
                  .count());
          // Update how many nodes have been optimized
          bar.update(nodes_in_problem.size());
        } catch (...) {
          SYNC_LOG(ERROR) << "Child process threw exception in problem "
                          << it.first << "." << std::endl;
          if (!child_process_threw_exception) {
            child_process_threw_exception = true;
            child_exception = std::current_exception();
          }
        }
      });
    } else {
      auto t1_c = std::chrono::high_resolution_clock::now();

      // Perform copy so that suboptimizer has similar configs
      Optimizer suboptimizer = Optimizer::Create(static_cast<Optimizer*>(this));
      // Solve independent subset using suboptimizer
      results[it.first] = suboptimizer.template RunSubset<Ns...>(
          nodes_in_problem, parameters...);

      auto t2_c = std::chrono::high_resolution_clock::now();
      cpu_time += static_cast<size_t>(
          std::chrono::duration_cast<std::chrono::milliseconds>(t2_c - t1_c)
              .count());
      if (py_interrupt.Raised()) {
        break;
      }
    }
  }

  if (parallel) {
    // We frequently check for interruptions through pybind (ctrl+c) and
    // terminate the optimization if an interruption is detected.
    thread_pool.Wait([&py_interrupt]() { return py_interrupt.Raised(); });
    bar.update();  // Guarantee to see the 100%;
    LOGCFG.silence_normal = false;
  }

  timer.Pause();

  if (child_process_threw_exception) {
    SYNC_LOG(ERROR) << "Parallel solver detected exception in child process:"
                    << std::endl;
    std::rethrow_exception(child_exception);
  }

  if (py_interrupt.Raised()) {
    py::gil_scoped_acquire acquire;
    throw py::error_already_set();
  }

  parallel_solver_time_ = timer.ElapsedSeconds();

  STDLOG(DEBUG) << "Parallel Optimizer time:"
                << " " << parallel_solver_time_ << "s"
                << ", CPU Time: " << cpu_time / 1000000.0 << "s" << std::endl;

  return results;
}

 protected:
  int n_threads_;
  double parallel_solver_time_;
};

}  // namespace pixsfm