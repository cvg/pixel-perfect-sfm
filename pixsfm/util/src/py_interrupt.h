#pragma once

#include <chrono>
#include <mutex>

#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace pixsfm {

struct PyInterrupt {
  using clock = std::chrono::steady_clock;
  using sec = std::chrono::duration<double>;
  PyInterrupt(double gap = 2.0);

  inline bool Raised();

 private:
  std::mutex mutex_;
  bool found = false;
  colmap::Timer timer_;
  clock::time_point start;
  double gap_;
};

PyInterrupt::PyInterrupt(double gap) : gap_(gap), start(clock::now()) {}

bool PyInterrupt::Raised() {
  const sec duration = clock::now() - start;
  if (!found && duration.count() > gap_) {
    std::lock_guard<std::mutex> lock(mutex_);
    py::gil_scoped_acquire acq;
    found = (PyErr_CheckSignals() != 0);
    start = clock::now();
  }
  return found;
}

}  // namespace pixsfm