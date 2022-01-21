#pragma once

#include "util/src/py_interrupt.h"
#include "util/src/simple_logger.h"

#include <ceres/ceres.h>

namespace pixsfm {

struct PyInterruptCallback : public ceres::IterationCallback {
 public:
  virtual ceres::CallbackReturnType operator()(
      const ceres::IterationSummary& summary) {
    if (PyInterrupt(-1).Raised()) {
      return ceres::SOLVER_TERMINATE_SUCCESSFULLY;
    } else {
      return ceres::SOLVER_CONTINUE;
    }
  }
};

struct ProgressBarIterationCallback : public ceres::IterationCallback {
 public:
  ProgressBarIterationCallback(size_t max_num_iterations)
      : bar_(max_num_iterations + 1, true, false) {}

  LogProgressbar& ProgressBar() { return bar_; }

  virtual ceres::CallbackReturnType operator()(
      const ceres::IterationSummary& summary) {
    bar_.update();
    return ceres::SOLVER_CONTINUE;
  }

 private:
  LogProgressbar bar_;
};

}  // namespace pixsfm
