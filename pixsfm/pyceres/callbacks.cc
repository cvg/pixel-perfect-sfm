// Ceres Solver Python Bindings
// Copyright Nikolaus Mitchell. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// * Redistributions of source code must retain the above copyright notice,
//   this list of conditions and the following disclaimer.
// * Redistributions in binary form must reproduce the above copyright notice,
//   this list of conditions and the following disclaimer in the documentation
//   and/or other materials provided with the distribution.
// * Neither the name of the copyright holder nor the names of its contributors
//   may be used to endorse or promote products derived from this software
//   without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//
// Author: nikolausmitchell@gmail.com (Nikolaus Mitchell)
// Edited by: philipp.lindenberger@math.ethz.ch (Philipp Lindenberger)
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

#include <ceres/ceres.h>

// // If the library is built by linking to ceres then we need to
// // access the defined compiler options(USE_SUITESPARSE,threading model
// // ...)
// #ifdef CERES_IS_LINKED
// #include<ceres/internal/port.h>
// #endif
//#include <pybind11/exception.h>
namespace py = pybind11;

#include <chrono>
#include <thread>

#include "_pixsfm/src/helpers.h"

// Trampoline class so we can create an EvaluationCallback in Python.
class PyEvaluationCallBack : public ceres::EvaluationCallback {
 public:
  /* Inherit the constructors */
  using ceres::EvaluationCallback::EvaluationCallback;

  void PrepareForEvaluation(bool evaluate_jacobians,
                            bool new_evaluation_point) override {
    PYBIND11_OVERLOAD_PURE(
        void,                                    /* Return type */
        ceres::EvaluationCallback,               /* Parent class */
        PrepareForEvaluation,                    // Name of function in C++ (fn)
        evaluate_jacobians, new_evaluation_point /* Argument(s) */
    );
  }
};

class PyIterationCallback : public ceres::IterationCallback {
 public:
  using ceres::IterationCallback::IterationCallback;

  ceres::CallbackReturnType operator()(
      const ceres::IterationSummary& summary) override {
    PYBIND11_OVERRIDE_PURE_NAME(
        ceres::CallbackReturnType,  // Return type (ret_type)
        ceres::IterationCallback,   // Parent class (cname)
        "__call__",                 // Name of method in Python (name)
        operator(),                 // Name of function in C++ (fn)
        summary);
  }
};

PYBIND11_MAKE_OPAQUE(std::vector<ceres::IterationCallback*>);

void init_callbacks(py::module& m) {
  py::class_<ceres::IterationCallback,
             PyIterationCallback /* <--- trampoline*/>(m, "IterationCallback")
      .def(py::init<>())
      .def("__call__", &ceres::IterationCallback::operator());

  py::class_<ceres::EvaluationCallback,
             PyEvaluationCallBack /* <--- trampoline*/>(m, "EvaluationCallback")
      .def(py::init<>());

  auto vec_it_cb = py::bind_vector<std::vector<ceres::IterationCallback*>>(
      m, "ListIterationCallback");

  vec_it_cb.def(
      py::init<>([](py::list list) {
        std::vector<ceres::IterationCallback*> callbacks;
        for (auto& handle : list) {
          callbacks.push_back(handle.cast<ceres::IterationCallback*>());
        }
        return callbacks;
      }),
      py::keep_alive<1, 2>());
  py::implicitly_convertible<py::list,
                             std::vector<ceres::IterationCallback*>>();
}