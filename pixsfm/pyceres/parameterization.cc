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
#include <ceres/ceres.h>
#include <pybind11/pybind11.h>

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
#include "util/src/log_exceptions.h"

class PyLocalParameterization : public ceres::LocalParameterization {
  /* Inherit the constructors */
  using ceres::LocalParameterization::LocalParameterization;
  bool Plus(const double* x, const double* delta, double* x_plus_delta) const {
    THROW_EXCEPTION(std::runtime_error, "<Plus> not implemented.");
    return true;
  }

  bool ComputeJacobian(const double* x, double* jacobian) const {
    THROW_EXCEPTION(std::runtime_error, "<Plus> not implemented.");
  }

  bool MultiplyByJacobian(const double* x, const int num_rows,
                          const double* global_matrix,
                          double* local_matrix) const {
    THROW_EXCEPTION(std::runtime_error,
                    "<MultiplyByJacobian> not implemented.");
    return true;
  }

  // Size of x.
  int GlobalSize() const override {
    PYBIND11_OVERLOAD_PURE_NAME(
        int,                          /* Return type */
        ceres::LocalParameterization, /* Parent class */
        "global_size",                /* Name of python function */
        GlobalSize /* Name of function in C++ (must match Python name) */
    );
  }

  // Size of delta.
  int LocalSize() const override {
    PYBIND11_OVERLOAD_PURE_NAME(
        int,                          /* Return type */
        ceres::LocalParameterization, /* Parent class */
        "local_size",                 /* Name of python function */
        LocalSize /* Name of function in C++ (must match Python name) */
    );
  }
};

using namespace ceres;

void init_parameterization(py::module& m) {
  py::class_<LocalParameterization,
             PyLocalParameterization /* <--- trampoline*/>(
      m, "LocalParameterization")
      .def(py::init<>())
      .def("global_size", &LocalParameterization::GlobalSize)
      .def("local_size", &LocalParameterization::LocalSize);

  py::class_<IdentityParameterization, LocalParameterization>(
      m, "IdentityParameterization")
      .def(py::init<int>());
  py::class_<QuaternionParameterization, LocalParameterization>(
      m, "QuaternionParameterization")
      .def(py::init<>());
  py::class_<HomogeneousVectorParameterization, LocalParameterization>(
      m, "HomogeneousVectorParameterization")
      .def(py::init<int>());
  py::class_<EigenQuaternionParameterization, LocalParameterization>(
      m, "EigenQuaternionParameterization")
      .def(py::init<>());
  py::class_<SubsetParameterization, LocalParameterization>(
      m, "SubsetParameterization")
      .def(py::init<int, const std::vector<int>&>());
}