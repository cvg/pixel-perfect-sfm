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
#include <iostream>
#include <pybind11/eigen.h>
#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <string>

// // If the library is built by linking to ceres then we need to
// // access the defined compiler options(USE_SUITESPARSE,threading model
// // ...)
// #ifdef CERES_IS_LINKED
// #include<ceres/internal/port.h>
// #endif

namespace py = pybind11;

#include <chrono>
#include <thread>

#include "callbacks.cc"
#include "cost_functions.cc"
#include "loss_functions.cc"
#include "parameterization.cc"
#include "problem.cc"
#include "solver.cc"
#include "types.cc"

void init_types(py::module& m);
void init_callbacks(py::module& m);
void init_solver(py::module& m);
void init_loss_functions(py::module& m);
void init_cost_functions(py::module& m);
void init_parameterization(py::module& m);
void init_problem(py::module& m);
void init_pyceres(py::module& m);

class MyProblem {
 public:
  ceres::Problem* GetProblem() { return problem_.get(); }

  void SetUp() {
    problem_.reset(new ceres::Problem(ceres::Problem::Options()));
  }

 protected:
  std::unique_ptr<ceres::Problem> problem_;
};

// extern py::dict custom_loss_functions;

void bind_pyceres(py::module& m) {
  init_types(m);
  init_callbacks(m);
  init_solver(m);
  init_loss_functions(m);
  init_cost_functions(m);
  init_parameterization(m);
  init_problem(m);
}

// PYBIND11_MODULE(pyceres, m) {
//     m.doc() = "Ceres-Solver plugin";
//     init_pyceres(m);
// }
