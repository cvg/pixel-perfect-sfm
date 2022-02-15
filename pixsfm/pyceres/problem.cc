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
#include <memory>
#include <thread>

#include "_pixsfm/src/helpers.h"
#include "util/src/log_exceptions.h"

// Function to create Problem::Options with DO_NOT_TAKE_OWNERSHIP
// This is cause we want Python to manage our memory not Ceres
ceres::Problem::Options CreateProblemOptions() {
  ceres::Problem::Options o;
  o.local_parameterization_ownership = ceres::Ownership::TAKE_OWNERSHIP;
  o.loss_function_ownership = ceres::Ownership::DO_NOT_TAKE_OWNERSHIP;
  o.cost_function_ownership = ceres::Ownership::TAKE_OWNERSHIP;
  return o;
}

// Function to create a ceres Problem with the default options that Ceres does
// NOT take ownership. Needed since Python expects to own the memory.
std::unique_ptr<ceres::Problem> CreatePythonProblem() {
  ceres::Problem::Options options = CreateProblemOptions();
  return std::unique_ptr<ceres::Problem>(new ceres::Problem(options));
}

// Wrapper around ceres ResidualBlockID. In Ceres a ResidualBlockId is
// actually just a pointer to internal::ResidualBlock. However, since Ceres
// uses a forward declaration we don't actually have the type definition.
// (Ceres doesn't make it part of its public API). Since pybind11 needs a type
// we use this class instead which simply holds the pointer.
struct ResidualBlockIDWrapper {
 public:
  ResidualBlockIDWrapper(const ceres::ResidualBlockId& id) : id(id) {}
  const ceres::ResidualBlockId id;
};

// Same as FirstOrderFunctionWrapper
class CostFunctionWrapper : public ceres::CostFunction {
 public:
  explicit CostFunctionWrapper(ceres::CostFunction* real_cost_function)
      : cost_function_(real_cost_function) {
    this->set_num_residuals(cost_function_->num_residuals());
    *(this->mutable_parameter_block_sizes()) =
        cost_function_->parameter_block_sizes();
  }

  bool Evaluate(double const* const* parameters, double* residuals,
                double** jacobians) const override {
    return cost_function_->Evaluate(parameters, residuals, jacobians);
  }

 private:
  ceres::CostFunction* cost_function_;
};

class LocalParameterizationWrapper : public ceres::LocalParameterization {
 public:
  explicit LocalParameterizationWrapper(
      ceres::LocalParameterization* real_parameterization)
      : parameterization_(real_parameterization) {}
  virtual ~LocalParameterizationWrapper() {}

  // Generalization of the addition operation,
  //
  //   x_plus_delta = Plus(x, delta)
  //
  // with the condition that Plus(x, 0) = x.
  bool Plus(const double* x, const double* delta,
            double* x_plus_delta) const override {
    return parameterization_->Plus(x, delta, x_plus_delta);
  }

  // The jacobian of Plus(x, delta) w.r.t delta at delta = 0.
  //
  // jacobian is a row-major GlobalSize() x LocalSize() matrix.
  bool ComputeJacobian(const double* x, double* jacobian) const override {
    return parameterization_->ComputeJacobian(x, jacobian);
  }

  // local_matrix = global_matrix * jacobian
  //
  // global_matrix is a num_rows x GlobalSize  row major matrix.
  // local_matrix is a num_rows x LocalSize row major matrix.
  // jacobian(x) is the matrix returned by ComputeJacobian at x.
  //
  // This is only used by GradientProblem. For most normal uses, it is
  // okay to use the default implementation.
  bool MultiplyByJacobian(const double* x, const int num_rows,
                          const double* global_matrix,
                          double* local_matrix) const override {
    return parameterization_->MultiplyByJacobian(x, num_rows, global_matrix,
                                                 local_matrix);
  }

  // Size of x.
  int GlobalSize() const override { return parameterization_->GlobalSize(); }

  // Size of delta.
  int LocalSize() const override { return parameterization_->LocalSize(); }

 private:
  ceres::LocalParameterization* parameterization_;
};

void init_problem(py::module& m) {
  using options = ceres::Problem::Options;
  py::class_<ceres::Problem::Options>(m, "ProblemOptions")
      .def(py::init(&CreateProblemOptions))  // Ensures default is that
                                             // Python manages memory
      .def_readonly("cost_function_ownership",
                    &options::cost_function_ownership)
      .def_readonly("loss_function_ownership",
                    &options::loss_function_ownership)
      .def_readonly("local_parameterization_ownership",
                    &options::local_parameterization_ownership)
      .def_readwrite("enable_fast_removal", &options::enable_fast_removal)
      .def_readwrite("disable_all_safety_checks",
                     &options::disable_all_safety_checks);

  py::class_<ceres::Problem::EvaluateOptions>(m, "EvaluateOptions")
      .def(py::init<>())
      // Doesn't make sense to wrap this as you can't see the pointers in python
      //.def_readwrite("parameter_blocks",&ceres::Problem::EvaluateOptions)
      .def_readwrite("apply_loss_function",
                     &ceres::Problem::EvaluateOptions::apply_loss_function)
      .def_readwrite("num_threads",
                     &ceres::Problem::EvaluateOptions::num_threads);

  py::class_<ResidualBlockIDWrapper> residual_block_wrapper(m, "ResidualBlock");

  py::class_<ceres::Problem>(m, "Problem")
      .def(py::init(&CreatePythonProblem))
      .def(py::init<ceres::Problem::Options>())
      .def("num_parameter_bocks", &ceres::Problem::NumParameterBlocks)
      .def("num_parameters", &ceres::Problem::NumParameters)
      .def("num_residual_blocks", &ceres::Problem::NumResidualBlocks)
      .def("num_residuals", &ceres::Problem::NumResiduals)
      .def("parameter_block_size", &ceres::Problem::ParameterBlockSize)
      .def("set_parameter_block_constant",
           [](ceres::Problem& self, py::array_t<double>& np_arr) {
             py::buffer_info info = np_arr.request();
             THROW_CHECK(self.HasParameterBlock((double*)info.ptr));
             self.SetParameterBlockConstant((double*)info.ptr);
           })
      .def("set_parameter_block_variable",
           [](ceres::Problem& self, py::array_t<double>& np_arr) {
             py::buffer_info info = np_arr.request();
             THROW_CHECK(self.HasParameterBlock((double*)info.ptr));
             self.SetParameterBlockVariable((double*)info.ptr);
           })
      .def("is_parameter_block_constant",
           [](ceres::Problem& self, py::array_t<double>& np_arr) {
             py::buffer_info info = np_arr.request();
             THROW_CHECK(self.HasParameterBlock((double*)info.ptr));
             return self.IsParameterBlockConstant((double*)info.ptr);
           })
      .def("set_parameter_lower_bound",
           [](ceres::Problem& self, py::array_t<double>& np_arr, int index,
              double lower_bound) {
             py::buffer_info info = np_arr.request();
             THROW_CHECK(self.HasParameterBlock((double*)info.ptr));
             self.SetParameterLowerBound((double*)info.ptr, index, lower_bound);
           })
      .def("set_parameter_upper_bound",
           [](ceres::Problem& self, py::array_t<double>& np_arr, int index,
              double upper_bound) {
             py::buffer_info info = np_arr.request();
             THROW_CHECK(self.HasParameterBlock((double*)info.ptr));
             self.SetParameterUpperBound((double*)info.ptr, index, upper_bound);
           })
      // .def("get_parameter_lower_bound",
      //           [](ceres::Problem &self,
      //              py::array_t<double> &np_arr,
      //              int index) {
      //             py::buffer_info info = np_arr.request();
      //             return self.GetParameterLowerBound((double *) info.ptr,
      //                                                  index);
      //           })
      // .def("get_parameter_upper_bound",
      //           [](ceres::Problem &self,
      //              py::array_t<double> &np_arr,
      //              int index) {
      //             py::buffer_info info = np_arr.request();
      //             return self.GetParameterUpperBound((double *) info.ptr,
      //                                                  index);
      //           })
      .def("get_parameterization",
           [](ceres::Problem& self, py::array_t<double>& np_arr) {
             py::buffer_info info = np_arr.request();
             THROW_CHECK(self.HasParameterBlock((double*)info.ptr));
             return self.GetParameterization((double*)info.ptr);
           })
      .def("set_parameterization",
           [](ceres::Problem& self, py::array_t<double>& np_arr,
              ceres::LocalParameterization* local_parameterization) {
             py::buffer_info info = np_arr.request();
             THROW_CHECK(self.HasParameterBlock((double*)info.ptr));
             ceres::LocalParameterization* paramw = new LocalParameterizationWrapper(
                     local_parameterization);
             self.SetParameterization((double*)info.ptr, paramw);
           },
          py::keep_alive<1, 3>())  // LocalParameterization
      .def("parameter_block_size",
           [](ceres::Problem& self, py::array_t<double>& np_arr) {
             py::buffer_info info = np_arr.request();
             THROW_CHECK(self.HasParameterBlock((double*)info.ptr));
             return self.ParameterBlockSize((double*)info.ptr);
           })
      .def("has_parameter_block",
           [](ceres::Problem& self, py::array_t<double>& np_arr) {
             py::buffer_info info = np_arr.request();
             return self.HasParameterBlock((double*)info.ptr);
           })
      .def(
          "add_residual_block",
          [](ceres::Problem& self, ceres::CostFunction* cost,
             std::shared_ptr<ceres::LossFunction> loss, py::args& params) {
            THROW_CHECK_EQ(params.size(), cost->parameter_block_sizes().size());
            std::vector<double*> pointer_values(params.size());
            for (int i = 0; i < params.size(); i++) {
              py::buffer_info param_buf =
                  params[i]
                      .cast<py::array_t<double, py::array::c_style>>()
                      .request();
              pointer_values[i] = static_cast<double*>(param_buf.ptr);
              ssize_t num_dims = 1;
              std::vector<ssize_t> param_shape = param_buf.shape;
              for (int k = 0; k < param_shape.size(); k++) {
                num_dims *= param_shape[k];
              }
              THROW_CHECK_EQ(num_dims, cost->parameter_block_sizes()[i]);
            }
            ceres::CostFunction* costw = new CostFunctionWrapper(cost);
            return ResidualBlockIDWrapper(
                self.AddResidualBlock(costw, loss.get(), pointer_values));
          },
          py::keep_alive<1, 2>(),  // Cost Function
          py::keep_alive<1, 3>())  // Loss Function
      .def(
          "add_residual_block",
          [](ceres::Problem& self, ceres::CostFunction* cost,
             std::shared_ptr<ceres::LossFunction> loss,
             std::vector<py::array_t<double>>& paramv) {
            THROW_CHECK_EQ(paramv.size(), cost->parameter_block_sizes().size());
            std::vector<double*> pointer_values;
            for (int i = 0; i < paramv.size(); ++i) {
              py::buffer_info param_buf = paramv[i].request();
              pointer_values[i] = static_cast<double*>(param_buf.ptr);
              ssize_t num_dims = 1;
              std::vector<ssize_t> param_shape = param_buf.shape;
              for (int k = 0; k < param_shape.size(); k++) {
                num_dims *= param_shape[k];
              }
              THROW_CHECK_EQ(num_dims, cost->parameter_block_sizes()[i]);
            }
            ceres::CostFunction* costw = new CostFunctionWrapper(cost);
            return ResidualBlockIDWrapper(
                self.AddResidualBlock(costw, loss.get(), pointer_values));
          },
          py::keep_alive<1, 2>(),  // Cost Function
          py::keep_alive<1, 3>())  // Loss Function
      .def("add_parameter_block",
           [](ceres::Problem& self, py::array_t<double>& values, int size) {
             double* pointer = static_cast<double*>(values.request().ptr);
             self.AddParameterBlock(pointer, size);
           })
      .def(
          "add_parameter_block",
          [](ceres::Problem& self, py::array_t<double>& values, int size,
             ceres::LocalParameterization* local_parameterization) {
            double* pointer = static_cast<double*>(values.request().ptr);
            self.AddParameterBlock(pointer, size, local_parameterization);
          },
          py::keep_alive<1, 4>()  // LocalParameterization
          )
      .def("remove_parameter_block",
           [](ceres::Problem& self, py::array_t<double>& values) {
             double* pointer = static_cast<double*>(values.request().ptr);
             THROW_CHECK(self.HasParameterBlock(pointer));
             self.RemoveParameterBlock(pointer);
           })
      .def("remove_resdidual_block",
           [](ceres::Problem& self, ResidualBlockIDWrapper& residual_block_id) {
             self.RemoveResidualBlock(residual_block_id.id);
           });
}
