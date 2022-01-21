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

#include "_pixsfm/src/helpers.h"
#include "util/src/log_exceptions.h"

// Trampoline class so that we can create a LossFunction in Python.
class PyLossFunction : public ceres::LossFunction {
 public:
  /* Inherit the constructors */
  using ceres::LossFunction::LossFunction;

  void Evaluate(double sq_norm, double out[3]) const override {
    pybind11::gil_scoped_acquire gil;

    py::array_t<double> out_arr(3, out, no_copy);
    pybind11::function overload = pybind11::get_overload(
        static_cast<const ceres::LossFunction*>(this), "Evaluate");
    if (overload) {
      overload.operator()(sq_norm, out_arr);
    } else {
      THROW_EXCEPTION(std::runtime_error, "<Evaluate> not implemented.")
    }
  }

 private:
  mutable py::str no_copy;  // Dummy variable for pybind11 to avoid copy
};

using namespace ceres;

// py::dict custom_loss_functions;

ceres::LossFunction* CreateLossFunction(std::string loss_name,
                                        std::vector<double> scales) {
  ceres::LossFunction* loss_function = nullptr;
  if (loss_name == "trivial") {
    loss_function = new ceres::TrivialLoss();
  } else if (loss_name == "soft_l1") {
    loss_function = new ceres::SoftLOneLoss(scales.at(0));
  } else if (loss_name == "cauchy") {
    loss_function = new ceres::CauchyLoss(scales.at(0));
  } else if (loss_name == "tolerant") {
    loss_function = new ceres::TolerantLoss(scales.at(0), scales.at(1));
  } else if (loss_name == "huber") {
    loss_function = new ceres::HuberLoss(scales.at(0));
  } else if (loss_name == "arctan") {
    loss_function = new ceres::ArctanLoss(scales.at(0));
    // } else if (custom_loss_functions.contains(loss_name)) {
    //         loss_function =
    //         custom_loss_functions[loss_name.c_str()](py::cast(scales))
    //                             .cast<ceres::LossFunction*>();
  } else {
    std::string failure_message = "Unknown loss_name " + loss_name;
    throw py::index_error(failure_message.c_str());
  }
  return loss_function;
}

ceres::LossFunction* CreateScaledLossFunction(std::string loss_name,
                                              std::vector<double> scales,
                                              double magnitude) {
  return new ceres::ScaledLoss(CreateLossFunction(loss_name, scales), magnitude,
                               ceres::TAKE_OWNERSHIP);
}

std::shared_ptr<ceres::LossFunction> CreateLossFunctionFromDict(py::dict dict) {
  THROW_CHECK(dict.contains("name"));
  std::string loss_name = dict["name"].cast<std::string>();

  if (loss_name != std::string("trivial")) {
    THROW_CHECK(dict.contains("params"));
  }
  if (dict.contains("magnitude")) {
    return std::shared_ptr<ceres::LossFunction>(
        CreateScaledLossFunction(dict["name"].cast<std::string>(),
                                 dict["params"].cast<std::vector<double>>(),
                                 dict["magnitude"].cast<double>()));
  } else {
    return std::shared_ptr<ceres::LossFunction>(
        CreateLossFunction(dict["name"].cast<std::string>(),
                           dict["params"].cast<std::vector<double>>()));
  }
}

void init_loss_functions(py::module& m) {
  py::class_<LossFunction, PyLossFunction /*<--- trampoline*/,
             std::shared_ptr<LossFunction>>(m, "LossFunction")
      .def(py::init<>())
      .def(py::init(&CreateLossFunctionFromDict))
      .def("evaluate", [](ceres::LossFunction& self, float v) {
        Eigen::Vector3d rho;
        self.Evaluate(v, rho.data());
        return rho;
      });
  py::implicitly_convertible<py::dict, LossFunction>();

  py::class_<TrivialLoss, LossFunction, std::shared_ptr<TrivialLoss>>(
      m, "TrivialLoss")
      .def(py::init<>());

  py::class_<HuberLoss, LossFunction, std::shared_ptr<HuberLoss>>(m,
                                                                  "HuberLoss")
      .def(py::init<double>());

  py::class_<SoftLOneLoss, LossFunction, std::shared_ptr<SoftLOneLoss>>(
      m, "SoftLOneLoss")
      .def(py::init<double>());

  py::class_<CauchyLoss, LossFunction, std::shared_ptr<CauchyLoss>>(
      m, "CauchyLoss")
      .def(py::init<double>());

  // m.def("register_loss_function", [](std::string& name, py::object cls) {
  //     custom_loss_functions[name.c_str()] = cls;
  // });

  // m.attr("custom_loss_functions") = &custom_loss_functions;
}