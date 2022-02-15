#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "_pixsfm/src/helpers.h"

#include <ceres/ceres.h>

#include "residuals/src/feature_reference.h"
#include "residuals/src/featuremetric.h"
#include "residuals/src/geometric.h"
namespace pixsfm {

template <typename dtype>
void BindTemplate(py::module& m) {
  m.def("FeatureReferenceCostFunctor",
        &CreateFeatureReferenceCostFunctor<dtype>);
  m.def("FeatureReferenceConstantPoseCostFunctor",
        &CreateFeatureReferenceConstantPoseCostFunctor<dtype>);
  m.def("FeatureMetricCostFunctor", &CreateFeatureMetricCostFunctor<dtype>);
}
void bind_residuals(py::module& m) {
  BindTemplate<half>(m);
  BindTemplate<float>(m);
  BindTemplate<double>(m);

  m.def("GeometricCostFunctor", &CreateGeometricCostFunctor);
  m.def("GeometricConstantPoseCostFunctor",
        &CreateGeometricConstantPoseCostFunctor);
}

}  // namespace pixsfm
