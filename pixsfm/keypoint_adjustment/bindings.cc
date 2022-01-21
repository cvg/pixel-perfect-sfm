#include "_pixsfm/src/helpers.h"
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

#include "base/src/graph.h"
#include "keypoint_adjustment/src/featuremetric_keypoint_optimizer.h"
#include "keypoint_adjustment/src/keypoint_adjustment_options.h"
#include "keypoint_adjustment/src/keypoint_optimizer.h"
#include "keypoint_adjustment/src/topological_reference_keypoint_optimizer.h"
#include "util/src/log_exceptions.h"
#include "util/src/types.h"

namespace pixsfm {

template <class dtype, class T>
void BindParallelSolve(py::class_<T>& c) {
  c.def("run", &T::template Run<dtype>);
  c.def("run", &T::template RunParallel<dtype>,
        py::call_guard<py::gil_scoped_release>());
  c.def("run_subset", &T::template RunSubset<dtype>);
}

template <typename T>
void BindKeypointOptimizer(py::class_<T>& class_) {
  BindParallelSolve<half>(class_);
  BindParallelSolve<float>(class_);
  BindParallelSolve<double>(class_);
  class_.def("summary", &T::Summary);
  // class_.def_property_readonly("problem", &T::Problem);
  class_.def("solve_problem", &T::SolveProblem);
  // class_.def("reset", &T::Reset);
}

void bind_keypoint_adjustment(py::module& m) {
  py::class_<KeypointAdjustmentSetup, std::shared_ptr<KeypointAdjustmentSetup>>(
      m, "KeypointAdjustmentSetup")
      .def(py::init<>())
      .def("set_node_constant", &KeypointAdjustmentSetup::SetNodeConstant)
      .def("is_node_constant", &KeypointAdjustmentSetup::IsNodeConstant)
      .def("is_keypoint_constant", &KeypointAdjustmentSetup::IsKeypointConstant)
      .def("set_masked_nodes_constant",
           &KeypointAdjustmentSetup::SetMaskedNodesConstant)
      .def("set_keypoint_constant",
           &KeypointAdjustmentSetup::SetKeypointConstant)
      .def("set_image_constant", &KeypointAdjustmentSetup::SetImageConstant);

  auto ka_options =
      py::class_<KeypointOptimizerOptions>(m, "KeypointOptimizerOptions")
          .def(py::init<>())
          .def_readwrite("solver", &KeypointOptimizerOptions::solver_options)
          .def_readwrite("loss", &KeypointOptimizerOptions::loss,
                         py::keep_alive<1, 2>())
          .def_readwrite("print_summary",
                         &KeypointOptimizerOptions::print_summary)
          .def_readwrite("bound", &KeypointOptimizerOptions::bound);
  make_dataclass(ka_options);

  auto fmka_options =
      py::class_<FeatureMetricKeypointOptimizer::Options>(
          m, "FeatureMetricKeypointOptimizerOptions", ka_options)
          .def(py::init<>())
          .def_readwrite(
              "root_regularize_weight",
              &FeatureMetricKeypointOptimizer::Options::root_regularize_weight)
          .def_readwrite(
              "weight_by_sim",
              &FeatureMetricKeypointOptimizer::Options::weight_by_sim)
          .def_readwrite(
              "root_edges_only",
              &FeatureMetricKeypointOptimizer::Options::root_edges_only)
          .def_readwrite("num_threads",
                         &FeatureMetricKeypointOptimizer::Options::num_threads);
  make_dataclass(fmka_options);

  auto fm_solver = py::class_<FeatureMetricKeypointOptimizer>(
                       m, "FeatureMetricKeypointOptimizer")
                       .def(py::init<FeatureMetricKeypointOptimizer::Options&,
                                     std::shared_ptr<KeypointAdjustmentSetup>,
                                     InterpolationConfig&>());

  BindKeypointOptimizer(fm_solver);

  auto trka_options =
      py::class_<TopologicalReferenceKeypointOptimizer::Options>(
          m, "TopologicalReferenceKeypointOptimizerOptions", ka_options)
          .def(py::init<>())
          .def_readwrite(
              "num_threads",
              &TopologicalReferenceKeypointOptimizer::Options::num_threads);
  make_dataclass(trka_options);
  auto tr_solver =
      py::class_<TopologicalReferenceKeypointOptimizer>(
          m, "TopologicalReferenceKeypointOptimizer", fm_solver)
          .def(py::init<TopologicalReferenceKeypointOptimizer::Options&,
                        std::shared_ptr<KeypointAdjustmentSetup>,
                        InterpolationConfig&>());

  BindKeypointOptimizer(tr_solver);
}

}  // namespace pixsfm
