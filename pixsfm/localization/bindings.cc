#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "_pixsfm/src/helpers.h"

#include "localization/src/nearest_references.h"
#include "localization/src/query_refinement_options.h"

#include "localization/src/single_query_bundle_optimizer.h"
#include "localization/src/single_query_keypoint_optimizer.h"

namespace pixsfm {

template <typename dtype, typename T>
void BindSingleQueryBundleOptimizer(py::class_<T>& sqba) {
  sqba.def("run", &T::template RunQuery<dtype, std::vector<Reference>>,
           py::arg("qvec").noconvert(), py::arg("tvec").noconvert(),
           py::arg("camera").noconvert(), py::arg("points3D"),
           py::arg("fmap").noconvert(), py::arg("references"),
           py::arg("patch_idxs") = py::none(), py::arg("inliers") = py::none());
  sqba.def(
      "run",
      &T::template RunQuery<dtype, std::vector<Eigen::Ref<DescriptorMatrixXd>>>,
      py::arg("qvec").noconvert(), py::arg("tvec").noconvert(),
      py::arg("camera").noconvert(), py::arg("points3D"),
      py::arg("fmap").noconvert(), py::arg("references"),
      py::arg("patch_idxs") = py::none(), py::arg("inliers") = py::none());
  sqba.def("run",
           &T::template RunQuery<dtype,
                                 std::vector<std::vector<DescriptorMatrixXd>>>,
           py::arg("qvec").noconvert(), py::arg("tvec").noconvert(),
           py::arg("camera").noconvert(), py::arg("points3D"),
           py::arg("fmap").noconvert(), py::arg("references"),
           py::arg("patch_idxs") = py::none(), py::arg("inliers") = py::none());
}

template <typename dtype, typename T>
void BindSingleQueryKeypointOptimizer(py::class_<T>& sqka) {
  sqka.def(
      "run",
      &T::template RunQuery<dtype, std::vector<Eigen::Ref<DescriptorMatrixXd>>>,
      py::arg("keypoints").noconvert(), py::arg("fmap").noconvert(),
      py::arg("references").noconvert(), py::arg("patch_idxs") = py::none(),
      py::arg("inliers") = py::none());
  sqka.def("run",
           &T::template RunQuery<dtype,
                                 std::vector<std::vector<DescriptorMatrixXd>>>,
           py::arg("keypoints").noconvert(), py::arg("fmap").noconvert(),
           py::arg("references").noconvert(),
           py::arg("patch_idxs") = py::none(), py::arg("inliers") = py::none());
  sqka.def("run", &T::template RunQuery<dtype, std::vector<Reference>>,
           py::arg("keypoints").noconvert(), py::arg("fmap").noconvert(),
           py::arg("references").noconvert(),
           py::arg("patch_idxs") = py::none(), py::arg("inliers") = py::none());
}

template <typename dtype>
void BindQueryRefinementTemplate(py::module& m, std::string type_suffix) {
  m.def("find_nearest_references", &FindNearestReferences<dtype>,
        py::arg("fmap").noconvert(), py::arg("keypoints").noconvert(),
        py::arg("references").noconvert(), py::arg("point3D_ids"),
        py::arg("interpolation_config"), py::arg("patch_idxs") = py::none());
}

void bind_localization(py::module& m) {
  auto qba_options =
      py::class_<QueryBundleOptimizerOptions>(m, "QueryBundleOptimizerOptions")
          .def(py::init<>())
          .def_readwrite("loss", &QueryBundleOptimizerOptions::loss,
                         py::keep_alive<1, 2>())
          .def_readwrite("solver", &QueryBundleOptimizerOptions::solver_options)
          .def_readwrite("refine_focal_length",
                         &QueryBundleOptimizerOptions::refine_focal_length)
          .def_readwrite("refine_principal_point",
                         &QueryBundleOptimizerOptions::refine_principal_point)
          .def_readwrite("refine_extra_params",
                         &QueryBundleOptimizerOptions::refine_extra_params)
          .def_readwrite("print_summary",
                         &QueryBundleOptimizerOptions::print_summary);
  make_dataclass(qba_options);

  auto qka_options =
      py::class_<QueryKeypointOptimizerOptions>(m,
                                                "QueryKeypointOptimizerOptions")
          .def(py::init<>())
          .def_readwrite("loss", &QueryKeypointOptimizerOptions::loss,
                         py::keep_alive<1, 2>())
          .def_readwrite("solver",
                         &QueryKeypointOptimizerOptions::solver_options)
          .def_readwrite("bound", &QueryKeypointOptimizerOptions::bound)
          .def_readwrite("print_summary",
                         &QueryKeypointOptimizerOptions::print_summary);
  make_dataclass(qka_options);

  auto sqba =
      py::class_<SingleQueryBundleOptimizer>(m, "QueryBundleOptimizer")
          .def(py::init<QueryBundleOptimizerOptions, InterpolationConfig>());
  BindSingleQueryBundleOptimizer<float16>(sqba);
  BindSingleQueryBundleOptimizer<float>(sqba);
  BindSingleQueryBundleOptimizer<double>(sqba);
  // sqba.def("summary", &SingleQueryBundleOptimizer::Summary);

  auto sqka =
      py::class_<SingleQueryKeypointOptimizer>(m, "QueryKeypointOptimizer")
          .def(py::init<QueryKeypointOptimizerOptions, InterpolationConfig>());
  BindSingleQueryKeypointOptimizer<float16>(sqka);
  BindSingleQueryKeypointOptimizer<float>(sqka);
  BindSingleQueryKeypointOptimizer<double>(sqka);
  // sqka.def("summary", &SingleQueryKeypointOptimizer::Summary);

  BindQueryRefinementTemplate<float16>(m, "_f16");
  BindQueryRefinementTemplate<float>(m, "_f32");
  BindQueryRefinementTemplate<double>(m, "_f64");
}

}  // namespace pixsfm
