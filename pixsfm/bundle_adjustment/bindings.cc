#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "_pixsfm/src/helpers.h"

#include "bundle_adjustment/src/bundle_adjustment_options.h"
#include "bundle_adjustment/src/bundle_optimizer.h"
#include "bundle_adjustment/src/costmap_bundle_optimizer.h"
#include "bundle_adjustment/src/feature_reference_bundle_optimizer.h"
#include "bundle_adjustment/src/geometric_bundle_optimizer.h"
#include "bundle_adjustment/src/patch_warp_bundle_optimizer.h"

#include "bundle_adjustment/src/costmap_extractor.h"
#include "bundle_adjustment/src/reference_extractor.h"

namespace pixsfm {

template <typename dtype, typename T>
void BindCostMapExtractor(py::class_<T>& class_) {
  class_.def("run", &CostMapExtractor::Run<dtype, dtype>,
             py::arg("problem_labels"), py::arg("feature_set").noconvert(),
             py::arg("reconstruction").noconvert(),
             py::arg("ref_extractor").noconvert(),
             py::call_guard<py::gil_scoped_release>(), py::keep_alive<1, 3>());
}

template <typename dtype, typename T>
void BindReferenceExtractor(py::class_<T>& class_) {
  class_.def("run", &ReferenceExtractor::Run<dtype>, py::arg("problem_labels"),
             py::arg("reconstruction").noconvert(),
             py::arg("feature_set").noconvert(),
             py::call_guard<py::gil_scoped_release>(), py::keep_alive<1, 3>());
}

template <typename dtype, typename T>
void BindBundleOptimizer(py::class_<T>& class_) {
  class_.def("run", &T::template Run<dtype>);
  class_.def("set_up", &T::template SetUp<dtype>);
}

template <typename T>
void BindBundleOptimizer(py::class_<T>& class_) {
  BindBundleOptimizer<half>(class_);
  BindBundleOptimizer<float>(class_);
  BindBundleOptimizer<double>(class_);
  class_.def("summary", &T::Summary);
  class_.def_property_readonly("problem", &T::Problem);
  class_.def("solve_problem", &T::SolveProblem);
  class_.def("reset", &T::Reset);
}

void bind_bundle_adjustment(py::module& m) {
  auto costmapconf =
      py::class_<CostMapConfig>(m, "CostMapConfig")
          .def(py::init<>())
          .def_readwrite("as_gradientfield", &CostMapConfig::as_gradientfield)
          .def_readwrite("upsampling_factor", &CostMapConfig::upsampling_factor)
          .def_readwrite("compute_cross_derivative",
                         &CostMapConfig::compute_cross_derivative)
          .def("get_effective_channels", &CostMapConfig::GetEffectiveChannels)
          .def_readwrite("apply_sqrt", &CostMapConfig::apply_sqrt)
          .def_readwrite("num_threads", &CostMapConfig::num_threads)
          .def_readwrite("loss", &CostMapConfig::loss, py::keep_alive<1, 2>());
  make_dataclass(costmapconf);

  auto refconf =
      py::class_<ReferenceConfig>(m, "ReferenceConfig")
          .def(py::init<>())
          .def_readwrite("iters", &ReferenceConfig::iters)
          .def_readwrite("keep_observations",
                         &ReferenceConfig::keep_observations)
          .def_readwrite("num_threads", &ReferenceConfig::num_threads)
          .def_readwrite("compute_offsets3D",
                         &ReferenceConfig::compute_offsets3D)
          .def_readwrite("loss", &ReferenceConfig::loss,
                         py::keep_alive<1, 2>());
  make_dataclass(refconf);

  py::class_<BundleAdjustmentSetup>(m, "BundleAdjustmentSetup")
      .def(py::init<>())
      .def("add_image", &BundleAdjustmentSetup::AddImage)
      .def("add_images",
           [](BundleAdjustmentSetup& self,
              std::unordered_set<colmap::image_t> image_ids) {
             for (colmap::image_t image_id : image_ids) {
               self.AddImage(image_id);
             }
           })
      .def("set_constant_camera", &BundleAdjustmentSetup::SetConstantCamera)
      .def("set_variable_camera", &BundleAdjustmentSetup::SetVariableCamera)
      .def("is_constant_camera", &BundleAdjustmentSetup::IsConstantCamera)
      .def("set_constant_pose", &BundleAdjustmentSetup::SetConstantPose)
      .def("set_variable_pose", &BundleAdjustmentSetup::SetVariablePose)
      .def("has_constant_pose", &BundleAdjustmentSetup::HasConstantPose)
      .def("set_constant_tvec", &BundleAdjustmentSetup::SetConstantTvec)
      .def("remove_constant_tvec", &BundleAdjustmentSetup::RemoveConstantTvec)
      .def("has_constant_tvec", &BundleAdjustmentSetup::HasConstantTvec)
      .def("add_variable_point", &BundleAdjustmentSetup::AddVariablePoint)
      .def("add_constant_point", &BundleAdjustmentSetup::AddConstantPoint)
      .def("has_point", &BundleAdjustmentSetup::HasPoint)
      .def("has_variable_point", &BundleAdjustmentSetup::HasVariablePoint)
      .def("has_constant_point", &BundleAdjustmentSetup::HasConstantPoint)
      .def("remove_variable_point", &BundleAdjustmentSetup::RemoveVariablePoint)
      .def("remove_constant_point", &BundleAdjustmentSetup::RemoveConstantPoint)
      .def_property_readonly("image_ids", &BundleAdjustmentSetup::Images)
      .def_property_readonly("variable_point3D_ids",
                             &BundleAdjustmentSetup::VariablePoints)
      .def_property_readonly("constant_point3D_ids",
                             &BundleAdjustmentSetup::ConstantPoints);

  auto ba_options =
      py::class_<BundleOptimizerOptions>(m, "BundleOptimizerOptions")
          .def(py::init<>([]() {
            BundleOptimizerOptions options = BundleOptimizerOptions();
            options.register_pyinterrupt_callback = true;
            return options;
          }))
          .def_readwrite("refine_focal_length",
                         &BundleOptimizerOptions::refine_focal_length)
          .def_readwrite("refine_extra_params",
                         &BundleOptimizerOptions::refine_extra_params)
          .def_readwrite("refine_extrinsics",
                         &BundleOptimizerOptions::refine_extrinsics)
          .def_readwrite("refine_principal_point",
                         &BundleOptimizerOptions::refine_principal_point)
          .def_readwrite("solver", &BundleOptimizerOptions::solver_options)
          .def_readwrite("min_track_length",
                         &BundleOptimizerOptions::min_track_length)
          .def_readwrite("print_summary",
                         &BundleOptimizerOptions::print_summary)
          .def_readwrite("loss", &BundleOptimizerOptions::loss,
                         py::keep_alive<1, 2>());
  make_dataclass(ba_options);

  auto frba = py::class_<FeatureReferenceBundleOptimizer>(
                  m, "FeatureReferenceBundleOptimizer")
                  .def(py::init<BundleOptimizerOptions, BundleAdjustmentSetup,
                                InterpolationConfig>());
  BindBundleOptimizer(frba);

  auto cmba = py::class_<CostMapBundleOptimizer>(m, "CostMapBundleOptimizer")
                  .def(py::init<BundleOptimizerOptions, BundleAdjustmentSetup,
                                InterpolationConfig>());
  BindBundleOptimizer(cmba);

  auto pwba =
      py::class_<PatchWarpBundleOptimizer>(m, "PatchWarpBundleOptimizer")
          .def(py::init<PatchWarpBundleOptimizer::Options,
                        BundleAdjustmentSetup, InterpolationConfig>());
  BindBundleOptimizer(pwba);

  auto gba =
      py::class_<GeometricBundleOptimizer>(m, "GeometricBundleOptimizer")
          .def(py::init<BundleOptimizerOptions, BundleAdjustmentSetup>())
          .def("run", &GeometricBundleOptimizer::Run)
          .def("set_up", &GeometricBundleOptimizer::SetUp)
          .def("summary", &GeometricBundleOptimizer::Summary)
          .def_property_readonly("problem", &GeometricBundleOptimizer::Problem)
          .def("solve_problem", &GeometricBundleOptimizer::SolveProblem)
          .def("reset", &GeometricBundleOptimizer::Reset);

  auto pw_options =
      py::class_<PatchWarpBundleOptimizer::Options>(
          m, "PatchWarpBundleOptimizerOptions", ba_options)  // inherit
          .def(py::init<>())
          .def_readwrite("regularize_source",
                         &PatchWarpBundleOptimizer::Options::regularize_source);
  make_dataclass(pw_options);

  auto r = py::class_<ReferenceExtractor>(m, "ReferenceExtractor")
               .def(py::init<ReferenceConfig&, InterpolationConfig&>());

  BindReferenceExtractor<float16>(r);
  BindReferenceExtractor<float>(r);
  BindReferenceExtractor<double>(r);

  auto c = py::class_<CostMapExtractor>(m, "CostMapExtractor")
               .def(py::init<CostMapConfig&, InterpolationConfig&>());

  BindCostMapExtractor<float16>(c);
  BindCostMapExtractor<float>(c);
  BindCostMapExtractor<double>(c);
}

}  // namespace pixsfm
