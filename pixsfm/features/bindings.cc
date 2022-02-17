#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

namespace py = pybind11;

#include "_pixsfm/src/helpers.h"

#include "util/src/types.h"

#include "features/src/dynamic_patch_interpolator.h"
#include "features/src/featuremanager.h"
#include "features/src/featuremap.h"
#include "features/src/featurepatch.h"
#include "features/src/featureset.h"
#include "features/src/featureview.h"
#include "features/src/references.h"

#include "base/src/graph.h"

PYBIND11_MAKE_OPAQUE(pixsfm::VecFSet<float16>);
PYBIND11_MAKE_OPAQUE(pixsfm::VecFSet<float>);
PYBIND11_MAKE_OPAQUE(pixsfm::VecFSet<double>);

PYBIND11_MAKE_OPAQUE(pixsfm::MapStringFMap<float16>);
PYBIND11_MAKE_OPAQUE(pixsfm::MapStringFMap<float>);
PYBIND11_MAKE_OPAQUE(pixsfm::MapStringFMap<double>);

PYBIND11_MAKE_OPAQUE(pixsfm::MapIdFPatch<float16>);
PYBIND11_MAKE_OPAQUE(pixsfm::MapIdFPatch<float>);
PYBIND11_MAKE_OPAQUE(pixsfm::MapIdFPatch<double>);

PYBIND11_MAKE_OPAQUE(std::unordered_map<colmap::point3D_t, pixsfm::Reference>);

namespace pixsfm {

template <typename dtype>
void BindFeatureTemplate(py::module& m, std::string type_suffix) {
  using FPatch = FeaturePatch<dtype>;
  using FMap = FeatureMap<dtype>;
  using FSet = FeatureSet<dtype>;
  using FView = FeatureView<dtype>;
  using FManager = FeatureManager<dtype>;

  // FeaturePatch
  py::class_<FPatch>(m, ("FeaturePatch" + type_suffix).c_str())
      .def(py::init<py::array_t<dtype, py::array::c_style>, Eigen::Vector2i&,
                    Eigen::Vector2d&, bool>(),
           py::arg("inarray").noconvert(), py::arg("offset"), py::arg("scale"),
           py::arg("do_copy") = false)
      .def_property_readonly("data", &FPatch::AsNumpyArray)
      .def_property_readonly("shape", &FPatch::Shape)
      .def_property_readonly("corner", &FPatch::Corner)
      .def_property_readonly("scale", &FPatch::Scale)
      .def_property_readonly("height", &FPatch::Height)
      .def_property_readonly("width", &FPatch::Width)
      .def_property_readonly("channels", &FPatch::Channels)
      .def_property_readonly("size", &FPatch::Size)
      .def("get_entry", &FPatch::GetEntry)
      .def("is_reference", &FPatch::IsReference)
      .def("has_data", &FPatch::HasData)
      .def("data_ptr", overload_cast_<>()(&FPatch::Data))
      .def("get_pixel_coords", &FPatch::GetPixelCoordinatesVec)
      .def("lock", &FPatch::Lock)
      .def("flush", &FPatch::Flush)
      .def_property_readonly("status", &FPatch::Status)
      .def("current_memory", &FPatch::CurrentMemory,
           "Estimate current memory consumption in bytes.")
      .def("num_bytes", &FPatch::NumBytes,
           "Estimate theoretical memory consumption in bytes.")
      .def_property("upsampling_factor", &FPatch::UpsamplingFactor,
                    &FPatch::SetUpsamplingFactor);

  m.def(
      "FeaturePatch",
      [](py::array_t<dtype, py::array::c_style> inarray,
         Eigen::Vector2i& offset, Eigen::Vector2d& scale,
         bool do_copy) { return FPatch(inarray, offset, scale, do_copy); },
      py::arg("inarray").noconvert(), py::arg("offset"), py::arg("scale"),
      py::arg("do_copy") = false);

  // FeatureMap
  py::class_<FMap>(m, ("FeatureMap" + type_suffix).c_str())
      .def(py::init<py::array_t<dtype, py::array::c_style>, std::vector<int>&,
                    std::vector<Eigen::Vector2i>&, py::dict>(),
           py::arg("patches").noconvert(), py::arg("point2D_ids"),
           py::arg("corners"), py::arg("metadata"))
      .def("fpatch", &FMap::GetFeaturePatch,
           py::return_value_policy::reference_internal)
      .def("add_fpatch", &FMap::AddFeaturePatch)
      .def_property_readonly("fpatches", &FMap::Patches)
      .def("num_fpatches", &FMap::NumPatches)
      .def("shape", &FMap::Shape)
      .def_property_readonly("channels", &FMap::Channels)
      .def_property_readonly("is_sparse", &FMap::IsSparse)
      .def_property_readonly("size", &FMap::Size)
      .def("keys", &FMap::Keys)
      .def("flush", &FMap::Flush)
      .def("lock", &FMap::Lock)
      .def("current_memory", &FMap::CurrentMemory,
           "Estimate current memory consumption in bytes.")
      .def("num_bytes", &FMap::NumBytes,
           "Estimate theoretical memory consumption in bytes.");

  m.def(
      "FeatureMap",
      [](py::array_t<dtype, py::array::c_style> patches,
         std::vector<int>& point2D_ids, std::vector<Eigen::Vector2i>& corners,
         py::dict metadata) {
        return FMap(patches, point2D_ids, corners, metadata);
      },
      py::arg("patches").noconvert(), py::arg("point2D_ids"),
      py::arg("corners"), py::arg("metadata"));

  // FeatureSet
  py::class_<FSet>(m, ("FeatureSet" + type_suffix).c_str())
      .def(py::init<MapStringFMap<dtype>&, int>(),
           py::arg("feature_dict").noconvert(), py::arg("channels"))
      .def(py::init<int>(), py::arg("channels"))
      .def(py::init<std::string&, std::string&, int, bool>())  // H5 Constructor
      .def_property_readonly("fmaps", &FSet::FeatureMaps)
      .def("emplace", &FSet::Emplace)
      .def("add_fmap", &FSet::AddFeatureMap)
      .def("fmap", &FSet::GetFeatureMap,
           py::return_value_policy::reference_internal)
      .def("keys", &FSet::Keys)
      .def("has_fmap", &FSet::HasFeatureMap)
      .def_property_readonly("channels", &FSet::Channels)
      .def("flush", &FSet::Flush)
      .def("lock", &FSet::Lock)
      .def("use_parallel_io", &FSet::UseParallelIO)
      .def("flush_every_n", &FSet::FlushEveryN)
      .def("current_memory", &FSet::CurrentMemory,
           "Estimate current memory consumption in bytes.")
      .def("num_bytes", &FSet::NumBytes,
           "Estimate theoretical memory consumption in bytes.");

  m.def(
      "FeatureSet",
      [](MapStringFMap<dtype>& feature_dict, int channels) {
        return FSet(feature_dict, channels);
      },
      py::arg("feature_dict"), py::arg("channels"));

  // FeatureView
  py::class_<FView, std::shared_ptr<FView>>(
      m, ("FeatureView" + type_suffix).c_str())
      .def(py::init<FSet*, std::unordered_map<colmap::image_t, std::string>&>(),
           py::arg("feature_set").noconvert(), py::arg("image_id_to_name"))
      .def(py::init<FSet*, colmap::Reconstruction*>(),
           py::arg("feature_set").noconvert(), py::arg("reconstruction"))
      .def(py::init<FSet*, const Graph*>(), py::arg("feature_set").noconvert(),
           py::arg("graph"))
      .def(py::init<FSet*, const Graph*, const std::unordered_set<size_t>&>(),
           py::arg("feature_set").noconvert(), py::arg("graph"),
           py::arg("required_nodes"))
      .def(py::init<FSet*, colmap::Reconstruction*,
                    std::unordered_set<colmap::point3D_t>&>(),
           py::arg("feature_set").noconvert(), py::arg("reconstruction"),
           py::arg("point3D_ids"))
      .def("fmap", overload_cast_<colmap::image_t>()(&FView::GetFeatureMap),
           py::return_value_policy::reference_internal)
      .def("fmap", overload_cast_<const std::string&>()(&FView::GetFeatureMap),
           py::return_value_policy::reference_internal)
      .def("fpatch",
           overload_cast_<colmap::image_t, colmap::point2D_t>()(
               &FView::GetFeaturePatch),
           py::return_value_policy::reference_internal)
      .def("fpatch",
           overload_cast_<const std::string&, colmap::point2D_t>()(
               &FView::GetFeaturePatch),
           py::return_value_policy::reference_internal)
      .def("has_fpatch", overload_cast_<colmap::image_t, colmap::point2D_t>()(
                             &FView::HasFeaturePatch))
      .def("has_fpatch",
           overload_cast_<const std::string&, colmap::point2D_t>()(
               &FView::HasFeaturePatch))
      .def_property_readonly("channels", &FView::Channels)
      .def("mapping", &FView::Mapping)
      .def("find_image_id", &FView::FindImageId)
      .def("reserved_memory", &FView::ReservedMemory,
           "Estimate reserved memory of this view.");

  m.def(
      "FeatureView",
      [](FSet* feature_set, colmap::Reconstruction* reconstruction) {
        return std::move(FView(feature_set, reconstruction));
      },
      py::arg("feature_set").noconvert(), py::arg("reconstruction"),
      py::keep_alive<0, 2>());

  m.def(
      "FeatureView",
      [](FSet* feature_set, const Graph* graph) {
        return std::move(FView(feature_set, graph));
      },
      py::arg("feature_set").noconvert(), py::arg("graph").noconvert(),
      py::keep_alive<0, 2>());

  m.def(
      "FeatureView",
      [](FSet* feature_set, const Graph* graph,
         const std::unordered_set<size_t>& required_nodes) {
        return std::move(FView(feature_set, graph, required_nodes));
      },
      py::arg("feature_set").noconvert(), py::arg("graph").noconvert(),
      py::arg("required_nodes"), py::keep_alive<0, 2>());

  m.def(
      "FeatureView",
      [](FSet* feature_set, std::unordered_set<std::string>& image_names) {
        return std::move(FView(feature_set, image_names));
      },
      py::arg("feature_set").noconvert(), py::arg("image_names"),
      py::keep_alive<0, 2>());

  m.def(
      "FeatureView",
      [](FSet* feature_set, colmap::Reconstruction* reconstruction,
         std::unordered_set<colmap::point3D_t>& point3D_ids) {
        return std::move(FView(feature_set, reconstruction, point3D_ids));
      },
      py::arg("feature_set").noconvert(), py::arg("reconstruction"),
      py::arg("point3D_ids"), py::keep_alive<0, 2>());

  py::class_<FManager>(m, ("FeatureManager" + type_suffix).c_str())
      .def(py::init<std::vector<int>, py::array_t<dtype>>())
      .def(py::init<std::string&, bool, std::string>())  // H5 Constructor
      .def("fset", &FManager::FeatureSet,
           py::return_value_policy::reference_internal)
      .def("fsets", &FManager::FeatureSets,
           py::return_value_policy::reference_internal)
      .def_property_readonly("num_levels", &FManager::NumLevels)
      .def("lock", &FManager::Lock)
      .def("current_memory", &FManager::CurrentMemory,
           "Estimate current memory consumption in bytes.")
      .def("num_bytes", &FManager::NumBytes,
           "Estimate theoretical memory consumption in bytes.");

  m.def("FeatureManager",
        [](std::vector<int> channels_per_level, py::array_t<dtype> dummy) {
          return FManager(channels_per_level, dummy);
        });

  py::bind_vector<VecFSet<dtype>>(m, ("Vec_FSet" + type_suffix).c_str());
  py::bind_map<MapStringFMap<dtype>>(m,
                                     ("Map_StringFMap" + type_suffix).c_str());
  py::bind_map<MapIdFPatch<dtype>>(m, ("Map_IdFPatch" + type_suffix).c_str());
}

void bind_features(py::module& m) {
  m.attr("kDenseId") = &kDensePatchId;
  py::class_<PatchStatus>(m, "PatchStatus")
      .def_readwrite("is_locked", &PatchStatus::is_locked)
      .def_readwrite("reference_count", &PatchStatus::reference_count);

  py::class_<Reference>(m, "Reference")
      .def_readwrite("source", &Reference::source)
      .def_property_readonly("descriptor", &Reference::NpArray)
      .def_property_readonly("channels", &Reference::Channels)
      .def_property_readonly("n_nodes", &Reference::NumNodes)
      .def_readwrite("track", &Reference::track)
      .def_readwrite("observations", &Reference::observations)
      .def_readwrite("costs", &Reference::costs)
      .def("has_observations", &Reference::HasObservations);

  py::class_<DynamicPatchInterpolator>(m, "PatchInterpolator")
      .def(py::init<const InterpolationConfig&>())
      .def("interpolate", &DynamicPatchInterpolator::Interpolate<-1, float16>)
      .def("interpolate", &DynamicPatchInterpolator::Interpolate<-1, float>)
      .def("interpolate", &DynamicPatchInterpolator::Interpolate<-1, double>)
      .def("interpolate_nodes",
           &DynamicPatchInterpolator::InterpolateNodes<-1, -1, float16>)
      .def("interpolate_nodes",
           &DynamicPatchInterpolator::InterpolateNodes<-1, -1, float>)
      .def("interpolate_nodes",
           &DynamicPatchInterpolator::InterpolateNodes<-1, -1, double>)
      .def("interpolate_local",
           &DynamicPatchInterpolator::InterpolateLocal<-1, float16>)
      .def("interpolate_local",
           &DynamicPatchInterpolator::InterpolateLocal<-1, float>)
      .def("interpolate_local",
           &DynamicPatchInterpolator::InterpolateLocal<-1, double>);

  py::bind_map<std::unordered_map<colmap::point3D_t, Reference>>(
      m, "Map_IdReference");

  BindFeatureTemplate<float16>(m, "_f16");
  BindFeatureTemplate<double>(m, "_f64");
  BindFeatureTemplate<float>(m, "_f32");
}

}  // namespace pixsfm
