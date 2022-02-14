#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

namespace py = pybind11;

#include "_pixsfm/src/helpers.h"

#include <ceres/ceres.h>

#include "base/src/graph.h"
#include "base/src/interpolation.h"
#include "util/src/types.h"

#include <colmap/util/threading.h>

PYBIND11_MAKE_OPAQUE(pixsfm::MapNameKeypoints);
PYBIND11_MAKE_OPAQUE(std::vector<pixsfm::FeatureNode*>);
PYBIND11_MAKE_OPAQUE(std::vector<pixsfm::Match>);
PYBIND11_MAKE_OPAQUE(std::unordered_map<std::string, colmap::image_t>);
PYBIND11_MAKE_OPAQUE(std::unordered_map<colmap::image_t, std::string>);
PYBIND11_MAKE_OPAQUE(std::vector<size_t>);
PYBIND11_MAKE_OPAQUE(std::vector<colmap::point2D_t>);
PYBIND11_MAKE_OPAQUE(std::vector<bool>);

namespace pixsfm {

void bind_graph(py::module& m) {
  py::class_<FeatureNode>(m, "FeatureNode")
      .def(py::init<colmap::image_t, colmap::point2D_t>())
      .def_readonly("image_id", &FeatureNode::image_id)
      .def_readonly("feature_idx", &FeatureNode::feature_idx)
      .def_readonly("node_idx", &FeatureNode::node_idx)
      .def_readwrite("out_matches", &FeatureNode::out_matches)
      .def("__repr__", [](const FeatureNode& self) {
        std::stringstream ss;
        ss << "<FeatureNode <"
           << "node_idx=" << self.node_idx << ", "
           << "image_id=" << self.image_id << ", "
           << "feature_idx=" << self.feature_idx << ", "
           << "num_matches=" << self.out_matches.size() << ">";
        return ss.str();
      });

  py::class_<Match>(m, "Match")
      .def(py::init([](size_t node_idx, double sim) {
             Match match;
             match.node_idx = node_idx;
             match.sim = sim;
             return match;
           }),
           py::arg("node_idx"), py::arg("similarity"))
      .def_readwrite("node_idx", &Match::node_idx)
      .def_readwrite("similarity", &Match::sim)
      .def("__repr__", [](const Match& self) {
        std::stringstream ss;
        ss << "<Match <"
           << "node_idx=" << self.node_idx << ", "
           << "similarity=" << self.sim << ">";
        return ss.str();
      });

  py::class_<Graph>(m, "Graph")
      .def(py::init<>())
      .def_readwrite("image_name_to_id", &Graph::image_name_to_id)
      .def_readwrite("image_id_to_name", &Graph::image_id_to_name)
      .def_readonly("nodes", &Graph::nodes)
      .def("find_or_create_node", &Graph::FindOrCreateNode,
           py::return_value_policy::reference_internal)
      .def("add_node",
           overload_cast_<std::string, colmap::point2D_t>()(&Graph::AddNode))
      .def("add_node", overload_cast_<colmap::image_t, colmap::point2D_t>()(
                           &Graph::AddNode))
      .def("add_edge", &Graph::AddEdge)
      .def("degrees", &Graph::GetDegrees)
      .def("scores", &Graph::GetScores)
      .def("edges", &Graph::GetEdges)
      .def(
          "register_matches",
          [](Graph& self, std::string imname1, std::string imname2,
             py::array_t<size_t, py::array::c_style>& matches,
             py::array_t<double, py::array::c_style | py::array::forcecast>& similarities) {
            py::buffer_info matches_info = matches.request();

            THROW_CHECK_EQ(matches_info.ndim, 2);

            size_t* matches_ptr = static_cast<size_t*>(matches_info.ptr);
            std::vector<ssize_t> matches_shape = matches_info.shape;

            THROW_CHECK_EQ(matches_shape[1], 2);

            size_t n_matches = static_cast<size_t>(matches_shape[0]);
            double* sim_ptr = nullptr;
            py::buffer_info sim_info = similarities.request();
            if (sim_info.ndim > 0) {  // Not None
              size_t num_scores = static_cast<size_t>(sim_info.shape[0]);
              THROW_CHECK_EQ(n_matches, num_scores);
              sim_ptr = static_cast<double*>(sim_info.ptr);
            }

            self.RegisterMatches(imname1, imname2, matches_ptr, sim_ptr,
                                 n_matches);
          },
          py::arg("imagename1"), py::arg("imagename2"), py::arg("matches"),
          py::arg("similarities") = py::none());

  m.def("compute_track_labels", &ComputeTrackLabels);
  m.def("compute_score_labels", &ComputeScoreLabels);
  m.def("compute_root_labels", &ComputeRootLabels);
  m.def("count_track_edges", &CountTrackEdges);
  m.def("count_edges_AB", &CountEdgesAB);
}

void bind_base(py::module& m) {
  py::bind_map<MapNameKeypoints>(m, "Map_NameKeypoints", py::buffer_protocol());
  py::bind_map<std::unordered_map<std::string, colmap::image_t>>(m,
                                                                 "Map_NameId");
  py::bind_map<std::unordered_map<colmap::image_t, std::string>>(m,
                                                                 "Map_IdName");
  py::bind_vector<std::vector<FeatureNode*>>(m, "Vector_FeatureNode");
  py::bind_vector<std::vector<Match>>(m, "Vector_Match");

  auto vec_sizet = py::bind_vector<std::vector<size_t>>(m, "Vector_SizeT");
  AddListToVectorConstructor(vec_sizet);
  auto vec_p2D_idx =
      py::bind_vector<std::vector<colmap::point2D_t>>(m, "Vector_Point2DIdxT");
  AddListToVectorConstructor(vec_p2D_idx);
  auto vec_bool = py::bind_vector<std::vector<bool>>(m, "Vector_Bool");
  AddListToVectorConstructor(vec_bool);

  bind_graph(m);

  auto pyinterptype =
      py::enum_<InterpolatorType>(m, "InterpolatorType")
          .value("BICUBIC", InterpolatorType::BICUBIC)
          .value("BILINEAR", InterpolatorType::BILINEAR)
          .value("POLYGRADIENTFIELD", InterpolatorType::POLYGRADIENTFIELD)
          .value("BICUBICGRADIENTFIELD", InterpolatorType::BICUBICGRADIENTFIELD)
          .value("BICUBICCHAIN", InterpolatorType::BICUBICCHAIN)
          .value("CERES_BICUBIC", InterpolatorType::CERES_BICUBIC);
  AddStringToEnumConstructor(pyinterptype);

  auto pyinterp =
      py::class_<InterpolationConfig>(m, "InterpolationConfig")
          .def(py::init<>())
          .def_readwrite("l2_normalize", &InterpolationConfig::l2_normalize)
          .def_readwrite("ncc_normalize", &InterpolationConfig::ncc_normalize)
          .def_readwrite("nodes", &InterpolationConfig::nodes)
          .def_readwrite("mode", &InterpolationConfig::mode)
          .def_readwrite("check_bounds", &InterpolationConfig::check_bounds)
          .def_readwrite("use_float_simd",
                         &InterpolationConfig::use_float_simd);
  make_dataclass(pyinterp);

  m.def("get_effective_num_threads", &colmap::GetEffectiveNumThreads);
}

}  // namespace pixsfm
