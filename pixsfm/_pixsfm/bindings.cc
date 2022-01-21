#include <pybind11/iostream.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
//#include <pybind11/exception.h>
namespace py = pybind11;

#include "_pixsfm/src/helpers.h"

#include "base/bindings.cc"
#include "bundle_adjustment/bindings.cc"
#include "features/bindings.cc"
#include "keypoint_adjustment/bindings.cc"
#include "localization/bindings.cc"
#include "pyceres/pyceres.cc"
#include "residuals/bindings.cc"
#include "util/bindings.cc"

#include "util/src/types.h"
#include <third-party/progressbar.h>

#include <chrono>
#include <thread>

void bind_base(py::module&);
void bind_keypoint_adjustment(py::module&);
void bind_bundle_adjustment(py::module&);
void bind_localization(py::module&);
void bind_features(py::module&);
void bind_residuals(py::module&);
void bind_util(py::module&);
void bind_pyceres(py::module&);

namespace pixsfm {

PYBIND11_MODULE(_pixsfm, m) {
  m.doc() = "Pixel-Perfect SfM plugin";

  py::class_<structlog>(m, "CppLogConfig")
      .def(py::init<>())
      .def_readwrite("silence", &structlog::silence_normal)
      .def_readwrite("headers", &structlog::headers)
      .def_readwrite("level", &structlog::level);

  m.attr("cpplog") = &LOGCFG;
  py::module::import("pycolmap");  // Load symbols for colmap classes
  py::add_ostream_redirect(m, "ostream_redirect");

  pybind11::module_ f = m.def_submodule("_features");
  pybind11::module_ b = m.def_submodule("_base");
  pybind11::module_ ba = m.def_submodule("_bundle_adjustment");
  pybind11::module_ ka = m.def_submodule("_keypoint_adjustment");
  pybind11::module_ loc = m.def_submodule("_localization");
  pybind11::module_ res = m.def_submodule("_residuals");
  pybind11::module_ util = m.def_submodule("_util");
  pybind11::module_ pyceres = m.def_submodule("pyceres");

  bind_base(b);
  bind_keypoint_adjustment(ka);
  bind_bundle_adjustment(ba);
  bind_localization(loc);
  bind_features(f);
  bind_residuals(res);
  bind_util(util);
  bind_pyceres(pyceres);
}

}  // namespace pixsfm