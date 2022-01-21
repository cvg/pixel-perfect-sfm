#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "_pixsfm/src/helpers.h"

#include <ceres/ceres.h>

#include "util/src/glog.cc"
#include "util/src/memory.h"
namespace pixsfm {

void bind_util(py::module& m) {
  m.def("total_memory", &TotalPhysicalMemory);
  m.def("used_memory", &UsedPhysicalMemory);
  m.def("free_memory", &FreePhysicalMemory);

  init_glog(m);
}

}  // namespace pixsfm