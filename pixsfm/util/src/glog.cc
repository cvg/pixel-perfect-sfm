#include <colmap/util/logging.h>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <pybind11/iostream.h>

namespace py = pybind11;

#include <glog/logging.h>

#include <chrono>
#include <iostream>
#include <thread>

class glog_dummy {};  // dummy class

namespace pixsfm {

// Issue #7: Glog version > 0.5.0 requires T=size_t, <= 0.5.0 T=int
template <typename T>
void PyBindLogStack(const char* data, T size) {
  std::chrono::milliseconds timespan(2000);  // or whatever
  py::scoped_estream_redirect stream(
      std::cerr,                                // std::ostream&
      py::module::import("sys").attr("stderr")  // Python output
  );
  std::this_thread::sleep_for(timespan);
  std::this_thread::sleep_for(timespan);

  std::cerr << data << std::endl;
  std::cerr << std::endl;
  std::cerr
      << "ERROR: C++ code terminated. Kernel Died. See log files for details.";
  std::cerr << std::endl << std::endl << std::endl;
}

void PyBindLogTermination() {
  std::chrono::milliseconds timespan(2000);  // or whatever
  py::scoped_estream_redirect stream(
      std::cerr,                                // std::ostream&
      py::module::import("sys").attr("stderr")  // Python output
  );
  std::this_thread::sleep_for(timespan);
  std::this_thread::sleep_for(timespan);

  std::cerr << std::endl;
  std::cerr
      << "ERROR: C++ code terminated. Kernel Died. See log files for details.";
  std::cerr << std::endl << std::endl << std::endl;
  exit(1);
}

void init_glog(py::module& m) {
  // google::InstallFailureSignalHandler();
  // google::InitGoogleLogging("");
  // google::InstallFailureFunction(&PyBindLogTermination); //Important to warn
  // User in jupyter-notebook about FATAL failure (segfault, LOG(FATAL),
  // CHECK(), ...)

  py::class_<glog_dummy>(m, "glog")
      .def_property_static(
          "minloglevel", [](py::object) { return FLAGS_minloglevel; },
          [](py::object, int a) { FLAGS_minloglevel = a; })
      .def_property_static(
          "stderrthreshold", [](py::object) { return FLAGS_stderrthreshold; },
          [](py::object, int a) { FLAGS_stderrthreshold = a; })
      .def_property_static(
          "log_dir", [](py::object) { return FLAGS_log_dir; },
          [](py::object, std::string a) { FLAGS_log_dir = a; })
      .def_property_static(
          "logtostderr", [](py::object) { return FLAGS_logtostderr; },
          [](py::object, bool a) { FLAGS_logtostderr = a; })
      .def_property_static(
          "alsologtostderr", [](py::object) { return FLAGS_alsologtostderr; },
          [](py::object, bool a) { FLAGS_alsologtostderr = a; })
      .def("init",
           [](std::string path) {
             google::ShutdownGoogleLogging();
             google::InitGoogleLogging(path.c_str());
           })
      .def("init",
           []() {
             google::InstallFailureSignalHandler();
             google::InitGoogleLogging("");
             google::InstallFailureFunction(&PyBindLogTermination);
           })
      .def("install_failure_writer",
           []() { google::InstallFailureWriter(&PyBindLogStack); })
      .def("install_failure_function",
           []() { google::InstallFailureFunction(&PyBindLogTermination); });
}

}  // namespace pixsfm