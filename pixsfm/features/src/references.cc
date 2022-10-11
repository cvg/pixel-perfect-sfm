#include "features/src/references.h"

namespace pixsfm {

py::array_t<double> Reference::NpArray() const {
  py::str dummy_data_owner;  // Workaround, we need a valid pybind object as
                             // base
  return py::array_t<double, py::array::c_style>(
      {descriptor.rows(), descriptor.cols()},  // shape
      {descriptor.cols() * sizeof(double), sizeof(double)},
      // C-style contiguous strides for double
      descriptor.data(),
      dummy_data_owner);  // numpy array references this parent
}

void Reference::SetNpArray(const py::array_t<double, py::array::c_style>& desc) {
  const py::buffer_info info = desc.request();
  THROW_CHECK_EQ(info.ndim, 2);
  auto *ptr = static_cast<double *>(info.ptr);
  auto rows = static_cast<size_t>(info.shape[0]);
  auto cols = static_cast<size_t>(info.shape[1]);

  descriptor = Eigen::Map<DescriptorMatrixXd>(ptr, rows, cols); // Copy
}

const double* Reference::NodeOffsets3DData() const {
  return node_offsets3D.data();
}

const double* Reference::DescriptorData() const { return descriptor.data(); }

const DescriptorMatrixXd& Reference::Descriptor() const { return descriptor; }

double* Reference::NodeOffsets3DData() { return node_offsets3D.data(); }

double* Reference::DescriptorData() { return descriptor.data(); }

DescriptorMatrixXd& Reference::Descriptor() { return descriptor; }

colmap::image_t Reference::SourceImageId() const { return source.image_id; }

colmap::point2D_t Reference::SourcePoint2DIdx() const {
  return source.point2D_idx;
}

int Reference::Channels() const { return descriptor.cols(); }
int Reference::NumNodes() const { return descriptor.rows(); }

bool Reference::HasObservations() const {
  return (track.Length() > 0 && !observations.empty() && !costs.empty());
}

}  // namespace pixsfm