#pragma once
#include "_pixsfm/src/helpers.h"
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

#include <highfive/H5DataSet.hpp>
#include <highfive/H5DataSpace.hpp>
#include <highfive/H5File.hpp>
#include <highfive/H5Group.hpp>

#include <colmap/scene/track.h>
#include <colmap/util/types.h>

#include "util/src/log_exceptions.h"
#include "util/src/types.h"

#include <unordered_map>

namespace pixsfm {

struct ReferenceData {
  colmap::Track track;
  std::vector<DescriptorMatrixXd> observations;
  std::vector<double> costs;
};

struct Reference {
  Reference() : source(colmap::TrackElement(0, 0)) {}

  Reference(colmap::TrackElement source_, DescriptorMatrixXd descriptor_,
            const ReferenceData* refdata_ = nullptr,
            OffsetMatrix3d<Eigen::Dynamic> node_offsets3D_ =
                OffsetMatrix3d<Eigen::Dynamic>())
      : descriptor(std::move(descriptor_)), source(source_) {
    if (node_offsets3D_.data()) {
      node_offsets3D = std::move(node_offsets3D_);
    }
    if (refdata_) {
      track = refdata_->track;
      observations = refdata_->observations;
      costs = refdata_->costs;
    }
  }

  const double* NodeOffsets3DData() const;
  const double* DescriptorData() const;
  const DescriptorMatrixXd& Descriptor() const;
  double* NodeOffsets3DData();
  double* DescriptorData();
  DescriptorMatrixXd& Descriptor();

  colmap::image_t SourceImageId() const;
  colmap::point2D_t SourcePoint2DIdx() const;
  py::array_t<double> NpArray() const;

  void SetNpArray(const py::array_t<double, py::array::c_style>& desc);

  int Channels() const;
  int NumNodes() const;
  bool HasObservations() const;

  colmap::TrackElement source;
  DescriptorMatrixXd descriptor;
  OffsetMatrix3d<Eigen::Dynamic> node_offsets3D;

  // optional data
  colmap::Track track;
  std::vector<DescriptorMatrixXd> observations;
  std::vector<double> costs;
};

}  // namespace pixsfm
