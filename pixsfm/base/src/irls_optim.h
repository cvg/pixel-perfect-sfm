#pragma once
#include "_pixsfm/src/helpers.h"
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

#include <highfive/H5DataSet.hpp>
#include <highfive/H5DataSpace.hpp>
#include <highfive/H5File.hpp>
#include <highfive/H5Group.hpp>

#include <ceres/ceres.h>
#include <colmap/base/projection.h>
#include <colmap/util/types.h>

#include "base/src/interpolation.h"
#include "util/src/types.h"

namespace pixsfm {

// Actually only makes sense for 1D
template <int CHANNELS, int N_NODES>
inline DescriptorMatrixd<N_NODES, CHANNELS> RobustMeanIRLS(
    const std::vector<DescriptorMatrixd<N_NODES, CHANNELS>>& descriptor_track,
    ceres::LossFunction* loss_function, int num_iterations,
    const InterpolationConfig& interpolation_config) {
  THROW_CHECK_GT(descriptor_track.size(), 0);
  DescriptorMatrixd<N_NODES, CHANNELS> mean;

  int channels_ = descriptor_track[0].cols();
  int n_nodes_ = descriptor_track[0].rows();

  THROW_CHECK_EQ(interpolation_config.nodes.size(), n_nodes_);

  if (CHANNELS == Eigen::Dynamic || N_NODES == Eigen::Dynamic) {
    mean.resize(n_nodes_, channels_);
  }

  Eigen::Matrix<double, 1, Eigen::Dynamic> weights =
      Eigen::VectorXd::Ones(descriptor_track.size()).transpose();

  for (int k = 0; k < num_iterations; k++) {
    weights = weights / weights.sum();
    mean.setZero();
    for (int i = 0; i < descriptor_track.size(); i++) {
      mean += descriptor_track[i] * weights[i];
    }

    if (interpolation_config.ncc_normalize) {
      NCCNormalize<double>(mean.data(), channels_, n_nodes_);
    }

    if (interpolation_config.l2_normalize) {
      for (int j = 0; j < interpolation_config.nodes.size(); j++) {
        mean.row(j).normalize();
      }
    }

    for (int i = 0; i < descriptor_track.size(); i++) {
      double rho[3];
      loss_function->Evaluate((descriptor_track[i] - mean).squaredNorm(), rho);
      if (rho[0] > 0.0) {
        weights(i) = 1.0 / rho[0];
      } else {
        return descriptor_track[i];
      }
    }
  }
  return mean;
}

}  // namespace pixsfm