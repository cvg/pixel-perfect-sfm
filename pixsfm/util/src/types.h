#pragma once

#include <Eigen/Core>
#include <colmap/util/types.h>
#include <third-party/half.h>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include "util/src/log_exceptions.h"

namespace py = pybind11;

using float16 = half_float::half;

namespace pixsfm {

template <typename T>
using VectorX = Eigen::Matrix<T, Eigen::Dynamic, 1>;

template <int rows>
using VectorNd = Eigen::Matrix<double, rows, 1>;

template <int n_nodes, int channels>
using DescriptorMatrixd =
    Eigen::Matrix<double, n_nodes, channels, Eigen::RowMajor>;

template <int n_nodes>
using OffsetMatrix3d = Eigen::Matrix<double, n_nodes, 3, Eigen::RowMajor>;

using DescriptorMatrixXd = DescriptorMatrixd<Eigen::Dynamic, Eigen::Dynamic>;

const colmap::point2D_t kDensePatchId = 1000000;

using KeypointMatrixd = Eigen::Matrix<double, -1, 2, Eigen::RowMajor>;

using MapNameKeypoints = std::unordered_map<std::string, KeypointMatrixd>;

using CorrespondenceVec =
    std::vector<std::pair<colmap::point2D_t, colmap::point3D_t>>;

template <int rows, int cols>
DescriptorMatrixd<rows, cols> CreateDescriptorMatrixd(int n_nodes,
                                                      int channels) {
  if (rows == Eigen::Dynamic || cols == Eigen::Dynamic) {
    return DescriptorMatrixXd(n_nodes, channels);
  } else {
    THROW_CHECK_EQ(rows, n_nodes);
    THROW_CHECK_EQ(cols, channels);
    return DescriptorMatrixd<rows, cols>();
  }
}

}  // namespace pixsfm