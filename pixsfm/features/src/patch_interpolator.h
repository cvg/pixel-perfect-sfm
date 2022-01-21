#pragma once

#include "_pixsfm/src/helpers.h"
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

#include <highfive/H5DataSet.hpp>
#include <highfive/H5DataSpace.hpp>
#include <highfive/H5File.hpp>
#include <highfive/H5Group.hpp>

#include <boost/serialization/vector.hpp>
#include <utility>
#include <vector>

#include <colmap/util/logging.h>

#include "features/src/featurepatch.h"
#include "features/src/patch_interpolator.h"

#include "base/src/grid2d.h"
#include "base/src/interpolation.h"

#include "util/src/log_exceptions.h"
#include "util/src/math.h"
#include "util/src/types.h"

#include <atomic>
#include <mutex>

namespace pixsfm {

/*******************************************************************************
FeaturePatchInterpolator:
*******************************************************************************/

// Does not support dynamic!
template <typename dtype, int channels, int N_NODES = 1>
class PatchInterpolator : public PixelInterpolator<Grid2D<dtype, channels>> {
  using Grid = Grid2D<dtype, channels>;
  using PixelInterpolator<Grid>::config_;

 public:
  PatchInterpolator(const InterpolationConfig& config,
                    const FeaturePatch<dtype>& patch);

  template <typename T>
  bool EvaluateNodes(const T* xy, T* f);

  template <typename T>
  bool EvaluateNNodes(const T* xys, T* f);

  template <typename T>
  bool Evaluate(const T* xy, T* f);

  template <typename T>
  void AddScaledNodeCoords(const T* x0y0, T* xys);

  template <typename T>
  inline bool CheckBounds(const T* uv) const;

  // Evaluate without pixel_coords
  template <typename T>
  bool EvaluateLocal(const T* xy, T* f);
  bool EvaluateLocal(double* xy, double* f, double* dfdr, double* dfdc,
                     double* dfdrc = NULL);

 private:
  Grid grid2D_;
  const FeaturePatch<dtype>& patch_;
};

/*******************************************************************************
Methods:
*******************************************************************************/

template <typename dtype, int channels, int N_NODES>
PatchInterpolator<dtype, channels, N_NODES>::PatchInterpolator(
    const InterpolationConfig& config, const FeaturePatch<dtype>& patch)
    : grid2D_(patch.Data(), 0, patch.Height(), 0, patch.Width()),
      PixelInterpolator<Grid>(config, grid2D_),
      patch_(patch) {}

template <typename dtype, int channels, int N_NODES>
template <typename T>
bool PatchInterpolator<dtype, channels, N_NODES>::EvaluateNodes(const T* xy,
                                                                T* f) {
  Eigen::Matrix<T, 2, 1> uv = patch_.GetPixelCoordinates(xy);
  PixelInterpolator<Grid>::EvaluateNodes(uv[1], uv[0], f);
  if (config_.check_bounds) {
    return CheckBounds(uv.data());
  } else {
    return true;
  }
}

template <typename dtype, int channels, int N_NODES>
template <typename T>
void PatchInterpolator<dtype, channels, N_NODES>::AddScaledNodeCoords(
    const T* x0y0, T* xys) {
  const Eigen::Vector2d& scale = patch_.Scale();

  for (int i = 0; i < N_NODES; i++) {
    xys[2 * i] = x0y0[0] + T(config_.nodes.at(i)[0]) / T(scale(0));
    xys[2 * i + 1] = x0y0[1] + T(config_.nodes.at(i)[1]) / T(scale(1));
  }
}
template <typename dtype, int channels, int N_NODES>
template <typename T>
bool PatchInterpolator<dtype, channels, N_NODES>::EvaluateNNodes(const T* xys,
                                                                 T* f) {
  T uvs[2 * N_NODES];
  bool is_inside = true;
  for (int i = 0; i < N_NODES; i++) {
    patch_.ToPixelCoordinates(xys + 2 * i, uvs + 2 * i);
    is_inside =
        is_inside && (config_.check_bounds ? CheckBounds(uvs + 2 * i) : true);
  }
  PixelInterpolator<Grid>::template EvaluateNNodes<T>(uvs, f, N_NODES);
  return is_inside;
}

template <typename dtype, int channels, int N_NODES>
template <typename T>
bool PatchInterpolator<dtype, channels, N_NODES>::Evaluate(const T* xy, T* f) {
  Eigen::Matrix<T, 2, 1> uv = patch_.GetPixelCoordinates(xy);
  PixelInterpolator<Grid>::Evaluate(uv[1], uv[0], f);
  if (config_.check_bounds) {
    return CheckBounds<T>(uv.data());
  } else {
    return true;
  }
}

template <typename dtype, int channels, int N_NODES>
bool PatchInterpolator<dtype, channels, N_NODES>::EvaluateLocal(
    double* xy, double* f, double* dfdr, double* dfdc, double* dfdrc) {
  PixelInterpolator<Grid>::Evaluate(xy[1], xy[0], f, dfdr, dfdc, dfdrc);
  if (config_.check_bounds) {
    return CheckBounds<double>(xy);
  } else {
    return true;
  }
}

template <typename dtype, int channels, int N_NODES>
template <typename T>
bool PatchInterpolator<dtype, channels, N_NODES>::EvaluateLocal(const T* xy,
                                                                T* f) {
  PixelInterpolator<Grid>::Evaluate(xy[1], xy[0], f);
  if (config_.check_bounds) {
    return CheckBounds<T>(xy);
  } else {
    return true;
  }
}

template <typename dtype, int channels, int N_NODES>
template <typename T>
bool PatchInterpolator<dtype, channels, N_NODES>::CheckBounds(
    const T* uv) const {
  return (IsInsideZeroL(uv[0], static_cast<double>(patch_.Width())) &&
          IsInsideZeroL(uv[1], static_cast<double>(patch_.Height())));
}

}  // namespace pixsfm