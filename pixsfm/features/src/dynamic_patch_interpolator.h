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
DynamicPatchInterpolator:
*******************************************************************************/

class DynamicPatchInterpolator {
#define INTERPOLATION_CASES                                            \
  REGISTER_INTERPOLATION(1)                                            \
  REGISTER_INTERPOLATION(3)                                            \
  REGISTER_INTERPOLATION(8)                                            \
  REGISTER_INTERPOLATION(16)                                           \
  REGISTER_INTERPOLATION(32)                                           \
  REGISTER_INTERPOLATION(64)                                           \
  REGISTER_INTERPOLATION(128)                                          \
  REGISTER_INTERPOLATION(256)                                          \
  throw(std::runtime_error("Unknown channel size in interpolation: " + \
                           std::to_string(patch.Channels())));         \
  return false;                                                        \
  // High-Level Functions
 public:
  DynamicPatchInterpolator(const InterpolationConfig& config)
      : config_(config) {}

  template <int CHANNELS, typename dtype>
  DescriptorMatrixd<1, CHANNELS> Interpolate(const FeaturePatch<dtype>& patch,
                                             const Eigen::Vector2d& xy);

  template <int CHANNELS, int N_NODES, typename dtype>
  DescriptorMatrixd<N_NODES, CHANNELS> InterpolateNodes(
      const FeaturePatch<dtype>& patch, const Eigen::Vector2d& xy);

  template <int CHANNELS, typename dtype>
  DescriptorMatrixd<1, CHANNELS> InterpolateLocal(
      const FeaturePatch<dtype>& patch, const Eigen::Vector2d& xy);

  // Dynamic/Static overloads
  template <int CHANNELS, typename dtype,
            typename std::enable_if<CHANNELS == -1, void>::type* = nullptr>
  bool Evaluate(const FeaturePatch<dtype>& patch, const double* xy,
                double* result_ptr);

  template <int CHANNELS, typename dtype,
            typename std::enable_if<CHANNELS != -1, void>::type* = nullptr>
  bool Evaluate(const FeaturePatch<dtype>& patch, const double* xy,
                double* result_ptr);

  template <int CHANNELS, typename dtype,
            typename std::enable_if<CHANNELS == -1, void>::type* = nullptr>
  bool EvaluateNodes(const FeaturePatch<dtype>& patch, const double* xys,
                     double* result_ptr);

  template <int CHANNELS, typename dtype,
            typename std::enable_if<CHANNELS != -1, void>::type* = nullptr>
  bool EvaluateNodes(const FeaturePatch<dtype>& patch, const double* xys,
                     double* result_ptr);

  template <int CHANNELS, typename dtype,
            typename std::enable_if<CHANNELS == -1, void>::type* = nullptr>
  bool EvaluateLocal(const FeaturePatch<dtype>& patch, const double* xy,
                     double* result_ptr);

  template <int CHANNELS, typename dtype,
            typename std::enable_if<CHANNELS != -1, void>::type* = nullptr>
  bool EvaluateLocal(const FeaturePatch<dtype>& patch, const double* xy,
                     double* result_ptr);

 private:
  InterpolationConfig config_;
};

/*******************************************************************************
Methods:
*******************************************************************************/

template <int CHANNELS, typename dtype>
DescriptorMatrixd<1, CHANNELS> DynamicPatchInterpolator::Interpolate(
    const FeaturePatch<dtype>& patch, const Eigen::Vector2d& xy) {
  auto descriptor_mat =
      CreateDescriptorMatrixd<1, CHANNELS>(1, patch.Channels());
  bool res = Evaluate<CHANNELS>(patch, xy.data(), descriptor_mat.data());
  return descriptor_mat;
}

template <int CHANNELS, int N_NODES, typename dtype>
DescriptorMatrixd<N_NODES, CHANNELS> DynamicPatchInterpolator::InterpolateNodes(
    const FeaturePatch<dtype>& patch, const Eigen::Vector2d& xy) {
  auto descriptor_mat = CreateDescriptorMatrixd<N_NODES, CHANNELS>(
      config_.nodes.size(), patch.Channels());
  bool res = EvaluateNodes<CHANNELS>(patch, xy.data(), descriptor_mat.data());
  return descriptor_mat;
}

template <int CHANNELS, typename dtype>
DescriptorMatrixd<1, CHANNELS> DynamicPatchInterpolator::InterpolateLocal(
    const FeaturePatch<dtype>& patch, const Eigen::Vector2d& xy) {
  auto descriptor_mat =
      CreateDescriptorMatrixd<1, CHANNELS>(1, patch.Channels());
  bool res = EvaluateLocal<CHANNELS>(patch, xy.data(), descriptor_mat.data());
  return descriptor_mat;
}

/*******************************************************************************
Dynamic Patch Interpolator: static overloads
*******************************************************************************/

template <int CHANNELS, typename dtype,
          typename std::enable_if<CHANNELS != -1, void>::type*>
bool DynamicPatchInterpolator::Evaluate(const FeaturePatch<dtype>& patch,
                                        const double* xy, double* result_ptr) {
  THROW_CHECK_EQ(CHANNELS, patch.Channels());
  return PatchInterpolator<dtype, CHANNELS>(config_, patch)
      .Evaluate(xy, result_ptr);
}

template <int CHANNELS, typename dtype,
          typename std::enable_if<CHANNELS != -1, void>::type*>
bool DynamicPatchInterpolator::EvaluateNodes(const FeaturePatch<dtype>& patch,
                                             const double* xy,
                                             double* result_ptr) {
  THROW_CHECK_EQ(CHANNELS, patch.Channels());
  return PatchInterpolator<dtype, CHANNELS>(config_, patch)
      .EvaluateNodes(xy, result_ptr);
}

template <int CHANNELS, typename dtype,
          typename std::enable_if<CHANNELS != -1, void>::type*>
bool DynamicPatchInterpolator::EvaluateLocal(const FeaturePatch<dtype>& patch,
                                             const double* xy,
                                             double* result_ptr) {
  THROW_CHECK_EQ(CHANNELS, patch.Channels());
  return PatchInterpolator<dtype, CHANNELS>(config_, patch)
      .EvaluateLocal(xy, result_ptr);
}

/*******************************************************************************
Dynamic Patch Interpolator: dynamic overloads
*******************************************************************************/

template <int CHANNELS, typename dtype,
          typename std::enable_if<CHANNELS == -1, void>::type*>
bool DynamicPatchInterpolator::Evaluate(const FeaturePatch<dtype>& patch,
                                        const double* xy, double* result_ptr) {
#define REGISTER_INTERPOLATION(CHANNELS_)                      \
  if (patch.Channels() == CHANNELS_) {                         \
    return PatchInterpolator<dtype, CHANNELS_>(config_, patch) \
        .Evaluate(xy, result_ptr);                             \
  }

  INTERPOLATION_CASES
#undef REGISTER_INTERPOLATION
}

template <int CHANNELS, typename dtype,
          typename std::enable_if<CHANNELS == -1, void>::type*>
bool DynamicPatchInterpolator::EvaluateNodes(const FeaturePatch<dtype>& patch,
                                             const double* xy,
                                             double* result_ptr) {
#define REGISTER_INTERPOLATION(CHANNELS_)                      \
  if (patch.Channels() == CHANNELS_) {                         \
    return PatchInterpolator<dtype, CHANNELS_>(config_, patch) \
        .EvaluateNodes(xy, result_ptr);                        \
  }

  INTERPOLATION_CASES
#undef REGISTER_INTERPOLATION
}

template <int CHANNELS, typename dtype,
          typename std::enable_if<CHANNELS == -1, void>::type*>
bool DynamicPatchInterpolator::EvaluateLocal(const FeaturePatch<dtype>& patch,
                                             const double* xy,
                                             double* result_ptr) {
#define REGISTER_INTERPOLATION(CHANNELS_)                      \
  if (patch.Channels() == CHANNELS_) {                         \
    return PatchInterpolator<dtype, CHANNELS_>(config_, patch) \
        .EvaluateLocal(xy, result_ptr);                        \
  }

  INTERPOLATION_CASES
#undef REGISTER_INTERPOLATION
}

#undef INTERPOLATION_CASES

}  // namespace pixsfm