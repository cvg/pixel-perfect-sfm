#pragma once

#include <ceres/ceres.h>
#include <colmap/scene/projection.h>
#include <colmap/scene/reconstruction.h>
#include <colmap/util/types.h>

#include "features/src/featurepatch.h"
#include "features/src/patch_interpolator.h"

#include "base/src/interpolation.h"
#include "base/src/projection.h"

#include "util/src/simple_logger.h"
#include "util/src/types.h"

namespace pixsfm {

/*******************************************************************************
Keypoint Adjustment Costs:
*******************************************************************************/

template <typename dtype, int CHANNELS, int N_NODES = 1>
struct FeatureReference2DCostFunctor {
  FeatureReference2DCostFunctor(const FeaturePatch<dtype>& patch,
                                const InterpolationConfig& interpolation_config,
                                const double* ref_descriptor)
      : ref_descriptor_(ref_descriptor),
        interpolation_config_(interpolation_config) {
    patch_interpolator_.reset(
        new PatchInterpolator<dtype, CHANNELS>(interpolation_config_, patch));
  }

  static ceres::CostFunction* Create(
      const FeaturePatch<dtype>& patch,
      const InterpolationConfig& interpolation_config,
      const double* ref_descriptor) {
    return (new ceres::AutoDiffCostFunction<FeatureReference2DCostFunctor,
                                            N_NODES * CHANNELS, 2>(
        new FeatureReference2DCostFunctor(patch, interpolation_config,
                                          ref_descriptor)));
  }

  template <typename T>
  bool operator()(const T* const keypoint, T* residuals) const {
    T target_descriptor[CHANNELS * N_NODES];

    if (N_NODES == 1) {
      // We assume keypoints already in COLMAP coords
      patch_interpolator_->Evaluate(keypoint, target_descriptor);
    } else {
      patch_interpolator_->EvaluateNodes(keypoint, target_descriptor);
    }

    for (int i = 0; i < CHANNELS * N_NODES; i++) {
      residuals[i] = target_descriptor[i] - ref_descriptor_[i];
    }

    return true;
  }

 protected:
  const double* ref_descriptor_;
  std::unique_ptr<PatchInterpolator<dtype, CHANNELS>> patch_interpolator_;
  const InterpolationConfig& interpolation_config_;
};

/*******************************************************************************
Feature Bundle Adjustment Costs:
*******************************************************************************/
template <typename CameraModel, typename dtype, int CHANNELS, int N_NODES,
          int OUT_CHANNELS = -1>
struct FeatureReferenceCostFunctor {
 public:
  static constexpr bool WARP_PATCH = N_NODES > 1;
  FeatureReferenceCostFunctor(const FeaturePatch<dtype>& patch,
                              InterpolationConfig& interpolation_config,
                              const double* ref_descriptor = NULL,
                              const double* node_offsets3D = NULL)
      : ref_descriptor_(ref_descriptor),
        interpolation_config_(interpolation_config),
        node_offsets3D_(node_offsets3D) {
    patch_interpolator_.reset(new PatchInterpolator<dtype, CHANNELS, N_NODES>(
        interpolation_config_, patch));
  }

  static ceres::CostFunction* Create(const FeaturePatch<dtype>& patch,
                                     InterpolationConfig& interpolation_config,
                                     const double* reference_descriptor = NULL,
                                     const double* node_offsets3D = NULL) {
    return (new ceres::AutoDiffCostFunction<FeatureReferenceCostFunctor,
                                            N_RESIDUALS, 4, 3, 3,
                                            CameraModel::kNumParams>(
        new FeatureReferenceCostFunctor(patch, interpolation_config,
                                        reference_descriptor, node_offsets3D)));
  }

  template <typename T>
  bool operator()(const T* const qvec, const T* const tvec,
                  const T* const point3D, const T* const camera_params,
                  T* residuals) const {
    bool is_inside = true;
    if (WARP_PATCH) {
      // CHECK_NOTNULL(node_offsets3D_);
      if (node_offsets3D_) {
        T projections[2 * N_NODES];
        for (int i = 0; i < N_NODES; i++) {
          T point3D_node[3];
          for (int j = 0; j < 3; j++) {
            point3D_node[j] = point3D[j] + node_offsets3D_[3 * i + j];
          }
          WorldToPixel<CameraModel>(camera_params, qvec, tvec, point3D_node,
                                    projections + 2 * i);
        }
        is_inside = patch_interpolator_->EvaluateNNodes(projections, residuals);
      } else {
        T projection[2];
        WorldToPixel<CameraModel>(camera_params, qvec, tvec, point3D,
                                  projection);
        patch_interpolator_->EvaluateNodes(projection, residuals);
      }
    } else {
      T projection[2];
      WorldToPixel<CameraModel>(camera_params, qvec, tvec, point3D, projection);
      is_inside = patch_interpolator_->Evaluate(projection, residuals);
    }

    if (!ref_descriptor_) {
      return is_inside;
    }

    for (int i = 0; i < N_RESIDUALS; i++) {
      residuals[i] -= ref_descriptor_[i];
    }

    return true;
  }

 protected:
  static constexpr int N_RESIDUALS =
      N_NODES * (OUT_CHANNELS > 0 ? OUT_CHANNELS : CHANNELS);
  const double* ref_descriptor_ = NULL;
  const double* node_offsets3D_ = NULL;

  std::unique_ptr<PatchInterpolator<dtype, CHANNELS, N_NODES>>
      patch_interpolator_;
  const InterpolationConfig& interpolation_config_;
};

template <typename CameraModel, typename dtype, int CHANNELS,
          int OUT_CHANNELS = -1>
using CostMapFunctor =
    FeatureReferenceCostFunctor<CameraModel, dtype, CHANNELS, 1, OUT_CHANNELS>;

// For constant pose cases we create a separate cost functor to reduce jacobian
// size
template <typename CameraModel, typename dtype, int CHANNELS, int N_NODES,
          int OUT_CHANNELS = -1>
struct FeatureReferenceConstantPoseCostFunctor
    : public FeatureReferenceCostFunctor<CameraModel, dtype, CHANNELS, N_NODES,
                                         OUT_CHANNELS> {
  using Parent = FeatureReferenceCostFunctor<CameraModel, dtype, CHANNELS,
                                             N_NODES, OUT_CHANNELS>;

 public:
  static constexpr bool WARP_PATCH = N_NODES > 1;
  FeatureReferenceConstantPoseCostFunctor(
      const FeaturePatch<dtype>& patch,
      InterpolationConfig& interpolation_config, const double* qvec,
      const double* tvec, const double* ref_descriptor = NULL,
      const double* node_offsets3D = NULL)
      : Parent(patch, interpolation_config, ref_descriptor, node_offsets3D),
        qvec_(qvec),
        tvec_(tvec) {}

  static ceres::CostFunction* Create(const FeaturePatch<dtype>& patch,
                                     InterpolationConfig& interpolation_config,
                                     const double* qvec, const double* tvec,
                                     const double* reference_descriptor = NULL,
                                     const double* node_offsets3D = NULL) {
    return (
        new ceres::AutoDiffCostFunction<FeatureReferenceConstantPoseCostFunctor,
                                        N_RESIDUALS, 3,
                                        CameraModel::kNumParams>(
            new FeatureReferenceConstantPoseCostFunctor(
                patch, interpolation_config, qvec, tvec, reference_descriptor,
                node_offsets3D)));
  }

  template <typename T>
  bool operator()(const T* const point3D, const T* const camera_params,
                  T* residuals) const {
    // TYPE cast
    CHECK_NOTNULL(qvec_);
    CHECK_NOTNULL(tvec_);

    T tvec[3] = {T(tvec_[0]), T(tvec_[1]), T(tvec_[2])};
    T qvec[4] = {T(qvec_[0]), T(qvec_[1]), T(qvec_[2]), T(qvec_[3])};
    return Parent::operator()(qvec, tvec, point3D, camera_params, residuals);
  }

 protected:
  static constexpr int N_RESIDUALS =
      N_NODES * (OUT_CHANNELS > 0 ? OUT_CHANNELS : CHANNELS);
  const double* qvec_;
  const double* tvec_;
};

template <typename CameraModel, typename dtype, int CHANNELS,
          int OUT_CHANNELS = -1>
using CostMapConstantPoseFunctor =
    FeatureReferenceConstantPoseCostFunctor<CameraModel, dtype, CHANNELS, 1,
                                            OUT_CHANNELS>;

/*******************************************************************************
Initialization Wrappers: (resolving camera model templates)
*******************************************************************************/

template <int CHANNELS, int N_NODES, int OUT_CHANNELS, typename dtype>
ceres::CostFunction* CreateFeatureReferenceCostFunctor(
    int camera_model_id, const FeaturePatch<dtype>& patch,
    const double* reference_descriptor, const double* node_offsets3D,
    InterpolationConfig& interpolation_config) {
  switch (camera_model_id) {
#define CAMERA_MODEL_CASE(CameraModel)                               \
  case colmap::CameraModel::kModelId:                                \
    return FeatureReferenceCostFunctor<                              \
        colmap::CameraModel, dtype, CHANNELS, N_NODES,               \
        OUT_CHANNELS>::Create(patch, interpolation_config,           \
                              reference_descriptor, node_offsets3D); \
    break;
    CAMERA_MODEL_SWITCH_CASES
#undef CAMERA_MODEL_CASE
  }
}

template <int CHANNELS, int N_NODES, int OUT_CHANNELS, typename dtype>
ceres::CostFunction* CreateFeatureReferenceConstantPoseCostFunctor(
    int camera_model_id, double* qvec, double* tvec,
    const FeaturePatch<dtype>& patch, const double* reference_descriptor,
    const double* node_offsets3D, InterpolationConfig& interpolation_config) {
  switch (camera_model_id) {
#define CAMERA_MODEL_CASE(CameraModel)                                 \
  case colmap::CameraModel::kModelId:                                  \
    return FeatureReferenceConstantPoseCostFunctor<                    \
        colmap::CameraModel, dtype, CHANNELS, N_NODES,                 \
        OUT_CHANNELS>::Create(patch, interpolation_config, qvec, tvec, \
                              reference_descriptor, node_offsets3D);   \
    break;
    CAMERA_MODEL_SWITCH_CASES
#undef CAMERA_MODEL_CASE
  }
}

// PyBind interfaces
template <typename dtype>
ceres::CostFunction* CreateFeatureReferenceCostFunctor(
    int camera_model_id, const FeaturePatch<dtype>& patch,
    Eigen::Ref<DescriptorMatrixXd> reference_descriptor,
    InterpolationConfig& interpolation_config) {
  int channels = patch.Channels();
  int n_nodes = interpolation_config.nodes.size();

  THROW_CHECK_EQ(reference_descriptor.rows(), n_nodes);
  THROW_CHECK_EQ(reference_descriptor.cols(), channels);

#define REGISTER_METHOD(CHANNELS, N_NODES)                               \
  if (channels == CHANNELS && n_nodes == N_NODES) {                      \
    return CreateFeatureReferenceCostFunctor<CHANNELS, N_NODES, -1>(     \
        camera_model_id, patch, nullptr, nullptr, interpolation_config); \
  }

  REGISTER_METHOD(128, 1)
  // REGISTER_METHOD(64,1)
  // REGISTER_METHOD(32,1)
  // REGISTER_METHOD(16,1)
  // REGISTER_METHOD(8,1)
  // REGISTER_METHOD(3,1)
  REGISTER_METHOD(1, 1)

  THROW_EXCEPTION(std::invalid_argument,
                  "Unsupported dimensions (CHANNELS,N_NODES).");

#undef REGISTER_METHOD
  return nullptr;
}

template <typename dtype>
ceres::CostFunction* CreateFeatureReferenceConstantPoseCostFunctor(
    int camera_model_id, Eigen::Ref<Eigen::Vector4d> qvec,
    Eigen::Ref<Eigen::Vector3d> tvec, const FeaturePatch<dtype>& patch,
    Eigen::Ref<DescriptorMatrixXd> reference_descriptor,
    InterpolationConfig& interpolation_config) {
  int channels = patch.Channels();
  int n_nodes = interpolation_config.nodes.size();

  THROW_CHECK_EQ(reference_descriptor.rows(), n_nodes);
  THROW_CHECK_EQ(reference_descriptor.cols(), channels);

#define REGISTER_METHOD(CHANNELS, N_NODES)                                  \
  if (channels == CHANNELS && n_nodes == N_NODES) {                         \
    return CreateFeatureReferenceConstantPoseCostFunctor<CHANNELS, N_NODES, \
                                                         -1>(               \
        camera_model_id, qvec.data(), tvec.data(), patch,                   \
        reference_descriptor.data(), nullptr, interpolation_config);        \
  }

  REGISTER_METHOD(128, 1)
  // REGISTER_METHOD(64,1)
  // REGISTER_METHOD(32,1)
  // REGISTER_METHOD(16,1)
  // REGISTER_METHOD(8,1)
  // REGISTER_METHOD(3,1)
  REGISTER_METHOD(1, 1)

  THROW_EXCEPTION(std::invalid_argument,
                  "Unsupported dimensions (CHANNELS,N_NODES).");

#undef REGISTER_METHOD
  return nullptr;
}

}  // namespace pixsfm
