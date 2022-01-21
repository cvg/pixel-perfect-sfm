
#pragma once

#include <ceres/ceres.h>
#include <colmap/base/projection.h>
#include <colmap/base/reconstruction.h>
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

template <typename dtype, int CHANNELS, int N_NODES>
struct FeatureMetric2DCostFunctor {
  FeatureMetric2DCostFunctor(const FeaturePatch<dtype>& patch1,
                             const FeaturePatch<dtype>& patch2,
                             const InterpolationConfig& interpolation_config)
      : interpolation_config_(interpolation_config) {
    patch1_interpolator_.reset(
        new PatchInterpolator<dtype, CHANNELS>(interpolation_config_, patch1));
    patch2_interpolator_.reset(
        new PatchInterpolator<dtype, CHANNELS>(interpolation_config_, patch2));
  }

  static ceres::CostFunction* Create(
      const FeaturePatch<dtype>& patch1, const FeaturePatch<dtype>& patch2,
      const InterpolationConfig& interpolation_config) {
    return (new ceres::AutoDiffCostFunction<FeatureMetric2DCostFunctor,
                                            N_NODES * CHANNELS, 2, 2>(
        new FeatureMetric2DCostFunctor(patch1, patch2, interpolation_config)));
  }

  template <typename T>
  bool operator()(const T* const keypoint1, const T* const keypoint2,
                  T* residuals) const {
    T target_descriptor1[CHANNELS * N_NODES];
    T target_descriptor2[CHANNELS * N_NODES];

    if (N_NODES == 1) {
      patch1_interpolator_->Evaluate(keypoint1, target_descriptor1);
      patch2_interpolator_->Evaluate(keypoint2, target_descriptor2);
    } else {
      patch1_interpolator_->EvaluateNodes(keypoint1, target_descriptor1);
      patch2_interpolator_->EvaluateNodes(keypoint2, target_descriptor2);
    }

    for (int i = 0; i < CHANNELS * N_NODES; i++) {
      residuals[i] = target_descriptor1[i] - target_descriptor2[i];
    }

    return true;
  }

 protected:
  std::unique_ptr<PatchInterpolator<dtype, CHANNELS>> patch2_interpolator_;
  std::unique_ptr<PatchInterpolator<dtype, CHANNELS>> patch1_interpolator_;
  const InterpolationConfig& interpolation_config_;
};

/*******************************************************************************
Patch Warping Cost Functors:
*******************************************************************************/

// TODO: Support surface normals (atm fronto-parallel only)
// Fronto-Parallel Warping using source observation
template <typename CameraModel, typename SourceCameraModel, typename dtype,
          int CHANNELS, int N_NODES = 1>
struct FeatureMetricCostFunctor {
  FeatureMetricCostFunctor(
      const FeaturePatch<dtype>& patch,
      const FeaturePatch<dtype>* src_patch,  // Can be NULL here
      InterpolationConfig& interpolation_config,
      const double* reference_descriptor = NULL)
      : interpolation_config_(interpolation_config),
        ref_descriptor_(reference_descriptor) {
    THROW_CHECK_EQ(N_NODES, interpolation_config.nodes.size());
    if (src_patch) {
      src_patch_interpolator_.reset(
          new PatchInterpolator<dtype, CHANNELS, N_NODES>(interpolation_config_,
                                                          *src_patch));
    }

    patch_interpolator_.reset(new PatchInterpolator<dtype, CHANNELS, N_NODES>(
        interpolation_config_, patch));
  }

  static ceres::CostFunction* Create(
      const FeaturePatch<dtype>& patch, const FeaturePatch<dtype>& src_patch,
      InterpolationConfig& interpolation_config) {
    return (new ceres::AutoDiffCostFunction<
            FeatureMetricCostFunctor, N_NODES * CHANNELS, 4, 3, 4, 3, 3,
            CameraModel::kNumParams, SourceCameraModel::kNumParams>(
        new FeatureMetricCostFunctor(patch, &src_patch, interpolation_config)));
  }

  static ceres::CostFunction* Create(const FeaturePatch<dtype>& target_patch,
                                     InterpolationConfig& interpolation_config,
                                     const double* reference_descriptor) {
    return (new ceres::AutoDiffCostFunction<
            FeatureMetricCostFunctor, N_NODES * CHANNELS, 4, 3, 4, 3, 3,
            CameraModel::kNumParams, SourceCameraModel::kNumParams>(
        new FeatureMetricCostFunctor(target_patch, NULL, interpolation_config,
                                     reference_descriptor)));
  }

  template <typename T>
  bool operator()(const T* const target_qvec, const T* const target_tvec,
                  const T* const source_qvec, const T* const source_tvec,
                  const T* const point3D, const T* const target_camera_params,
                  const T* const source_camera_params, T* residuals) const {
    T source_projection[2];

    WorldToPixel<SourceCameraModel>(source_camera_params, source_qvec,
                                    source_tvec, point3D, source_projection);

    T target_descriptor[CHANNELS * N_NODES];

    // TODO: if constexpr (warp_patch_) with warp_patch_ as static const!
    if (WARP_PATCH) {
      T depth;
      CalculateDepth(source_qvec, source_tvec, point3D, &depth);

      T source_patch_coords[2 * N_NODES];
      src_patch_interpolator_->AddScaledNodeCoords(source_projection,
                                                   source_patch_coords);
      T target_projections[2 * N_NODES];
      for (int k = 0; k < N_NODES; k++) {
        T world[3];

        PixelToWorld<SourceCameraModel>(source_camera_params, source_qvec,
                                        source_tvec, source_patch_coords[2 * k],
                                        source_patch_coords[2 * k + 1], &depth,
                                        world);

        WorldToPixel<CameraModel>(target_camera_params, target_qvec,
                                  target_tvec, world,
                                  target_projections + 2 * k);
      }

      patch_interpolator_->EvaluateNNodes(target_projections,
                                          target_descriptor);
    } else {
      T target_projection[2];
      WorldToPixel<CameraModel>(target_camera_params, target_qvec, target_tvec,
                                point3D, target_projection);
      for (int k = 0; k < N_NODES; k++) {
        patch_interpolator_->Evaluate(target_projection,
                                      target_descriptor + k * CHANNELS);
      }
    }

    if (!ref_descriptor_) {
      T source_descriptor[CHANNELS * N_NODES];
      src_patch_interpolator_->EvaluateNodes(source_projection,
                                             source_descriptor);

      for (int i = 0; i < CHANNELS * N_NODES; i++) {
        residuals[i] = target_descriptor[i] - source_descriptor[i];
      }
    } else {
      for (int i = 0; i < CHANNELS * N_NODES; i++) {
        residuals[i] = target_descriptor[i] - ref_descriptor_[i];
      }
    }
    return true;
  }

  static constexpr bool WARP_PATCH = N_NODES > 1;

 private:
  const double* ref_descriptor_ = NULL;
  std::unique_ptr<PatchInterpolator<dtype, CHANNELS, N_NODES>>
      patch_interpolator_;
  std::unique_ptr<PatchInterpolator<dtype, CHANNELS, N_NODES>>
      src_patch_interpolator_;
  const InterpolationConfig& interpolation_config_;
};

// Same Camera Model type, but different intrinsic parameters
template <typename CameraModel, typename dtype, int CHANNELS, int N_NODES>
using FeatureMetricSameModelCostFunctor =
    FeatureMetricCostFunctor<CameraModel, CameraModel, dtype, CHANNELS,
                             N_NODES>;

// Same Camera Model and shared intrinsics
// To avoid having duplicate parameter blocks in residual
template <typename CameraModel, typename dtype, int CHANNELS, int N_NODES>
struct FeatureMetricSharedIntrinsicsCostFunctor
    : public FeatureMetricCostFunctor<CameraModel, CameraModel, dtype, CHANNELS,
                                      N_NODES> {
  using Parent = FeatureMetricCostFunctor<CameraModel, CameraModel, dtype,
                                          CHANNELS, N_NODES>;
  FeatureMetricSharedIntrinsicsCostFunctor(
      const FeaturePatch<dtype>& patch,
      const FeaturePatch<dtype>* src_patch,  // Can be NULL here
      InterpolationConfig& interpolation_config,
      const double* reference_descriptor = NULL)
      : Parent(patch, src_patch, interpolation_config, reference_descriptor) {}

  static ceres::CostFunction* Create(
      const FeaturePatch<dtype>& patch, const FeaturePatch<dtype>& src_patch,
      InterpolationConfig& interpolation_config) {
    return (new ceres::AutoDiffCostFunction<
            FeatureMetricSharedIntrinsicsCostFunctor, N_NODES * CHANNELS, 4, 3,
            4, 3, 3, CameraModel::kNumParams>(
        new FeatureMetricSharedIntrinsicsCostFunctor(patch, &src_patch,
                                                     interpolation_config)));
  }

  static ceres::CostFunction* Create(InterpolationConfig& interpolation_config,
                                     const FeaturePatch<dtype>& target_patch,
                                     const double* reference_descriptor) {
    return (new ceres::AutoDiffCostFunction<
            FeatureMetricSharedIntrinsicsCostFunctor, N_NODES * CHANNELS, 4, 3,
            4, 3, 3, CameraModel::kNumParams>(
        new FeatureMetricSharedIntrinsicsCostFunctor(
            target_patch, NULL, interpolation_config, reference_descriptor)));
  }

  template <typename T>
  bool operator()(const T* const target_qvec, const T* const target_tvec,
                  const T* const source_qvec, const T* const source_tvec,
                  const T* const point3D, const T* const camera_params,
                  T* residuals) const {
    return Parent::operator()(target_qvec, target_tvec, source_qvec,
                              source_tvec, point3D, camera_params,
                              camera_params, residuals);
  }
};

/*******************************************************************************
Initialization Wrappers: (resolving camera model templates)
*******************************************************************************/

#define PW_CAMERA_MODEL_CASES                 \
  CAMERA_MODEL_CASE(SimplePinholeCameraModel) \
  CAMERA_MODEL_CASE(PinholeCameraModel)       \
  CAMERA_MODEL_CASE(SimpleRadialCameraModel)  \
  CAMERA_MODEL_CASE(RadialCameraModel)

#ifndef PW_CAMERA_MODEL_SWITCH_CASES
#define PW_CAMERA_MODEL_SWITCH_CASES      \
  PW_CAMERA_MODEL_CASES                   \
  default:                                \
    CAMERA_MODEL_DOES_NOT_EXIST_EXCEPTION \
    break;
#endif

template <int CHANNELS, int N_NODES, typename dtype>
ceres::CostFunction* CreateFeatureMetricCostFunctor(
    int camera_model_id,               // Src
    const FeaturePatch<dtype>& patch,  // Src
    int src_camera_model_id, const FeaturePatch<dtype>& src_patch,
    InterpolationConfig& interpolation_config) {
  // Temporary
  THROW_CHECK_EQ(camera_model_id, src_camera_model_id);

  switch (camera_model_id) {
#define CAMERA_MODEL_CASE(CameraModel)                            \
  case colmap::CameraModel::kModelId:                             \
    return FeatureMetricSameModelCostFunctor<                     \
        colmap::CameraModel, dtype, CHANNELS,                     \
        N_NODES>::Create(patch, src_patch, interpolation_config); \
    break;
    PW_CAMERA_MODEL_SWITCH_CASES
#undef CAMERA_MODEL_CASE
  }

  // TODO: Add different camera model cases
}

template <int CHANNELS, int N_NODES, typename dtype>
ceres::CostFunction* CreateFeatureMetricRegularizerCostFunctor(
    int camera_model_id, int src_camera_model_id,
    const FeaturePatch<dtype>& patch, const double* reference_descriptor,
    InterpolationConfig& interpolation_config) {
  // Temporary
  THROW_CHECK_EQ(camera_model_id, src_camera_model_id);

  switch (camera_model_id) {
#define CAMERA_MODEL_CASE(CameraModel)                                       \
  case colmap::CameraModel::kModelId:                                        \
    return FeatureMetricSameModelCostFunctor<                                \
        colmap::CameraModel, dtype, CHANNELS,                                \
        N_NODES>::Create(patch, interpolation_config, reference_descriptor); \
    break;
    PW_CAMERA_MODEL_SWITCH_CASES
#undef CAMERA_MODEL_CASE
  }

  // TODO: Add different camera model cases
}

template <int CHANNELS, int N_NODES, typename dtype>
ceres::CostFunction* CreateFeatureMetricSharedIntrinsicsCostFunctor(
    int camera_model_id, const FeaturePatch<dtype>& patch,
    const FeaturePatch<dtype>& src_patch,
    InterpolationConfig& interpolation_config) {
  switch (camera_model_id) {
#define CAMERA_MODEL_CASE(CameraModel)                            \
  case colmap::CameraModel::kModelId:                             \
    return FeatureMetricSharedIntrinsicsCostFunctor<              \
        colmap::CameraModel, dtype, CHANNELS,                     \
        N_NODES>::Create(patch, src_patch, interpolation_config); \
    break;
    PW_CAMERA_MODEL_SWITCH_CASES
#undef CAMERA_MODEL_CASE
  }
}

template <int CHANNELS, int N_NODES, typename dtype>
ceres::CostFunction* CreateFeatureMetricSharedIntrinsicsRegularizerCostFunctor(
    int camera_model_id, const FeaturePatch<dtype>& patch,
    const double* reference_descriptor,
    InterpolationConfig& interpolation_config) {
  switch (camera_model_id) {
#define CAMERA_MODEL_CASE(CameraModel)                                       \
  case colmap::CameraModel::kModelId:                                        \
    return FeatureMetricSharedIntrinsicsCostFunctor<                         \
        dtype, colmap::CameraModel, CHANNELS,                                \
        N_NODES>::Create(patch, interpolation_config, reference_descriptor); \
    break;
    PW_CAMERA_MODEL_SWITCH_CASES
#undef CAMERA_MODEL_CASE
  }
}

#undef PW_CAMERA_MODEL_SWITCH_CASES
#undef PW_CAMERA_MODEL_CASES

template <typename dtype>
ceres::CostFunction* CreateFeatureMetricCostFunctor(
    colmap::Camera& camera, const FeaturePatch<dtype>& patch,
    colmap::Camera& src_camera, const FeaturePatch<dtype>& src_patch,
    InterpolationConfig& interpolation_config) {
  int channels = patch.Channels();
  int n_nodes = interpolation_config.nodes.size();

#define REGISTER_METHOD(CHANNELS, N_NODES)                            \
  if (channels == CHANNELS && n_nodes == N_NODES) {                   \
    if (camera.Params() == src_camera.Params()) {                     \
      return CreateFeatureMetricSharedIntrinsicsCostFunctor<CHANNELS, \
                                                            N_NODES>( \
          camera.ModelId(), patch, src_patch, interpolation_config);  \
    } else {                                                          \
      return CreateFeatureMetricCostFunctor<CHANNELS, N_NODES>(       \
          camera.ModelId(), patch, src_camera.ModelId(), src_patch,   \
          interpolation_config);                                      \
    }                                                                 \
  }

  // REGISTER_METHOD(128,4)
  // REGISTER_METHOD(128,1)
  REGISTER_METHOD(3, 16)
  // REGISTER_METHOD(3,4)
  // REGISTER_METHOD(3,1)
  REGISTER_METHOD(1, 16)
  // REGISTER_METHOD(1,4)
  // REGISTER_METHOD(1,1)

  THROW_EXCEPTION(std::invalid_argument,
                  "Unsupported dimensions (CHANNELS,N_NODES).");

#undef REGISTER_METHOD
  return nullptr;
}

}  // namespace pixsfm
