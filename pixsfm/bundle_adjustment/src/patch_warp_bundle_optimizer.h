#pragma once
#include "bundle_adjustment/src/bundle_optimizer.h"

#include "residuals/src/feature_reference.h"
#include "residuals/src/featuremetric.h"

#include "features/src/featureview.h"
#include "features/src/references.h"

namespace pixsfm {

// REGISTER_METHOD(channels, n_nodes)
#define PATCH_WARP_CASES                 \
  REGISTER_METHOD(128, 1)                \
  REGISTER_METHOD(1, 16)                 \
  REGISTER_METHOD(3, 16)                 \
  THROW_EXCEPTION(std::invalid_argument, \
                  "Unsupported dimensions (CHANNELS,N_NODES).");

// Feature Bundle adjustment based on Ceres-Solver.
class PatchWarpBundleOptimizer
    : public BundleOptimizer<PatchWarpBundleOptimizer> {
  using Base = BundleOptimizer<PatchWarpBundleOptimizer>;

 public:
  struct Options : public BundleOptimizerOptions {
    using BundleOptimizerOptions::BundleOptimizerOptions;  // inh. constr.
    bool regularize_source = false;
  };
  PatchWarpBundleOptimizer(const Options& options,
                           const BundleAdjustmentSetup& setup,
                           const InterpolationConfig& interpolation_config)
      : BundleOptimizer(options, setup),
        interpolation_config_(interpolation_config) {
    STDLOG(INFO) << "Start patch-warp bundle adjustment." << std::endl;
  }

  template <typename dtype>
  bool Run(colmap::Reconstruction* reconstruction,
           FeatureView<dtype>& feature_view,
           std::unordered_map<colmap::point3D_t, Reference>& references);

  template <typename dtype>
  void SetUp(colmap::Reconstruction* reconstruction,
             std::shared_ptr<ceres::LossFunction> loss_function,
             FeatureView<dtype>& feature_view,
             std::unordered_map<colmap::point3D_t, Reference>& references);

  // Constant reference, patch warping if n_nodes > 1 using fronto-parallel
  // assumption
  template <int CHANNELS, int N_NODES, typename dtype>
  int AddResiduals(
      const colmap::image_t image_id, const colmap::point2D_t point2D_id,
      colmap::Reconstruction* reconstruction,
      ceres::LossFunction* loss_function, FeatureView<dtype>& feature_view,
      std::unordered_map<colmap::point3D_t, Reference>& references);

 protected:
  InterpolationConfig interpolation_config_;
  Options options_;
};

template <typename dtype>
bool PatchWarpBundleOptimizer::Run(
    colmap::Reconstruction* reconstruction, FeatureView<dtype>& feature_view,
    std::unordered_map<colmap::point3D_t, Reference>& references) {
  size_t channels = feature_view.Channels();
  size_t n_nodes = interpolation_config_.nodes.size();
#define REGISTER_METHOD(CHANNELS, N_NODES)                            \
  if (channels == CHANNELS && n_nodes == N_NODES) {                   \
    return Base::Run<CHANNELS, N_NODES>(reconstruction, feature_view, \
                                        references);                  \
  }
  PATCH_WARP_CASES
#undef REGISTER_METHOD
}

template <typename dtype>
void PatchWarpBundleOptimizer::SetUp(
    colmap::Reconstruction* reconstruction,
    std::shared_ptr<ceres::LossFunction> loss_function,
    FeatureView<dtype>& feature_view,
    std::unordered_map<colmap::point3D_t, Reference>& references) {
  size_t channels = feature_view.Channels();
  size_t n_nodes = interpolation_config_.nodes.size();
#define REGISTER_METHOD(CHANNELS, N_NODES)                                     \
  if (channels == CHANNELS && n_nodes == N_NODES) {                            \
    return Base::SetUp<CHANNELS, N_NODES>(reconstruction, loss_function.get(), \
                                          feature_view, references);           \
  }
  PATCH_WARP_CASES
#undef REGISTER_METHOD
}

template <int CHANNELS, int N_NODES, typename dtype>
int PatchWarpBundleOptimizer::AddResiduals(
    const colmap::image_t image_id, const colmap::point2D_t point2D_idx,
    colmap::Reconstruction* reconstruction, ceres::LossFunction* loss_function,
    FeatureView<dtype>& feature_view,
    std::unordered_map<colmap::point3D_t, Reference>& references) {
  const bool constant_cam_pose =
      !options_.refine_extrinsics || setup_.HasConstantCamPose(image_id);

  colmap::Image& image = reconstruction->Image(image_id);
  colmap::Camera& camera = reconstruction->Camera(image.CameraId());
  colmap::Point2D& point2D = image.Point2D(point2D_idx);

  // Check whether the requested point is associated to a point3D
  if (!point2D.HasPoint3D()) {
    return 0;
  }

  colmap::point3D_t point3D_id = point2D.point3D_id;
  colmap::Point3D& point3D = reconstruction->Point3D(point3D_id);

  double* cam_from_world_rotation = image.CamFromWorld().rotation.coeffs().data();
  double* cam_from_world_translation = image.CamFromWorld().translation.data();
  double* camera_params = camera.params.data();
  double* xyz = point3D.xyz.data();

  ceres::ResidualBlockId block_id;

  colmap::image_t src_image_id = references.at(point3D_id).SourceImageId();
  colmap::point2D_t src_point2D_idx =
      references.at(point3D_id).SourcePoint2DIdx();
  colmap::Image& src_image = reconstruction->Image(src_image_id);
  colmap::Camera& src_camera = reconstruction->Camera(src_image.CameraId());

  double* src_qvec_data = src_image.CamFromWorld().rotation.coeffs().data();
  double* src_tvec_data = src_image.CamFromWorld().translation.data();
  double* src_camera_params_data = src_camera.params.data();

  if (src_image_id == image_id) {
    // INTERNAL FUNCTION //
    if (src_point2D_idx == point2D_idx && options_.regularize_source) {
      ceres::CostFunction* cost_function =
          CreateFeatureReferenceCostFunctor<CHANNELS, N_NODES, -1>(
              camera.model_id,
              feature_view.GetFeaturePatch(image_id, src_point2D_idx),
              references.at(point3D_id).DescriptorData(),
              references.at(point3D_id).NodeOffsets3DData(),
              interpolation_config_);
      block_id =
          problem_->AddResidualBlock(cost_function, loss_function, cam_from_world_rotation,
                                     cam_from_world_translation, xyz, camera_params);
      image_num_residuals_[image_id] += 1;
    } else {
      // Do nothing in this case
    }

  } else {
    // TODO: reduced system when poses are fixed
    // ATM: Just set pose parameters constant (see bundle_adjuster.h)

    if (camera.camera_id == src_camera.camera_id) {
      // Shared intrinsics
      ceres::CostFunction* cost_function =
          CreateFeatureMetricSharedIntrinsicsCostFunctor<CHANNELS, N_NODES>(
              camera.model_id,
              feature_view.GetFeaturePatch(image_id, point2D_idx),
              feature_view.GetFeaturePatch(src_image_id, src_point2D_idx),
              interpolation_config_);
      block_id = problem_->AddResidualBlock(
          cost_function, loss_function, cam_from_world_rotation, cam_from_world_translation, src_qvec_data,
          src_tvec_data, xyz, camera_params);
    } else {
      ceres::CostFunction* cost_function =
          CreateFeatureMetricCostFunctor<CHANNELS, N_NODES>(
              camera.model_id,
              feature_view.GetFeaturePatch(image_id, point2D_idx),
              src_camera.model_id,
              feature_view.GetFeaturePatch(src_image_id, src_point2D_idx),
              interpolation_config_);
      block_id = problem_->AddResidualBlock(
          cost_function, loss_function, cam_from_world_rotation, cam_from_world_translation, src_qvec_data,
          src_tvec_data, xyz, camera_params, src_camera_params_data);
    }

    image_num_residuals_[image_id] += 1;
  }

  // REGISTER BLOCK ID
  point3D_num_residuals_[point3D_id] += 1;
  RegisterPoint3DObservation(point3D_id, image_id, point2D_idx, reconstruction);
  RegisterPoint3DObservation(point3D_id, src_image_id, src_point2D_idx,
                             reconstruction);

  camera_num_residuals_[image.CameraId()] += 1;
  return 1;
}

#undef REGISTER_METHOD
#undef PATCH_WARP_CASES

}  // namespace pixsfm