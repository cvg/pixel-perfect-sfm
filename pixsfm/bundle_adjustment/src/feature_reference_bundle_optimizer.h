#pragma once

#include "bundle_adjustment/src/bundle_optimizer.h"

#include "residuals/src/feature_reference.h"

#include "features/src/featureview.h"
#include "features/src/references.h"

namespace pixsfm {

#define FEATURE_REFERENCE_CASES          \
  REGISTER_METHOD(128, 1)                \
  REGISTER_METHOD(64, 1)                 \
  REGISTER_METHOD(3, 1)                  \
  REGISTER_METHOD(1, 1)                  \
  THROW_EXCEPTION(std::invalid_argument, \
                  "Unsupported dimensions (CHANNELS,N_NODES).");

// Feature Bundle adjustment based on Ceres-Solver.
class FeatureReferenceBundleOptimizer
    : public BundleOptimizer<FeatureReferenceBundleOptimizer> {
  using Base = BundleOptimizer<FeatureReferenceBundleOptimizer>;

 public:
  FeatureReferenceBundleOptimizer(
      const BundleOptimizerOptions& options, const BundleAdjustmentSetup& setup,
      const InterpolationConfig& interpolation_config)
      : BundleOptimizer(options, setup),
        interpolation_config_(interpolation_config) {
    // @TODO: Verify if feature and reference view match
    // @TODO: Verify if everything required from config is available
    STDLOG(INFO) << "Start feature-reference bundle adjustment." << std::endl;
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

  template <int CHANNELS, int N_NODES, typename dtype>
  int AddResiduals(
      const colmap::image_t image_id, const colmap::point2D_t point2D_id,
      colmap::Reconstruction* reconstruction,
      ceres::LossFunction* loss_function, FeatureView<dtype>& feature_view,
      std::unordered_map<colmap::point3D_t, Reference>& references);

 protected:
  InterpolationConfig interpolation_config_;
};

template <typename dtype>
bool FeatureReferenceBundleOptimizer::Run(
    colmap::Reconstruction* reconstruction, FeatureView<dtype>& feature_view,
    std::unordered_map<colmap::point3D_t, Reference>& references) {
  size_t channels = feature_view.Channels();
  size_t n_nodes = interpolation_config_.nodes.size();
#define REGISTER_METHOD(CHANNELS, N_NODES)                            \
  if (channels == CHANNELS && n_nodes == N_NODES) {                   \
    return Base::Run<CHANNELS, N_NODES>(reconstruction, feature_view, \
                                        references);                  \
  }
  FEATURE_REFERENCE_CASES
#undef REGISTER_METHOD
}

template <typename dtype>
void FeatureReferenceBundleOptimizer::SetUp(
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
  FEATURE_REFERENCE_CASES
#undef REGISTER_METHOD
}

template <int CHANNELS, int N_NODES, typename dtype>
int FeatureReferenceBundleOptimizer::AddResiduals(
    const colmap::image_t image_id, const colmap::point2D_t point2D_idx,
    colmap::Reconstruction* reconstruction, ceres::LossFunction* loss_function,
    FeatureView<dtype>& feature_view,
    std::unordered_map<colmap::point3D_t, Reference>& references) {
  const bool constant_pose =
      !options_.refine_extrinsics || setup_.HasConstantPose(image_id);

  colmap::Image& image = reconstruction->Image(image_id);
  colmap::Camera& camera = reconstruction->Camera(image.CameraId());
  colmap::Point2D& point2D = image.Point2D(point2D_idx);

  // Check whether the requested point is associated to a point3D
  if (!point2D.HasPoint3D()) {
    return 0;
  }

  colmap::point3D_t point3D_id = point2D.Point3DId();
  colmap::Point3D& point3D = reconstruction->Point3D(point3D_id);

  double* qvec_data = image.Qvec().data();
  double* tvec_data = image.Tvec().data();
  double* camera_params_data = camera.ParamsData();
  double* xyz = point3D.XYZ().data();

  ceres::ResidualBlockId block_id;

  if (constant_pose) {
    ceres::CostFunction* cost_function =
        CreateFeatureReferenceConstantPoseCostFunctor<CHANNELS, N_NODES, -1>(
            camera.ModelId(), qvec_data, tvec_data,
            feature_view.GetFeaturePatch(image_id, point2D_idx),
            references.at(point3D_id).DescriptorData(),
            references.at(point3D_id).NodeOffsets3DData(),
            interpolation_config_);
    block_id = problem_->AddResidualBlock(cost_function, loss_function, xyz,
                                          camera_params_data);
  } else {
    ceres::CostFunction* cost_function =
        CreateFeatureReferenceCostFunctor<CHANNELS, N_NODES, -1>(
            camera.ModelId(),
            feature_view.GetFeaturePatch(image_id, point2D_idx),
            references.at(point3D_id).DescriptorData(),
            references.at(point3D_id).NodeOffsets3DData(),
            interpolation_config_);

    block_id =
        problem_->AddResidualBlock(cost_function, loss_function, qvec_data,
                                   tvec_data, xyz, camera_params_data);
    image_num_residuals_[image_id] += 1;
  }

  // REGISTER BLOCK ID
  point3D_num_residuals_[point3D_id] += 1;
  RegisterPoint3DObservation(point3D_id, image_id, point2D_idx, reconstruction);

  camera_num_residuals_[image.CameraId()] += 1;
  return 1;
}

#undef FEATURE_REFERENCE_CASES

}  // namespace pixsfm