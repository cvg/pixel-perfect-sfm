#pragma once
#include "bundle_adjustment/src/bundle_optimizer.h"

#include "features/src/featureview.h"
#include "residuals/src/feature_reference.h"

namespace pixsfm {

#define COSTMAP_CASES                       \
  REGISTER_METHOD(1)                        \
  REGISTER_METHOD(3)                        \
  THROW_EXCEPTION(std::invalid_argument,    \
                  "Unsupported dimensions " \
                  "(CHANNELS).");

// Feature Bundle adjustment based on Ceres-Solver.
class CostMapBundleOptimizer : public BundleOptimizer<CostMapBundleOptimizer> {
  using Base = BundleOptimizer<CostMapBundleOptimizer>;

 public:
  CostMapBundleOptimizer(const BundleOptimizerOptions& options,
                         const BundleAdjustmentSetup& setup,
                         const InterpolationConfig& interpolation_config)
      : BundleOptimizer(options, setup),
        interpolation_config_(interpolation_config) {
    STDLOG(INFO) << "Start costmap bundle adjustment." << std::endl;
  }

  template <typename dtype>
  inline bool Run(colmap::Reconstruction* reconstruction,
                  FeatureView<dtype>& feature_view);

  template <typename dtype>
  inline void SetUp(colmap::Reconstruction* reconstruction,
                    std::shared_ptr<ceres::LossFunction> loss_function,
                    FeatureView<dtype>& feature_view);

  template <int CHANNELS, typename dtype>
  int AddResiduals(const colmap::image_t image_id,
                   const colmap::point2D_t point2D_id,
                   colmap::Reconstruction* reconstruction,
                   ceres::LossFunction* loss_function,
                   FeatureView<dtype>& feature_view);

 protected:
  InterpolationConfig interpolation_config_;
};

template <typename dtype>
bool CostMapBundleOptimizer::Run(colmap::Reconstruction* reconstruction,
                                 FeatureView<dtype>& feature_view) {
  size_t channels = feature_view.Channels();
#define REGISTER_METHOD(CHANNELS)                             \
  if (channels == CHANNELS) {                                 \
    return Base::Run<CHANNELS>(reconstruction, feature_view); \
  }
  COSTMAP_CASES
#undef REGISTER_METHOD
}

template <typename dtype>
void CostMapBundleOptimizer::SetUp(
    colmap::Reconstruction* reconstruction,
    std::shared_ptr<ceres::LossFunction> loss_function,
    FeatureView<dtype>& feature_view) {
  size_t channels = feature_view.Channels();
#define REGISTER_METHOD(CHANNELS)                                     \
  if (channels == CHANNELS) {                                         \
    return Base::SetUp<CHANNELS>(reconstruction, loss_function.get(), \
                                 feature_view);                       \
  }
  COSTMAP_CASES
#undef REGISTER_METHOD
}

template <int CHANNELS, typename dtype>
int CostMapBundleOptimizer::AddResiduals(const colmap::image_t image_id,
                                         const colmap::point2D_t point2D_idx,
                                         colmap::Reconstruction* reconstruction,
                                         ceres::LossFunction* loss_function,
                                         FeatureView<dtype>& feature_view) {
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
        CreateFeatureReferenceConstantPoseCostFunctor<CHANNELS, 1, -1>(
            camera.ModelId(), qvec_data, tvec_data,
            feature_view.GetFeaturePatch(image_id, point2D_idx),
            nullptr,  // Just minimize
            nullptr, interpolation_config_);
    block_id = problem_->AddResidualBlock(cost_function, loss_function, xyz,
                                          camera_params_data);
  } else {
    ceres::CostFunction* cost_function =
        CreateFeatureReferenceCostFunctor<CHANNELS, 1, -1>(
            camera.ModelId(),
            feature_view.GetFeaturePatch(image_id, point2D_idx),
            nullptr,  // Just minimize
            nullptr, interpolation_config_);
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

#undef REGISTER_METHOD
#undef COSTMAP_CASES
}  // namespace pixsfm