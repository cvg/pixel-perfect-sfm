#pragma once
#include "bundle_adjustment/src/bundle_optimizer.h"

#include "residuals/src/geometric.h"

#include "features/src/featureview.h"
#include "features/src/references.h"

namespace pixsfm {

// Feature Bundle adjustment based on Ceres-Solver.
class GeometricBundleOptimizer
    : public BundleOptimizer<GeometricBundleOptimizer> {
  using Base = BundleOptimizer<GeometricBundleOptimizer>;

 public:
  GeometricBundleOptimizer(const BundleOptimizerOptions& options,
                           const BundleAdjustmentSetup& setup)
      : BundleOptimizer(options, setup) {
    STDLOG(INFO) << "Start geometric bundle adjustment." << std::endl;
  }

  bool Run(colmap::Reconstruction* reconstruction) {
    return Base::Run(reconstruction);
  }

  void SetUp(colmap::Reconstruction* reconstruction,
             std::shared_ptr<ceres::LossFunction> loss_function) {
    return Base::SetUp(reconstruction, loss_function.get());
  }

  template <int T = 0>
  int AddResiduals(const colmap::image_t image_id,
                   const colmap::point2D_t point2D_id,
                   colmap::Reconstruction* reconstruction,
                   ceres::LossFunction* loss_function);
};

template <>
int GeometricBundleOptimizer::AddResiduals<>(
    const colmap::image_t image_id, const colmap::point2D_t point2D_idx,
    colmap::Reconstruction* reconstruction,
    ceres::LossFunction* loss_function) {
  const bool constant_pose =
      (!options_.refine_extrinsics || setup_.HasConstantPose(image_id) ||
       !setup_.HasImage(image_id));

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
    ceres::CostFunction* cost_function = CreateGeometricConstantPoseCostFunctor(
        camera.ModelId(), image.Qvec(), image.Tvec(), point2D.XY());
    block_id = problem_->AddResidualBlock(cost_function, loss_function, xyz,
                                          camera_params_data);
  } else {
    ceres::CostFunction* cost_function =
        CreateGeometricCostFunctor(camera.ModelId(), point2D.XY());

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

}  // namespace pixsfm