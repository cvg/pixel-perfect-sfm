#pragma once

#include <ceres/ceres.h>
#include <colmap/geometry/rigid3.h>
#include <colmap/sensor/models.h>
#include <colmap/util/types.h>

#include "base/src/projection.h"

namespace pixsfm {

/*******************************************************************************
Reprojection Error Cost Functors:
*******************************************************************************/

// Standard bundle adjustment cost function for variable camera pose
// (qvec in Eigen order, tvec), point and calibration parameters.
template <typename CameraModel>
struct GeometricCostFunctor {
  explicit GeometricCostFunctor(const Eigen::Vector2d& point2D)
      : point2D_(point2D) {}

  static ceres::CostFunction* Create(const Eigen::Vector2d& point2D) {
    return (new ceres::AutoDiffCostFunction<GeometricCostFunctor, 2, 4, 3, 3,
                                            CameraModel::num_params>(
        new GeometricCostFunctor(point2D)));
  }

  template <typename T>
  bool operator()(const T* const qvec, const T* const tvec,
                  const T* const point3D, const T* const camera_params,
                  T* residuals) const {
    WorldToPixel<CameraModel>(camera_params, qvec, tvec, point3D, residuals);
    residuals[0] -= T(point2D_(0));
    residuals[1] -= T(point2D_(1));
    return true;
  }

 private:
  const Eigen::Vector2d point2D_;
};

// Bundle adjustment cost function for variable point and calibration
// parameters, and fixed camera pose.
template <typename CameraModel>
struct GeometricConstantPoseCostFunctor {
  GeometricConstantPoseCostFunctor(const colmap::Rigid3d& cam_from_world,
                                   const Eigen::Vector2d& point2D)
      : cam_from_world_(cam_from_world), point2D_(point2D) {}

  static ceres::CostFunction* Create(const colmap::Rigid3d& cam_from_world,
                                     const Eigen::Vector2d& point2D) {
    return (new ceres::AutoDiffCostFunction<GeometricConstantPoseCostFunctor,
                                            2, 3, CameraModel::num_params>(
        new GeometricConstantPoseCostFunctor(cam_from_world, point2D)));
  }

  template <typename T>
  bool operator()(const T* const point3D, const T* const camera_params,
                  T* residuals) const {
    const Eigen::Matrix<T, 7, 1> cam_from_world =
        cam_from_world_.params.cast<T>();
    WorldToPixel<CameraModel>(camera_params, cam_from_world.data(),
                              cam_from_world.data() + 4, point3D, residuals);
    residuals[0] -= T(point2D_(0));
    residuals[1] -= T(point2D_(1));
    return true;
  }

 private:
  const colmap::Rigid3d cam_from_world_;
  const Eigen::Vector2d point2D_;
};

/*******************************************************************************
Initialization Wrappers: (resolving camera model templates)
*******************************************************************************/

inline ceres::CostFunction* CreateGeometricCostFunctor(
    colmap::CameraModelId camera_model_id, const Eigen::Vector2d& point2D) {
  switch (camera_model_id) {
#define CAMERA_MODEL_CASE(CameraModel)                                    \
  case colmap::CameraModel::model_id:                                     \
    return GeometricCostFunctor<colmap::CameraModel>::Create(point2D);    \
    break;
    PIXSFM_CAMERA_MODEL_SWITCH_CASES
#undef CAMERA_MODEL_CASE
  }
  return nullptr;
}

inline ceres::CostFunction* CreateGeometricConstantPoseCostFunctor(
    colmap::CameraModelId camera_model_id,
    const colmap::Rigid3d& cam_from_world, const Eigen::Vector2d& point2D) {
  switch (camera_model_id) {
#define CAMERA_MODEL_CASE(CameraModel)                                     \
  case colmap::CameraModel::model_id:                                      \
    return GeometricConstantPoseCostFunctor<colmap::CameraModel>::Create(  \
        cam_from_world, point2D);                                          \
    break;
    PIXSFM_CAMERA_MODEL_SWITCH_CASES
#undef CAMERA_MODEL_CASE
  }
  return nullptr;
}

}  // namespace pixsfm
