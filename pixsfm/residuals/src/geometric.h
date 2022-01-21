#pragma once

#include <ceres/ceres.h>
#include <colmap/base/cost_functions.h>
#include <colmap/base/projection.h>
#include <colmap/util/types.h>

#include "base/src/projection.h"

namespace pixsfm {

/*******************************************************************************
Initialization Wrappers: (resolving camera model templates)
*******************************************************************************/

ceres::CostFunction* CreateGeometricCostFunctor(
    int camera_model_id, const Eigen::Vector2d& point2D) {
  switch (camera_model_id) {
#define CAMERA_MODEL_CASE(CameraModel)                                        \
  case colmap::CameraModel::kModelId:                                         \
    return colmap::BundleAdjustmentCostFunction<colmap::CameraModel>::Create( \
        point2D);                                                             \
    break;
    CAMERA_MODEL_SWITCH_CASES
#undef CAMERA_MODEL_CASE
  }
}

ceres::CostFunction* CreateGeometricConstantPoseCostFunctor(
    int camera_model_id, const Eigen::Vector4d& qvec,
    const Eigen::Vector3d& tvec, const Eigen::Vector2d& point2D) {
  switch (camera_model_id) {
#define CAMERA_MODEL_CASE(CameraModel)                       \
  case colmap::CameraModel::kModelId:                        \
    return colmap::BundleAdjustmentConstantPoseCostFunction< \
        colmap::CameraModel>::Create(qvec, tvec, point2D);   \
    break;
    CAMERA_MODEL_SWITCH_CASES
#undef CAMERA_MODEL_CASE
  }
}

}  // namespace pixsfm