#pragma once

#include <ceres/ceres.h>
#include <colmap/geometry/rigid3.h>
#include <colmap/sensor/models.h>
#include <colmap/scene/projection.h>
#include <colmap/estimators/cost_functions.h>
#include <colmap/util/types.h>

#include "base/src/projection.h"

namespace pixsfm {

/*******************************************************************************
Initialization Wrappers: (resolving camera model templates)
*******************************************************************************/

ceres::CostFunction* CreateGeometricCostFunctor(
    const colmap::CameraModelId model_id, const Eigen::Vector2d& point2D) {
  switch (model_id) {
#define CAMERA_MODEL_CASE(CameraModel)                                   \
  case colmap::CameraModel::model_id:                                    \
    return colmap::ReprojErrorCostFunction<colmap::CameraModel>::Create( \
        point2D);                                                        \
    break;
    CAMERA_MODEL_SWITCH_CASES
#undef CAMERA_MODEL_CASE
  }
}

ceres::CostFunction* CreateGeometricConstantPoseCostFunctor(
    const colmap::CameraModelId model_id, const colmap::Rigid3d& cam_from_world,
    const Eigen::Vector2d& point2D) {
  switch (model_id) {
#define CAMERA_MODEL_CASE(CameraModel)                           \
  case colmap::CameraModel::model_id:                            \
    return colmap::ReprojErrorConstantPoseCostFunction<          \
        colmap::CameraModel>::Create(cam_from_world, point2D);   \
    break;
    CAMERA_MODEL_SWITCH_CASES
#undef CAMERA_MODEL_CASE
  }
}

}  // namespace pixsfm