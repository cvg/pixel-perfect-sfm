
#pragma once

#include <colmap/geometry/rigid3.h>
#include <colmap/scene/camera.h>
#include <colmap/sensor/models.h>

#include <ceres/ceres.h>

#include <Eigen/Geometry>

#include "base/src/colmap_compat.h"
#include "base/src/interpolation.h"
#include "util/src/types.h"

#include "base/src/undistortion.h"

namespace pixsfm {

// Quaternions (qvec) are stored in Eigen order (x, y, z, w), matching
// colmap::Rigid3d::params.

// Deep tested, equal to colmap::CalculateDepth, but templated
template <typename T>
inline void CalculateDepth(const T* const qvec, const T* const tvec,
                           const T* const point3D, T* depth) {
  const Eigen::Map<const Eigen::Quaternion<T>> q(qvec);
  const Eigen::Map<const Eigen::Matrix<T, 3, 1>> xyz(point3D);
  *depth = (q * xyz)[2] + tvec[2];
}

// Deep tested
template <typename CameraModel, typename T>
void PixelToWorld(const T* camera_params, const T* qvec, const T* tvec,
                  const T x, const T y, const T* depth, T* xyz) {
  T local_xyz[3];
  UndistortionAutodiffModel<CameraModel>::ImageToWorld(
      camera_params, x, y, &local_xyz[0], &local_xyz[1]);

  local_xyz[2] = T(1.0);
  for (int i = 0; i < 3; i++) {
    local_xyz[i] = local_xyz[i] * depth[0] - tvec[i];
  }
  const Eigen::Map<const Eigen::Quaternion<T>> q(qvec);
  Eigen::Map<Eigen::Matrix<T, 3, 1>> map(xyz);
  map = q.conjugate() * Eigen::Map<const Eigen::Matrix<T, 3, 1>>(local_xyz);
}

// Deep tested
template <typename CameraModel, typename T>
inline void WorldToPixel(const T* camera_params, const T* qvec, const T* tvec,
                         const T* xyz, T* xy) {
  const Eigen::Map<const Eigen::Quaternion<T>> q(qvec);
  const Eigen::Matrix<T, 3, 1> projection =
      q * Eigen::Map<const Eigen::Matrix<T, 3, 1>>(xyz) +
      Eigen::Map<const Eigen::Matrix<T, 3, 1>>(tvec);

  // Project to image plane. Like COLMAP 3.8, no cheirality check is done here.
  CameraModel::ImgFromCam(camera_params, projection[0], projection[1],
                          projection[2], &xy[0], &xy[1],
                          /*check_cheirality=*/false);
}

inline void WorldToPixel(const colmap::Camera& camera,
                         const colmap::Rigid3d& cam_from_world,
                         const Eigen::Vector3d& xyz, double* xy) {
  const double* qvec = cam_from_world.params.data();
  const double* tvec = cam_from_world.params.data() + 4;
  switch (camera.model_id) {
#define CAMERA_MODEL_CASE(CameraModel)                                      \
  case colmap::CameraModel::model_id:                                       \
    WorldToPixel<colmap::CameraModel>(camera.params.data(), qvec, tvec,     \
                                      xyz.data(), xy);                      \
    break;
    PIXSFM_CAMERA_MODEL_SWITCH_CASES
#undef CAMERA_MODEL_CASE
  }
}

}  // namespace pixsfm
