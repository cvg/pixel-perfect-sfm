
#pragma once

#include <colmap/base/camera_models.h>
#include <colmap/base/image.h>
#include <colmap/base/projection.h>
#include <colmap/util/math.h>

#include <ceres/ceres.h>
#include <ceres/rotation.h>

#include "base/src/interpolation.h"
#include "util/src/types.h"

#include "base/src/undistortion.h"

namespace pixsfm {

// Deep tested, equal to colmap::CalculateDepth, but templated
template <typename T>
inline void CalculateDepth(const T* const qvec, const T* const tvec,
                           const T* const point3D, T* depth) {
  T rotation_matrix[3 * 3];
  ceres::QuaternionToRotation(qvec, rotation_matrix);

  *depth = tvec[2];

  T norm = T(0.0);
  for (int i = 0; i < 3; i++) {
    *depth += rotation_matrix[2 * 3 + i] * point3D[i];
    norm += rotation_matrix[i * 3 + 2] * rotation_matrix[i * 3 + 2];
  }

  *depth /= ceres::sqrt(norm);

  // const double proj_z = proj_matrix.row(2).dot(point3D.homogeneous());
  // return proj_z * proj_matrix.col(2).norm();
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
  // local_xyz[2] = depth[0] - tvec[2];
  Eigen::Quaternion<T> q(qvec[0], qvec[1], qvec[2], qvec[3]);
  Eigen::Map<Eigen::Matrix<T, 3, 1>> map(xyz);
  map = q.conjugate() * Eigen::Map<const Eigen::Matrix<T, 3, 1>>(local_xyz);
  // ceres::QuaternionRotatePoint(q, local_xyz, xyz);
}

// Deep tested
template <typename CameraModel, typename T>
inline void WorldToPixel(const T* camera_params, const T* qvec, const T* tvec,
                         const T* xyz, T* xy) {
  T projection[3];
  ceres::QuaternionRotatePoint(qvec, xyz, projection);
  projection[0] += tvec[0];
  projection[1] += tvec[1];
  projection[2] += tvec[2];

  // Project to image plane.
  projection[0] /= projection[2];  // u
  projection[1] /= projection[2];  // v

  CameraModel::WorldToImage(camera_params, projection[0], projection[1], &xy[0],
                            &xy[1]);
}

inline void WorldToPixel(const colmap::Camera& camera,
                         const Eigen::Vector4d& qvec,
                         const Eigen::Vector3d& tvec,
                         const Eigen::Vector3d& xyz, double* xy) {
  switch (camera.ModelId()) {
#define CAMERA_MODEL_CASE(CameraModel)                                  \
  case colmap::CameraModel::kModelId:                                   \
    WorldToPixel<colmap::CameraModel>(camera.ParamsData(), qvec.data(), \
                                      tvec.data(), xyz.data(), xy);     \
    break;
    CAMERA_MODEL_SWITCH_CASES
#undef CAMERA_MODEL_CASE
  }
}

}  // namespace pixsfm