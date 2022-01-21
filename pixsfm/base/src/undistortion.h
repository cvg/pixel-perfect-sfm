#pragma once

#include <colmap/base/camera_models.h>
#include <colmap/base/image.h>
#include <colmap/base/projection.h>
#include <colmap/util/math.h>

#include <ceres/ceres.h>
#include <ceres/rotation.h>

// Autodiff compatible undistortion (COLMAP does not support this atm)
template <typename CameraModel, typename T>
inline void CeresIterativeUndistortion(const T* params, T* u, T* v) {
  // Parameters for Newton iteration using numerical differentiation with
  // central differences, 100 iterations should be enough even for complex
  // camera models with higher order terms.
  const size_t kNumIterations = 100;
  const T kMaxStepNorm = T(1e-10);
  const T kRelStepSize = T(1e-6);

  Eigen::Matrix<T, 2, 2> J;
  const Eigen::Matrix<T, 2, 1> x0(*u, *v);
  Eigen::Matrix<T, 2, 1> x(*u, *v);
  Eigen::Matrix<T, 2, 1> dx;
  Eigen::Matrix<T, 2, 1> dx_0b;
  Eigen::Matrix<T, 2, 1> dx_0f;
  Eigen::Matrix<T, 2, 1> dx_1b;
  Eigen::Matrix<T, 2, 1> dx_1f;

  const T epsilon = T(std::numeric_limits<double>::epsilon());

  for (size_t i = 0; i < kNumIterations; ++i) {
    T step0 = ceres::abs(kRelStepSize * x(0));
    T step1 = ceres::abs(kRelStepSize * x(1));

    if (step0 < epsilon) {
      step0 = epsilon;
    }
    if (step1 < epsilon) {
      step1 = epsilon;
    }
    // const T step0 = T(std::max(std::numeric_limits<double>::epsilon(),
    //                               std::abs(kRelStepSize * double(x(0)))));
    // const T step1 = T(std::max(std::numeric_limits<double>::epsilon(),
    //                               std::abs(kRelStepSize * double(x(1)))));
    CameraModel::Distortion(params, x(0), x(1), &dx(0), &dx(1));
    CameraModel::Distortion(params, x(0) - step0, x(1), &dx_0b(0), &dx_0b(1));
    CameraModel::Distortion(params, x(0) + step0, x(1), &dx_0f(0), &dx_0f(1));
    CameraModel::Distortion(params, x(0), x(1) - step1, &dx_1b(0), &dx_1b(1));
    CameraModel::Distortion(params, x(0), x(1) + step1, &dx_1f(0), &dx_1f(1));
    J(0, 0) = T(1.0) + (dx_0f(0) - dx_0b(0)) / (T(2.0) * step0);
    J(0, 1) = (dx_1f(0) - dx_1b(0)) / (T(2.0) * step1);
    J(1, 0) = (dx_0f(1) - dx_0b(1)) / (T(2.0) * step0);
    J(1, 1) = T(1.0) + (dx_1f(1) - dx_1b(1)) / (T(2.0) * step1);
    const Eigen::Matrix<T, 2, 1> step_x = J.inverse() * (x + dx - x0);
    x -= step_x;
    if (step_x.squaredNorm() < kMaxStepNorm) {
      break;
    }
  }

  *u = x(0);
  *v = x(1);
}

template <typename CameraModel>
struct UndistortionAutodiffModel {
  template <typename T>
  static inline void ImageToWorld(const T* params, const T x, const T y, T* u,
                                  T* v);
};

template <>
template <typename T>
void UndistortionAutodiffModel<colmap::SimplePinholeCameraModel>::ImageToWorld(
    const T* params, const T x, const T y, T* u, T* v) {
  const T f = params[0];
  const T c1 = params[1];
  const T c2 = params[2];

  *u = (x - c1) / f;
  *v = (y - c2) / f;
}

template <>
template <typename T>
void UndistortionAutodiffModel<colmap::PinholeCameraModel>::ImageToWorld(
    const T* params, const T x, const T y, T* u, T* v) {
  const T f1 = params[0];
  const T f2 = params[1];
  const T c1 = params[2];
  const T c2 = params[3];

  *u = (x - c1) / f1;
  *v = (y - c2) / f2;
}

template <>
template <typename T>
void UndistortionAutodiffModel<colmap::SimpleRadialCameraModel>::ImageToWorld(
    const T* params, const T x, const T y, T* u, T* v) {
  const T f = params[0];
  const T c1 = params[1];
  const T c2 = params[2];

  // Lift points to normalized plane
  *u = (x - c1) / f;
  *v = (y - c2) / f;

  CeresIterativeUndistortion<colmap::SimpleRadialCameraModel>(&params[3], u, v);
}

template <>
template <typename T>
void UndistortionAutodiffModel<colmap::RadialCameraModel>::ImageToWorld(
    const T* params, const T x, const T y, T* u, T* v) {
  const T f = params[0];
  const T c1 = params[1];
  const T c2 = params[2];

  // Lift points to normalized plane
  *u = (x - c1) / f;
  *v = (y - c2) / f;

  CeresIterativeUndistortion<colmap::RadialCameraModel>(&params[3], u, v);
}

template <>
template <typename T>
void UndistortionAutodiffModel<colmap::OpenCVCameraModel>::ImageToWorld(
    const T* params, const T x, const T y, T* u, T* v) {
  const T f1 = params[0];
  const T f2 = params[1];
  const T c1 = params[2];
  const T c2 = params[3];

  // Lift points to normalized plane
  *u = (x - c1) / f1;
  *v = (y - c2) / f2;

  CeresIterativeUndistortion<colmap::OpenCVCameraModel>(&params[4], u, v);
}