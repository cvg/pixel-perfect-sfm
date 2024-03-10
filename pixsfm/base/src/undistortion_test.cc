// Check undistortion
#define TEST_NAME "base/undistortion"
#include <gtest/gtest.h>

#include <colmap/image/undistortion.h>

#include "base/src/undistortion.h"

namespace pixsfm {

// Check if new CamFromImg are correct
template <typename CameraModel>
void TestCamToCamFromImg(const std::vector<double> params, const Eigen::Vector3d& uvw) {
  double u, v, x, y;
  CameraModel::ImgFromCam(params.data(), uvw.x(), uvw.y(), uvw.z(), &x, &y);
  Eigen::Vector2d xxyy = colmap::CameraModelImgFromCam(CameraModel::model_id, params, uvw);
  EXPECT_EQ(x, xxyy.x());
  EXPECT_EQ(y, xxyy.y());
  UndistortionAutodiffModel<CameraModel>::ImageToCam(params.data(), x, y, &u,
                                                       &v);
  CHECK_LT(std::abs(u - uvw.x()), 1e-6);
  CHECK_LT(std::abs(v - uvw.y()), 1e-6);
  CHECK_LT(std::abs(v - uvw.z()), 1e-6);
}

template <typename CameraModel>
void TestUndistortion(const std::vector<double> params, const double u0,
                      const double v0) {
  double uc, vc, up, vp;
  uc = u0;
  vc = v0;
  up = u0;
  vp = v0;
  CameraModel::IterativeUndistortion(params.data(), &uc, &vc);
  CeresIterativeUndistortion<CameraModel>(params.data(), &up, &vp);
  CHECK_NEAR(uc, up, 1.0e-6);
  CHECK_NEAR(vc, vp, 1.0e-6);

  double du, dv;
  CameraModel::Distortion(params.data(), u0, v0, &du, &dv);
  double uu = u0 + du;
  double vv = v0 + dv;
  CeresIterativeUndistortion<CameraModel>(params.data(), &uu, &vv);
  CHECK_NEAR(u0, uu, 1.0e-6);
  CHECK_NEAR(v0, vv, 1.0e-6);
}

template <typename CameraModel>
void TestModel(const std::vector<double> params) {
  for (double u = -0.5; u <= 0.5; u += 0.1) {
    for (double v = -0.5; v <= 0.5; v += 0.1) {
        for (double w = -0.5; w <= 0.5; w += 0.1) {
            Eigen::Vector3d uvw{u,v,w};
            TestCamToCamFromImg<CameraModel>(params, uvw);
        }
    }
  }
}

template <typename CameraModel>
void TestUndistortion(const std::vector<double> params) {
  for (double u = -0.5; u <= 0.5; u += 0.1) {
    for (double v = -0.5; v <= 0.5; v += 0.1) {
      TestUndistortion<CameraModel>(params, u, v);
    }
  }
}

TEST(SimplePinhole, Nominal) {
  std::vector<double> params = {655.123, 386.123, 511.123};
  TestModel<colmap::SimplePinholeCameraModel>(params);
}

TEST(Pinhole, Nominal) {
  std::vector<double> params = {651.123, 655.123, 386.123, 511.123};
  TestModel<colmap::PinholeCameraModel>(params);
}

TEST(SimpleRadial, Nominal) {
  std::vector<double> params = {651.123, 386.123, 511.123, 0};
  TestModel<colmap::SimpleRadialCameraModel>(params);
  TestUndistortion<colmap::SimpleRadialCameraModel>(params);
  params[3] = 0.1;
  TestModel<colmap::SimpleRadialCameraModel>(params);
  TestUndistortion<colmap::SimpleRadialCameraModel>(params);
}

TEST(Radial, Nominal) {
  std::vector<double> params = {651.123, 386.123, 511.123, 0, 0};
  TestModel<colmap::RadialCameraModel>(params);
  TestUndistortion<colmap::RadialCameraModel>(params);
  params[3] = 0.1;
  TestModel<colmap::RadialCameraModel>(params);
  TestUndistortion<colmap::RadialCameraModel>(params);
  params[3] = 0.05;
  TestModel<colmap::RadialCameraModel>(params);
  TestUndistortion<colmap::RadialCameraModel>(params);
  params[4] = 0.03;
  TestModel<colmap::RadialCameraModel>(params);
  TestUndistortion<colmap::RadialCameraModel>(params);
}

}  // namespace pixsfm
