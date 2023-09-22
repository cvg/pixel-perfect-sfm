// Check undistortion
#define TEST_NAME "base/undistortion"
#include <colmap/util/testing.h>

#include <colmap/image/undistortion.h>

#include "base/src/undistortion.h"

namespace pixsfm {

// Check if new ImageToWorld are correct
template <typename CameraModel>
void TestWorldToImageToWorld(const std::vector<double> params, const double u0,
                             const double v0) {
  double u, v, x, y, xx, yy;
  CameraModel::WorldToImage(params.data(), u0, v0, &x, &y);
  colmap::CameraModelWorldToImage(CameraModel::model_id, params, u0, v0, &xx,
                                  &yy);
  BOOST_CHECK_EQUAL(x, xx);
  BOOST_CHECK_EQUAL(y, yy);
  UndistortionAutodiffModel<CameraModel>::ImageToWorld(params.data(), x, y, &u,
                                                       &v);
  BOOST_CHECK_LT(std::abs(u - u0), 1e-6);
  BOOST_CHECK_LT(std::abs(v - v0), 1e-6);
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
  BOOST_CHECK_CLOSE(uc, up, 1.0e-6);
  BOOST_CHECK_CLOSE(vc, vp, 1.0e-6);

  double du, dv;
  CameraModel::Distortion(params.data(), u0, v0, &du, &dv);
  double uu = u0 + du;
  double vv = v0 + dv;
  CeresIterativeUndistortion<CameraModel>(params.data(), &uu, &vv);
  BOOST_CHECK_CLOSE(u0, uu, 1.0e-6);
  BOOST_CHECK_CLOSE(v0, vv, 1.0e-6);
}

template <typename CameraModel>
void TestModel(const std::vector<double> params) {
  for (double u = -0.5; u <= 0.5; u += 0.1) {
    for (double v = -0.5; v <= 0.5; v += 0.1) {
      TestWorldToImageToWorld<CameraModel>(params, u, v);
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

BOOST_AUTO_TEST_CASE(TestSimplePinhole) {
  std::vector<double> params = {655.123, 386.123, 511.123};
  TestModel<colmap::SimplePinholeCameraModel>(params);
}

BOOST_AUTO_TEST_CASE(TestPinhole) {
  std::vector<double> params = {651.123, 655.123, 386.123, 511.123};
  TestModel<colmap::PinholeCameraModel>(params);
}

BOOST_AUTO_TEST_CASE(TestSimpleRadial) {
  std::vector<double> params = {651.123, 386.123, 511.123, 0};
  TestModel<colmap::SimpleRadialCameraModel>(params);
  TestUndistortion<colmap::SimpleRadialCameraModel>(params);
  params[3] = 0.1;
  TestModel<colmap::SimpleRadialCameraModel>(params);
  TestUndistortion<colmap::SimpleRadialCameraModel>(params);
}

BOOST_AUTO_TEST_CASE(TestRadial) {
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
