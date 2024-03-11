#include <gtest/gtest.h>

#include "base/src/projection.h"

namespace pixsfm {

template <typename CameraModel>
void TestWorldToPixelToWorld(const double* params) {
  double qvec[4] = {1, 0, 0, 0};
  double tvec[3] = {0, 0, 0};
  double point3D[3] = {0, 0, 1};
  double depth;
  double xy[2];
  double world[3];
  WorldToPixel<CameraModel>(params, qvec, tvec, point3D, xy);
  CalculateDepth(qvec, tvec, point3D, &depth);
  PixelToWorld<CameraModel>(params, qvec, tvec, xy[0], xy[1], &depth, world);
  for (int i = 0; i < 3; i++) {
    EXPECT_NEAR(point3D[i], world[i], 1.0e-6);
  }
}

TEST(Projection, SimplePinhole) {
  std::vector<double> params = {655.123, 386.123, 511.123};
  TestWorldToPixelToWorld<colmap::SimplePinholeCameraModel>(params.data());
}

TEST(Projection, Radial) {
  std::vector<double> params = {651.123, 386.123, 511.123, 0, 0};
  TestWorldToPixelToWorld<colmap::RadialCameraModel>(params.data());
  params[3] = 0.1;
  TestWorldToPixelToWorld<colmap::RadialCameraModel>(params.data());
  params[3] = 0.05;
  TestWorldToPixelToWorld<colmap::RadialCameraModel>(params.data());
  params[4] = 0.03;
  TestWorldToPixelToWorld<colmap::RadialCameraModel>(params.data());
}

}  // namespace pixsfm
