#define TEST_NAME "base/interpolation"
#include <colmap/util/testing.h>
#include <colmap/util/timer.h>

#include <ceres/ceres.h>

#include "base/src/irls_optim.h"

namespace pixsfm {

template <int CHANNELS>
void TestSimilarity(int n_descriptors,
                    InterpolationConfig interpolation_config) {
  std::vector<DescriptorMatrixXd> descriptors(n_descriptors);
  std::vector<DescriptorMatrixd<1, CHANNELS>> descriptors_t(n_descriptors);
  Eigen::MatrixXd track_matrix(CHANNELS, n_descriptors);
  int i = 0;

  for (auto& desc : descriptors) {
    desc = DescriptorMatrixXd::Random(1, CHANNELS);
    track_matrix.col(i) = Eigen::Map<Eigen::VectorXd>(desc.data(), CHANNELS);
    descriptors_t[i] = Eigen::Map<Eigen::VectorXd>(desc.data(), CHANNELS);
    ++i;
  }

  std::unique_ptr<ceres::LossFunction> loss_function;
  loss_function.reset(new ceres::CauchyLoss(0.25));

  colmap::Timer timer;
  timer.Start();
  DescriptorMatrixXd robust_mean1 = RobustMeanIRLS<-1, -1>(
      descriptors, loss_function.get(), 100, interpolation_config);
  timer.PrintSeconds();
  timer.Restart();
  DescriptorMatrixXd robust_mean2 = RobustMeanIRLS<CHANNELS, 1>(
      descriptors_t, loss_function.get(), 100, interpolation_config);
  timer.PrintSeconds();

  for (int i = 0; i < CHANNELS; i++) {
    BOOST_CHECK_LT(std::abs(robust_mean1(0, i) - robust_mean2(i)), 1.0e-8);
  }
}

BOOST_AUTO_TEST_CASE(TestSimilar) {
  InterpolationConfig interpolation_config;
  interpolation_config.l2_normalize = false;

  TestSimilarity<128>(10, interpolation_config);
  TestSimilarity<128>(100, interpolation_config);
  TestSimilarity<128>(1000, interpolation_config);

  TestSimilarity<3>(10, interpolation_config);
  TestSimilarity<3>(100, interpolation_config);
  TestSimilarity<3>(1000, interpolation_config);
}

}  // namespace pixsfm