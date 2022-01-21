#define TEST_NAME "base/interpolation"
#include <colmap/util/testing.h>

#include <ceres/ceres.h>

#include "base/src/grid2d.h"
#include "base/src/interpolation.h"

#include <random>

namespace pixsfm {

// Source:
// https://github.com/ceres-solver/ceres-solver/blob/master/internal/ceres/cubic_interpolation_test.cc
class TestInterpolation2D {
 public:
  // This class needs to have an Eigen aligned operator new as it contains
  // fixed-size Eigen types.
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  template <int kDataDimension>
  void RunPolynomialInterpolationTest(InterpolationConfig& config,
                                      const Eigen::Matrix3d& coeff) {
    values_.reset(new double[kNumRows * kNumCols * kDataDimension]);
    coeff_ = coeff;
    double* v = values_.get();
    for (int r = 0; r < kNumRows; ++r) {
      for (int c = 0; c < kNumCols; ++c) {
        for (int dim = 0; dim < kDataDimension; ++dim) {
          *v++ = (dim * dim + 1) * EvaluateF(r, c);
        }
      }
    }

    Grid2D<double, kDataDimension> grid(values_.get(), 0, kNumRows, 0,
                                        kNumCols);
    PixelInterpolator<Grid2D<double, kDataDimension>> interpolator(config,
                                                                   grid);

    for (int j = 0; j < kNumRowSamples; ++j) {
      const double r = 1.0 + 7.0 / (kNumRowSamples - 1) * j;
      for (int k = 0; k < kNumColSamples; ++k) {
        const double c = 1.0 + 7.0 / (kNumColSamples - 1) * k;
        double f[kDataDimension], dfdr[kDataDimension], dfdc[kDataDimension];
        interpolator.Evaluate(r, c, f, dfdr, dfdc);
        for (int dim = 0; dim < kDataDimension; ++dim) {
          BOOST_CHECK_LT(std::abs(f[dim] - (dim * dim + 1) * EvaluateF(r, c)),
                         1.0e-8);
          BOOST_CHECK_LT(
              std::abs(dfdr[dim] - (dim * dim + 1) * EvaluatedFdr(r, c)),
              1.0e-8);
          BOOST_CHECK_LT(
              std::abs(dfdc[dim] - (dim * dim + 1) * EvaluatedFdc(r, c)),
              1.0e-8);
        }
      }
    }
  }

 private:
  double EvaluateF(double r, double c) {
    Eigen::Vector3d x;
    x(0) = r;
    x(1) = c;
    x(2) = 1;
    return x.transpose() * coeff_ * x;
  }

  double EvaluatedFdr(double r, double c) {
    Eigen::Vector3d x;
    x(0) = r;
    x(1) = c;
    x(2) = 1;
    return (coeff_.row(0) + coeff_.col(0).transpose()) * x;
  }

  double EvaluatedFdc(double r, double c) {
    Eigen::Vector3d x;
    x(0) = r;
    x(1) = c;
    x(2) = 1;
    return (coeff_.row(1) + coeff_.col(1).transpose()) * x;
  }

  Eigen::Matrix3d coeff_;
  static constexpr int kNumRows = 10;
  static constexpr int kNumCols = 10;
  static constexpr int kNumRowSamples = 100;
  static constexpr int kNumColSamples = 100;
  std::unique_ptr<double[]> values_;
};

BOOST_AUTO_TEST_CASE(TestZeroFunction) {
  Eigen::Matrix3d coeff = Eigen::Matrix3d::Zero();
  InterpolationConfig configs[2];
  configs[0].mode = InterpolatorType::BICUBIC;
  configs[0].l2_normalize = false;
  configs[1].mode = InterpolatorType::BILINEAR;
  configs[1].l2_normalize = false;

  TestInterpolation2D test;
  for (int i = 0; i < 2; i++) {
    test.RunPolynomialInterpolationTest<1>(configs[i], coeff);
    test.RunPolynomialInterpolationTest<2>(configs[i], coeff);
    test.RunPolynomialInterpolationTest<3>(configs[i], coeff);
  }
}

BOOST_AUTO_TEST_CASE(TestDegree00Function) {
  Eigen::Matrix3d coeff = Eigen::Matrix3d::Zero();
  coeff(2, 2) = 1.0;
  InterpolationConfig configs[3];
  configs[0].mode = InterpolatorType::BICUBIC;
  configs[0].l2_normalize = false;
  configs[1].mode = InterpolatorType::BILINEAR;
  configs[1].l2_normalize = false;
  configs[2].mode = InterpolatorType::NEARESTNEIGHBOR;
  configs[2].l2_normalize = false;

  TestInterpolation2D test;
  for (int i = 0; i < 3; i++) {
    test.RunPolynomialInterpolationTest<1>(configs[i], coeff);
    test.RunPolynomialInterpolationTest<2>(configs[i], coeff);
    test.RunPolynomialInterpolationTest<3>(configs[i], coeff);
  }
}

BOOST_AUTO_TEST_CASE(TestDegree01Function) {
  Eigen::Matrix3d coeff = Eigen::Matrix3d::Zero();
  coeff(2, 2) = 1.0;
  coeff(0, 2) = 0.1;
  coeff(2, 0) = 0.1;
  InterpolationConfig configs[2];
  configs[0].mode = InterpolatorType::BICUBIC;
  configs[0].l2_normalize = false;
  configs[1].mode = InterpolatorType::BILINEAR;
  configs[1].l2_normalize = false;

  TestInterpolation2D test;
  for (int i = 0; i < 2; i++) {
    test.RunPolynomialInterpolationTest<1>(configs[i], coeff);
    test.RunPolynomialInterpolationTest<2>(configs[i], coeff);
    test.RunPolynomialInterpolationTest<3>(configs[i], coeff);
  }
}

BOOST_AUTO_TEST_CASE(TestDegree10Function) {
  Eigen::Matrix3d coeff = Eigen::Matrix3d::Zero();
  coeff(2, 2) = 1.0;
  coeff(0, 1) = 0.1;
  coeff(1, 0) = 0.1;
  InterpolationConfig configs[2];
  configs[0].mode = InterpolatorType::BICUBIC;
  configs[0].l2_normalize = false;
  configs[1].mode = InterpolatorType::BILINEAR;
  configs[1].l2_normalize = false;

  TestInterpolation2D test;
  for (int i = 0; i < 2; i++) {
    test.RunPolynomialInterpolationTest<1>(configs[i], coeff);
    test.RunPolynomialInterpolationTest<2>(configs[i], coeff);
    test.RunPolynomialInterpolationTest<3>(configs[i], coeff);
  }
}

BOOST_AUTO_TEST_CASE(TestDegree11Function) {
  Eigen::Matrix3d coeff = Eigen::Matrix3d::Zero();
  coeff(2, 2) = 1.0;
  coeff(0, 1) = 0.1;
  coeff(1, 0) = 0.1;
  coeff(0, 2) = 0.2;
  coeff(2, 0) = 0.2;
  InterpolationConfig configs[2];
  configs[0].mode = InterpolatorType::BICUBIC;
  configs[0].l2_normalize = false;
  configs[1].mode = InterpolatorType::BILINEAR;
  configs[1].l2_normalize = false;

  TestInterpolation2D test;
  for (int i = 0; i < 2; i++) {
    test.RunPolynomialInterpolationTest<1>(configs[i], coeff);
    test.RunPolynomialInterpolationTest<2>(configs[i], coeff);
    test.RunPolynomialInterpolationTest<3>(configs[i], coeff);
  }
}

BOOST_AUTO_TEST_CASE(TestL2Normalize) {
  InterpolationConfig interpolation_config;
  interpolation_config.l2_normalize = true;

  const double values[] = {1.0, 5.0, 2.0, 10.0, 2.0, 6.0, 3.0, 5.0,
                           1.0, 2.0, 2.0, 2.0,  2.0, 2.0, 3.0, 1.0};
  Grid2D<double, 2> grid(values, 0, 2, 0, 4);

  double f[2], dfdr[2], dfdc[2];

  PixelInterpolator<Grid2D<double, 2>> interpolator(interpolation_config, grid);

  interpolator.Evaluate(0.5, 2.5, f, dfdr, dfdc);
  BOOST_CHECK_LT(std::abs(1.0 - f[0] * f[0] - f[1] * f[1]), 1.0e-10);

  interpolator.Evaluate(1.5, 1.5, f, dfdr, dfdc);
  BOOST_CHECK_LT(std::abs(1.0 - f[0] * f[0] - f[1] * f[1]), 1.0e-10);

  interpolator.Evaluate(0.0, 3.0, f, dfdr, dfdc);
  BOOST_CHECK_LT(std::abs(1.0 - f[0] * f[0] - f[1] * f[1]), 1.0e-10);
}

// Assumes stacked format
Eigen::VectorXd compute_mean_per_channel(double* data, int channels,
                                         int n_nodes) {
  Eigen::VectorXd mean(channels);
  for (int i = 0; i < channels; i++) {
    mean(i) = 0.0;
    for (int j = 0; j < n_nodes; j++) {
      mean(i) += data[j * channels + i];
    }
    mean(i) /= n_nodes;
  }
  return mean;
}

// Assumes stacked format
Eigen::VectorXd compute_std_per_channel(double* data, int channels,
                                        int n_nodes) {
  Eigen::VectorXd std(channels);
  Eigen::VectorXd mean = compute_mean_per_channel(data, channels, n_nodes);
  for (int i = 0; i < channels; i++) {
    std(i) = 0.0;
    for (int j = 0; j < n_nodes; j++) {
      double d = (data[j * channels + i] - mean(i));
      std(i) += d * d;
    }
    std(i) = std::sqrt(std(i) / (n_nodes));
  }
  return std;
}

BOOST_AUTO_TEST_CASE(TestNCCNormalize) {
  InterpolationConfig interpolation_config;
  interpolation_config.l2_normalize = false;
  interpolation_config.ncc_normalize = true;

  interpolation_config.nodes.resize(4);
  interpolation_config.nodes[0] = {0.5, 0.5};
  interpolation_config.nodes[1] = {-0.5, 0.5};
  interpolation_config.nodes[2] = {0.5, -0.5};
  interpolation_config.nodes[3] = {-0.5, -0.5};

  const double values[] = {1.0, 5.0, 2.0, 10.0, 2.0, 6.0, 3.0, 5.0,
                           1.0, 2.0, 2.0, 2.0,  2.0, 2.0, 3.0, 1.0};
  Grid2D<double, 2> grid(values, 0, 2, 0, 4);

  double f[2 * 4];
  Eigen::VectorXd mean, stddev;

  PixelInterpolator<Grid2D<double, 2>> interpolator(interpolation_config, grid);

  double r = 0.5;
  double c = 2.5;
  interpolator.EvaluateNodes(r, c, f);

  mean = compute_mean_per_channel(f, 2, 4);
  stddev = compute_std_per_channel(f, 2, 4);

  BOOST_CHECK_LT(std::abs(mean(0)), 1.0e-8);
  BOOST_CHECK_LT(std::abs(mean(1)), 1.0e-8);

  BOOST_CHECK_LT(std::abs(stddev(0) - 1.0), 1.0e-8);
  BOOST_CHECK_LT(std::abs(stddev(1) - 1.0), 1.0e-8);
}
void TestInterpolationJetEvaluation(InterpolationConfig interpolation_config) {
  // clang-format off
    const double values[] = {1.0, 5.0, 2.0, 10.0, 2.0, 6.0, 3.0, 5.0,
                            1.0, 2.0, 2.0,  2.0, 2.0, 2.0, 3.0, 1.0};
  // clang-format on

  Grid2D<double, 2> grid(values, 0, 2, 0, 4);

  PixelInterpolator<Grid2D<double, 2>> interpolator(interpolation_config, grid);

  double f[2], dfdr[2], dfdc[2];
  const double r = 0.5;
  const double c = 2.5;
  interpolator.Evaluate(r, c, f, dfdr, dfdc);

  // Create a Jet with the same scalar part as x, so that the output
  // Jet will be evaluated at x.
  ceres::Jet<double, 4> r_jet;
  r_jet.a = r;
  r_jet.v(0) = 1.0;
  r_jet.v(1) = 1.1;
  r_jet.v(2) = 1.2;
  r_jet.v(3) = 1.3;

  ceres::Jet<double, 4> c_jet;
  c_jet.a = c;
  c_jet.v(0) = 2.0;
  c_jet.v(1) = 3.1;
  c_jet.v(2) = 4.2;
  c_jet.v(3) = 5.3;

  ceres::Jet<double, 4> f_jets[2];
  interpolator.Evaluate(r_jet, c_jet, f_jets);
  BOOST_CHECK_EQUAL(f_jets[0].a, f[0]);
  BOOST_CHECK_EQUAL(f_jets[1].a, f[1]);
  BOOST_CHECK_LT((f_jets[0].v - dfdr[0] * r_jet.v - dfdc[0] * c_jet.v).norm(),
                 1.0e-10);
  BOOST_CHECK_LT((f_jets[1].v - dfdr[1] * r_jet.v - dfdc[1] * c_jet.v).norm(),
                 1.0e-10);
}

BOOST_AUTO_TEST_CASE(TestBiCubicInterpolation) {
  InterpolationConfig interpolation_config;
  interpolation_config.mode = InterpolatorType::BILINEAR;
  interpolation_config.l2_normalize = false;
  TestInterpolationJetEvaluation(interpolation_config);
}

template <typename T>
void rands(T* m, size_t size) {
  std::random_device rd;
  std::mt19937 gen{rd()};
  std::uniform_real_distribution<double> dist(-1.0, 1.0);
  for (size_t i = 0; i < size; ++i) m[i] = static_cast<T>(dist(gen));
}
template <typename dtype, int CHANNELS>
void TestSimilarToCeres() {
  using Grid2D = Grid2D<dtype, CHANNELS>;
  int h = 10;
  int w = 10;

  std::vector<dtype> data(h * w * CHANNELS);
  rands(data.data(), data.size());

  Grid2D grid_2D(data.data(), 0, h, 0, w);

  double f[CHANNELS], dfdc[CHANNELS], dfdr[CHANNELS];
  double f2[CHANNELS], dfdc2[CHANNELS], dfdr2[CHANNELS];

  std::unique_ptr<Interpolator<Grid2D>> base_interpolator;
  base_interpolator.reset(new CeresBiCubicInterpolator<Grid2D>(grid_2D));
  std::unique_ptr<Interpolator<Grid2D>> simd_interpolator;
  simd_interpolator.reset(new BiCubicInterpolator<Grid2D>(grid_2D));

  for (int r = 0; r < 100; r++) {
    for (int c = 0; c < 100; c++) {
      // bench.run_benchmark(r/10.0,c/10.0, f, dfdc, dfdr);
      base_interpolator->Evaluate(r / 10.0, c / 10.0, f, dfdr, dfdc);
      simd_interpolator->Evaluate(r / 10.0, c / 10.0, f2, dfdr2, dfdc2);
      for (int i = 0; i < CHANNELS; i++) {
        BOOST_CHECK_LT(std::abs(f2[i] - f[i]), 1.0e-5);
        BOOST_CHECK_LT(std::abs(dfdr2[i] - dfdr[i]), 1.0e-5);
        BOOST_CHECK_LT(std::abs(dfdc2[i] - dfdc[i]), 1.0e-5);
      }
    }
  }
}

BOOST_AUTO_TEST_CASE(TestBiCubicSimilarCeres) {
  TestSimilarToCeres<float, 128>();
  TestSimilarToCeres<double, 128>();
  TestSimilarToCeres<half, 128>();
}

}  // namespace pixsfm
