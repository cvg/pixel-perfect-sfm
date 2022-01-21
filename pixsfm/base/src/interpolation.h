#pragma once

#include "_pixsfm/src/helpers.h"
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

#include <highfive/H5DataSet.hpp>
#include <highfive/H5DataSpace.hpp>
#include <highfive/H5File.hpp>
#include <highfive/H5Group.hpp>

#include <boost/serialization/vector.hpp>
#include <ceres/ceres.h>
#include <ceres/cubic_interpolation.h>
#include <exception>
#include <string>
#include <utility>
#include <vector>

#include "base/src/cubic_hermite_spline_simd.h"

#include "util/src/log_exceptions.h"
#include "util/src/types.h"

namespace pixsfm {

enum class InterpolatorType {
  BICUBIC = 0,
  BILINEAR,
  NEARESTNEIGHBOR,
  POLYGRADIENTFIELD,
  BICUBICGRADIENTFIELD,
  BICUBICCHAIN,
  CERES_BICUBIC
};

struct InterpolationConfig {
  InterpolationConfig() {}

  InterpolatorType mode = InterpolatorType::BICUBIC;
  bool l2_normalize = true;
  bool ncc_normalize = false;
  std::vector<std::array<double, 2>> nodes = {{0, 0}};
  bool fill_channel_differences = true;

  bool check_bounds = false;

  bool use_float_simd = false;  // by default we apply SIMD on doubles
};

template <typename T>
inline void NCCNormalizePerChannel(T* data, int channels, int Npixels) {
  T sum = T(0.0);
  for (int i = 0; i < Npixels; i++) {
    sum += data[i * channels];
  }
  // Eigen::Map<Eigen::Matrix<T, channels, 1>> vec(data);
  T mu = sum / T(Npixels);

  // Eigen::Matrix<T, channels, 1> vec_copy = vec;
  // T sigma = (vec_copy - mu).norm();
  T sigma = T(0.0);

  for (int i = 0; i < Npixels; i++) {
    sigma += (data[i * channels] - mu) * (data[i * channels] - mu);
  }

  sigma = ceres::sqrt(sigma / T(Npixels));

  sigma = sigma > T(0.0) ? sigma : T(1.0);

  for (int i = 0; i < Npixels; i++) {
    data[i * channels] = (data[i * channels] - mu) / sigma;
  }
}

// Suboptimal, strided memory access
template <typename T>
inline void NCCNormalize(T* data, int channels, int Npixels) {
  for (int j = 0; j < channels; j++) {
    NCCNormalizePerChannel(data + j, channels, Npixels);
  }
}

// Interface
template <typename Grid>
class Interpolator {
 public:
  explicit Interpolator(const Grid& grid) : grid_(grid) {
    // The + casts the enum into an int before doing the
    // comparison. It is needed to prevent
    // "-Wunnamed-type-template-args" related errors.
    CHECK_GE(+Grid::DATA_DIMENSION, 1);
  }

  // Evaluate the interpolated function value and/or its
  // derivative. Returns false if r or c is out of bounds.
  virtual void Evaluate(double r, double c, double* f, double* dfdr,
                        double* dfdc) const {
    throw(
        std::runtime_error("Default Interpolator::Evaluate not implemented."));
  }

  // Evaluate the interpolated function value and/or its
  // derivative. Returns false if r or c is out of bounds.
  virtual void Evaluate(double r, double c, double* f, double* dfdr,
                        double* dfdc, double* dfdrc) const {
    if (!dfdrc) {
      Evaluate(r, c, f, dfdr, dfdc);  // Fall back to the default Evaluate
    } else {
      throw(std::runtime_error(
          "Default Interpolator::Evaluate dfdrc not implemented."));
    }
  }

  // The following two Evaluate overloads are needed for interfacing
  // with automatic differentiation. The first is for when a scalar
  // evaluation is done, and the second one is for when Jets are used.
  void Evaluate(const double& r, const double& c, double* f) const {
    return Evaluate(r, c, f, NULL, NULL);
  }

  void IEvaluate(double r, double c, double* f, double* dfdr,
                 double* dfdc) const {
    return Evaluate(r, c, f, dfdr, dfdc);
  }

  template <typename JetT>
  void Evaluate(const JetT& r, const JetT& c, JetT* f) const {
    double frc[Grid::DATA_DIMENSION];
    double dfdr[Grid::DATA_DIMENSION];
    double dfdc[Grid::DATA_DIMENSION];
    Evaluate(r.a, c.a, frc, dfdr, dfdc);
    for (int i = 0; i < Grid::DATA_DIMENSION; ++i) {
      f[i].a = frc[i];
      f[i].v = dfdr[i] * r.v + dfdc[i] * c.v;
    }
  }

  virtual int OutputDimension() const { return Grid::DATA_DIMENSION; }

 protected:
  const Grid& grid_;
};

// Interface to ceres bicubic interpolation
template <typename Grid>
class CeresBiCubicInterpolator : public Interpolator<Grid> {
 public:
  explicit CeresBiCubicInterpolator(const Grid& grid)
      : Interpolator<Grid>(grid) {
    ceres_bicubic_.reset(new ceres::BiCubicInterpolator<Grid>(grid));
  }

  void Evaluate(double r, double c, double* f, double* dfdr,
                double* dfdc) const override {
    ceres_bicubic_->Evaluate(r, c, f, dfdr, dfdc);
  }

 private:
  std::unique_ptr<ceres::BiCubicInterpolator<Grid>> ceres_bicubic_;
};

// https://ceres-solver.googlesource.com/ceres-solver/+/1.12.0/include/ceres/cubic_interpolation.h
// Extended to compute dfdrc
template <typename Grid, typename dtype = double>
class BiCubicInterpolator : public Interpolator<Grid> {
  using Interpolator<Grid>::grid_;

 public:
  explicit BiCubicInterpolator(const Grid& grid) : Interpolator<Grid>(grid) {
    CHECK_GE(+Grid::DATA_DIMENSION, 1);
  }
#if AVX2_ENABLED
  inline void EvaluateSIMD(double r, double c, double* f, double* dfdr,
                           double* dfdc, double* dfdrc) const {
    const int row = std::floor(r);
    const int col = std::floor(c);
    // Eigen::Matrix<dtype, Grid::DATA_DIMENSION, 1> p0, p1, p2, p3;

    Eigen::Matrix<dtype, Grid::DATA_DIMENSION, 1> f0, f1, f2, f3;
    Eigen::Matrix<dtype, Grid::DATA_DIMENSION, 1> df0dc, df1dc, df2dc, df3dc;
    auto p0 = grid_.GetPointer(row - 1, col - 1);
    auto p1 = grid_.GetPointer(row - 1, col);
    auto p2 = grid_.GetPointer(row - 1, col + 1);
    auto p3 = grid_.GetPointer(row - 1, col + 2);
    CubicHermiteSplineSIMD<Grid::DATA_DIMENSION>(p0, p1, p2, p3, c - col,
                                                 f0.data(), df0dc.data());
    p0 = grid_.GetPointer(row, col - 1);
    p1 = grid_.GetPointer(row, col);
    p2 = grid_.GetPointer(row, col + 1);
    p3 = grid_.GetPointer(row, col + 2);
    CubicHermiteSplineSIMD<Grid::DATA_DIMENSION>(p0, p1, p2, p3, c - col,
                                                 f1.data(), df1dc.data());
    p0 = grid_.GetPointer(row + 1, col - 1);
    p1 = grid_.GetPointer(row + 1, col);
    p2 = grid_.GetPointer(row + 1, col + 1);
    p3 = grid_.GetPointer(row + 1, col + 2);
    CubicHermiteSplineSIMD<Grid::DATA_DIMENSION>(p0, p1, p2, p3, c - col,
                                                 f2.data(), df2dc.data());
    p0 = grid_.GetPointer(row + 2, col - 1);
    p1 = grid_.GetPointer(row + 2, col);
    p2 = grid_.GetPointer(row + 2, col + 1);
    p3 = grid_.GetPointer(row + 2, col + 2);
    CubicHermiteSplineSIMD<Grid::DATA_DIMENSION>(p0, p1, p2, p3, c - col,
                                                 f3.data(), df3dc.data());

    CubicHermiteSplineSIMD<Grid::DATA_DIMENSION>(
        f0.data(), f1.data(), f2.data(), f3.data(), r - row, f, dfdr);
    if (dfdc != NULL) {
      // The gradient here is d(dfdc) / dr, which is the cross derivative dfdrc
      CubicHermiteSplineSIMD<Grid::DATA_DIMENSION>(
          (dtype*)df0dc.data(), (dtype*)df1dc.data(), (dtype*)df2dc.data(),
          (dtype*)df3dc.data(), r - row, dfdc, dfdrc);
    }
  }
#endif

  void Evaluate(double r, double c, double* f, double* dfdr, double* dfdc,
                double* dfdrc) const override {
#if AVX2_ENABLED
    if (Grid::DATA_DIMENSION >= 8) {
      EvaluateSIMD(r, c, f, dfdr, dfdc, dfdrc);
      return;
    }

#endif
    const int row = std::floor(r);
    const int col = std::floor(c);
    Eigen::Matrix<double, Grid::DATA_DIMENSION, 1> p0, p1, p2, p3;

    Eigen::Matrix<double, Grid::DATA_DIMENSION, 1> f0, f1, f2, f3;
    Eigen::Matrix<double, Grid::DATA_DIMENSION, 1> df0dc, df1dc, df2dc, df3dc;
    grid_.GetValue(row - 1, col - 1, p0.data());
    grid_.GetValue(row - 1, col, p1.data());
    grid_.GetValue(row - 1, col + 1, p2.data());
    grid_.GetValue(row - 1, col + 2, p3.data());
    ceres::CubicHermiteSpline<Grid::DATA_DIMENSION>(p0, p1, p2, p3, c - col,
                                                    f0.data(), df0dc.data());
    grid_.GetValue(row, col - 1, p0.data());
    grid_.GetValue(row, col, p1.data());
    grid_.GetValue(row, col + 1, p2.data());
    grid_.GetValue(row, col + 2, p3.data());
    ceres::CubicHermiteSpline<Grid::DATA_DIMENSION>(p0, p1, p2, p3, c - col,
                                                    f1.data(), df1dc.data());
    grid_.GetValue(row + 1, col - 1, p0.data());
    grid_.GetValue(row + 1, col, p1.data());
    grid_.GetValue(row + 1, col + 1, p2.data());
    grid_.GetValue(row + 1, col + 2, p3.data());
    ceres::CubicHermiteSpline<Grid::DATA_DIMENSION>(p0, p1, p2, p3, c - col,
                                                    f2.data(), df2dc.data());
    grid_.GetValue(row + 2, col - 1, p0.data());
    grid_.GetValue(row + 2, col, p1.data());
    grid_.GetValue(row + 2, col + 1, p2.data());
    grid_.GetValue(row + 2, col + 2, p3.data());
    ceres::CubicHermiteSpline<Grid::DATA_DIMENSION>(p0, p1, p2, p3, c - col,
                                                    f3.data(), df3dc.data());

    ceres::CubicHermiteSpline<Grid::DATA_DIMENSION>(f0, f1, f2, f3, r - row, f,
                                                    dfdr);
    if (dfdc != NULL) {
      // The gradient here is d(dfdc) / dr, which is the cross derivative dfdrc
      ceres::CubicHermiteSpline<Grid::DATA_DIMENSION>(
          df0dc, df1dc, df2dc, df3dc, r - row, dfdc, dfdrc);
    }
  }

  inline void Evaluate(double r, double c, double* f, double* dfdr,
                       double* dfdc) const override {
    Evaluate(r, c, f, dfdr, dfdc, nullptr);
  }
};

// Format constraints: p(0), p(1), s(0), s(1)
inline Eigen::Vector4d FitCubicPolynomial(const Eigen::Vector4d& rhs) {
  Eigen::Matrix4d lhs;
  lhs << 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0,
      2.0, 3.0;
  return lhs.colPivHouseholderQr().solve(rhs);
}

inline void EvalCubicPolynomial(Eigen::Vector4d& params, double x, double* f,
                                double* dfdx) {
  f[0] = params(0) + x * (params(1) + x * (params(2) + params(3) * x));
  if (dfdx) {
    dfdx[0] = params(1) + x * (2.0 * params(2) + 3.0 * x * params(3));
  }
}

// Gradient field interpolation for advanced costmaps
// Grid must have 3 channels, where channel 1 is the value f, channel 1 is dfdr,
// channel2 is dfdc

// Return value is 1D!
template <typename Grid>
class PolyGradientFieldInterpolator : public Interpolator<Grid> {
  using Interpolator<Grid>::grid_;

 public:
  explicit PolyGradientFieldInterpolator(const Grid& grid)
      : Interpolator<Grid>(grid) {
    THROW_CHECK(Grid::DATA_DIMENSION == 4 || Grid::DATA_DIMENSION == 3);
  }

  // Note: here Grid::DATA_DIMENSION is not equal to the output dimension
  // --> Output channels are Grid::DATA_DIMENSION / 3 !
  void Evaluate(double r, double c, double* f, double* dfdr,
                double* dfdc) const override {
    const int row = std::floor(r);
    const int col = std::floor(c);

    double dy = r - row;
    double dx = c - col;
    typedef Eigen::Matrix<double, Grid::DATA_DIMENSION, 1> VType;
    VType ll, lr, ul, ur;

    grid_.GetValue(row, col, ll.data());
    grid_.GetValue(row, col + 1, lr.data());
    grid_.GetValue(row + 1, col, ul.data());
    grid_.GetValue(row + 1, col + 1, ur.data());

    Eigen::Vector4d upper_poly_rhs{ul(0), ur(0), ul(2), ur(2)};
    Eigen::Vector4d lower_poly_rhs{ll(0), lr(0), ll(2), lr(2)};

    Eigen::Vector4d upper_poly_coeffs = FitCubicPolynomial(upper_poly_rhs);
    Eigen::Vector4d lower_poly_coeffs = FitCubicPolynomial(lower_poly_rhs);

    double uf = 0;
    double lf = 0;

    double upper_dfdr = ul(1) * (1.0 - dx) + ur(1) * dx;
    double lower_dfdr = ll(1) * (1.0 - dx) + lr(1) * dx;

    double upper_dfdc = 0;
    double lower_dfdc = 0;

    EvalCubicPolynomial(upper_poly_coeffs, dx, &uf, &upper_dfdc);
    EvalCubicPolynomial(lower_poly_coeffs, dx, &lf, &lower_dfdc);

    Eigen::Vector4d vertical_poly_rhs{lf, uf, lower_dfdr, upper_dfdr};
    Eigen::Vector4d vertical_poly_coeffs =
        FitCubicPolynomial(vertical_poly_rhs);

    EvalCubicPolynomial(vertical_poly_coeffs, dy, f, dfdr);

    if (dfdc) {
      dfdc[0] = upper_dfdc * dy + (1.0 - dy) * lower_dfdc;
    }
  }

  int OutputDimension() const override {
    if (+Grid::DATA_DIMENSION == 3) {
      return 1;
    }
    if (+Grid::DATA_DIMENSION == 4) {
      return 1;
    }
    return Grid::DATA_DIMENSION / 3;
  }
};

inline Eigen::Matrix<double, 16, 1> FitBicubic(
    Eigen::Matrix<double, 16, 1>& rhs) {
  Eigen::Matrix<double, 16, 16> A_inv;
  A_inv << 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
      0.0, 0.0, 0.0, -3., 3.0, 0.0, 0.0, -2., -1., 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
      0.0, 0.0, 0.0, 0.0, 2.0, -2., 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0,
      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0,
      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
      0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
      -3., 3.0, 0.0, 0.0, -2., -1., 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
      0.0, 2.0, -2., 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, -3., 0.0, 3.0, 0.0, 0.0, 0.0,
      0.0, 0.0, -2., 0.0, -1., 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -3.,
      0.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0, -2., 0.0, -1., 0.0, 9.0, -9., -9., 9.0,
      6.0, 3.0, -6., -3., 6.0, -6., 3.0, -3., 4.0, 2.0, 2.0, 1.0, -6., 6.0, 6.0,
      -6., -3., -3., 3.0, 3.0, -4., 4.0, -2., 2.0, -2., -2., -1., -1., 2.0, 0.0,
      -2., 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
      0.0, 0.0, 0.0, 2.0, 0.0, -2., 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0,
      -6., 6.0, 6.0, -6., -4., -2., 4.0, 2.0, -3., 3.0, -3., 3.0, -2., -1., -2.,
      -1., 4.0, -4., -4., 4.0, 2.0, 2.0, -2., -2., 2.0, -2., 2.0, -2., 1.0, 1.0,
      1.0, 1.0;
  return A_inv * rhs;
}

inline void EvalBicubic(Eigen::Matrix<double, 16, 1>& coeffs, double y,
                        double x, double* f, double* dfdr, double* dfdc,
                        double* dfdrc) {
  double x2 = x * x;
  double x3 = x2 * x;
  double y2 = y * y;
  double y3 = y2 * y;
  double xy = x * y;
  double x2y = x2 * y;
  double x3y = x3 * y;
  double xy2 = x * y2;
  double xy3 = x * y3;
  double x2y2 = x2 * y2;
  double x3y2 = x3 * y2;
  double x2y3 = x2 * y3;
  double x3y3 = x3 * y3;

  if (f) {
    Eigen::Matrix<double, 16, 1> query;
    query << 1.0, x, x2, x3, y, xy, x2y, x3y, y2, xy2, x2y2, x3y2, y3, xy3,
        x2y3, x3y3;
    f[0] = coeffs.dot(query);
  }

  if (dfdc) {
    Eigen::Matrix<double, 16, 1> query;
    query << 0.0, 1.0, 2.0 * x, 3.0 * x2, 0.0, y, 2.0 * xy, 3.0 * x2y, 0.0, y2,
        2.0 * xy2, 3.0 * x2y2, 0.0, y3, 2.0 * xy3, 3.0 * x2y3;
    dfdc[0] = coeffs.dot(query);
  }

  if (dfdr) {
    Eigen::Matrix<double, 16, 1> query;
    query << 0.0, 0.0, 0.0, 0.0, 1.0, x, x2, x3, 2.0 * y, 2.0 * xy, 2.0 * x2y,
        2.0 * x3y, 3.0 * y2, 3.0 * xy2, 3.0 * x2y2, 3.0 * x3y2;
    dfdr[0] = coeffs.dot(query);
  }

  if (dfdrc) {
    Eigen::Matrix<double, 16, 1> query;
    query << 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 2.0 * x, 3.0 * x2, 0.0, 2.0 * y,
        4.0 * x * y, 6.0 * x2y, 0.0, 3.0 * y2, 6.0 * xy2, 9.0 * x2y2;
    dfdrc[0] = coeffs.dot(query);
  }
}

// Return value is 1D!
template <typename Grid>
class BiCubicGradientFieldInterpolator : public Interpolator<Grid> {
  using Interpolator<Grid>::grid_;

 public:
  explicit BiCubicGradientFieldInterpolator(const Grid& grid)
      : Interpolator<Grid>(grid) {
    THROW_CHECK_EQ(Grid::DATA_DIMENSION, 4);
  }

  // Note: here Grid::DATA_DIMENSION is not equal to the output dimension
  // --> Output channels are Grid::DATA_DIMENSION / 3 !
  void Evaluate(double r, double c, double* f, double* dfdr,
                double* dfdc) const override {
    Evaluate(r, c, f, dfdr, dfdc, NULL);
  }

  void Evaluate(double r, double c, double* f, double* dfdr, double* dfdc,
                double* dfdrc) const override {
    const int row = std::floor(r);
    const int col = std::floor(c);

    double dy = r - row;
    double dx = c - col;
    typedef Eigen::Matrix<double, Grid::DATA_DIMENSION, 1> VType;
    VType ll, lr, ul, ur;

    grid_.GetValue(row, col, ll.data());
    grid_.GetValue(row, col + 1, lr.data());
    grid_.GetValue(row + 1, col, ul.data());
    grid_.GetValue(row + 1, col + 1, ur.data());

    Eigen::Matrix<double, 16, 1> rhs;
    rhs << ll(0), lr(0), ul(0), ur(0), ll(2), lr(2), ul(2), ur(2), ll(1), lr(1),
        ul(1), ur(1), ll(3), lr(3), ul(3), ur(3);

    Eigen::Matrix<double, 16, 1> coeffs = FitBicubic(rhs);

    EvalBicubic(coeffs, dy, dx, f, dfdr, dfdc, dfdrc);
  }

  int OutputDimension() const override { return Grid::DATA_DIMENSION / 4; }
};

// Return value is 1D!
template <typename Grid>
class BiCubicChainInterpolator : public Interpolator<Grid> {
  using Interpolator<Grid>::grid_;

 public:
  explicit BiCubicChainInterpolator(const Grid& grid)
      : Interpolator<Grid>(grid) {
    THROW_CHECK_EQ(Grid::DATA_DIMENSION, 3);
    bicubic_interpolator_.reset(new BiCubicInterpolator<Grid>(grid_));
  }

  // Note: here Grid::DATA_DIMENSION is not equal to the output dimension
  // --> Output channels are Grid::DATA_DIMENSION / 3 !
  void Evaluate(double r, double c, double* f, double* dfdr,
                double* dfdc) const override {
    Evaluate(r, c, f, dfdr, dfdc, NULL);
  }

  void Evaluate(double r, double c, double* f, double* dfdr, double* dfdc,
                double* dfdrc) const override {
    double res[3];

    bicubic_interpolator_->Evaluate(r, c, res, NULL, NULL);

    f[0] = res[0];

    if (dfdr) {
      dfdr[0] = res[1];
    }
    if (dfdc) {
      dfdc[0] = res[2];
    }
    if (dfdrc) {
      dfdrc[0] = 0.0;
    }
  }

  int OutputDimension() const override { return Grid::DATA_DIMENSION / 3; }

 private:
  std::unique_ptr<BiCubicInterpolator<Grid>> bicubic_interpolator_;
};

template <typename Grid>
class BiLinearInterpolator : public Interpolator<Grid> {
  using Interpolator<Grid>::grid_;

 public:
  explicit BiLinearInterpolator(const Grid& grid) : Interpolator<Grid>(grid) {}
  void Evaluate(double r, double c, double* f, double* dfdr,
                double* dfdc) const override {
    const int row = std::floor(r);
    const int col = std::floor(c);

    double dy = r - row;
    double dx = c - col;
    typedef Eigen::Matrix<double, Grid::DATA_DIMENSION, 1> VType;
    VType ll, lr, ul, ur;

    grid_.GetValue(row, col, ll.data());
    grid_.GetValue(row, col + 1, lr.data());
    grid_.GetValue(row + 1, col, ul.data());
    grid_.GetValue(row + 1, col + 1, ur.data());

    VType fvec = GetInterpolation(dx, dy, ll, lr, ul, ur);
    Eigen::Map<VType>(f, Grid::DATA_DIMENSION) = fvec;

    if (dfdc != NULL) {
      // Forward difference
      Eigen::Matrix<double, Grid::DATA_DIMENSION, 1> lrx, uly, urx, ury;

      grid_.GetValue(row, col + 2, lrx.data());
      grid_.GetValue(row + 1, col + 2, urx.data());
      grid_.GetValue(row + 2, col, uly.data());
      grid_.GetValue(row + 2, col + 1, ury.data());

      Eigen::Map<VType>(dfdr, Grid::DATA_DIMENSION) =
          GetInterpolation(dx, dy, ul, ur, uly, ury) - fvec;
      Eigen::Map<VType>(dfdc, Grid::DATA_DIMENSION) =
          GetInterpolation(dx, dy, lr, lrx, ur, urx) - fvec;
    }
  }

  Eigen::Matrix<double, Grid::DATA_DIMENSION, 1> GetInterpolation(
      double dx, double dy,
      const Eigen::Matrix<double, Grid::DATA_DIMENSION, 1>& ll,
      const Eigen::Matrix<double, Grid::DATA_DIMENSION, 1>& lr,
      const Eigen::Matrix<double, Grid::DATA_DIMENSION, 1>& ul,
      const Eigen::Matrix<double, Grid::DATA_DIMENSION, 1>& ur) const {
    Eigen::Matrix<double, Grid::DATA_DIMENSION, 1> v0 =
        (1.0 - dx) * ll + dx * lr;
    Eigen::Matrix<double, Grid::DATA_DIMENSION, 1> v1 =
        (1.0 - dx) * ul + dx * ur;

    return (1.0 - dy) * v0 + dy * v1;
  }
};

template <typename Grid>
class NearestNeighborInterpolator : public Interpolator<Grid> {
  using Interpolator<Grid>::grid_;

 public:
  explicit NearestNeighborInterpolator(const Grid& grid)
      : Interpolator<Grid>(grid) {}
  void Evaluate(double r, double c, double* f, double* dfdr,
                double* dfdc) const override {
    const int row = static_cast<int>(std::round(r));
    const int col = static_cast<int>(std::round(c));

    typedef Eigen::Matrix<double, Grid::DATA_DIMENSION, 1> VType;

    grid_.GetValue(row, col, f);

    if (dfdc != NULL) {
      // Forward difference
      grid_.GetValue(row, col + 1, dfdc);
      Eigen::Map<VType>(dfdc, Grid::DATA_DIMENSION) -=
          Eigen::Map<VType>(f, Grid::DATA_DIMENSION);
    }

    if (dfdr != NULL) {
      grid_.GetValue(row + 1, col, dfdr);
      Eigen::Map<VType>(dfdr, Grid::DATA_DIMENSION) -=
          Eigen::Map<VType>(f, Grid::DATA_DIMENSION);
    }
  }
};

template <typename Grid>
class PixelInterpolator : public Interpolator<Grid> {
  // template <typename JetT> using Interpolator<Grid>::Evaluate<JetT>;
  using Interpolator<Grid>::grid_;

 public:
  explicit PixelInterpolator(const InterpolationConfig& config,
                             const Grid& grid)
      : config_(config), Interpolator<Grid>(grid) {
    if (config_.mode == InterpolatorType::BICUBIC) {
#if AVX2_ENABLED
      if (config_.use_float_simd) {
        interpolator_.reset(new BiCubicInterpolator<Grid, float>(grid_));
        return;
      }
#endif
      interpolator_.reset(new BiCubicInterpolator<Grid>(grid_));
    } else if (config_.mode == InterpolatorType::BILINEAR) {
      interpolator_.reset(new BiLinearInterpolator<Grid>(grid_));
    } else if (config_.mode == InterpolatorType::NEARESTNEIGHBOR) {
      interpolator_.reset(new NearestNeighborInterpolator<Grid>(grid_));
    } else if (config_.mode == InterpolatorType::POLYGRADIENTFIELD) {
      interpolator_.reset(new PolyGradientFieldInterpolator<Grid>(grid_));
    } else if (config_.mode == InterpolatorType::BICUBICGRADIENTFIELD) {
      interpolator_.reset(new BiCubicGradientFieldInterpolator<Grid>(grid_));
    } else if (config_.mode == InterpolatorType::BICUBICCHAIN) {
      interpolator_.reset(new BiCubicChainInterpolator<Grid>(grid_));
    } else if (config_.mode == InterpolatorType::CERES_BICUBIC) {
      interpolator_.reset(new CeresBiCubicInterpolator<Grid>(grid_));
    } else {
      throw(std::runtime_error("Unknown InterpolatorType."));
    }
  }
  void Evaluate(double r, double c, double* f, double* dfdr, double* dfdc,
                double* dfdrc) const override {
    interpolator_->Evaluate(r, c, f, dfdr, dfdc, dfdrc);
    THROW_CHECK_NE(Grid::DATA_DIMENSION, -1);

    typedef Eigen::Matrix<double, Grid::DATA_DIMENSION, 1> VType;
    if (config_.l2_normalize) {
      double norm_inv = 1.0 / Eigen::Map<VType>(f).norm();
      Eigen::Map<VType>(f) *= norm_inv;

      // Chain rule for l2 normalization derivatives
      if (dfdc) {
        Eigen::Map<VType>(dfdc) *= norm_inv;
        Eigen::Map<VType>(dfdc) -=
            Eigen::Map<VType>(f).dot(Eigen::Map<VType>(dfdc)) *
            Eigen::Map<VType>(f);
      }

      if (dfdr) {
        Eigen::Map<VType>(dfdr) *= norm_inv;
        Eigen::Map<VType>(dfdr) -=
            Eigen::Map<VType>(f).dot(Eigen::Map<VType>(dfdr)) *
            Eigen::Map<VType>(f);
      }
    }

    // WARNING: THIS MIGHT RESULT IN SEGFAULTS IF NOT USED WITH CAUTION!!
    if (config_.fill_channel_differences) {
      for (int i = OutputDimension(); i < Grid::DATA_DIMENSION; i++) {
        f[i] = 0.0;
        if (dfdr) dfdr[i] = 0.0;
        if (dfdc) dfdc[i] = 0.0;
        if (dfdrc) dfdrc[i] = 0.0;
      }
    }
  }

  void Evaluate(double r, double c, double* f, double* dfdr,
                double* dfdc) const override {
    Evaluate(r, c, f, dfdr, dfdc, NULL);
  }

  int OutputDimension() const override {
    return interpolator_->OutputDimension();
  }

  // The following two Evaluate overloads are needed for interfacing
  // with automatic differentiation. The first is for when a scalar
  // evaluation is done, and the second one is for when Jets are used.
  void Evaluate(const double& r, const double& c, double* f) const {
    Interpolator<Grid>::template Evaluate(r, c, f);
    // typedef Eigen::Matrix<double, Grid::DATA_DIMENSION, 1> VType;
    // Eigen::Map<VType>(f).normalize();
  }

  template <typename JetT>
  void Evaluate(const JetT& r, const JetT& c, JetT* f) const {
    // interpolator_->Evaluate<JetT>(r,c,f);
    Interpolator<Grid>::template Evaluate<JetT>(r, c, f);
    // typedef Eigen::Matrix<JetT, Grid::DATA_DIMENSION, 1> VType;
    // Eigen::Map<VType>(f).normalize();
  }

  // @TODO: Interpolate on a set of nodes = pixels around the interpolation
  // point as specified in config_. Functions should only rely on methods
  // defined in Interpolator<Grid>
  template <typename T>
  void EvaluateNodes(const T& r, const T& c, T* f) {
    for (int i = 0; i < config_.nodes.size(); i++) {
      Evaluate(r + config_.nodes[i][1], c + config_.nodes[i][0],
               f + i * Grid::DATA_DIMENSION);
    }
    // Apply NCC Normalization here
    if (config_.ncc_normalize) {
      NCCNormalize<T>(f, Grid::DATA_DIMENSION, config_.nodes.size());
    }
  }

  template <typename T>
  void EvaluateNNodes(const T* uvs, T* f, int n_nodes) {
    for (int i = 0; i < n_nodes; i++) {
      Evaluate(uvs[2 * i + 1], uvs[2 * i], f + i * Grid::DATA_DIMENSION);
    }
    // Apply NCC Normalization here
    if (config_.ncc_normalize) {
      NCCNormalize<T>(f, Grid::DATA_DIMENSION, n_nodes);
    }
  }

 protected:
  const InterpolationConfig& config_;

 private:
  std::unique_ptr<Interpolator<Grid>> interpolator_;
};

}  // namespace pixsfm