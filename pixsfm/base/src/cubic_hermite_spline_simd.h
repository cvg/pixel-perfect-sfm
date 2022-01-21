#include "Eigen/Core"

#include "third-party/half.h"

// The following definition enables or disables fast interpolation
// On gcc we need -mavx2, -mfma, -mf16c
// #if (defined(__AVX2__) && defined(__FMA__) && defined(__F16C__))
// #define AVX2_ENABLED 1
// #include <immintrin.h>
// #else
// #define AVX2_ENABLED 0
// #endif

#if AVX2_ENABLED
#include <immintrin.h>
#endif

namespace pixsfm {

#if AVX2_ENABLED
inline void _mm256_storeu_pd_T(double* data, const __m256d& val) {
  _mm256_storeu_pd(data, val);
}

inline void _mm256_storeu_pd_T(float* data, const __m256d& val) {
  __m128 val_f = _mm256_cvtpd_ps(val);
  _mm_storeu_ps(data, val_f);
}

inline void _mm256_storeu_ps_T(float* target, const __m256& val) {
  _mm256_storeu_ps(target, val);
}

inline void _mm256_storeu_ps_T(double* target, const __m256& val) {
  __m128 lo, hi;
  __m256d lod, hid;
  lo = _mm256_extractf128_ps(val, 0);
  hi = _mm256_extractf128_ps(val, 1);
  lod = _mm256_cvtps_pd(lo);
  hid = _mm256_cvtps_pd(hi);
  _mm256_storeu_pd(target, lod);
  _mm256_storeu_pd(target + 4, hid);
}

inline __m256 _mm256_loadu_ps_T(const float* data) {
  return _mm256_loadu_ps(data);
}

inline __m256 _mm256_loadu_ps_T(const half* data) {
  __m128i dat_h = _mm_loadu_si128((__m128i*)(data));
  return _mm256_cvtph_ps(dat_h);
}

template <int kDataDimension, typename T>
inline void CubicHermiteSplineSIMD(const double* p0v, const double* p1v,
                                   const double* p2v, const double* p3v,
                                   const double x, T* f, T* dfdx) {
  int offset = 0;
  const int step = 4;
  const double x2s = x * x;
  const __m256d onehalf = _mm256_set1_pd(0.5);
  const __m256d three = _mm256_set1_pd(3.0);
  const __m256d two = _mm256_set1_pd(2.0);
  const __m256d four = _mm256_set1_pd(4.0);
  const __m256d twofive = _mm256_set1_pd(2.5);
  const __m256d min1 = _mm256_set1_pd(-1.0);
  const __m256d fourx = _mm256_set1_pd(4.0 * x);
  const __m256d xhalf = _mm256_set1_pd(x * 0.5);
  const __m256d x2 = _mm256_set1_pd(x2s);
  const __m256d onefivex2 = _mm256_set1_pd(1.5 * x2s);

  const bool compute_f = f != NULL;
  const bool compute_dfdx = dfdx != NULL;

  for (int i = 0; i + step <= kDataDimension; i += step) {
    __m256d p0 = _mm256_loadu_pd(p0v + i);
    __m256d p1 = _mm256_loadu_pd(p1v + i);
    __m256d p2 = _mm256_loadu_pd(p2v + i);
    __m256d p3 = _mm256_loadu_pd(p3v + i);

    __m256d t1 = _mm256_fmsub_pd(three, p1, p0);
    __m256d t2 = _mm256_fmsub_pd(three, p2, p3);
    __m256d t4 = _mm256_fmsub_pd(four, p2, p3);
    __m256d t5 = _mm256_fmsub_pd(twofive, p1, p0);
    __m256d t6 = _mm256_fmadd_pd(min1, p0, p2);
    __m256d t3 = _mm256_sub_pd(t1, t2);
    __m256d b = _mm256_fmsub_pd(onehalf, t4, t5);

    if (compute_f) {
      __m256d t7 = _mm256_fmadd_pd(xhalf, t6, p1);  // d = p1
      __m256d t8 = _mm256_fmadd_pd(xhalf, t3, b);
      __m256d rf = _mm256_fmadd_pd(x2, t8, t7);
      _mm256_storeu_pd_T(f + i, rf);
    }
    if (compute_dfdx) {
      __m256d t9 = _mm256_fmadd_pd(fourx, b, t6);
      __m256d t10 = _mm256_mul_pd(onefivex2, t3);
      __m256d rdfdx = _mm256_fmadd_pd(onehalf, t9, t10);
      _mm256_storeu_pd_T(dfdx + i, rdfdx);
    }
    offset = i + step;
  }
  for (int i = offset; i < kDataDimension; i++) {
    double p0 = p0v[i];
    double p1 = p1v[i];
    double p2 = p2v[i];
    double p3 = p3v[i];
    double a = 0.5 * (-p0 + 3.0 * (p1 - p2) + p3);
    double b = 0.5 * (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3);
    double c = 0.5 * (-p0 + p2);
    double d = p1;
    if (f != NULL) {
      f[i] = static_cast<T>(d + x * (c + x * (b + x * a)));
    }
    if (dfdx != NULL) {
      dfdx[i] = static_cast<T>(c + x * (2.0 * b + 3.0 * a * x));
    }
  }
}

template <int kDataDimension, typename IN_T, typename T>
inline void CubicHermiteSplineSIMD(const IN_T* p0v, const IN_T* p1v,
                                   const IN_T* p2v, const IN_T* p3v,
                                   const double x, T* f, T* dfdx) {
  int offset = 0;
  const int step = 8;
  const float x2s = static_cast<float>(x * x);
  const __m256 onehalf = _mm256_set1_ps(0.5);
  const __m256 three = _mm256_set1_ps(3.0);
  const __m256 two = _mm256_set1_ps(2.0);
  const __m256 four = _mm256_set1_ps(4.0);
  const __m256 twofive = _mm256_set1_ps(2.5);
  const __m256 min1 = _mm256_set1_ps(-1.0);
  const __m256 fourx = _mm256_set1_ps(4.0f * x);
  const __m256 xhalf = _mm256_set1_ps(x * 0.5f);
  const __m256 x2 = _mm256_set1_ps(x2s);
  const __m256 onefivex2 = _mm256_set1_ps(1.5f * x2s);

  const bool compute_f = f != NULL;
  const bool compute_dfdx = dfdx != NULL;

  for (int i = 0; i + step <= kDataDimension; i += step) {
    __m256 p0 = _mm256_loadu_ps_T(p0v + i);
    __m256 p1 = _mm256_loadu_ps_T(p1v + i);
    __m256 p2 = _mm256_loadu_ps_T(p2v + i);
    __m256 p3 = _mm256_loadu_ps_T(p3v + i);

    __m256 t1 = _mm256_fmsub_ps(three, p1, p0);
    __m256 t2 = _mm256_fmsub_ps(three, p2, p3);
    __m256 t4 = _mm256_fmsub_ps(four, p2, p3);
    __m256 t5 = _mm256_fmsub_ps(twofive, p1, p0);
    __m256 t6 = _mm256_fmadd_ps(min1, p0, p2);

    __m256 t3 = _mm256_sub_ps(t1, t2);

    // const __m256 a = _mm256_mul_ps(onehalf, t3);
    __m256 b = _mm256_fmsub_ps(onehalf, t4, t5);
    // const __m256 c = _mm256_mul_ps(onehalf,t6);

    if (compute_f) {
      __m256 t7 = _mm256_fmadd_ps(xhalf, t6, p1);  // d = p1
      __m256 t8 = _mm256_fmadd_ps(xhalf, t3, b);
      __m256 rf = _mm256_fmadd_ps(x2, t8, t7);
      _mm256_storeu_ps_T(f + i, rf);
    }
    if (compute_dfdx) {
      __m256 t9 = _mm256_fmadd_ps(fourx, b, t6);
      __m256 t10 = _mm256_mul_ps(onefivex2, t3);
      __m256 rdfdx = _mm256_fmadd_ps(onehalf, t9, t10);
      _mm256_storeu_ps_T(dfdx + i, rdfdx);
    }
    offset = i + step;
  }
  for (int i = offset; i < kDataDimension; i++) {
    double p0 = static_cast<double>(p0v[i]);
    double p1 = static_cast<double>(p1v[i]);
    double p2 = static_cast<double>(p2v[i]);
    double p3 = static_cast<double>(p3v[i]);
    double a = 0.5 * (-p0 + 3.0 * (p1 - p2) + p3);
    double b = 0.5 * (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3);
    double c = 0.5 * (-p0 + p2);
    double d = p1;
    if (f != NULL) {
      f[i] = static_cast<T>(d + x * (c + x * (b + x * a)));
    }
    if (dfdx != NULL) {
      dfdx[i] = static_cast<T>(c + x * (2.0 * b + 3.0 * a * x));
    }
  }
}
#endif

}  // namespace pixsfm