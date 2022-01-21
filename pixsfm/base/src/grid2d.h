#pragma once

#include <ceres/ceres.h>
#include <ceres/cubic_interpolation.h>

#include "util/src/log_exceptions.h"

namespace pixsfm {

template <typename T, int kDataDimension = 1, bool kRowMajor = true,
          bool kInterleaved = true>
struct Grid2D {
 public:
  enum { DATA_DIMENSION = kDataDimension };
  Grid2D(const T* data, const int row_begin, const int row_end,
         const int col_begin, const int col_end)
      : data_(data),
        row_begin_(row_begin),
        row_end_(row_end),
        col_begin_(col_begin),
        col_end_(col_end),
        num_rows_(row_end - row_begin),
        num_cols_(col_end - col_begin),
        num_values_(num_rows_ * num_cols_) {
    CHECK_GE(kDataDimension, 1);
    CHECK_LT(row_begin, row_end);
    CHECK_LT(col_begin, col_end);
  }
  EIGEN_STRONG_INLINE void GetValue(const int r, const int c, double* f) const {
    const int row_idx =
        std::min(std::max(row_begin_, r), row_end_ - 1) - row_begin_;
    const int col_idx =
        std::min(std::max(col_begin_, c), col_end_ - 1) - col_begin_;
    const int n = (kRowMajor) ? num_cols_ * row_idx + col_idx
                              : num_rows_ * col_idx + row_idx;
    if (kInterleaved) {
      for (int i = 0; i < kDataDimension; ++i) {
        f[i] = static_cast<double>(data_[kDataDimension * n + i]);
      }
    } else {
      for (int i = 0; i < kDataDimension; ++i) {
        f[i] = static_cast<double>(data_[i * num_values_ + n]);
      }
    }
  }
  EIGEN_STRONG_INLINE void GetValue(const int r, const int c, float* f) const {
    const int row_idx =
        std::min(std::max(row_begin_, r), row_end_ - 1) - row_begin_;
    const int col_idx =
        std::min(std::max(col_begin_, c), col_end_ - 1) - col_begin_;
    const int n = (kRowMajor) ? num_cols_ * row_idx + col_idx
                              : num_rows_ * col_idx + row_idx;
    if (kInterleaved) {
      for (int i = 0; i < kDataDimension; ++i) {
        f[i] = static_cast<float>(data_[kDataDimension * n + i]);
      }
    } else {
      for (int i = 0; i < kDataDimension; ++i) {
        f[i] = static_cast<float>(data_[i * num_values_ + n]);
      }
    }
  }

  EIGEN_STRONG_INLINE const T* GetPointer(const int r, const int c) const {
    THROW_CHECK(kInterleaved);
    const int row_idx =
        std::min(std::max(row_begin_, r), row_end_ - 1) - row_begin_;
    const int col_idx =
        std::min(std::max(col_begin_, c), col_end_ - 1) - col_begin_;
    const int n = (kRowMajor) ? num_cols_ * row_idx + col_idx
                              : num_rows_ * col_idx + row_idx;
    return data_ + kDataDimension * n;
  }

 private:
  const T* data_;
  const int row_begin_;
  const int row_end_;
  const int col_begin_;
  const int col_end_;
  const int num_rows_;
  const int num_cols_;
  const int num_values_;
};

}  // namespace pixsfm