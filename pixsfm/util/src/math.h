#pragma once

#include <colmap/util/math.h>
template <typename T>
inline bool IsInsideZeroL(const T& value, double L) {
  return (value > 0.0 && value < L);
}