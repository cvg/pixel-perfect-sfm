#pragma once

#include <colmap/math/math.h>
template <typename T>
inline bool IsInsideZeroL(const T& value, double L) {
  return (value > 0.0 && value < L);
}