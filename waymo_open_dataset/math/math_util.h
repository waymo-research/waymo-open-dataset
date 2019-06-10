/* Copyright 2019 The Waymo Open Dataset Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef WAYMO_OPEN_DATASET_MATH_MATH_UTIL_H_
#define WAYMO_OPEN_DATASET_MATH_MATH_UTIL_H_

#include <cmath>
#include <type_traits>

namespace waymo {
namespace open_dataset {

// Square a value.
template <class T>
constexpr inline T Sqr(T x) {
  return x * x;
}

template <typename T>
constexpr T Min(T a, T b) {
  return (a < b) ? a : b;
}

template <typename T>
constexpr T Max(T a, T b) {
  return (a < b) ? b : a;
}

// Modulo operation that never returns a negative result if the second
// argument is positive. Works for integral and floating point types. Not well
// defined for negative second argument. Second argument should be less than
// half the maximum representable value of T.
template <typename T>
constexpr typename std::enable_if<std::is_integral<T>::value, T>::type
PositiveModulo(T x, T y) {
  // If x is in [0, y), return it without doing math as a fast path.
  // If x in in [-y, 2*y), do a single add/subtract.
  // Otherwise use % operator with correction for negative values.
  return x >= 0 ? (x < y + y ? (x < y ? x : x - y) : x % y)
                : (x >= -y ? x + y : y - (-(x + 1)) % y - 1);
}

template <typename T>
inline typename std::enable_if<std::is_floating_point<T>::value, T>::type
PositiveModulo(T x, T y) {
  // Specialize for the case y is a constant (which it most often is). In this
  // case, the division 1 / y is optimized away.
  const T x_div_y = x * (1 / y);
  const T truncated_result = std::trunc(x_div_y);
  const T modulo = x - truncated_result * y;
  const T modulo_shifted = y + modulo;
  return modulo >= 0 ? modulo : (modulo_shifted >= y ? 0 : modulo_shifted);
}

// Wraps a value to be in [min_val, max_val).
template <typename T>
inline T WrapToRange(T min_val, T max_val, T val) {
  return PositiveModulo(val - min_val, max_val - min_val) + min_val;
}

// Wraps an angle in radians to be in [-pi, pi).
template <class T>
inline T NormalizeAngle(T rad) {
  return WrapToRange<T>(-M_PI, M_PI, rad);
}

// This function clamps val to range [min_val, max_val], i.e., if val is
// in [min_val, max_val] it is returned unchanged. Otherwise, either min_val or
// max_val is returned as appropriate.
template <class T>
constexpr T BoundToRange(T min_val, T max_val, T val) {
  // This expansion compiles to a branchless sequence using min/max
  // instructions.
  return Max<T>(min_val, Min<T>(max_val, val));
}

}  // namespace open_dataset
}  // namespace waymo

#endif  // WAYMO_OPEN_DATASET_MATH_MATH_UTIL_H_
