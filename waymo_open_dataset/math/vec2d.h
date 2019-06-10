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

#ifndef WAYMO_OPEN_DATASET_MATH_VEC2D_H_
#define WAYMO_OPEN_DATASET_MATH_VEC2D_H_

#include <algorithm>
#include <cmath>
#include <iostream>
#include <string>
#include <tuple>

#include "absl/strings/str_cat.h"

namespace waymo {
namespace open_dataset {

// A 2D real-valued vector. Represents 2D points & vectors.
// Prefer to pass by value rather than const reference.
class Vec2d {
 public:
  constexpr Vec2d() : xy_{0.0, 0.0} {}
  constexpr Vec2d(double x, double y) : xy_{x, y} {}
  constexpr explicit Vec2d(std::tuple<double, double> v)
      : xy_{std::get<0>(v), std::get<1>(v)} {}
  explicit Vec2d(double theta) : xy_{std::cos(theta), std::sin(theta)} {}

  void Set(double x, double y) { *this = Vec2d(x, y); }

  static Vec2d FromScalar(double s) { return Vec2d(s, s); }
  void UnitFromAngle(double theta) { *this = Vec2d(theta); }
  static Vec2d CreateUnitFromAngle(double theta) {
    Vec2d v;
    v.UnitFromAngle(theta);
    return v;
  }

  // Set to a vector that extends a given _signed_ length along the given
  // heading.
  void FromAngleAndLength(double theta, double len) {
    UnitFromAngle(theta);
    *this *= len;
  }
  static Vec2d CreateFromAngleAndLength(double theta, double len) {
    Vec2d v;
    v.FromAngleAndLength(theta, len);
    return v;
  }

  double x() const { return xy_[0]; }
  double y() const { return xy_[1]; }

  void set_x(double x) { xy_[0] = x; }
  void set_y(double y) { xy_[1] = y; }

  Vec2d &Shift(double dx, double dy) {
    *this += Vec2d(dx, dy);
    return *this;
  }

  bool IsZero() const { return x() == 0.0 && y() == 0.0; }

  double L1Norm() const { return std::abs(x()) + std::abs(y()); }
  double Length() const { return std::hypot(x(), y()); }
  double Sqr() const { return Dot(*this); }

  Vec2d &operator*=(double scale) {
    xy_[0] *= scale;
    xy_[1] *= scale;
    return *this;
  }

  Vec2d &operator/=(double scale) {
    const double scale_inv = 1.0 / scale;
    return operator*=(scale_inv);
  }

  Vec2d &operator+=(const Vec2d pp) {
    xy_[0] += pp.x();
    xy_[1] += pp.y();
    return *this;
  }

  Vec2d &operator-=(const Vec2d pp) {
    xy_[0] -= pp.x();
    xy_[1] -= pp.y();
    return *this;
  }

  Vec2d operator+(Vec2d pp) const {
    pp += *this;
    return pp;
  }

  Vec2d operator-(const Vec2d pp) const {
    Vec2d tmp(*this);
    tmp -= pp;
    return tmp;
  }

  Vec2d operator-() const { return Vec2d(-x(), -y()); }

  Vec2d operator*(double scale) const {
    Vec2d tmp(*this);
    tmp *= scale;
    return tmp;
  }

  Vec2d operator/(double scale) const {
    Vec2d tmp(*this);
    tmp /= scale;
    return tmp;
  }

  bool operator==(const Vec2d pp) const {
    return xy_[0] == pp.x() && xy_[1] == pp.y();
  }

  bool operator!=(const Vec2d pp) const { return !(*this == pp); }

  // Returns a lexicographical comparison over Vec2ds, producing a total order.
  // This is primarily useful for map/set sorting.
  bool operator<(const Vec2d pp) const {
    return std::make_tuple(x(), y()) < std::make_tuple(pp.x(), pp.y());
  }

  // Return the dot product of this and another vector.
  double Dot(Vec2d pp) const { return xy_[0] * pp.x() + xy_[1] * pp.y(); }

  // Return the dot product of this and Vec2d(x, y).
  double Dot(double x, double y) const { return Dot(Vec2d(x, y)); }

  // Return the signed magnitude of cross product of this and another vector.
  // It is computed as x * y' - y * x'
  double CrossProd(const Vec2d vec) const {
    return xy_[0] * vec.y() - xy_[1] * vec.x();
  }

  // Return a perpendicular vector with the same length as the current vector.
  // The returned vector points left, or in other words, if the current vector
  // represents a segment of a counter-clockwise polygon, the returned vector,
  // if normalized, will represent the inward normal.
  Vec2d Perp() const { return Vec2d(-y(), x()); }

  // Return a copy of the vector that's transformed and rotated relative to the
  // given origin and theta.
  Vec2d RelativeToOrigin(const Vec2d origin_pos, double origin_theta) const {
    return (*this - origin_pos).Rotate(-origin_theta);
  }

  // Return the projection of this vector onto a line with the given heading.
  double Project(double theta) const { return Dot(CreateUnitFromAngle(theta)); }

  // Return the projection of this vector onto the line along 'vec'.
  double Project(const Vec2d vec) const { return Dot(vec) / vec.Length(); }

  // Return the projection of this vector onto another (assumed unit) vector.
  double ProjectOntoUnit(const Vec2d unit_vec) const { return Dot(unit_vec); }

  // Rotate the vector by the given angle theta (in radians). Positive
  // is counter-clockwise. Return reference to self.
  Vec2d &Rotate(double theta) {
    return RotateWithTransform(std::cos(theta), std::sin(theta));
  }

  // Rotate the vector using precomputed trig.
  inline Vec2d &RotateWithTransform(double cos_theta, double sin_theta);

  // Same functions as above, but they return copies of a vector instead of
  // mutating this vector.
  ABSL_MUST_USE_RESULT Vec2d Rotated(double theta) const {
    return RotatedWithTransform(std::cos(theta), std::sin(theta));
  }

  ABSL_MUST_USE_RESULT Vec2d RotatedWithTransform(double cos_theta,
                                                  double sin_theta) const;

  // Return the distance to the given point (either Vec2d or (x, y)).
  double DistanceToPoint(const Vec2d query) const {
    const Vec2d tmp = query - *this;
    return tmp.Length();
  }
  double DistanceToPoint(double x, double y) const {
    return DistanceToPoint(Vec2d(x, y));
  }

  // Return the squared distance to the given point (either Vec2d or (x, y)).
  double DistanceSqrToPoint(const Vec2d query) const {
    const Vec2d tmp = query - *this;
    return tmp.Sqr();
  }
  double DistanceSqrToPoint(double x, double y) const {
    return DistanceSqrToPoint(Vec2d(x, y));
  }

  // Heading of this vector.
  double Angle() const { return std::atan2(y(), x()); }

  // Normalize this vector to unit length.
  void Normalize() {
    const double len = Length();
    if (len > 0.0) {
      (*this) /= len;
    }
  }

  // Return a copy of the current vector normalized to unit length.
  ABSL_MUST_USE_RESULT Vec2d Normalized() const {
    Vec2d tmp(*this);
    tmp.Normalize();
    return tmp;
  }

  // Flip the direction of the vector.
  void Flip() { (*this) = -(*this); }

  // Return the vector with the absolute values of the current vector.
  Vec2d Abs() const { return {std::abs(x()), std::abs(y())}; }

  // Returns true if the distance to the other vector is less than epsilon on
  // each dimension.
  bool NearlyEquals(const Vec2d other, double epsilon) const {
    const Vec2d d = *this - other;
    const Vec2d eps(epsilon, epsilon);

    const Vec2d d_abs = d.Abs();
    return d_abs.x() < epsilon && d_abs.y() < epsilon;
  }

  // Returns true, if 'this' vector is left of the directed line
  // defined by 'start' and 'end'.
  bool IsLeftOfLine(const Vec2d start, const Vec2d end) const {
    // Leverage the cross product:
    //  . v_dstart = v_start - v_this
    //  . v_dend = v_end - v_this
    //  . sin(a) = (v_dstart x v_dend) / |v_dstart| * |v_dend|.
    //  . If sin(a) > 0.0, 0 < a < M_PI.
    const Vec2d v_dstart = start - *this;
    const Vec2d v_dend = end - *this;
    return v_dstart.CrossProd(v_dend) > 0.0;
  }

  // Returns an element-wise max between the current and the provided vector.
  ABSL_MUST_USE_RESULT Vec2d ElemWiseMax(const Vec2d vec) const {
    return {std::max(x(), vec.x()), std::max(y(), vec.y())};
  }

  // Returns an element-wise min between the current and the provided vector.
  ABSL_MUST_USE_RESULT Vec2d ElemWiseMin(const Vec2d vec) const {
    return {std::min(x(), vec.x()), std::min(y(), vec.y())};
  }

  // Returns a Vec2i object with each element being floored.
  ABSL_MUST_USE_RESULT Vec2d Floor() const {
    return {std::floor(x()), std::floor(y())};
  }

  // Returns a Vec2i object with the ceiling of each element.
  ABSL_MUST_USE_RESULT Vec2d Ceil() const {
    return {std::ceil(x()), std::ceil(y())};
  }

  // Returns true if this vector's elements have a finite value (i.e., not
  // infinite or NaN).
  bool IsFinite() const { return std::isfinite(x()) && std::isfinite(y()); }

  std::string DebugString() const {
    return absl::StrCat("[", x(), ",", y(), "]");
  }

 private:
  double xy_[2];
};

inline Vec2d &Vec2d::RotateWithTransform(double cos_theta, double sin_theta) {
  *this = RotatedWithTransform(cos_theta, sin_theta);
  return *this;
}

inline Vec2d Vec2d::RotatedWithTransform(double cos_theta,
                                         double sin_theta) const {
  const Vec2d sincos(sin_theta, cos_theta);
  return Vec2d(CrossProd(sincos), Dot(sincos));
}

inline Vec2d operator*(double scale, const Vec2d pp) { return pp * scale; }

inline std::ostream &operator<<(std::ostream &os, const Vec2d v) {
  return os << v.DebugString();
}

}  // namespace open_dataset
}  // namespace waymo

#endif  // WAYMO_OPEN_DATASET_MATH_VEC2D_H_
