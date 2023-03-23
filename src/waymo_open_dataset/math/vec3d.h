/* Copyright 2022 The Waymo Open Dataset Authors.

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

#ifndef WAYMO_OPEN_DATASET_MATH_VEC3D_H_
#define WAYMO_OPEN_DATASET_MATH_VEC3D_H_

#include <cmath>

namespace waymo {
namespace open_dataset {

// A simple 3 vector class with arithmetic, scaling and normalization.
class Vec3d {
 public:
  constexpr Vec3d() : x_(0.0), y_(0.0), z_(0.0) {}
  constexpr Vec3d(double x, double y, double z) : x_(x), y_(y), z_(z) {}

  constexpr double x() const { return x_; }
  constexpr double y() const { return y_; }
  constexpr double z() const { return z_; }

  constexpr double Sqr() const { return x_ * x_ + y_ * y_ + z_ * z_; }
  double Length() const { return std::sqrt(Sqr()); }

  constexpr Vec3d operator+(const Vec3d& p) const {
    return Vec3d(x_ + p.x_, y_ + p.y_, z_ + p.z_);
  }

  constexpr Vec3d operator-() const { return Vec3d(-x_, -y_, -z_); }

  constexpr Vec3d operator-(const Vec3d& p) const {
    return Vec3d(x_ - p.x_, y_ - p.y_, z_ - p.z_);
  }

  constexpr Vec3d operator*(double s) const {
    return Vec3d(x_ * s, y_ * s, z_ * s);
  }

  constexpr Vec3d operator/(double s) const {
    return Vec3d(x_ / s, y_ / s, z_ / s);
  }

  Vec3d Normalized() const {
    const double len = Length();
    if (len > 0) {
      return Vec3d(x_ / len, y_ / len, z_ / len);
    }
    return Vec3d(0, 0, 0);
  }

  constexpr bool operator==(const Vec3d& pp) const {
    return x_ == pp.x() && y_ == pp.y() && z_ == pp.z();
  }

 private:
  double x_, y_, z_;
};

constexpr Vec3d operator*(double scale, const Vec3d& pp) { return pp * scale; }

}  // namespace open_dataset
}  // namespace waymo

#endif  // WAYMO_OPEN_DATASET_MATH_VEC3D_H_
