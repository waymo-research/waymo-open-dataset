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

#include "waymo_open_dataset/math/box2d.h"

#include <float.h>
#include <math.h>

#include <algorithm>
#include <limits>
#include <memory>

namespace waymo {
namespace open_dataset {

namespace {

// Epsilon used in this file to avoid double precision problem
// when we compare two numbers.
const double kEpsilon = 1e-10;

}  // namespace

void Box2d::Set(int corner_index, const Vec2d corner, const Vec2d direction,
                double length, double width) {
  axis_ = Segment2d(corner, direction, length);
  width_ = width;

  Vec2d w2 = direction.Perp() * (width * 0.5);
  Vec2d l2 = direction * (length * 0.5);
  Vec2d shift;
  switch (corner_index) {
    case 0:  // bottom-left.
      shift = -w2 + l2;
      break;
    case 1:  // top-left.
      shift = -w2 - l2;
      break;
    case 2:  // top-right.
      shift = w2 - l2;
      break;
    case 3:  // bottom-right.
      shift = w2 + l2;
      break;
  }
  ShiftCenter(shift);
}

void Box2d::Set(const Vec2d bottom_left, const Vec2d top_right) {
  const double length_x = top_right.x() - bottom_left.x();
  const double length_y = top_right.y() - bottom_left.y();
  const double heading = length_x >= length_y ? 0 : M_PI_2;

  Vec2d center((bottom_left + top_right) / 2.0);
  axis_ = Segment2d(center, heading, std::max(length_x, length_y));
  width_ = std::min(length_x, length_y);
  CHECK_LE(0.0, width_);
}

void Box2d::GetCorners(double *xx, double *yy) const {
  const Vec2d w2 = axis_.PerpUnit() * (width_ / 2.0);

  const double x1 = axis_.start().x();
  const double y1 = axis_.start().y();
  const double x2 = axis_.end().x();
  const double y2 = axis_.end().y();

  xx[0] = x1 + w2.x();
  yy[0] = y1 + w2.y();

  xx[1] = x2 + w2.x();
  yy[1] = y2 + w2.y();

  xx[2] = x2 - w2.x();
  yy[2] = y2 - w2.y();

  xx[3] = x1 - w2.x();
  yy[3] = y1 - w2.y();
}

void Box2d::GetCornersInVectorCounterClockwise(
    std::vector<Vec2d> *corners) const {
  double xx[4], yy[4];
  GetCorners(xx, yy);
  corners->resize(4);
  (*corners)[0].Set(xx[0], yy[0]);
  (*corners)[1].Set(xx[3], yy[3]);
  (*corners)[2].Set(xx[2], yy[2]);
  (*corners)[3].Set(xx[1], yy[1]);
}

std::string Box2d::DebugString() const {
  std::vector<Vec2d> corners;
  GetCornersInVectorCounterClockwise(&corners);
  std::string debug_str;
  for (const auto &corner : corners) {
    absl::StrAppend(&debug_str, ", ", corner.DebugString());
  }
  return debug_str;
}

}  // namespace open_dataset
}  // namespace waymo
