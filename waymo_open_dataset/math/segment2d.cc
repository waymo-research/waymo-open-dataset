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

#include "waymo_open_dataset/math/segment2d.h"

#include <algorithm>

#include "absl/base/optimization.h"
#include "absl/strings/str_format.h"

namespace waymo {
namespace open_dataset {

namespace {

void RotateVectorsAroundOrigin(double theta, Vec2d *start, Vec2d *end) {
  const double cs = cos(theta);
  const double sn = sin(theta);
  start->Set(cs * start->x() - sn * start->y(),
             sn * start->x() + cs * start->y());
  end->Set(cs * end->x() - sn * end->y(), sn * end->x() + cs * end->y());
}

}  // namespace

// See header for value;
constexpr double Segment2d::kEpsilon;

Segment2d Segment2d::RotateAroundOrigin(double theta) const {
  Vec2d new_start = start_;
  Vec2d new_end = end_;
  RotateVectorsAroundOrigin(theta, &new_start, &new_end);
  const double heading = heading_.load(std::memory_order_relaxed);
  if (IsCached(heading)) {
    return Segment2d(new_start, new_end, NormalizeAngle(heading + theta),
                     *this);
  } else {
    return Segment2d(new_start, new_end, *this);
  }
}

Segment2d Segment2d::RotateAroundPoint(const Vec2d point, double theta) const {
  // Move the axis of rotation to the origin.
  Vec2d new_start = start_ - point;
  Vec2d new_end = end_ - point;

  // Rotate.
  RotateVectorsAroundOrigin(theta, &new_start, &new_end);

  // Shift back.
  new_start += point;
  new_end += point;

  const double heading = heading_.load(std::memory_order_relaxed);
  if (IsCached(heading)) {
    return Segment2d(new_start, new_end, NormalizeAngle(heading + theta),
                     *this);
  } else {
    return Segment2d(new_start, new_end, *this);
  }
}

void Segment2d::trig_heading(double *cos_heading, double *sin_heading) const {
  DCHECK(cos_heading != nullptr);
  DCHECK(sin_heading != nullptr);
  const double inv_len = inv_length();
  if (inv_len == 0.0) {
    const double theta = heading();
    *cos_heading = cos(theta);
    *sin_heading = sin(theta);
  } else {
    *cos_heading = inv_len * (end_.x() - start_.x());
    *sin_heading = inv_len * (end_.y() - start_.y());
  }
}

bool Segment2d::Intersect(const Segment2d &other, Vec2d *point) const {
  // We want to find point P where:
  // P = start + t1 * (end - start);
  // P = other.start + t2 * (other.start - other.end)
  // such that both t1 and t2 are within [0.0, 1.0].

  const double denominator =
      (end_ - start_).CrossProd(other.end() - other.start());

  // Two segments are parallel to each other, happens very rarely.
  if (std::abs(denominator) < kEpsilon) {
    // Two segments are parallel or coincident to each other.
    return false;
  }

  const double numerator1 =
      (other.end() - other.start()).CrossProd(start_ - other.start());
  const double numerator2 = (end_ - start_).CrossProd(start_ - other.start());

  constexpr double kEpsilonSqr = kEpsilon * kEpsilon;
  const double t = numerator1 / denominator;
  const double t2 = numerator2 / denominator;
  const Vec2d pt = start_ + (end_ - start_) * t;

  if ((t < 0 && start_.DistanceSqrToPoint(pt) > kEpsilonSqr) ||
      (t > 1.0 && end_.DistanceSqrToPoint(pt) > kEpsilonSqr) ||
      (t2 < 0 && other.start_.DistanceSqrToPoint(pt) > kEpsilonSqr) ||
      (t2 > 1.0 && other.end_.DistanceSqrToPoint(pt) > kEpsilonSqr)) {
    return false;
  }

  if (point != nullptr) {
    *point = pt;
  }
  return true;
}

bool Segment2d::IntersectOrCoincide(const Segment2d &other) const {
  // We want to find point P where:
  // P = start + t1 * (end - start);
  // P = other.start + t2 * (other.start - other.end)
  // such that both t1 and t2 are within [0.0, 1.0].

  constexpr double kEpsilonSqr = kEpsilon * kEpsilon;
  const double denominator =
      (end_ - start_).CrossProd(other.end() - other.start());

  // Two segments are parallel to each other, happens very rarely.
  if (ABSL_PREDICT_FALSE(std::abs(denominator) < kEpsilon)) {
    return DistanceSqrToPoint(other.start()) < kEpsilonSqr ||
           DistanceSqrToPoint(other.end()) < kEpsilonSqr ||
           other.DistanceSqrToPoint(start_) < kEpsilonSqr ||
           other.DistanceSqrToPoint(end_) < kEpsilonSqr;
  }

  const double numerator1 =
      (other.end() - other.start()).CrossProd(start_ - other.start());
  const double numerator2 = (end_ - start_).CrossProd(start_ - other.start());

  const double t = numerator1 / denominator;
  const double t2 = numerator2 / denominator;
  const Vec2d pt = start_ + (end_ - start_) * t;

  if ((t < 0 && start_.DistanceSqrToPoint(pt) > kEpsilonSqr) ||
      (t > 1.0 && end_.DistanceSqrToPoint(pt) > kEpsilonSqr) ||
      (t2 < 0 && other.start_.DistanceSqrToPoint(pt) > kEpsilonSqr) ||
      (t2 > 1.0 && other.end_.DistanceSqrToPoint(pt) > kEpsilonSqr)) {
    return false;
  }

  return true;
}

bool Segment2d::Sample(double dist, Vec2d *point) const {
  CHECK(point);
  CHECK_LE(0, dist);
  const Vec2d dp = end_ - start_;
  const double len = length();
  CHECK_LE(0, len);
  if (len == 0) {
    *point = start_;
    return false;
  }
  const double frac = dist / len;
  *point = start_ + dp * frac;
  return frac < 1.0;
}

std::vector<Vec2d> Segment2d::SamplePoints(const Vec2d &start, const Vec2d &end,
                                           double dist) {
  CHECK_LT(0, dist);
  const Vec2d dp = end - start;
  const double len = dp.Length();
  if (len == 0) {
    return {start};
  }
  const double frac = dist / len;
  std::vector<Vec2d> sample_points;
  for (double ratio = 0; ratio < 1.0; ratio += frac) {
    sample_points.push_back(start + dp * ratio);
  }
  return sample_points;
}

std::string Segment2d::DebugString() const {
  return absl::StrFormat("%s, %s", start_.DebugString(), end_.DebugString());
}

}  // namespace open_dataset
}  // namespace waymo
