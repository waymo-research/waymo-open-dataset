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

#ifndef WAYMO_OPEN_DATASET_MATH_SEGMENT2D_H_
#define WAYMO_OPEN_DATASET_MATH_SEGMENT2D_H_

#include <atomic>
#include <cmath>
#include <iostream>
#include <limits>

#include <glog/logging.h>
#include "waymo_open_dataset/math/math_util.h"
#include "waymo_open_dataset/math/vec2d.h"

namespace waymo {
namespace open_dataset {

// A line segment in 2d.
class Segment2d {
 public:
  // Epsilon used in this class to avoid double precision problems when we
  // compare two numbers.
  static constexpr double kEpsilon = 1e-10;

  Segment2d() : heading_(0.0), length_(0.0), inv_length_(0.0) {}

  // Construct from two endpoints. As a speed optimization, this will not
  // compute and store the heading.
  Segment2d(const Vec2d start, const Vec2d end) : start_(start), end_(end) {}

  // Construct from center point, heading, and length.
  Segment2d(const Vec2d center, double heading, double len)
      : Segment2d(center, Vec2d::CreateUnitFromAngle(heading), len) {
    heading_.store(heading, std::memory_order_relaxed);
  }

  // Construct from center point, direction (cos/sin of heading), and length.
  Segment2d(const Vec2d center, const Vec2d direction, double len)
      : length_(len) {
    const Vec2d offset = direction * (len / 2.0);
    start_ = center - offset;
    end_ = center + offset;
  }

  // Copy construction and copy assignment are not threadsafe.
  Segment2d(const Segment2d &other)
      : Segment2d(other.start(), other.end(), other) {}
  Segment2d &operator=(const Segment2d &other) {
    if (this != &other) {
      start_ = other.start_;
      end_ = other.end_;
      CopyCachedValues(other);
    }
    return *this;
  }

  // Accessors for start & end points of the segment.
  Vec2d start() const { return start_; }
  Vec2d end() const { return end_; }

  // Swap start & end points.
  ABSL_MUST_USE_RESULT Segment2d Flip() const {
    const double heading = heading_.load(std::memory_order_relaxed);
    if (IsCached(heading)) {
      return Segment2d(end_, start_, NormalizeAngle(heading + M_PI), *this);
    } else {
      return Segment2d(end_, start_, *this);
    }
  }

  // Dot product between the segment (end-start) and vec.
  double Dot(const Vec2d vec) const { return vec.Dot(end_ - start_); }

  // Rotate the segment around the center.
  ABSL_MUST_USE_RESULT Segment2d RotateAroundCenter(double theta) const {
    return RotateAroundPoint(center(), theta);
  }

  // Rotate the segment around the origin. This rotates both endpoints by
  // the given angle. This is useful, for example, when changing to a
  // different, rotated, coordinate frame.
  ABSL_MUST_USE_RESULT Segment2d RotateAroundOrigin(double theta) const;

  // Rotate the segment around an arbitrary point.
  ABSL_MUST_USE_RESULT Segment2d RotateAroundPoint(const Vec2d point,
                                                   double theta) const;

  // Return a perpendicular unit vector (i.e. rotate by 90 degree
  // counter-clockwise).
  ABSL_MUST_USE_RESULT Vec2d PerpUnit() const {
    return length() == 0.0 ? Vec2d() : (end_ - start_).Perp() / length();
  }

  // Return a unit vector parallel to the segment.
  ABSL_MUST_USE_RESULT Vec2d Tangent() const {
    return length() == 0.0 ? Vec2d() : (end_ - start_) / length();
  }

  // Return the length of the segment.
  double length() const {
    double length = length_.load(std::memory_order_relaxed);
    if (!IsCached(length)) {
      length = start_.DistanceToPoint(end_);
      length_.store(length, std::memory_order_relaxed);
    }
    return length;
  }

  // Returns 1 / length if length != 0 and 0 otherwise.
  double inv_length() const {
    double inv_length = inv_length_.load(std::memory_order_relaxed);
    if (!IsCached(inv_length)) {
      const double len = length();
      inv_length = (len == 0.0 ? 0.0 : 1.0 / len);
      inv_length_.store(inv_length, std::memory_order_relaxed);
    }
    return inv_length;
  }

  // Return the squared length of the segment.
  double Sqr() const {
    const double length = length_.load(std::memory_order_relaxed);
    if (IsCached(length)) {
      return open_dataset::Sqr(length);
    } else {
      return (start_ - end_).Sqr();
    }
  }

  // Return the center point of the segment.
  double center_x() const { return (start_.x() + end_.x()) / 2; }
  double center_y() const { return (start_.y() + end_.y()) / 2; }
  Vec2d center() const { return Vec2d(center_x(), center_y()); }

  // Slide the segment so that it's centered around the given coordinates.
  ABSL_MUST_USE_RESULT Segment2d MoveCenter(const Vec2d new_center) const {
    return ShiftCenter(new_center - center());
  }

  // Slide the segment by the given (dx, dy) vector.
  ABSL_MUST_USE_RESULT Segment2d ShiftCenter(const Vec2d offset) const {
    const Vec2d new_start = start_ + offset;
    const Vec2d new_end = end_ + offset;
    return Segment2d(new_start, new_end, *this);
  }

  // Return heading value (this does an atan the first time if the heading
  // is not cached).
  double heading() const {
    double heading = heading_.load(std::memory_order_relaxed);
    if (!IsCached(heading)) {
      heading = atan2(end_.y() - start_.y(), end_.x() - start_.x());
      heading_.store(heading, std::memory_order_relaxed);
    }
    return heading;
  }

  // Cosine and sine of heading.
  double cos_heading() const {
    const double inv_len = inv_length();
    if (inv_len == 0.0) {
      return cos(heading());
    } else {
      return inv_len * (end_.x() - start_.x());
    }
  }

  double sin_heading() const {
    const double inv_len = inv_length();
    if (inv_len == 0.0) {
      return sin(heading());
    } else {
      return inv_len * (end_.y() - start_.y());
    }
  }

  void trig_heading(double *cos_heading, double *sin_heading) const;

  // Same as above but returns the point on the segment closest to the query
  // point.
  double DistanceToPoint(Vec2d query, Vec2d *closest_point) const;

  // Same as above but returns the square distance instead to avoid square roots
  // and be slightly faster.
  double DistanceSqrToPoint(const Vec2d query) const {
    if (start_ == end_) {
      return start_.DistanceSqrToPoint(query);
    }
    return PointToSegmentDistanceSqr(query, start_, end_);
  }

  // Compute the distance from query_point to this line segment.
  // The output fields are set as follows:
  // - closest_point is the point on the segment closest to the query point
  //
  // - parallel_norm is the *signed* distance from the projection of the query
  //   point onto the (infinite) line this segment lies on. This is given in
  //   units of the length of this segment. I.e., if parallel_norm is
  //   in [0,1], the projection lies inside the segment. IF (parallel_norm < 0),
  //   the closest point is outside of the segment  and is closer to start_.
  //   If parallel_norm > 1, the projection is outside of this segment and is
  //   closer to end_.
  //
  // - perp_dist is the unsigned distance from the query point q to
  //   (closest_x, closest_y)
  void DistanceToPoint(Vec2d query, double *parallel_norm, double *perp_dist,
                       Vec2d *closest_point) const;

  // Return true if this line segment intersects the given line segment. Return
  // false if they don't intersect each other or they are parallel to each
  // other or they are coincident to each other. If point is not nullptr, then
  // it is set to where the line segments intersect.
  //
  // NOTE: due to our use of kEpsilon, point might not actually be on either
  // segment, merely within kEpsilon of both.
  bool Intersect(const Segment2d &other, Vec2d *point = nullptr) const;

  // Similar to above, but returns true for coinciding segements.
  bool IntersectOrCoincide(const Segment2d &other) const;

  // Checks if two segments might overlap given their bounding boxes, useful for
  // avoiding unnecessary Intersect() calls.
  // WARNING: this function is about half the cost of Intersect(), so only use
  // it when Intersection is unlikely.
  bool MightIntersect(const Segment2d &other) const {
    return std::max(start_.x(), end_.x()) + kEpsilon >=
               std::min(other.start_.x(), other.end_.x()) &&
           std::min(start_.x(), end_.x()) - kEpsilon <=
               std::max(other.start_.x(), other.end_.x()) &&
           std::max(start_.y(), end_.y()) + kEpsilon >=
               std::min(other.start_.y(), other.end_.y()) &&
           std::min(start_.y(), end_.y()) - kEpsilon <=
               std::max(other.start_.y(), other.end_.y());
  }

  // Generate a sample dist-distance along the line. Returns false if the sample
  // is past the end of the segment. If the segment length is 0, returns false
  // and sets the output to the segment start/end.
  bool Sample(double dist, Vec2d *point) const;

  // Generate samples with a maximum sampling distance along the line. The
  // samples include start point, but exclude end point.
  static std::vector<Vec2d> SamplePoints(const Vec2d &start, const Vec2d &end,
                                         double dist);

  enum PointToSegmentLocation {
    // If the point is to the left (counter-clockwise) of the line going
    // through this segment.
    ON_LEFT_SIDE,
    // If the point is to the right (counter-clockwise) of the line going
    // through this segment.
    ON_RIGHT_SIDE,
    // If the point is on the line going through this segment.
    COLINEAR
  };

  static std::string SideToString(Segment2d::PointToSegmentLocation side) {
    if (side == Segment2d::ON_LEFT_SIDE) {
      return "left";
    } else if (side == Segment2d::ON_RIGHT_SIDE) {
      return "right";
    } else {
      return "colinear";
    }
  }

  // Compute where the point lies with respect to the line that goes
  // through this segment. Optionally returns the signed perpendicular distance
  // to the point (positive corresponds to ON_RIGHT_SIDE and vice versa)
  PointToSegmentLocation SideOfPoint(const Vec2d pp,
                                     double *perp_dist = nullptr) const {
    const Vec2d v = pp - start_;
    const double unnormalized_signed_dist = (end_ - start_).CrossProd(v);
    PointToSegmentLocation result = COLINEAR;
    if (std::abs(unnormalized_signed_dist) >= kEpsilon) {
      result = unnormalized_signed_dist > 0.0 ? ON_LEFT_SIDE : ON_RIGHT_SIDE;
    }
    // Only compute the segment length when perp_dist is not nullptr
    if (perp_dist != nullptr) {
      *perp_dist = -unnormalized_signed_dist / length();
    }
    return result;
  }

  std::string DebugString() const;

 private:
  // Represent an invalid cache value, i.e. the value is not cached.
  static constexpr double kInvalidCacheValue =
      std::numeric_limits<double>::lowest();

  // Construct from two endpoints. Copy any cached data (heading, length,
  // inv_length) from the other segment as an optimization. Only safe to use if
  // the new segment is the same length and heading as the other segment.
  Segment2d(const Vec2d start, const Vec2d end, const Segment2d &other)
      : start_(start), end_(end) {
    CopyCachedValues(other);
  }

  // Construct from two endpoints and pre-computed heading. Copy any cached data
  // (length, inv_length) from the other segment as an optimization. Only safe
  // to use if the new segment is the same length as the other segment.
  Segment2d(const Vec2d start, const Vec2d end, double heading,
            const Segment2d &other)
      : Segment2d(start, end, other) {
    heading_.store(heading, std::memory_order_relaxed);
  }

  // Check if the given cache value is valid. If not, then the value is not
  // cached.
  static bool IsCached(double val) { return val != kInvalidCacheValue; }

  // Copy cached results of expensive computations from another segment. Assumes
  // that both segments should share the same length and heading.
  void CopyCachedValues(const Segment2d &other) {
    if (this == &other) return;
    const auto relaxed = std::memory_order_relaxed;
    heading_.store(other.heading_.load(relaxed), relaxed);
    length_.store(other.length_.load(relaxed), relaxed);
    inv_length_.store(other.inv_length_.load(relaxed), relaxed);
  }

  static double PointToSegmentDistanceSqr(const Vec2d query,
                                          const Vec2d segment_p1,
                                          const Vec2d segment_p2);
  // The segment is parameterized by the coordinates of its two endpoints.
  Vec2d start_;
  Vec2d end_;

  // An alternative common parameterization is the coordinates of the segment
  // center, its length, and heading.
  // - center point is cheap to compute from end points.
  // - length is fairly cheap to compute on the fly (hypot)
  // - heading is more expensive (atan), so we cache it.
  mutable std::atomic<double> heading_{kInvalidCacheValue};
  mutable std::atomic<double> length_{kInvalidCacheValue};
  mutable std::atomic<double> inv_length_{kInvalidCacheValue};

  // We want this class to be copyable, and shallow copy is OK.
};

inline std::ostream &operator<<(std::ostream &os, const Segment2d &seg) {
  return os << seg.DebugString();
}

// Same as PointToSegmentDistance but compute square perpendicular distance
// instead of distance to potentially avoid square root computation.
inline double Segment2d::PointToSegmentDistanceSqr(const Vec2d query,
                                                   const Vec2d segment_p1,
                                                   const Vec2d segment_p2) {
  // Vector from the first line point to the query point.
  const Vec2d v1 = query - segment_p1;
  // Vector between line points.
  const Vec2d v2 = segment_p2 - segment_p1;
  const double v2len_sqr = segment_p1.DistanceSqrToPoint(segment_p2);
  DCHECK_GT(v2len_sqr, 0) << "query=" << query << " p1=" << segment_p1
                          << " p2=" << segment_p2;
  const double inv_v2len_sqr = 1.0 / v2len_sqr;
  // Parallel norm is dot product normalized by square segment length.
  const double parallel_norm = v1.Dot(v2) * inv_v2len_sqr;

  // Square perpendicular distance is square cross product normalized by
  // square segment length.
  double dist = open_dataset::Sqr(v1.CrossProd(v2)) * inv_v2len_sqr;

  if (parallel_norm < 0) {
    // We are outside, closer to segment_p1.
    dist = query.DistanceSqrToPoint(segment_p1);
  } else if (parallel_norm > 1) {
    // We are outside, closer to segment_p2.
    dist = query.DistanceSqrToPoint(segment_p2);
  }

  return dist;
}

}  // namespace open_dataset
}  // namespace waymo

#endif  // WAYMO_OPEN_DATASET_MATH_SEGMENT2D_H_
