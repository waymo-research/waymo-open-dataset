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

#ifndef WAYMO_OPEN_DATASET_MATH_POLYGON2D_H_
#define WAYMO_OPEN_DATASET_MATH_POLYGON2D_H_

#include <utility>
#include <vector>

#include "waymo_open_dataset/math/box2d.h"
#include "waymo_open_dataset/math/segment2d.h"
#include "waymo_open_dataset/math/vec2d.h"

namespace waymo {
namespace open_dataset {

// A class for *convex* polygons only.
//
// Used for computing intersections with other polygons only.
class Polygon2d {
 public:
  // Create an empty polygon.
  Polygon2d() = default;
  // Make sure this class remains movable and copiable.
  Polygon2d(Polygon2d &&) = default;
  Polygon2d(const Polygon2d &) = default;
  Polygon2d &operator=(Polygon2d &&) = default;
  Polygon2d &operator=(const Polygon2d &) = default;

  // Create a polygon from the given box.
  explicit Polygon2d(const Box2d &box) {
    box.GetCornersInVectorCounterClockwise(&points_);
    CHECK_EQ(points_.size(), 4);
    BuildFromPoints();
  }

  // Create a convex polygon with the given circular array of 2d points. There
  // should be at least three 2d points. 2d points must be in counter-clockwise
  // order.
  explicit Polygon2d(std::vector<Vec2d> points);

  // Return the number of points.
  int NumPoints() const { return num_points_; }

  // Return the point array.
  const std::vector<Vec2d> &points() const { return points_; }

  // Performs a fast check to see if the two polygons are potentially
  // intersecting. A false return value guarantees that there's no intersection.
  bool MaybeHasIntersectionWith(const Polygon2d &other) const;

  // Return the segments that make up this polygon. Each segment ii
  // is between points ii -> (ii + 1) % num_points_.
  Vec2d Segment(int ii) const {
    DCHECK_LE(0, ii);
    DCHECK_LT(ii, num_points_);
    return points_[Next(ii)] - points_[ii];
  }

  // Return the index of the previous (or the next) point of the polygon.
  int Prev(int index) const {
    return index == 0 ? num_points_ - 1 : (index - 1);
  }
  int Next(int index) const {
    return index == num_points_ - 1 ? 0 : (index + 1);
  }

  // The corners of the axis aligned bounding box for this polygon.
  Vec2d bbox_bottom_left() const { return bbox_bottom_left_; }
  Vec2d bbox_top_right() const { return bbox_top_right_; }

  double Area() const { return AreaNoValidityCheck(); }
  // Like Area(), but performs no check on the validity of the polygon/area. May
  // return a negative area.
  double AreaNoValidityCheck() const;

  // Return true if the given point is inside the polygon. The implementation
  // uses a check on how many times a ray originating from (x,y) intersects the
  // polygon, and works with both convex and concave polygons.
  bool PointInside(Vec2d p) const;
  bool PointInside(double x, double y) const {
    return PointInside(Vec2d(x, y));
  }

  // Returns the area of the intersection between this polygon and the given
  // polygon.
  double ComputeIntersectionArea(const Polygon2d &other) const;

  void AxisAlignedBoundingBox(Vec2d *bottom_left, Vec2d *top_right) const;

  // Checks whether the given set of points form a convex hull, and if yes
  // return whether they are in clockwise or counter-clockwise order. It does
  // not check whether the polygon is degenerate, has repeated points.
  static bool IsConvexHull(const std::vector<Vec2d> &points,
                           bool *counter_clockwise = nullptr);

  // Slides the polygon by the given (dx, dy) vector.
  void ShiftCenter(Vec2d offset);

  // Rotates the polygon around the given center, then shift by the given
  // offset.
  void Transform(double angle, Vec2d center, Vec2d offset);

  // Debugging function for printing the polygon's vertices to a string.
  static std::string PrintPointsToString(const std::vector<Vec2d> &points,
                                         bool full_precision);
  std::string DebugString() const;

  // Epsilon used in this file to avoid double precision problem
  // when we compare two numbers.
  static constexpr double kEpsilon = 1e-10;

 private:
  // Construct other useful data, given the points_ vector is filled.
  void BuildFromPoints();

  // Return all vectices of the intersection between this polygon and the given
  // one, which should form a convex polygon. Both this polygon and the given
  // polygon must be convex ones. This function is used to either compute the
  // intersection polygon or the intersection area.
  std::vector<Vec2d> ComputeIntersectionVertices(const Polygon2d &other) const;

  // The corners of the axis-aligned bounding box for this polygon.
  Vec2d bbox_bottom_left_, bbox_top_right_;

  // The polygon is parameterized by this circular array of 2d points (corners).
  // The points are in counter-clockwise order.
  int num_points_ = 0;
  std::vector<Vec2d> points_;
};

namespace internal {

// Compute the intersection of two segments. Returns true if the segments
// intersect or coincide.
bool ExactSegmentIntersection(const Segment2d &s1, const Segment2d &s2,
                              Vec2d *intersection);

}  // namespace internal

}  // namespace open_dataset
}  // namespace waymo

#endif  // WAYMO_OPEN_DATASET_MATH_POLYGON2D_H_
