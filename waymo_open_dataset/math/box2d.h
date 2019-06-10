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

#ifndef WAYMO_OPEN_DATASET_MATH_BOX2D_H_
#define WAYMO_OPEN_DATASET_MATH_BOX2D_H_

#include <type_traits>
#include <vector>

#include "waymo_open_dataset/math/segment2d.h"
#include "waymo_open_dataset/math/vec2d.h"

namespace waymo {
namespace open_dataset {

// A 2d box (rectangle). Parameterized internally as a 2d line segment with
// width.
class Box2d {
 public:
  // Convenience names for accessing specific corners of the box.
  enum Corner : int {
    kRearLeft = 0,
    kFrontLeft = 1,
    kFrontRight = 2,
    kRearRight = 3
  };

  Box2d() {}

  // Initialize the box given the end points of its "long" axis and width. The
  // axis endpoints are the points in the middle of the two "wide" sides.
  Box2d(const Vec2d axis_start, const Vec2d axis_end, double width)
      : axis_(axis_start, axis_end), width_(width) {}

  // Initialize the box given the coordinates of its center, the heading of the
  // box axis ("long" side) in radians, its length and width.
  Box2d(const Vec2d center, double axis_heading, double length, double width)
      : axis_(center, axis_heading, length), width_(width) {}

  // Initialize the box given the coordinates of its center, a unit vector in
  // the direction of the axis heading, its length and width. This constructor
  // accepts the direction components (the sine and cosine of axis_heading) to
  // avoid expensive recalculations.
  Box2d(const Vec2d center, const Vec2d direction, double length, double width)
      : axis_(center, direction, length), width_(width) {}

  // Initialize the box given one corner, the heading of the box axis ("long"
  // side) in radians, its length, and width. Corners are ordered clockwise.
  // The first one is the bottom-left corner (when heading axis points up).
  Box2d(int corner_index, const Vec2d corner, double axis_heading,
        double length, double width) {
    Set(corner_index, corner, axis_heading, length, width);
  }

  // Initialize the box as axis-aligned with specified bottom-left and top-right
  // corners.
  Box2d(const Vec2d bottom_left, const Vec2d top_right) {
    Set(bottom_left, top_right);
  }

  // The following Set functions mirror the constructors above.
  void Set(const Vec2d center, double axis_heading, double length,
           double width) {
    width_ = width;
    axis_ = Segment2d(center, axis_heading, length);
  }

  void Set(const Vec2d axis_start, const Vec2d axis_end, double width) {
    axis_ = Segment2d(axis_start, axis_end);
    width_ = width;
  }

  void Set(int corner_index, const Vec2d corner, double axis_heading,
           double length, double width) {
    Set(corner_index, corner, Vec2d::CreateUnitFromAngle(axis_heading), length,
        width);
  }

  void Set(Vec2d bottom_left, Vec2d top_right);

  // Those two Set functions accept the already calculated direction (cos/sin of
  // axis_heading) to avoid the expensive calculation of them.
  void Set(const Vec2d center, const Vec2d direction, double length,
           double width) {
    width_ = width;
    axis_ = Segment2d(center, direction, length);
  }

  void Set(int corner_index, Vec2d corner, Vec2d direction, double length,
           double width);

  // Resets the box so that it keeps the same center and heading, but changing
  // its size to the specified length and width.
  void SetSize(double length, double width) {
    axis_ = Segment2d(axis_.center(), axis_.heading(), length);
    set_width(width);
  }

  // Slide the box so that it's centered around the given coordinates.
  void MoveCenter(const Vec2d new_center) {
    axis_ = axis_.MoveCenter(new_center);
  }

  // Slide the box by the given (dx, dy) vector.
  void ShiftCenter(const Vec2d offset) { axis_ = axis_.ShiftCenter(offset); }

  // Rotate the box around the center.
  void RotateAroundCenter(double theta) {
    axis_ = axis_.RotateAroundCenter(theta);
  }
  // Rotate the box around the origin. This is useful when changing to a
  // different, rotated, coordinate frame.
  void RotateAroundOrigin(double theta) {
    axis_ = axis_.RotateAroundOrigin(theta);
  }

  // Rotate the box around an arbitrary point.
  void RotateAroundPoint(const Vec2d point, double theta) {
    axis_ = axis_.RotateAroundPoint(point, theta);
  }

  double axis_heading() const { return axis_.heading(); }
  double cos_heading() const { return axis_.cos_heading(); }
  double sin_heading() const { return axis_.sin_heading(); }

  // Return the length&  width of the box.
  double length() const { return axis_.length(); }
  double width() const { return width_; }

  // Set the width of the box.
  void set_width(double w) { width_ = w; }

  // Return the area of the box.
  double GetArea() const { return width_ * axis_.length(); }

  // Return the segment that defines the main axis of the box.
  const Segment2d &axis() const { return axis_; }
  void SetAxis(const Vec2d start, const Vec2d end) {
    axis_ = Segment2d(start, end);
  }

  Vec2d axis_start() const { return axis_.start(); }
  Vec2d axis_end() const { return axis_.end(); }

  // Return the coordinates of the center of the box.
  Vec2d center() const { return axis_.center(); }

  // Get the coordinates of the four corners of the box. The arrays
  // "xx" and "yy" must have memory for at least 4 elements each.
  // Corners are ordered clockwise. The first one is the bottom-left
  // corner (when axis points up).
  void GetCorners(double *xx, double *yy) const;

  // Get the four corners of the box in the format of Vec2d. The corners
  // are ordered counter-clockwise. The first one is the bottom-left corner
  // (when axis points up).
  void GetCornersInVectorCounterClockwise(std::vector<Vec2d> *corners) const;

  std::string DebugString() const;

  // Comparison operators for Box2d.
  bool operator==(const Box2d &other) const {
    return axis_start() == other.axis_start() &&
           axis_end() == other.axis_end() && width() == other.width();
  }
  bool operator!=(const Box2d &other) const { return !(*this == other); }

 private:
  // The box is parameterized by the coordinates of the centers of its "wide"
  // sides and width. I.e., the line segment between axis_start and axis_end
  // goes down the middle of the box parallel to its "long" side.
  Segment2d axis_;
  double width_ = 0.0;

  // Shallow copying of this class is OK.
};

static_assert(std::is_trivially_destructible<Box2d>::value,
              "Box2d should be trivially destructible");

}  // namespace open_dataset
}  // namespace waymo

#endif  // WAYMO_OPEN_DATASET_MATH_BOX2D_H_
