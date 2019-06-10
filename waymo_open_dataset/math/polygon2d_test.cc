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

#include "waymo_open_dataset/math/polygon2d.h"

#include <ctype.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <memory>
#include <tuple>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

namespace waymo {
namespace open_dataset {

namespace {
// Epsilon used in this file to avoid double precision problem
// when we compare two numbers.
static constexpr double kEpsilon = 1e-10;

// Test Polygon basic methods such as constructors.
TEST(PolygonTest, Basics) {
  std::vector<Vec2d> points;
  points.push_back(Vec2d(1.0, -1.0));
  points.push_back(Vec2d(2.0, 0.0));
  points.push_back(Vec2d(1.0, 1.0));
  points.push_back(Vec2d(0.0, 1.0));
  points.push_back(Vec2d(0.0, 0.0));
  Polygon2d polygon(points);
  EXPECT_EQ(polygon.NumPoints(), points.size());
  Polygon2d empty;
  EXPECT_EQ(empty.NumPoints(), 0);
  Box2d box1(Vec2d(0, 0), Vec2d(0, 1), 1);
  Polygon2d box2(box1);
  EXPECT_NEAR(box2.Area(), 1.0, 1e-6);
  EXPECT_EQ(box2.NumPoints(), 4);
  Vec2d bottom_left, top_right;
  box2.AxisAlignedBoundingBox(&bottom_left, &top_right);
  EXPECT_EQ(bottom_left.x(), box2.bbox_bottom_left().x());
  EXPECT_EQ(bottom_left.y(), box2.bbox_bottom_left().y());
  EXPECT_EQ(top_right.x(), box2.bbox_top_right().x());
  EXPECT_EQ(top_right.y(), box2.bbox_top_right().y());
}

// Test Polygon::Area().
TEST(PolygonTest, Area) {
  Polygon2d empty;
  EXPECT_DOUBLE_EQ(empty.Area(), 0.0);
  std::vector<Vec2d> points;
  points.push_back(Vec2d(1.0, 1.0));
  points.push_back(Vec2d(0.0, 1.0));
  points.push_back(Vec2d(0.0, 0.0));
  Polygon2d triangle(points);
  EXPECT_DOUBLE_EQ(triangle.Area(), 0.5);
  points.push_back(Vec2d(1.0, 0.0));
  Polygon2d box(points);
  EXPECT_DOUBLE_EQ(box.Area(), 1.0);
  points.push_back(Vec2d(2.0, 0.0));
  points.push_back(Vec2d(2.0, 1.0));
  Polygon2d hexagon(points);
  EXPECT_DOUBLE_EQ(hexagon.Area(), 2.0);
  points.clear();
  points.push_back(Vec2d(-27395.400000000001, 31650.000104521929));
  points.push_back(Vec2d(-27395.400078237279, 31650.000000000004));
  points.push_back(Vec2d(-27395.400000000001, 31650.000000000004));
  Polygon2d very_small_polygon(points);
  EXPECT_TRUE(very_small_polygon.Area() >= 0.0);
}

TEST(PolygonTest, NonEmptyZeroAreaPolygons) {
  // Construct a polygon that consists of many points on a line in an order
  // that's neither clockwise nor counterclockwise. This polygon should not be
  // empty (it has points), but is has no area and has a non-zero perimeter.
  const std::vector<Vec2d> points = {{0.0, 0.0}, {0.0, 2.0}, {0.0, 5.0},
                                     {0.0, 1.0}, {0.0, 2.0}, {0.0, 0.0},
                                     {0.0, 5.0}, {0.0, -1.0}};
  Polygon2d empty_complex(points);
  EXPECT_EQ(empty_complex.points().size(), points.size());
  EXPECT_DOUBLE_EQ(empty_complex.Area(), 0.0);
}

// Test Polygon::PointInside().
TEST(PolygonTest, PointInside) {
  std::vector<Vec2d> points;
  points.push_back(Vec2d(1.0, 0.0));
  points.push_back(Vec2d(0.0, 1.0));
  points.push_back(Vec2d(0.0, 0.0));
  // Triangle.
  Polygon2d triangle(points);
  // Points that on the edge are inside poly.
  EXPECT_TRUE(triangle.PointInside(0.5, 0.5));
  // Vertices are considered inside poly.
  EXPECT_TRUE(triangle.PointInside(0.0, 0.0));
  EXPECT_TRUE(triangle.PointInside(1.0, 0.0));
  // Points outside poly.
  EXPECT_FALSE(triangle.PointInside(0.5, 1.0));
  EXPECT_FALSE(triangle.PointInside(0.5, -1.0));
  EXPECT_FALSE(triangle.PointInside(1.0, 0.00001));
  // Quadrilateral.
  points.push_back(Vec2d(1.0, -1.0));
  Polygon2d quad(points);
  EXPECT_TRUE(quad.PointInside(0.5, -0.2));
  EXPECT_FALSE(quad.PointInside(0.0, -1.0));
  EXPECT_FALSE(quad.PointInside(-5.0, -1.0));
  EXPECT_FALSE(quad.PointInside(1.0, 0.00001));
  // Pentagon.
  points.push_back(Vec2d(2.0, -1.0));
  Polygon2d pentagon(points);
  EXPECT_TRUE(pentagon.PointInside(1.5, -0.7));
  EXPECT_FALSE(pentagon.PointInside(-0.5, 0.0));
  EXPECT_FALSE(pentagon.PointInside(-15.0, 10.0));
  EXPECT_FALSE(pentagon.PointInside(1.0, 0.00001));

  points.clear();
  // Expect true when checking vertices of the polygon.
  points = {Vec2d(0.0, 0.0), Vec2d(1.0, 0.0), Vec2d(1.0, 1.0), Vec2d(0.0, 2.0)};
  Polygon2d quad_1(points);
  for (const Vec2d point : points) {
    EXPECT_TRUE(quad_1.PointInside(point));
  }
}


TEST(PolygonTest, IntersectionPreciseCase) {
  // Two boxes that touch along one edge, should detect no intersection.
  Polygon2d poly_a({{-0.99999997615023983322, -4.99999995230047744599},
                    {-1.00000002384976149905, -2.99999995230047877826},
                    {-5.00000002384975950065, -3.00000004769952210992},
                    {-4.99999997615023872299, -5.00000004769952077766}});

  Polygon2d poly_b({{-0.99999997615023983322, -2.99999995230047744599},
                    {-1.00000002384976149905, -0.99999995230047855621},
                    {-5.00000002384976038883, -1.00000004769952210992},
                    {-4.99999997615023783482, -3.00000004769952122174}});

  // Rotate the two polygons over the same center to test robustness.
  for (double angle = 0.0; angle < M_PI; angle += M_PI / 4) {
    poly_a.Transform(angle, Vec2d(0, 0), Vec2d(0, 0));
    poly_b.Transform(angle, Vec2d(0, 0), Vec2d(0, 0));
    const double intersection_area = poly_a.ComputeIntersectionArea(poly_b);
    EXPECT_NEAR(intersection_area, 0.0, 1e-6);
  }
}

TEST(PolygonTest, IntersectionEdgeCase) {
  // Two polygons that substantially overlap.
  std::vector<Vec2d> points_a(
      {{42.59999990910291955970, 6.60000031143426824087},
       {42.59999990910291955970, 6.40000030845403600210},
       {48.59999999850988672279, -14.40000000149011682993},
       {49.20000000745058343909, -14.40000000149011682993},
       {49.60000001341104791663, -13.99999999552965235239},
       {50.00000001937151239417, -12.19999996870756220346},
       {50.20000002235174463294, -9.99999993592500757700},
       {50.20000002235174463294, -7.99999990612268518930},
       {50.00000001937151239417, -6.39999988228082727915},
       {49.80000001639128015540, -5.19999986439943384653},
       {49.60000001341104791663, -4.39999985247850489145},
       {48.59999999850988672279, -1.39999980777502130991},
       {47.79999998658895776771, 0.20000021606683660025},
       {45.99999995976686761878, 3.40000026375055242056},
       {45.59999995380640314124, 4.00000027269124913687},
       {43.19999991804361627601, 6.40000030845403600210},
       {42.79999991208315179847, 6.60000031143426824087}});

  std::vector<Vec2d> points_b(
      {{48.19999999254942224525, -1.99999981671571802622},
       {48.19999999254942224525, -2.19999981969595026499},
       {48.59999999850988672279, -14.40000000149011682993},
       {48.80000000149011896156, -14.40000000149011682993},
       {49.60000001341104791663, -13.99999999552965235239},
       {50.00000001937151239417, -12.59999997466802668100},
       {50.20000002235174463294, -11.39999995678663324838},
       {50.20000002235174463294, -7.39999989718198847299},
       {50.00000001937151239417, -5.79999987334013056284},
       {49.80000001639128015540, -4.39999985247850489145},
       {49.00000000447035120033, -2.19999981969595026499},
       {48.80000000149011896156, -1.99999981671571802622}});

  Polygon2d poly_a(points_a);
  Polygon2d poly_b(points_b);
  const double area_a = poly_a.Area();
  const double area_b = poly_b.Area();
  for (int i = 0; i < 32; ++i) {
    if (i > 0) {
      poly_a.Transform(2 * M_PI / 32.0, {}, {});
      poly_b.Transform(2 * M_PI / 32.0, {}, {});
    }
    const double intersection_area = poly_a.ComputeIntersectionArea(poly_b);
    const double min_area = std::min(area_a, area_b);
    EXPECT_LE(intersection_area, min_area)
        << "Intersection area " << intersection_area
        << " should be less than minimum of individual areas: " << area_a
        << ", " << area_b;
  }
}

TEST(PolygonTest, IntersectionUnordered) {
  Polygon2d poly_a({{0, 1}, {1, 1}, {1, 3}, {0, 3}});
  Polygon2d poly_b({{0, 2}, {0, 0}, {1, 0}, {1, 2}});

  EXPECT_NEAR(poly_a.ComputeIntersectionArea(poly_b), 1.0, 1e-6);
  EXPECT_NEAR(poly_b.ComputeIntersectionArea(poly_a), 1.0, 1e-6);
}

TEST(PolygonTest, ExactIntersection) {
  {
    const Segment2d s1{{0, 3}, {0, 1}};
    const Segment2d s2{{0, 2}, {0, 0}};
    Vec2d intx;
    EXPECT_TRUE(internal::ExactSegmentIntersection(s1, s2, &intx));
  }
  {
    const Segment2d s1{{0, 3}, {0, 1}};
    const Segment2d s2{{0, 0}, {0, 2}};
    Vec2d intx;
    EXPECT_TRUE(internal::ExactSegmentIntersection(s1, s2, &intx));
  }
  {
    const Segment2d s1{{0, 3}, {0, 1}};
    const Segment2d s2{{0, 0}, {0, -2}};
    Vec2d intx;
    EXPECT_FALSE(internal::ExactSegmentIntersection(s1, s2, &intx));
  }
  {
    const Segment2d s1{{3, 0}, {1, 0}};
    const Segment2d s2{{2, 0}, {0, 0}};
    Vec2d intx;
    EXPECT_TRUE(internal::ExactSegmentIntersection(s1, s2, &intx));
  }
  {
    const Segment2d s1{{3, 0}, {1, 0}};
    const Segment2d s2{{0, 0}, {2, 0}};
    Vec2d intx;
    EXPECT_TRUE(internal::ExactSegmentIntersection(s1, s2, &intx));
  }
  {
    const Segment2d s1{{3, 0}, {1, 0}};
    const Segment2d s2{{0, 0}, {-2, 0}};
    Vec2d intx;
    EXPECT_FALSE(internal::ExactSegmentIntersection(s1, s2, &intx));
  }
}

}  // namespace
}  // namespace open_dataset
}  // namespace waymo
