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

#include "waymo_open_dataset/math/vec2d.h"

#include <limits>
#include <memory>
#include <vector>

#include <gtest/gtest.h>

namespace waymo {
namespace open_dataset {
namespace {

TEST(Vec2dTest, FromTuple) {
  constexpr Vec2d v1 = {1, 2};
  EXPECT_EQ(1.0, v1.x());
  EXPECT_EQ(2.0, v1.y());

  constexpr Vec2d v2{std::make_tuple(3.0, 4.0)};
  EXPECT_EQ(3.0, v2.x());
  EXPECT_EQ(4.0, v2.y());
}

TEST(Vec2dTest, BasicsOperation) {
  Vec2d v;
  EXPECT_TRUE(v.IsZero());

  v.Set(1.0, 0.0);
  EXPECT_FALSE(v.IsZero());
  EXPECT_DOUBLE_EQ(v.L1Norm(), 1.0);
  EXPECT_DOUBLE_EQ(v.Length(), 1.0);
  EXPECT_DOUBLE_EQ(v.Sqr(), 1.0);
  v.set_x(0.0);
  v.set_y(1.0);
  v.Shift(0.0, -1.0);
  EXPECT_DOUBLE_EQ(v.L1Norm(), 0.0);
  EXPECT_DOUBLE_EQ(v.Length(), 0.0);
  EXPECT_DOUBLE_EQ(v.Sqr(), 0.0);
  v.set_x(3.0);
  v.set_y(0.0);
  EXPECT_DOUBLE_EQ(v.L1Norm(), 3.0);
  EXPECT_DOUBLE_EQ(v.Length(), 3.0);
  EXPECT_DOUBLE_EQ(v.Sqr(), 9.0);
  v.set_y(-4.0);
  EXPECT_DOUBLE_EQ(v.L1Norm(), 7.0);
  EXPECT_DOUBLE_EQ(v.Length(), 5.0);
  EXPECT_DOUBLE_EQ(v.Sqr(), 25.0);

  v.Set(0, 0);
  EXPECT_TRUE(v.IsZero());
}

TEST(Vec2dTest, FromAngles) {
  const double kEpsilon = 1e-13;
  Vec2d v;
  v.UnitFromAngle(0);
  EXPECT_NEAR(v.x(), 1, kEpsilon);
  EXPECT_NEAR(v.y(), 0, kEpsilon);
  EXPECT_EQ(v, Vec2d::CreateUnitFromAngle(0));

  Vec2d v1(0);
  EXPECT_NEAR(v1.x(), 1, kEpsilon);
  EXPECT_NEAR(v1.y(), 0, kEpsilon);

  v.UnitFromAngle(M_PI / 2);
  EXPECT_NEAR(v.x(), 0, kEpsilon);
  EXPECT_NEAR(v.y(), 1, kEpsilon);
  EXPECT_EQ(v, Vec2d::CreateUnitFromAngle(M_PI / 2));

  Vec2d v2(M_PI_2);
  EXPECT_NEAR(v2.x(), 0, kEpsilon);
  EXPECT_NEAR(v2.y(), 1, kEpsilon);

  v.UnitFromAngle(M_PI);
  EXPECT_NEAR(v.x(), -1, kEpsilon);
  EXPECT_NEAR(v.y(), 0, kEpsilon);

  Vec2d v3(M_PI);
  EXPECT_NEAR(v3.x(), -1, kEpsilon);
  EXPECT_NEAR(v3.y(), 0, kEpsilon);

  v.FromAngleAndLength(M_PI, 15.0);
  EXPECT_NEAR(v.x(), -15, kEpsilon);
  EXPECT_NEAR(v.y(), 0, kEpsilon);
  EXPECT_EQ(v, Vec2d::CreateFromAngleAndLength(M_PI, 15.0));

  v.FromAngleAndLength(0, 15.0);
  EXPECT_NEAR(v.x(), 15.0, kEpsilon);
  EXPECT_NEAR(v.y(), 0, kEpsilon);
  EXPECT_EQ(v, Vec2d::CreateFromAngleAndLength(0, 15.0));
}

TEST(Vec2dTest, Projection) {
  Vec2d v1(1, 0);
  Vec2d v2(0, 1);
  Vec2d v3(2, 0);
  EXPECT_DOUBLE_EQ(v1.Dot(v2), 0.0);
  EXPECT_DOUBLE_EQ(v1.Dot(v3), 2.0);
  EXPECT_DOUBLE_EQ(v1.Project(v2), 0.0);
  EXPECT_DOUBLE_EQ(v1.Project(v3), 1.0);
  EXPECT_DOUBLE_EQ(v1.Project(0), 1.0);
  EXPECT_NEAR(v1.Project(M_PI/2), 0.0, 1e-15);
}

TEST(Vec2dTest, Normalize) {
  Vec2d v1(1.234, 0);
  Vec2d v2(10, 0);
  Vec2d v3(2, -2);
  Vec2d v1_norm = v1;
  Vec2d v2_norm = v2;
  Vec2d v3_norm = v3;
  v1_norm.Normalize();
  v2_norm.Normalize();
  v3_norm.Normalize();
  EXPECT_DOUBLE_EQ(v1_norm.Length(), 1.0);
  EXPECT_DOUBLE_EQ(v2_norm.Length(), 1.0);
  EXPECT_DOUBLE_EQ(v3_norm.Length(), 1.0);
  EXPECT_DOUBLE_EQ(v1_norm.Project(v1), 1.0);
  EXPECT_DOUBLE_EQ(v2_norm.Project(v2), 1.0);
  EXPECT_DOUBLE_EQ(v3_norm.Project(v3), 1.0);
}

TEST(Vec2dTest, Normalized) {
  Vec2d v1(1.234, 0);
  Vec2d v2(10, 0);
  Vec2d v3(2, -2);
  EXPECT_DOUBLE_EQ(v1.Normalized().Length(), 1.0);
  EXPECT_DOUBLE_EQ(v2.Normalized().Length(), 1.0);
  EXPECT_DOUBLE_EQ(v3.Normalized().Length(), 1.0);
  EXPECT_DOUBLE_EQ(v1.Normalized().Project(v1), 1.0);
  EXPECT_DOUBLE_EQ(v2.Normalized().Project(v2), 1.0);
  EXPECT_DOUBLE_EQ(v3.Normalized().Project(v3), 1.0);
}

TEST(Vec2dTest, Flip) {
  Vec2d v1(1, 2);
  Vec2d v2(-1, -2);
  Vec2d v3(0, 0);
  v1.Flip();
  v3.Flip();
  EXPECT_DOUBLE_EQ((v1 - v2).Length(), 0);
  EXPECT_DOUBLE_EQ(v3.Length(), 0);
}

TEST(Vec2dTest, CrossProd) {
  Vec2d v1(1, 0);
  Vec2d v2(0, 1);
  Vec2d v3(2, 0);
  EXPECT_DOUBLE_EQ(v1.CrossProd(v2), 1.0);
  EXPECT_DOUBLE_EQ(v2.CrossProd(v1), -1.0);
  EXPECT_DOUBLE_EQ(v1.CrossProd(v3), 0.0);
  EXPECT_DOUBLE_EQ(v3.CrossProd(v2), 2.0);
}

TEST(Vec2dTest, Perp) {
  const Vec2d v(3.0, 2.0);
  const Vec2d perp = v.Perp();

  // Check that the result is perpendicular.
  EXPECT_DOUBLE_EQ(0.0, perp.Dot(v));

  // Check that the result has the correct handiness.
  EXPECT_GT(v.CrossProd(perp), 0.0);
  EXPECT_LT(perp.CrossProd(v), 0.0);

  // Check that the result has the same length.
  EXPECT_DOUBLE_EQ(v.Length(), perp.Length());
}

TEST(Vec2dTest, Distances) {
  Vec2d v(1, 0);
  EXPECT_DOUBLE_EQ(v.DistanceToPoint(Vec2d(0, 0)), 1.0);
  EXPECT_DOUBLE_EQ(v.DistanceToPoint(Vec2d(0, 1)), sqrt(2.0));
  EXPECT_DOUBLE_EQ(v.DistanceToPoint(Vec2d(1, 1)), 1.0);
  EXPECT_DOUBLE_EQ(v.DistanceSqrToPoint(Vec2d(0, 1)), 2);
  EXPECT_DOUBLE_EQ(v.DistanceSqrToPoint(Vec2d(1, 0)), 0.0);
}

TEST(Vec2dTest, Operators) {
  const Vec2d v1(1.0, 2.0);
  // operator vec + vec
  const Vec2d v2 = v1 + v1;
  EXPECT_DOUBLE_EQ(v2.x(), 2.0);
  EXPECT_DOUBLE_EQ(v2.y(), 4.0);
  // operator vec - vec
  Vec2d v3 = v1 - v1;
  EXPECT_DOUBLE_EQ(v3.x(), 0.0);
  EXPECT_DOUBLE_EQ(v3.y(), 0.0);
  // operator vec * scalar
  Vec2d v4 = v1 * 5.0;
  EXPECT_DOUBLE_EQ(v4.x(), 5.0);
  EXPECT_DOUBLE_EQ(v4.y(), 10.0);
  // operator scalar * vec
  v4 = 5.0 * v1;
  EXPECT_DOUBLE_EQ(v4.x(), 5.0);
  EXPECT_DOUBLE_EQ(v4.y(), 10.0);
  // operator vec / scalar
  Vec2d v5 = v1 / 5.0;
  EXPECT_DOUBLE_EQ(v5.x(), 0.2);
  EXPECT_DOUBLE_EQ(v5.y(), 0.4);
  // operator = (implicit)
  Vec2d v6;
  v6 = v1;
  EXPECT_DOUBLE_EQ(v1.x(), v6.x());
  EXPECT_DOUBLE_EQ(v1.y(), v6.y());
  // operator -vec
  Vec2d v7 = -v1;
  EXPECT_DOUBLE_EQ(v7.x(), -1.0);
  EXPECT_DOUBLE_EQ(v7.y(), -2.0);
  // operator ==
  EXPECT_TRUE(v1 == v1);
  EXPECT_FALSE(v1 == v2);
  // operator !=
  EXPECT_FALSE(v1 != v1);
  EXPECT_TRUE(v1 != v2);
  // operator <
  EXPECT_FALSE(v1 < v1);
  EXPECT_TRUE(v1 < v2);
  EXPECT_FALSE(v2 < v1);
}

TEST(Vec2dTest, IsLeftOfLine) {
  // start
  // (0,1) +
  //       |  .   left
  //       |    .
  //       |      .
  //       +--------+
  //   (0,0)          end(100, 0)
  const Vec2d start(0.0, 1.0);
  const Vec2d end(100, 0.0);
  for (int i = -100; i <= 100; i += 10) {
    {
      Vec2d v1(0.0, i);
      EXPECT_EQ(v1.y() > start.y(), v1.IsLeftOfLine(start, end));
    }
    {
      Vec2d v2(i, 0.0);
      EXPECT_EQ(v2.x() > end.x(), v2.IsLeftOfLine(start, end));
    }
  }
}

TEST(Vec2dTest, ElemWiseMinMax) {
  const Vec2d min = Vec2d::FromScalar(std::numeric_limits<double>::max());
  const Vec2d max = Vec2d::FromScalar(std::numeric_limits<double>::lowest());

  EXPECT_EQ(Vec2d(1, 2), min.ElemWiseMin({1, 2}).ElemWiseMin({4, 3}));
  EXPECT_EQ(Vec2d(4, 3), max.ElemWiseMax({1, 2}).ElemWiseMax({4, 3}));
}

TEST(Vec2dTest, Rotate) {
  std::vector<Vec2d> test_vectors = {
      {0.5, 0.2}, {-0.3, 3.0}, {-2.3, -1.5}, {1.2, -2.1}};
  std::vector<double> rotate_theta = {
      0.0, M_PI / 4.0, M_PI / 2.0, 5.0 * M_PI / 8.0, M_PI, 3.0 * M_PI / 2.0};

  for (const auto& test_vector : test_vectors) {
    for (int jj = 0; jj < rotate_theta.size(); ++jj) {
      const double theta = rotate_theta[jj];
      Vec2d vec1 = test_vector;
      vec1.Rotate(theta);
      Vec2d vec2 = test_vector.Rotated(theta);
      EXPECT_EQ(vec1, vec2);
    }
  }
}

}  // namespace
}  // namespace open_dataset
}  // namespace waymo
