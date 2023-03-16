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

#include "waymo_open_dataset/math/vec3d.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

namespace waymo {
namespace open_dataset {
namespace {

TEST(Vec3dTest, Norms) {
  Vec3d v(1.234, 0.0, -1.234);
  Vec3d norm = v.Normalized();
  EXPECT_DOUBLE_EQ(norm.Length(), 1.0);
  EXPECT_NEAR(norm.x(), .707107, 1e-6);
  EXPECT_NEAR(norm.y(), 0, 1e-12);
  EXPECT_NEAR(norm.z(), -.707107, 1e-6);

  // Test zero input.
  Vec3d zero(0, 0, 0);
  Vec3d zero_norm = zero.Normalized();
  EXPECT_EQ(zero_norm.x(), 0);
  EXPECT_EQ(zero_norm.y(), 0);
  EXPECT_EQ(zero_norm.z(), 0);
}

TEST(VectorTest, Addition) {
  const Vec3d v1(1.0, 2.0, 3.0);
  // operator vec + vec
  const Vec3d v2 = v1 + v1;
  EXPECT_DOUBLE_EQ(v2.x(), 2.0);
  EXPECT_DOUBLE_EQ(v2.y(), 4.0);
  EXPECT_DOUBLE_EQ(v2.z(), 6.0);
}

TEST(VectorTest, Subtraction) {
  const Vec3d v1(1.0, 2.0, 3.0);
  // operator vec - vec
  Vec3d v3 = v1 - v1;
  EXPECT_DOUBLE_EQ(v3.x(), 0.0);
  EXPECT_DOUBLE_EQ(v3.y(), 0.0);
  EXPECT_DOUBLE_EQ(v3.z(), 0.0);
}

TEST(VectorTest, ScalarPostMultiplication) {
  const Vec3d v1(1.0, 2.0, 3.0);
  // operator vec * scalar
  Vec3d v4 = v1 * 5.0;
  EXPECT_DOUBLE_EQ(v4.x(), 5.0);
  EXPECT_DOUBLE_EQ(v4.y(), 10.0);
  EXPECT_DOUBLE_EQ(v4.z(), 15.0);
}

TEST(VectorTest, ScalarPreMultiplication) {
  const Vec3d v1(1.0, 2.0, 3.0);
  // operator scalar * vec
  Vec3d v5 = 5.0 * v1;
  EXPECT_DOUBLE_EQ(v5.x(), 5.0);
  EXPECT_DOUBLE_EQ(v5.y(), 10.0);
  EXPECT_DOUBLE_EQ(v5.z(), 15.0);
}

TEST(VectorTest, ScalarDivision) {
  const Vec3d v1(1.0, 2.0, 3.0);
  // operator vec / scalar
  Vec3d v6 = v1 / 5.0;
  EXPECT_DOUBLE_EQ(v6.x(), 0.2);
  EXPECT_DOUBLE_EQ(v6.y(), 0.4);
  EXPECT_DOUBLE_EQ(v6.z(), 0.6);
}

TEST(VectorTest, UnaryNegation) {
  const Vec3d v1(1.0, 2.0, 3.0);
  // operator -vec
  Vec3d v7 = -v1;
  EXPECT_DOUBLE_EQ(v7.x(), -1.0);
  EXPECT_DOUBLE_EQ(v7.y(), -2.0);
  EXPECT_DOUBLE_EQ(v7.z(), -3.0);
}

TEST(VectorTest, EqualsOperator) {
  const Vec3d v1(1.0, 2.0, 3.0);
  const Vec3d v2 = v1 + v1;
  // operator ==
  EXPECT_TRUE(v1 == v1);
  EXPECT_FALSE(v1 == v2);
}

}  // namespace
}  // namespace open_dataset
}  // namespace waymo
