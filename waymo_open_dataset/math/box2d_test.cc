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

#include <gtest/gtest.h>

namespace waymo {
namespace open_dataset {
namespace {

void TestAxis(const Segment2d &axis, double x1, double y1, double x2,
              double y2) {
  EXPECT_NEAR(x1, axis.start().x(), 0.0001);
  EXPECT_NEAR(y1, axis.start().y(), 0.0001);
  EXPECT_NEAR(x2, axis.end().x(), 0.0001);
  EXPECT_NEAR(y2, axis.end().y(), 0.0001);
}

TEST(Box2dTest, Constructor_Vec2d_Vec2d) {
  Box2d box;
  Segment2d axis;

  // Test points.
  box = Box2d(Vec2d(1.0, 1.0), Vec2d(1.0, 1.0));
  axis = box.axis();
  TestAxis(axis, 1.0, 1.0, 1.0, 1.0);
  EXPECT_EQ(0.0, box.width());

  // Test lines.
  box = Box2d(Vec2d(1.0, 1.0), Vec2d(2.0, 1.0));
  axis = box.axis();
  TestAxis(axis, 1.0, 1.0, 2.0, 1.0);
  EXPECT_EQ(0.0, box.width());

  box = Box2d(Vec2d(1.0, 1.0), Vec2d(1.0, 2.0));
  axis = box.axis();
  TestAxis(axis, 1.0, 1.0, 1.0, 2.0);
  EXPECT_EQ(0.0, box.width());

  // Test boxes.
  box = Box2d(Vec2d(1.0, 1.0), Vec2d(2.0, 2.0));
  axis = box.axis();
  TestAxis(axis, 1.0, 1.5, 2.0, 1.5);
  EXPECT_EQ(1.0, box.width());

  box = Box2d(Vec2d(1.0, 1.0), Vec2d(3.0, 2.0));
  axis = box.axis();
  TestAxis(axis, 1.0, 1.5, 3.0, 1.5);
  EXPECT_EQ(1.0, box.width());

  box = Box2d(Vec2d(1.0, 1.0), Vec2d(2.0, 3.0));
  axis = box.axis();
  TestAxis(axis, 1.5, 1.0, 1.5, 3.0);
  EXPECT_EQ(1.0, box.width());
}

// Tests the Box2d constructor that uses a unit vector in the direction of the
// box heading.
TEST(Box2dTest, Constructor_Direction) {
  // An axis-aligned box oriented north.
  Box2d box(/*center=*/Vec2d(10.0, 20.0), Vec2d::CreateUnitFromAngle(M_PI_2),
            /*length=*/2.0, /*width=*/1.0);
  TestAxis(box.axis(), 10.0, 19.0, 10.0, 21.0);
  EXPECT_EQ(1.0, box.width());

  // A box oriented north-east.
  box = Box2d(/*center=*/Vec2d(0.0, 0.0), Vec2d::CreateUnitFromAngle(M_PI_4),
              /*length=*/4.0, /*width=*/4.0);
  TestAxis(box.axis(), -M_SQRT2, -M_SQRT2, M_SQRT2, M_SQRT2);
  EXPECT_EQ(4.0, box.width());
}

}  // namespace
}  // namespace open_dataset
}  // namespace waymo
