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

#include "waymo_open_dataset/metrics/iou.h"

#include <cmath>

#include <gtest/gtest.h>
#include "waymo_open_dataset/label.pb.h"
#include "waymo_open_dataset/metrics/test_utils.h"

namespace waymo {
namespace open_dataset {
namespace {
constexpr double kError = 1e-6;

TEST(ComputeIoU, AABox2d) {
  const Label::Box b1 = BuildAA2dBox(0.0, 0.0, 1.0, 2.0);
  const Label::Box b2 = BuildAA2dBox(0.0, 0.0, 1.0, 2.0);
  const Label::Box b3 = BuildAA2dBox(10.0, 10.0, 1.0, 2.0);
  const Label::Box b4 = BuildAA2dBox(0.5, 0.0, 1.0, 4.0);
  // Zero-sized box.
  const Label::Box b5 = BuildAA2dBox(0.0, 0.0, 0.0, 0.0);

  EXPECT_NEAR(1.0, ComputeIoU(b1, b2, Label::Box::TYPE_AA_2D), kError);
  EXPECT_NEAR(0.0, ComputeIoU(b1, b3, Label::Box::TYPE_AA_2D), kError);
  EXPECT_NEAR(0.5 * 2.0 / (2.0 + 4.0 - 0.5 * 2.0),
              ComputeIoU(b1, b4, Label::Box::TYPE_AA_2D), kError);
  EXPECT_NEAR(0.0, ComputeIoU(b1, b5, Label::Box::TYPE_AA_2D), kError);
}

TEST(ComputeIoU, Box2d) {
  const Label::Box b1 = BuildBox2d(0.0, 0.0, 1.0, 2.0, 0.0);
  const Label::Box b2 = BuildBox2d(0.0, 0.0, 1.0, 2.0, 0.0);
  const Label::Box b3 = BuildBox2d(10.0, 10.0, 1.0, 2.0, 1.0);
  const Label::Box b4 = BuildBox2d(0.5, 0.0, 1.0, 4.0, 0.0);
  const Label::Box b5 = BuildBox2d(0.5, 0.0, 1.0, 4.0, M_PI * 0.5);
  const Label::Box b6 = BuildBox2d(0.5, 0.0, 1.0, 4.0, M_PI * 0.25);
  // Zero-sized box.
  const Label::Box b7 = BuildBox2d(0.0, 0.0, 0.0, 0.0, 0.0);

  EXPECT_NEAR(1.0, ComputeIoU(b1, b2, Label::Box::TYPE_2D), kError);
  EXPECT_NEAR(0.0, ComputeIoU(b1, b3, Label::Box::TYPE_2D), kError);
  EXPECT_NEAR(0.5 * 2.0 / (2.0 + 4.0 - 0.5 * 2.0),
              ComputeIoU(b1, b4, Label::Box::TYPE_2D), kError);
  EXPECT_NEAR(1.0 / (2.0 + 4.0 - 1.0), ComputeIoU(b1, b5, Label::Box::TYPE_2D),
              kError);
  EXPECT_NEAR(0.24074958176697656, ComputeIoU(b1, b6, Label::Box::TYPE_2D),
              kError);
  EXPECT_NEAR(0.0, ComputeIoU(b1, b7, Label::Box::TYPE_2D), kError);
}

TEST(ComputeIoU, Box3d) {
  const Label::Box b1 = BuildBox3d(0.0, 0.0, 0.0, 1.0, 2.0, 2.0, 0.0);
  // Same as b1.
  const Label::Box b2 = BuildBox3d(0.0, 0.0, 0.0, 1.0, 2.0, 2.0, 0.0);
  // Zero overlap.
  const Label::Box b3 = BuildBox3d(10.0, 10.0, 0.0, 1.0, 2.0, 2.0, 1.0);
  const Label::Box b4 = BuildBox3d(0.5, 0.0, 0.0, 1.0, 4.0, 4.0, 0.0);
  const Label::Box b5 = BuildBox3d(0.5, 0.0, 0.0, 1.0, 4.0, 4.0, M_PI * 0.5);
  const Label::Box b6 = BuildBox3d(0.5, 0.0, 0.0, 1.0, 4.0, 2.0, M_PI * 0.25);
  // Zero-sized box.
  const Label::Box b7 = BuildBox3d(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
  const Label::Box b8 = BuildBox3d(0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0);
  // Illegal box.
  const Label::Box b9 = BuildBox3d(0.0, 0.0, 0.0, 1.0, -1.0, 1.0, 0.0);
  // Tiny box
  const Label::Box b10 = BuildBox3d(0.0, 0.0, 0.0, 1.0, 1e-12, 1.0, 0.0);
  // Small box
  const Label::Box b11 = BuildBox3d(0.0, 0.0, 0.0, 1.0, 1.01e-2, 2.0, 0.0);

  EXPECT_NEAR(1.0, ComputeIoU(b1, b2, Label::Box::TYPE_3D), kError);
  EXPECT_NEAR(0.0, ComputeIoU(b1, b3, Label::Box::TYPE_3D), kError);
  EXPECT_NEAR(0.5 * 2.0 * 2.0 / (4.0 + 16.0 - 0.5 * 2.0 * 2.0),
              ComputeIoU(b1, b4, Label::Box::TYPE_3D), kError);
  EXPECT_NEAR(0.5 * 2.0 * 2.0 / (4.0 + 16.0 - 0.5 * 2.0 * 2.0),
              ComputeIoU(b1, b5, Label::Box::TYPE_3D), kError);
  EXPECT_NEAR(0.24074958176697656, ComputeIoU(b1, b6, Label::Box::TYPE_3D),
              kError);
  EXPECT_NEAR(0.0, ComputeIoU(b1, b7, Label::Box::TYPE_3D), kError);
  EXPECT_NEAR(0.0, ComputeIoU(b1, b8, Label::Box::TYPE_3D), kError);
  EXPECT_NEAR(0.0, ComputeIoU(b1, b10, Label::Box::TYPE_3D), kError);
  EXPECT_NEAR(0.0, ComputeIoU(b10, b10, Label::Box::TYPE_3D), kError);
  EXPECT_NEAR(0.00505, ComputeIoU(b1, b11, Label::Box::TYPE_3D), kError);
  EXPECT_NEAR(1.0, ComputeIoU(b11, b11, Label::Box::TYPE_3D), kError);
}

}  // namespace
}  // namespace open_dataset
}  // namespace waymo
