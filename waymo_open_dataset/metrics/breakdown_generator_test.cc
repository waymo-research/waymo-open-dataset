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

#include "waymo_open_dataset/metrics/breakdown_generator.h"

#include <gtest/gtest.h>
#include "waymo_open_dataset/label.pb.h"
#include "waymo_open_dataset/metrics/test_utils.h"
#include "waymo_open_dataset/protos/breakdown.pb.h"
#include "waymo_open_dataset/protos/metrics.pb.h"

namespace waymo {
namespace open_dataset {
namespace {
TEST(BreakdownGenerator, BreakdownGeneratorAll) {
  const auto generator = BreakdownGenerator::Create(Breakdown::ONE_SHARD);
  EXPECT_EQ(1, generator->NumShards());
  EXPECT_EQ(0, generator->Shard(Object()));
  EXPECT_EQ(Breakdown::ONE_SHARD, generator->Id());
}

TEST(BreakdownGenerator, BreakdownGeneratorObjectType) {
  const auto generator = BreakdownGenerator::Create(Breakdown::OBJECT_TYPE);
  EXPECT_EQ(Label::Type_MAX, generator->NumShards());
  Object object;
  object.mutable_object()->set_type(Label::TYPE_VEHICLE);
  EXPECT_EQ(Breakdown::OBJECT_TYPE, generator->Id());
}

TEST(BreakdownGenerator, BreakdownGeneratorRange) {
  const auto generator = BreakdownGenerator::Create(Breakdown::RANGE);
  EXPECT_EQ(3 * Label::Type_MAX, generator->NumShards());
  Object object1;
  *object1.mutable_object()->mutable_box() = BuildAA2dBox(1.0, 0.0, 1.0, 1.0);
  object1.mutable_object()->set_type(Label::TYPE_VEHICLE);
  Object object2;
  *object2.mutable_object()->mutable_box() = BuildAA2dBox(30.0, 0.0, 1.0, 1.0);
  object2.mutable_object()->set_type(Label::TYPE_SIGN);
  Object object3;
  *object3.mutable_object()->mutable_box() = BuildAA2dBox(50.0, 0.0, 1.0, 1.0);
  object3.mutable_object()->set_type(Label::TYPE_PEDESTRIAN);
  EXPECT_EQ(0, generator->Shard(object1));
  EXPECT_EQ(1 + 3 * (Label::TYPE_SIGN - 1), generator->Shard(object2));
  EXPECT_EQ(2 + 3 * (Label::TYPE_PEDESTRIAN - 1), generator->Shard(object3));
  EXPECT_EQ(Breakdown::RANGE, generator->Id());
}

TEST(BreakdownGenerator, BreakdownGeneratorVelocity) {
  const auto generator = BreakdownGenerator::Create(Breakdown::VELOCITY);
  EXPECT_EQ(4 * 5, generator->NumShards());
  Object object;
  object.mutable_object()->set_type(Label::TYPE_VEHICLE);
  object.mutable_object()->mutable_metadata()->set_speed_x(0.05);
  object.mutable_object()->mutable_metadata()->set_speed_y(0.05);
  EXPECT_EQ(0, generator->Shard(object));
  object.mutable_object()->set_type(Label::TYPE_PEDESTRIAN);
  object.mutable_object()->mutable_metadata()->set_speed_x(0.5);
  object.mutable_object()->mutable_metadata()->set_speed_y(0.5);
  EXPECT_EQ(5 + 1, generator->Shard(object));
  object.mutable_object()->set_type(Label::TYPE_SIGN);
  object.mutable_object()->mutable_metadata()->set_speed_x(2.);
  object.mutable_object()->mutable_metadata()->set_speed_y(2.);
  EXPECT_EQ(10 + 2, generator->Shard(object));
  object.mutable_object()->set_type(Label::TYPE_CYCLIST);
  object.mutable_object()->mutable_metadata()->set_speed_x(5.);
  object.mutable_object()->mutable_metadata()->set_speed_y(5.);
  EXPECT_EQ(15 + 3, generator->Shard(object));
  object.mutable_object()->set_type(Label::TYPE_VEHICLE);
  object.mutable_object()->mutable_metadata()->set_speed_x(20.);
  object.mutable_object()->mutable_metadata()->set_speed_y(15.);
  EXPECT_EQ(4, generator->Shard(object));
}

}  // namespace
}  // namespace open_dataset
}  // namespace waymo
