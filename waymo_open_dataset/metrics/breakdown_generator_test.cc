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
}  // namespace
}  // namespace open_dataset
}  // namespace waymo
