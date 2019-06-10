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

#include "waymo_open_dataset/metrics/tracking_metrics.h"

#include <memory>
#include <string>
#include <utility>

#include <gtest/gtest.h>
#include "absl/strings/str_cat.h"
#include "waymo_open_dataset/metrics/test_utils.h"
#include "waymo_open_dataset/protos/breakdown.pb.h"
#include "waymo_open_dataset/protos/metrics.pb.h"

// This test does not try to extensively test the MOT computation logic. That is
// covered in mot_test.cc.

namespace waymo {
namespace open_dataset {
namespace {
// Builds an object that has an IoU of 'iou' with a 3d box parameterized as:
// center: (0, 0, center_x), length: 100, width: 1, height: 1.
Object BuildObject(const std::string& id, float iou, float center_x,
                   float score) {
  Label::Box box = BuildBox3d(center_x, 0.0, 0.0, iou * 100, 1.0, 1.0, 0.0);
  Object object;
  *object.mutable_object()->mutable_box() = std::move(box);
  object.mutable_object()->set_type(Label::TYPE_VEHICLE);
  object.mutable_object()->set_id(id);
  object.set_score(score);
  return object;
}

// Creates a vector of objects by specifying the number of objects for each
// object type.
std::vector<Object> CreateObjects(
    const std::vector<std::pair<Label::Type, int>>& sizes) {
  std::vector<Object> objects;
  for (const auto& kv : sizes) {
    for (int i = 0; i < kv.second; ++i) {
      objects.emplace_back(BuildObject(absl::StrCat(i, "_", kv.first), 0.8,
                                       0.0 * 100, i * 1.0 / sizes.size()));
    }
  }
  return objects;
}

Config BuildConfig() {
  Config config = BuildDefaultConfig();
  config.set_matcher_type(MatcherProto::TYPE_HUNGARIAN);
  return config;
}

TEST(TrackingMetricsTest, OneShard) {
  Config config = BuildConfig();
  const std::vector<std::vector<Object>> pds{
      {
          BuildObject("h1", 0.9, 0.0, 0.6),
          BuildObject("h2", 0.9, 200, 0.7),
      },
      {
          BuildObject("h2", 0.9, 0.0, 0.7),
      },
  };
  const std::vector<std::vector<Object>> gts{
      {
          BuildObject("o1", 1.0, 0.0, 1.0),
          BuildObject("o2", 1.0, 200, 1.0),
      },
      {
          BuildObject("o1", 1.0, 0.0, 1.0),
          BuildObject("o2", 1.0, 200, 1.0),
      },
  };

  const std::vector<TrackingMeasurements> measurements =
      ComputeTrackingMeasurements(config, pds, gts);
  EXPECT_EQ(measurements.size(), 1);
  ASSERT_EQ(config.score_cutoffs_size(), 10);
  EXPECT_EQ(measurements[0].measurements_size(), 10);
  // score cutoff = 0.0, 0.1, .., 0.6, mota = 0.5
  for (int i = 0; i <= 6; ++i) {
    // t0: o1->h1, o2->h2
    // t1: o1->h2, o2->
    const auto& m = measurements[0].measurements(i);
    // o2 at t1.
    EXPECT_EQ(m.num_misses(), 1);
    // o1 at t1.
    EXPECT_EQ(m.num_mismatches(), 1);
    EXPECT_EQ(m.num_fps(), 0);
    EXPECT_EQ(m.num_matches(), 3);
    EXPECT_EQ(m.num_objects_gt(), 4);
    EXPECT_NEAR(m.matching_cost(), 0.1 * 3, 1e-6);
    EXPECT_NEAR(m.score_cutoff(), i * 0.1, 1e-6);
  }
  // score cutoff = 0.7. mota = 0.5
  {
    // t0: o1->, o2->h2
    // t1: o1->h2, o2->
    const auto& m = measurements[0].measurements(7);
    // o1 at t0, o2 at t1.
    EXPECT_EQ(m.num_misses(), 2);
    // NOTE: o1 at t1 is not a mismatch as o2 (matched to h2 at t0) appears in
    // t2.
    EXPECT_EQ(m.num_mismatches(), 0);
    EXPECT_EQ(m.num_fps(), 0);
    EXPECT_EQ(m.num_matches(), 2);
    EXPECT_EQ(m.num_objects_gt(), 4);
    EXPECT_NEAR(m.matching_cost(), 0.1 * 2, 1e-6);
    EXPECT_NEAR(m.score_cutoff(), 0.7, 1e-6);
  }

  for (int i = 8; i < 10; ++i) {
    // t0: o1->, o2->
    // t1: o1->, o2->
    const auto& m = measurements[0].measurements(i);
    EXPECT_EQ(m.num_misses(), 4);
    EXPECT_EQ(m.num_mismatches(), 0);
    EXPECT_EQ(m.num_fps(), 0);
    EXPECT_EQ(m.num_matches(), 0);
    EXPECT_EQ(m.num_objects_gt(), 4);
    EXPECT_NEAR(m.matching_cost(), 0.0, 1e-6);
    EXPECT_NEAR(m.score_cutoff(), i * 0.1, 1e-6);
  }

  const std::vector<TrackingMetrics> metrics =
      ComputeTrackingMetrics(config, {pds, pds}, {gts, gts});
  EXPECT_EQ(metrics.size(), 1);
  EXPECT_NEAR(metrics[0].score_cutoff(), 0.0, 1e-6);
  EXPECT_NEAR(metrics[0].miss(), 2.0 / 8.0, 1e-6);
  EXPECT_NEAR(metrics[0].mismatch(), 2.0 / 8.0, 1e-6);
  EXPECT_NEAR(metrics[0].fp(), 0.0 / 8.0, 1e-6);
  EXPECT_NEAR(metrics[0].mota(), 4.0 / 8.0, 1e-6);
  EXPECT_NEAR(metrics[0].motp(), 0.1, 1e-6);
}

TEST(TrackingMetricsTest, MultipleShards) {
  Config config = BuildConfig();
  config.set_num_desired_score_cutoffs(5);
  config.clear_score_cutoffs();
  config.add_breakdown_generator_ids(Breakdown::OBJECT_TYPE);
  auto* d = config.add_difficulties();
  d->add_levels(Label::LEVEL_1);
  d->add_levels(Label::LEVEL_2);

  const std::vector<std::vector<Object>> pds{
      CreateObjects({{Label::TYPE_VEHICLE, 10}, {Label::TYPE_PEDESTRIAN, 10}}),
      CreateObjects({{Label::TYPE_VEHICLE, 20}, {Label::TYPE_PEDESTRIAN, 10}})};
  const std::vector<std::vector<Object>> gts{
      CreateObjects({{Label::TYPE_VEHICLE, 10}, {Label::TYPE_PEDESTRIAN, 10}}),
      CreateObjects({{Label::TYPE_VEHICLE, 30}, {Label::TYPE_PEDESTRIAN, 10}})};

  const std::vector<TrackingMetrics> metrics =
      ComputeTrackingMetrics(config, {pds, pds}, {gts, gts});
  EXPECT_EQ(metrics.size(), 1 + Label::Type_MAX * 2);
}

}  // namespace
}  // namespace open_dataset
}  // namespace waymo
