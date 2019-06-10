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

#include "waymo_open_dataset/metrics/detection_metrics.h"

#include <stdlib.h>

#include <memory>
#include <utility>

#include <gtest/gtest.h>
#include "waymo_open_dataset/label.pb.h"
#include "waymo_open_dataset/metrics/test_utils.h"
#include "waymo_open_dataset/protos/breakdown.pb.h"
#include "waymo_open_dataset/protos/metrics.pb.h"

namespace waymo {
namespace open_dataset {
namespace {
// Builds an object that has an IoU of 'iou' with a 3d box parameterized as:
// center: (0, 0, 0), length: 100, width: 1, height: 1.
Object BuildObject(float iou, Label::Type type) {
  Object object;
  *object.mutable_object()->mutable_box() =
      BuildBox3d(0.0, 0.0, 0.0, iou * 100, 1.0, 1.0, 0.0);
  object.mutable_object()->set_type(type);
  return object;
}

// Creates a vector of objects by specifying the number of objects for each
// object type.
std::vector<Object> CreateObjects(
    const std::vector<std::pair<Label::Type, int>>& sizes) {
  std::vector<Object> objects;
  for (const auto& kv : sizes) {
    for (int i = 0; i < kv.second; ++i) {
      objects.push_back(BuildObject(0.8, kv.first));
    }
  }
  return objects;
}

Config BuildConfig() {
  Config config = BuildDefaultConfig();
  config.set_matcher_type(MatcherProto::TYPE_HUNGARIAN);
  config.clear_score_cutoffs();
  config.add_score_cutoffs(0.0);
  config.add_score_cutoffs(0.5);
  config.add_score_cutoffs(1.0);
  return config;
}

TEST(DetectionMetricsTest, OneShard) {
  Config config = BuildConfig();
  std::vector<Object> pds{BuildObject(0.9, Label::TYPE_VEHICLE),
                          BuildObject(0.6, Label::TYPE_VEHICLE)};
  for (int i = 0, sz = pds.size(); i < sz; ++i) {
    pds[i].set_score(i * 1.0 / sz);
  }
  const std::vector<Object> gts{BuildObject(1.0, Label::TYPE_VEHICLE)};

  const std::vector<DetectionMeasurements> measurements =
      ComputeDetectionMeasurements(config, pds, gts);
  EXPECT_EQ(measurements.size(), 1);
  EXPECT_EQ(measurements[0].measurements_size(), 3);
  EXPECT_EQ(measurements[0].breakdown().generator_id(), Breakdown::ONE_SHARD);
  EXPECT_EQ(measurements[0].breakdown().shard(), 0);
  EXPECT_EQ(measurements[0].breakdown().difficulty_level(), Label::LEVEL_2);

  {
    const auto& m = measurements[0].measurements(0);
    // Score cutoff 0.0.
    EXPECT_EQ(m.num_tps(), 1);
    EXPECT_EQ(m.num_fps(), 1);
    EXPECT_EQ(m.num_fns(), 0);
    EXPECT_NEAR(m.sum_ha(), 1.0, 1e-10);
  }
  {
    // Score cutoff 0.5.
    const auto& m = measurements[0].measurements(1);
    EXPECT_EQ(m.num_tps(), 1);
    EXPECT_EQ(m.num_fps(), 0);
    EXPECT_EQ(m.num_fns(), 0);
    EXPECT_NEAR(m.sum_ha(), 1.0, 1e-10);
  }
  {
    // Score cutoff 1.0.
    const auto& m = measurements[0].measurements(2);
    EXPECT_EQ(m.num_tps(), 0);
    EXPECT_EQ(m.num_fps(), 0);
    EXPECT_EQ(m.num_fns(), 1);
    EXPECT_NEAR(m.sum_ha(), 0.0, 1e-10);
  }

  const std::vector<DetectionMetrics> metrics =
      ComputeDetectionMetrics(config, {pds, pds}, {gts, gts});
  EXPECT_EQ(metrics.size(), 1);
  // p/r curve:
  // p(r=1) = 0.5, p(r=1) = 1.0, p(r=0) = 1.0
  // mAP = 1.0
  EXPECT_NEAR(metrics[0].mean_average_precision(), 1.0, 1e-6);
  // Angles are perfectly matched.
  EXPECT_NEAR(metrics[0].mean_average_precision_ha_weighted(), 1.0, 1e-6);
}

TEST(DetectionMetricsTest, MultipleTypes) {
  Config config = BuildConfig();
  std::vector<Object> pds{BuildObject(0.9, Label::TYPE_VEHICLE),
                          BuildObject(0.6, Label::TYPE_VEHICLE),
                          BuildObject(0.6, Label::TYPE_PEDESTRIAN),
                          BuildObject(0.8, Label::TYPE_PEDESTRIAN)};
  for (int i = 0, sz = pds.size(); i < sz; ++i) {
    pds[i].set_score(i * 1.0 / sz);
  }
  const std::vector<Object> gts{BuildObject(1.0, Label::TYPE_VEHICLE),
                                BuildObject(1.0, Label::TYPE_PEDESTRIAN)};
  std::vector<DetectionMeasurements> measurements =
      ComputeDetectionMeasurements(config, pds, gts);
  EXPECT_EQ(measurements.size(), 1);
  EXPECT_EQ(measurements[0].measurements_size(), 3);
  EXPECT_EQ(measurements[0].breakdown().generator_id(), Breakdown::ONE_SHARD);
  EXPECT_EQ(measurements[0].breakdown().shard(), 0);
  EXPECT_EQ(measurements[0].breakdown().difficulty_level(), Label::LEVEL_2);

  {
    const auto& m = measurements[0].measurements(0);
    // Score cutoff 0.0.
    EXPECT_EQ(m.num_tps(), 2);
    EXPECT_EQ(m.num_fps(), 2);
    EXPECT_EQ(m.num_fns(), 0);
    EXPECT_NEAR(m.sum_ha(), 2.0, 1e-10);
  }
  {
    // Score cutoff 0.5.
    const auto& m = measurements[0].measurements(1);
    EXPECT_EQ(m.num_tps(), 1);
    EXPECT_EQ(m.num_fps(), 1);
    EXPECT_EQ(m.num_fns(), 1);
    EXPECT_NEAR(m.sum_ha(), 1.0, 1e-10);
  }
  {
    // Score cutoff 1.0.
    const auto& m = measurements[0].measurements(2);
    EXPECT_EQ(m.num_tps(), 0);
    EXPECT_EQ(m.num_fps(), 0);
    EXPECT_EQ(m.num_fns(), 2);
    EXPECT_NEAR(m.sum_ha(), 0.0, 1e-10);
  }

  const std::vector<DetectionMetrics> metrics =
      ComputeDetectionMetrics(config, {pds, pds}, {gts, gts});
  EXPECT_EQ(metrics.size(), 1);
  // p/r curve:
  // p(r=1) = 0.5, p(r=0.5) = 0.5
  // mAP = 0.5
  EXPECT_NEAR(metrics[0].mean_average_precision(), 0.5, 1e-6);
  // Angles are perfectly matched.
  EXPECT_NEAR(metrics[0].mean_average_precision_ha_weighted(), 0.5, 1e-6);
}

TEST(DetectionMetricsTest, MultipleShards) {
  Config config = BuildConfig();
  config.set_num_desired_score_cutoffs(5);
  config.clear_score_cutoffs();
  config.add_breakdown_generator_ids(Breakdown::OBJECT_TYPE);
  auto* d = config.add_difficulties();
  d->add_levels(Label::LEVEL_1);
  d->add_levels(Label::LEVEL_2);
  std::vector<Object> pds =
      CreateObjects({{Label::TYPE_VEHICLE, 10}, {Label::TYPE_PEDESTRIAN, 10}});
  for (int i = 0, sz = pds.size(); i < sz; ++i) {
    pds[i].set_score(std::abs(i - 10) * 2.0 / sz);
  }
  const std::vector<Object> gts =
      CreateObjects({{Label::TYPE_VEHICLE, 10}, {Label::TYPE_PEDESTRIAN, 10}});

  const std::vector<DetectionMetrics> metrics =
      ComputeDetectionMetrics(config, {pds, pds}, {gts, gts});
  EXPECT_EQ(metrics.size(), 1 + Label::Type_MAX * /*num difficulties*/ 2);
}
}  // namespace
}  // namespace open_dataset
}  // namespace waymo
