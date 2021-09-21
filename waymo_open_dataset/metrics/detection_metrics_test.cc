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
// center: (0, 0, 0), length: 100*iou, width: 1, height: 1.
Object BuildObject(const std::string& id, float iou, Label::Type type,
                   double x = 0.0, double vx = 0) {
  Object object;
  object.mutable_object()->set_id(id);
  *object.mutable_object()->mutable_box() =
      BuildBox3d(x, 0.0, 0.0, iou * 100, 1.0, 1.0, 0.0);
  object.mutable_object()->set_type(type);
  object.mutable_object()->mutable_metadata()->set_speed_x(vx);
  object.mutable_object()->mutable_metadata()->set_speed_y(0);
  return object;
}

// Creates a vector of objects by specifying the number of objects for each
// object type.
std::vector<Object> CreateObjects(
    const std::vector<std::pair<Label::Type, int>>& sizes) {
  std::vector<Object> objects;
  for (int s = 0; s < sizes.size(); s++) {
    const auto& kv = sizes.at(s);
    for (int i = 0; i < kv.second; ++i) {
      const std::string id = std::to_string(s) + "_" + std::to_string(i);
      objects.push_back(BuildObject(id, 0.8, kv.first));
    }
  }
  return objects;
}

Config BuildConfig(const bool add_details = false) {
  Config config = BuildDefaultConfig();
  config.set_matcher_type(MatcherProto::TYPE_HUNGARIAN);
  config.clear_score_cutoffs();
  config.add_score_cutoffs(0.0);
  config.add_score_cutoffs(0.5);
  config.add_score_cutoffs(1.0);
  config.set_include_details_in_measurements(add_details);
  return config;
}

TEST(DetectionMetricsTest, OneShard) {
  Config config = BuildConfig();
  std::vector<Object> pds{BuildObject("pd0", 0.9, Label::TYPE_VEHICLE),
                          BuildObject("pd1", 0.6, Label::TYPE_VEHICLE)};
  for (int i = 0, sz = pds.size(); i < sz; ++i) {
    pds[i].set_score(i * 1.0 / sz);
  }
  const std::vector<Object> gts{BuildObject("gt0", 1.0, Label::TYPE_VEHICLE)};

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
    EXPECT_EQ(m.details_size(), 0);
  }
  {
    // Score cutoff 0.5.
    const auto& m = measurements[0].measurements(1);
    EXPECT_EQ(m.num_tps(), 1);
    EXPECT_EQ(m.num_fps(), 0);
    EXPECT_EQ(m.num_fns(), 0);
    EXPECT_NEAR(m.sum_ha(), 1.0, 1e-10);
    EXPECT_EQ(m.details_size(), 0);
  }
  {
    // Score cutoff 1.0.
    const auto& m = measurements[0].measurements(2);
    EXPECT_EQ(m.num_tps(), 0);
    EXPECT_EQ(m.num_fps(), 0);
    EXPECT_EQ(m.num_fns(), 1);
    EXPECT_NEAR(m.sum_ha(), 0.0, 1e-10);
    EXPECT_EQ(m.details_size(), 0);
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

TEST(DetectionMetricsTest, OneShardWithDetails) {
  Config config = BuildConfig(/*add_details=*/true);
  std::vector<Object> pds{BuildObject("pd0", 0.9, Label::TYPE_VEHICLE),
                          BuildObject("pd1", 0.6, Label::TYPE_VEHICLE)};
  for (int i = 0, sz = pds.size(); i < sz; ++i) {
    pds[i].set_score(i * 1.0 / sz);
  }
  const std::vector<Object> gts{BuildObject("gt0", 1.0, Label::TYPE_VEHICLE)};

  const std::vector<DetectionMeasurements> measurements =
      ComputeDetectionMeasurements(config, pds, gts);
  EXPECT_EQ(measurements.size(), 1);
  EXPECT_EQ(measurements[0].measurements_size(), 3);

  {
    const auto& m = measurements[0].measurements(0);
    // Score cutoff 0.0.
    EXPECT_EQ(m.num_tps(), 1);
    EXPECT_EQ(m.num_fps(), 1);
    EXPECT_EQ(m.num_fns(), 0);

    ASSERT_EQ(m.details_size(), 1);
    const auto& details = m.details(0);
    ASSERT_EQ(details.tp_pr_ids().size(), 1);
    EXPECT_EQ(details.tp_pr_ids()[0], "pd0");
    ASSERT_EQ(details.tp_gt_ids().size(), 1);
    EXPECT_EQ(details.tp_gt_ids()[0], "gt0");
    ASSERT_EQ(details.fp_ids().size(), 1);
    EXPECT_EQ(details.fp_ids()[0], "pd1");
    EXPECT_TRUE(details.fn_ids().empty());
  }
  {
    // Score cutoff 0.5.
    const auto& m = measurements[0].measurements(1);
    EXPECT_EQ(m.num_tps(), 1);
    EXPECT_EQ(m.num_fps(), 0);
    EXPECT_EQ(m.num_fns(), 0);

    ASSERT_EQ(m.details_size(), 1);
    const auto& details = m.details(0);
    ASSERT_EQ(details.tp_pr_ids().size(), 1);
    EXPECT_EQ(details.tp_pr_ids()[0], "pd1");
    ASSERT_EQ(details.tp_gt_ids().size(), 1);
    EXPECT_EQ(details.tp_gt_ids()[0], "gt0");
    EXPECT_TRUE(details.fp_ids().empty());
    EXPECT_TRUE(details.fn_ids().empty());
  }
  {
    // Score cutoff 1.0.
    const auto& m = measurements[0].measurements(2);
    EXPECT_EQ(m.num_tps(), 0);
    EXPECT_EQ(m.num_fps(), 0);
    EXPECT_EQ(m.num_fns(), 1);

    ASSERT_EQ(m.details_size(), 1);
    const auto& details = m.details(0);
    EXPECT_TRUE(details.tp_pr_ids().empty());
    EXPECT_TRUE(details.tp_gt_ids().empty());
    EXPECT_TRUE(details.fp_ids().empty());
    ASSERT_EQ(details.fn_ids().size(), 1);
    EXPECT_EQ(details.fn_ids()[0], "gt0");
  }
}

TEST(DetectionMetricsTest, MultipleTypes) {
  Config config = BuildConfig();
  std::vector<Object> pds{BuildObject("pd0", 0.9, Label::TYPE_VEHICLE),
                          BuildObject("pd1", 0.6, Label::TYPE_VEHICLE),
                          BuildObject("pd2", 0.6, Label::TYPE_PEDESTRIAN),
                          BuildObject("pd3", 0.8, Label::TYPE_PEDESTRIAN)};
  for (int i = 0, sz = pds.size(); i < sz; ++i) {
    pds[i].set_score(i * 1.0 / sz);
  }
  const std::vector<Object> gts{
      BuildObject("gt0", 1.0, Label::TYPE_VEHICLE),
      BuildObject("gt1", 1.0, Label::TYPE_PEDESTRIAN)};
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

TEST(DetectionMetricsTest, VelocityBreakdown) {
  Config config = BuildConfig();
  config.set_num_desired_score_cutoffs(5);
  config.clear_score_cutoffs();
  config.clear_breakdown_generator_ids();
  config.clear_difficulties();
  config.add_breakdown_generator_ids(Breakdown::VELOCITY);
  auto* d = config.add_difficulties();
  d->add_levels(Label::LEVEL_2);
  std::vector<Object> pds = {
      // FP for shard 0, 1.
      BuildObject("pd0", 0.01, Label::TYPE_VEHICLE, -0.09),
      // FP for shard 1, ignored for shard 0.
      BuildObject("pd1", 0.01, Label::TYPE_VEHICLE, 0),
      // TP for shrad 0, ignored for shard 1.
      BuildObject("pd2", 0.01, Label::TYPE_VEHICLE, 2),
      // Ignored for shard 0, TP for shard 1.
      BuildObject("pd3", 0.01, Label::TYPE_VEHICLE, 4),
  };
  for (int i = 0, sz = pds.size(); i < sz; ++i) {
    pds[i].set_score(1.0);
  }
  const std::vector<Object> gts = {
      // Ignored for shard 0. FN for shard 1.
      BuildObject("gt0", 0.01, Label::TYPE_VEHICLE, 0.9, 0.5),
      // TP for shard 0, ignored for shard 1.
      BuildObject("gt1", 0.01, Label::TYPE_VEHICLE, 2.0, 0),
      // TP for shard 1, ignored for shrad 0.
      BuildObject("gt2", 0.01, Label::TYPE_VEHICLE, 4.0, 0.5),
      // Ignored for shard 0, FN for shard 1.
      BuildObject("gt3", 0.01, Label::TYPE_VEHICLE, 6.0, 0.5),
  };

  const std::vector<DetectionMetrics> metrics =
      ComputeDetectionMetrics(config, {pds}, {gts});
  EXPECT_EQ(metrics.size(), 5 * static_cast<int>(Label::Type_MAX));

  // Shard 0.
  {
    const auto& measurement_1 =
        *metrics[0].measurements().measurements().rbegin();
    EXPECT_EQ(measurement_1.num_tps(), 1);
    EXPECT_EQ(measurement_1.num_fps(), 1);
    EXPECT_EQ(measurement_1.num_fns(), 0);
  }
  // Shard 1.
  {
    const auto& measurement_1 =
        *metrics[1].measurements().measurements().rbegin();
    EXPECT_EQ(measurement_1.num_tps(), 1);
    EXPECT_EQ(measurement_1.num_fps(), 2);
    EXPECT_EQ(measurement_1.num_fns(), 2);
  }

  for (int i = 2; i < metrics.size(); ++i) {
    const auto& measurement_1 =
        *metrics[i].measurements().measurements().rbegin();
    EXPECT_EQ(measurement_1.num_tps(), 0);
    EXPECT_EQ(measurement_1.num_fps(), (i < 5 ? 1 : 0)) << i;
    EXPECT_EQ(measurement_1.num_fns(), 0);
  }
}
}  // namespace
}  // namespace open_dataset
}  // namespace waymo
