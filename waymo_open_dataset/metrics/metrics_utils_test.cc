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

#include "waymo_open_dataset/metrics/metrics_utils.h"

#include <math.h>

#include <memory>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "waymo_open_dataset/metrics/matcher.h"
#include "waymo_open_dataset/metrics/test_utils.h"
#include "waymo_open_dataset/protos/breakdown.pb.h"
#include "waymo_open_dataset/protos/metrics.pb.h"

namespace waymo {
namespace open_dataset {
namespace internal {
namespace {

TEST(MetricsUtilsTest, IsTP) {
  const std::vector<int> pd_matches = {1, 2, -1};
  EXPECT_TRUE(IsTP(pd_matches, 0));
  EXPECT_TRUE(IsTP(pd_matches, 1));
  EXPECT_FALSE(IsTP(pd_matches, 2));
}

TEST(MetricsUtilsTest, IsFP) {
  std::unique_ptr<Matcher> matcher = Matcher::Create(BuildDefaultConfig());
  std::vector<Object> gts(1);
  std::vector<Object> pds(3);
  pds[0].set_overlap_with_nlz(true);
  matcher->SetPredictions(pds);
  matcher->SetGroundTruths(gts);
  matcher->SetPredictionSubset({0, 1, 2});
  matcher->SetGroundTruthSubset({0});

  const std::vector<int> pd_matches{-1, 0, -1};
  // The 0-th prediction overlaps with NLZ.
  EXPECT_FALSE(IsFP(*matcher, pd_matches, 0));
  EXPECT_FALSE(IsFP(*matcher, pd_matches, 1));
  EXPECT_TRUE(IsFP(*matcher, pd_matches, 2));
}

TEST(MetricsUtilsTest, IsFN) {
  std::unique_ptr<Matcher> matcher = Matcher::Create(BuildDefaultConfig());
  std::vector<Object> gts(3);
  std::vector<Object> pds(1);
  for (int i = 0; i < 3; ++i) {
    Label::DifficultyLevel level = Label::LEVEL_1;
    if (i == 0) level = Label::LEVEL_2;
    gts[i].mutable_object()->set_detection_difficulty_level(level);
    gts[i].mutable_object()->set_tracking_difficulty_level(level);
  }
  matcher->SetPredictions(pds);
  matcher->SetGroundTruths(gts);
  matcher->SetPredictionSubset({0});
  matcher->SetGroundTruthSubset({0, 1, 2});

  const std::vector<int> gt_matches{-1, 0, -1};

  EXPECT_TRUE(IsDetectionFN(*matcher, gt_matches, 0, Label::LEVEL_2));
  EXPECT_TRUE(IsTrackingFN(*matcher, gt_matches, 0, Label::LEVEL_2));
  EXPECT_FALSE(IsDetectionFN(*matcher, gt_matches, 0, Label::LEVEL_1));
  EXPECT_FALSE(IsTrackingFN(*matcher, gt_matches, 0, Label::LEVEL_1));

  EXPECT_FALSE(IsDetectionFN(*matcher, gt_matches, 1, Label::LEVEL_2));
  EXPECT_FALSE(IsTrackingFN(*matcher, gt_matches, 1, Label::LEVEL_2));
  EXPECT_FALSE(IsDetectionFN(*matcher, gt_matches, 1, Label::LEVEL_1));
  EXPECT_FALSE(IsTrackingFN(*matcher, gt_matches, 1, Label::LEVEL_1));

  EXPECT_TRUE(IsDetectionFN(*matcher, gt_matches, 2, Label::LEVEL_2));
  EXPECT_TRUE(IsTrackingFN(*matcher, gt_matches, 2, Label::LEVEL_2));
  EXPECT_TRUE(IsDetectionFN(*matcher, gt_matches, 2, Label::LEVEL_1));
  EXPECT_TRUE(IsTrackingFN(*matcher, gt_matches, 2, Label::LEVEL_1));
}

TEST(MetricsUtilsTest, ComputeHeadingAccuracy) {
  std::unique_ptr<Matcher> matcher = Matcher::Create(BuildDefaultConfig());
  std::vector<Object> gts(1);
  std::vector<Object> pds(1);
  matcher->SetPredictions(pds);
  matcher->SetGroundTruths(gts);
  matcher->SetPredictionSubset({0});
  matcher->SetGroundTruthSubset({0});

  gts[0].mutable_object()->mutable_box()->set_heading(0.1);
  pds[0].mutable_object()->mutable_box()->set_heading(-0.1);
  EXPECT_NEAR(ComputeHeadingAccuracy(*matcher, 0, 0), 1.0 - 0.2 / M_PI, 1e-6);

  gts[0].mutable_object()->mutable_box()->set_heading(M_PI - 0.1);
  pds[0].mutable_object()->mutable_box()->set_heading(-M_PI + 0.1);
  EXPECT_NEAR(ComputeHeadingAccuracy(*matcher, 0, 0), 1.0 - 0.2 / M_PI, 1e-6);

  gts[0].mutable_object()->mutable_box()->set_heading(M_PI);
  pds[0].mutable_object()->mutable_box()->set_heading(0.0);
  EXPECT_NEAR(ComputeHeadingAccuracy(*matcher, 0, 0), 0.0, 1e-6);
}

TEST(MetricsUtilsTest, BuildSubsetsGroundTruth) {
  Config config;
  config.add_breakdown_generator_ids(Breakdown::ONE_SHARD);
  config.add_breakdown_generator_ids(Breakdown::OBJECT_TYPE);
  config.mutable_difficulties()->Add();
  config.mutable_difficulties()->Add();

  std::vector<Object> objects(6);
  std::vector<std::vector<int>> expected_indcies(Label::Type_MAX,
                                                 std::vector<int>());
  for (int i = 0, sz = objects.size(); i < sz; ++i) {
    const auto type = static_cast<Label::Type>(i % Label::Type_MAX + 1);
    objects[i].mutable_object()->set_type(type);
    expected_indcies[type - 1].push_back(i);
  }

  std::vector<BreakdownShardSubset> subsets =
      BuildSubsets(config, objects, /*is_gt=*/true, /*is_detection=*/false);

  EXPECT_EQ(subsets.size(), 1 + Label::Type_MAX);
  for (int i = 0; i <= Label::Type_MAX; ++i) {
    EXPECT_EQ(subsets[i].breakdown_generator_id_index, (i == 0 ? 0 : 1));
    EXPECT_EQ(subsets[i].breakdown_shard, (i == 0 ? 0 : i - 1));
    EXPECT_EQ(subsets[i].indices.size(), 1);
    if (i == 0) {
      EXPECT_THAT(subsets[0].indices[0],
                  testing::ElementsAre(0, 1, 2, 3, 4, 5));
    } else {
      EXPECT_THAT(subsets[i].indices,
                  testing::ElementsAre(expected_indcies[i - 1]));
    }
  }
}

TEST(MetricsUtilsTest, BuildSubsetsPredictions) {
  Config config;
  config.add_breakdown_generator_ids(Breakdown::ONE_SHARD);
  config.add_breakdown_generator_ids(Breakdown::OBJECT_TYPE);
  config.mutable_difficulties()->Add();
  config.mutable_difficulties()->Add();
  config.add_score_cutoffs(0.0);
  config.add_score_cutoffs(1.0);

  std::vector<Object> objects(12);
  std::vector<std::vector<int>> expected_indcies_one_shard(2);
  // Expected indices for OBJECT_TYPE generator.
  std::vector<std::vector<std::vector<int>>> expected_indcies_object_type(
      Label::Type_MAX, std::vector<std::vector<int>>(2, std::vector<int>()));
  for (int i = 0, sz = objects.size(); i < sz; ++i) {
    const auto type = static_cast<Label::Type>(i % Label::Type_MAX + 1);
    objects[i].mutable_object()->set_type(type);
    expected_indcies_one_shard[0].push_back(i);
    if (i % 2 == 1) {
      expected_indcies_one_shard[1].push_back(i);
    }
    expected_indcies_object_type[type - 1][0].push_back(i);
    if (i % 2 == 1) {
      expected_indcies_object_type[type - 1][1].push_back(i);
    }
    objects[i].set_score(i % 2 == 0 ? 0.0 : 1.0);
  }

  std::vector<BreakdownShardSubset> subsets =
      BuildSubsets(config, objects, /*is_gt=*/false, /*is_detection=*/false);

  EXPECT_EQ(subsets.size(), 1 + Label::Type_MAX);
  for (int i = 0; i <= Label::Type_MAX; ++i) {
    EXPECT_EQ(subsets[i].breakdown_generator_id_index, (i == 0 ? 0 : 1));
    EXPECT_EQ(subsets[i].breakdown_shard, (i == 0 ? 0 : i - 1));
    EXPECT_EQ(subsets[i].indices.size(), 2);
    if (i == 0) {
      EXPECT_THAT(subsets[0].indices,
                  testing::ElementsAre(expected_indcies_one_shard[0],
                                       expected_indcies_one_shard[1]));
    } else {
      EXPECT_THAT(subsets[i].indices,
                  testing::ElementsAre(expected_indcies_object_type[i - 1][0],
                                       expected_indcies_object_type[i - 1][1]));
    }
  }
}

TEST(MetricsUtilsTest, GetDifficultyLevels) {
  Config config;
  config.add_breakdown_generator_ids(Breakdown::ONE_SHARD);
  config.add_breakdown_generator_ids(Breakdown::RANGE);
  config.add_difficulties();
  config.add_difficulties()->add_levels(Label::LEVEL_1);
  config.mutable_difficulties(1)->add_levels(Label::LEVEL_2);

  EXPECT_THAT(GetDifficultyLevels(config, 0),
              testing::ElementsAre(Label::LEVEL_2));
  EXPECT_THAT(GetDifficultyLevels(config, 1),
              testing::ElementsAre(Label::LEVEL_1, Label::LEVEL_2));
}

TEST(MetricsUtilsTest, DecideScoreCutoffs) {
  EXPECT_THAT(DecideScoreCutoffs({}, 1), testing::ElementsAre(0.0));
  EXPECT_THAT(DecideScoreCutoffs({}, 3), testing::ElementsAre(0.0, 0.5, 1.0));
  EXPECT_THAT(DecideScoreCutoffs({0.4}, 3),
              testing::ElementsAre(0.0, 0.5, 1.0));
  EXPECT_THAT(DecideScoreCutoffs({0.1, 0.2, 0.3, 0.4}, 2),
              testing::ElementsAre(0.1, 1.0));
  EXPECT_THAT(DecideScoreCutoffs({0.1, 0.2, 0.3, 0.4}, 3),
              testing::ElementsAre(0.1, 0.3, 1.0));
  EXPECT_THAT(DecideScoreCutoffs({0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7}, 5),
              testing::ElementsAre(0.1, 0.2, 0.4, 0.6, 1.0));
  EXPECT_THAT(DecideScoreCutoffs({0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}, 5),
              testing::ElementsAre(0.0, 0.0, 0.0, 0.0, 1.0));
}

TEST(MetricsUtilsTest, ComputeMeanAveragePrecision) {
  EXPECT_NEAR(ComputeMeanAveragePrecision({}, {}, 0.05), 0.0, 1e-6);
  EXPECT_NEAR(ComputeMeanAveragePrecision({0.1}, {0.2}, 0.05), 0.2 * 0.1, 1e-6);
  EXPECT_NEAR(ComputeMeanAveragePrecision({0.1, 0.05}, {0.2, 1.0}, 0.05),
              0.05 * 0.8 + 0.1 * 0.2 + (0.1 - 0.05) * 0.5 * 0.05, 1e-6);
  EXPECT_NEAR(ComputeMeanAveragePrecision({0.1, 0.05}, {0.2, 1.0}, 1.0),
              0.2 * 0.1 + 0.5 * (0.1 + 0.05) * 0.8, 1e-6);
  // The r=0.4 has no impact as we override its precision to 0.05 (same as the
  // one for r=1.0).
  EXPECT_NEAR(
      ComputeMeanAveragePrecision({0.1, 0.0, 0.05}, {0.2, 0.4, 1.0}, 0.05),
      0.05 * 0.8 + 0.1 * 0.2 + (0.1 - 0.05) * 0.5 * 0.05, 1e-6);
}

TEST(MetricsUtilsTest, EstimateObjectSpeed) {
  Config config;
  config.add_breakdown_generator_ids(Breakdown::VELOCITY);
  auto build_object = [](const std::vector<float>& center_and_speed) {
    Object o;
    int i = 0;
    o.mutable_object()->mutable_box()->set_center_x(center_and_speed[i++]);
    o.mutable_object()->mutable_box()->set_center_y(center_and_speed[i++]);
    o.mutable_object()->mutable_box()->set_center_z(center_and_speed[i++]);
    if (center_and_speed.size() > 3) {
      o.mutable_object()->mutable_metadata()->set_speed_x(
          center_and_speed[i++]);
      o.mutable_object()->mutable_metadata()->set_speed_y(
          center_and_speed[i++]);
    }
    return o;
  };
  std::vector<Object> pds{
      build_object({0, 0, 0}),
      build_object({1, 0, 0}),
  };
  std::vector<Object> gts{
      build_object({0.1, 0, 0, 1, 0}),
      build_object({1.1, 0, 0, 2, 0}),
      build_object({1.2, 0, 0, 3, 0}),
  };
  auto pds_cp = EstimateObjectSpeed(pds, gts);
  EXPECT_DOUBLE_EQ(pds_cp[0].object().metadata().speed_x(), 1);
  EXPECT_DOUBLE_EQ(pds_cp[1].object().metadata().speed_x(), 2);
}


}  // namespace
}  // namespace internal
}  // namespace open_dataset
}  // namespace waymo
