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

#include "waymo_open_dataset/metrics/mot.h"

#include <memory>
#include <vector>

#include <gtest/gtest.h>
#include "absl/memory/memory.h"
#include "waymo_open_dataset/metrics/test_utils.h"
#include "waymo_open_dataset/protos/metrics.pb.h"

namespace waymo {
namespace open_dataset {
namespace {
// Builds an object that has an IoU of 'iou' with a 3d box parameterized as:
// center: (0, 0, center_x), length: 100, width: 1, height: 1.
Object BuildObject(const std::string& id, float iou, float center_x) {
  Object object;
  *object.mutable_object()->mutable_box() =
      BuildBox3d(center_x, 0.0, 0.0, iou * 100, 1.0, 1.0, 0.0);
  object.mutable_object()->set_type(Label::TYPE_VEHICLE);
  object.mutable_object()->set_id(id);
  return object;
}

class MOTTest : public ::testing::Test {
 protected:
  std::unique_ptr<Matcher> CreateMatcher(const std::vector<Object>& pds_input,
                                         const std::vector<Object>& gts_input) {
    Config config = BuildDefaultConfig();
    config.set_matcher_type(MatcherProto::TYPE_HUNGARIAN);
    auto matcher = Matcher::Create(config);
    object_stores_.emplace_back(absl::make_unique<std::vector<Object>>());
    std::vector<Object>& pds = *object_stores_.back().get();
    pds = pds_input;
    object_stores_.emplace_back(absl::make_unique<std::vector<Object>>());
    std::vector<Object>& gts = *object_stores_.back().get();
    gts = gts_input;
    std::vector<int> pd_subset, gt_subset;
    pd_subset.reserve(pds.size());
    gt_subset.reserve(gts.size());
    for (int i = 0, sz = pds.size(); i < sz; ++i) {
      pd_subset.push_back(i);
    }
    for (int i = 0, sz = gts.size(); i < sz; ++i) {
      gt_subset.push_back(i);
    }
    matcher->SetPredictions(pds);
    matcher->SetGroundTruths(gts);
    matcher->SetPredictionSubset(pd_subset);
    matcher->SetGroundTruthSubset(gt_subset);
    return matcher;
  }

  std::vector<std::unique_ptr<std::vector<Object>>> object_stores_;
};

// The following tests cover the 4 cases described in the MOT paper's Figure 2.

// Figure 2(a).
TEST_F(MOTTest, MissAndFalsePositive) {
  std::vector<std::unique_ptr<Matcher>> matchers;
  matchers.emplace_back(CreateMatcher(
      {
          BuildObject("h1", 0.8, 0.0),
      },
      {
          BuildObject("o1", 1.0, 0.0),
      }));
  matchers.emplace_back(CreateMatcher(
      {
          BuildObject("h1", 0.6, 0.0),
      },
      {
          BuildObject("o1", 1.0, 0.0),
      }));
  matchers.emplace_back(CreateMatcher(
      {
          BuildObject("h1", 0.4, 0.0),
      },
      {
          BuildObject("o1", 1.0, 0.0),
      }));
  MOT mot;
  for (auto& m : matchers) {
    mot.Eval(m.get(), Label::LEVEL_2);
  }
  const TrackingMeasurement m = mot.measurement();
  EXPECT_EQ(m.num_fps(), 1);
  EXPECT_EQ(m.num_misses(), 1);
  EXPECT_EQ(m.num_mismatches(), 0);
  EXPECT_EQ(m.num_matches(), 2);
  EXPECT_EQ(m.num_objects_gt(), 3);
  EXPECT_NEAR(m.matching_cost(), 0.2 + 0.4, 1e-6);
}

// Figure 2(b).
TEST_F(MOTTest, Mismatch) {
  std::vector<std::unique_ptr<Matcher>> matchers;
  // t0, t1, t2.
  for (int i = 0; i < 3; ++i) {
    matchers.emplace_back(CreateMatcher(
        {
            BuildObject("h1", 0.8, 0.0),
            BuildObject("h2", 0.8, 200.0),
            BuildObject("h3", 0.8, 400.0),
        },
        {
            BuildObject("o1", 1.0, 0.0),
            BuildObject("o2", 1.0, 200.0),
            BuildObject("o3", 1.0, 400.0),
        }));
  }

  // t3, t4.
  // h1, h2 switch at t3.
  for (int i = 0; i < 2; ++i) {
    matchers.emplace_back(CreateMatcher(
        {
            BuildObject("h1", 0.8, 0.0),
            BuildObject("h2", 0.8, 200.0),
            BuildObject("h3", 0.8, 400.0),
        },
        {
            BuildObject("o1", 1.0, 200.0),
            BuildObject("o2", 1.0, 0.0),
            BuildObject("o3", 1.0, 400.0),
        }));
  }

  // t5, t6, t7.
  // h2, h3 switch at t5.
  for (int i = 0; i < 3; ++i) {
    matchers.emplace_back(CreateMatcher(
        {
            BuildObject("h1", 0.8, 0.0),
            BuildObject("h2", 0.8, 200.0),
            BuildObject("h3", 0.8, 400.0),
        },
        {
            BuildObject("o1", 1.0, 200.0),
            BuildObject("o2", 1.0, 400.0),
            BuildObject("o3", 1.0, 0.0),
        }));
  }

  MOT mot;
  for (auto& m : matchers) {
    mot.Eval(m.get(), Label::LEVEL_2);
  }
  const TrackingMeasurement m = mot.measurement();
  EXPECT_EQ(m.num_fps(), 0);
  EXPECT_EQ(m.num_misses(), 0);
  // A switch causes 2 mismatches.
  EXPECT_EQ(m.num_mismatches(), 2 * 2);
  EXPECT_EQ(m.num_matches(), 8 * 3);
  EXPECT_EQ(m.num_objects_gt(), 8 * 3);
  EXPECT_NEAR(m.matching_cost(), 0.2 * 8 * 3, 1e-6);
}

// Figure 2(c). Case 1.
TEST_F(MOTTest, SequenceLevelBestIsNotIdeal_Case1) {
  std::vector<std::unique_ptr<Matcher>> matchers;
  // t0, t1.
  for (int i = 0; i < 2; ++i) {
    matchers.emplace_back(CreateMatcher(
        {
            BuildObject("h1", 0.8, 0.0),
        },
        {
            BuildObject("o1", 1.0, 0.0),
        }));
  }
  // t2 - t8
  for (int i = 0; i < 7; ++i) {
    matchers.emplace_back(CreateMatcher(
        {
            BuildObject("h2", 0.8, 0.0),
        },
        {
            BuildObject("o1", 1.0, 0.0),
        }));
  }
  MOT mot;
  for (auto& m : matchers) {
    mot.Eval(m.get(), Label::LEVEL_2);
  }
  const TrackingMeasurement m = mot.measurement();
  EXPECT_EQ(m.num_fps(), 0);
  EXPECT_EQ(m.num_misses(), 0);
  EXPECT_EQ(m.num_mismatches(), 1);
  EXPECT_EQ(m.num_matches(), 9);
  EXPECT_EQ(m.num_objects_gt(), 9);
  EXPECT_NEAR(m.matching_cost(), 0.2 * 9, 1e-6);
}

// Figure 2(c). Case 2.
TEST_F(MOTTest, SequenceLevelBestIsNotIdeal_Case2) {
  std::vector<std::unique_ptr<Matcher>> matchers;
  // t0 - t3.
  for (int i = 0; i < 4; ++i) {
    matchers.emplace_back(CreateMatcher(
        {
            BuildObject("h1", 0.8, 0.0),
        },
        {
            BuildObject("o1", 1.0, 0.0),
        }));
  }
  // t4 - t8
  for (int i = 0; i < 5; ++i) {
    matchers.emplace_back(CreateMatcher(
        {
            BuildObject("h2", 0.8, 0.0),
        },
        {
            BuildObject("o1", 1.0, 0.0),
        }));
  }
  MOT mot;
  for (auto& m : matchers) {
    mot.Eval(m.get(), Label::LEVEL_2);
  }
  const TrackingMeasurement m = mot.measurement();
  EXPECT_EQ(m.num_fps(), 0);
  EXPECT_EQ(m.num_misses(), 0);
  EXPECT_EQ(m.num_mismatches(), 1);
  EXPECT_EQ(m.num_matches(), 9);
  EXPECT_EQ(m.num_objects_gt(), 9);
  EXPECT_NEAR(m.matching_cost(), 0.2 * 9, 1e-6);
}

// Figure 2(d).
TEST_F(MOTTest, CorrectReinitialization) {
  std::vector<std::unique_ptr<Matcher>> matchers;
  // t0.
  matchers.emplace_back(CreateMatcher(
      {
          BuildObject("h1", 0.8, 0.0),
      },
      {
          BuildObject("o1", 1.0, 0.0),
      }));
  // t1
  matchers.emplace_back(CreateMatcher({}, {BuildObject("o1", 1.0, 0.0)}));
  // t2.
  matchers.emplace_back(CreateMatcher(
      {
          BuildObject("h1", 0.8, 0.0),
          BuildObject("h2", 0.9, 0.0),
      },
      {
          BuildObject("o1", 1.0, 0.0),
      }));

  MOT mot;
  for (auto& m : matchers) {
    mot.Eval(m.get(), Label::LEVEL_2);
  }
  const TrackingMeasurement m = mot.measurement();
  EXPECT_EQ(m.num_fps(), 1);
  EXPECT_EQ(m.num_misses(), 1);
  EXPECT_EQ(m.num_mismatches(), 0);
  EXPECT_EQ(m.num_matches(), 2);
  EXPECT_EQ(m.num_objects_gt(), 3);
  EXPECT_NEAR(m.matching_cost(), 0.2 * 2, 1e-6);
}

// t0: o1->h1, o2->h2
// t1: o1->h2,
TEST_F(MOTTest, Mismatch_GT_Eviction) {
  std::vector<std::unique_ptr<Matcher>> matchers;
  // t0
  matchers.emplace_back(CreateMatcher(
      {
          BuildObject("h1", 0.8, 0.0),
          BuildObject("h2", 0.8, 200.0),
      },
      {
          BuildObject("o1", 1.0, 0.0),
          BuildObject("o2", 1.0, 200.0),
      }));
  // t1
  matchers.emplace_back(CreateMatcher(
      {
          BuildObject("h2", 0.8, 0.0),
      },
      {
          BuildObject("o1", 1.0, 0.0),
      }));
  MOT mot;
  for (auto& m : matchers) {
    mot.Eval(m.get(), Label::LEVEL_2);
  }
  const TrackingMeasurement m = mot.measurement();
  EXPECT_EQ(m.num_fps(), 0);
  EXPECT_EQ(m.num_misses(), 0);
  EXPECT_EQ(m.num_mismatches(), 2);
  EXPECT_EQ(m.num_matches(), 3);
  EXPECT_EQ(m.num_objects_gt(), 3);
  EXPECT_NEAR(m.matching_cost(), 0.2 * 3, 1e-6);
}

// t0: o1->h1, o2->h2
// t1: o1->h2, o2->
TEST_F(MOTTest, Missmatch_Miss_GT_Eviction) {
  std::vector<std::unique_ptr<Matcher>> matchers;
  // t0
  matchers.emplace_back(CreateMatcher(
      {
          BuildObject("h1", 0.8, 0.0),
          BuildObject("h2", 0.8, 200.0),
      },
      {
          BuildObject("o1", 1.0, 0.0),
          BuildObject("o2", 1.0, 200.0),
      }));
  // t1
  matchers.emplace_back(CreateMatcher(
      {
          BuildObject("h2", 0.8, 0.0),
      },
      {
          BuildObject("o1", 1.0, 0.0),
          BuildObject("o2", 1.0, 200.0),
      }));
  MOT mot;
  for (auto& m : matchers) {
    mot.Eval(m.get(), Label::LEVEL_2);
  }
  const TrackingMeasurement m = mot.measurement();
  EXPECT_EQ(m.num_fps(), 0);
  EXPECT_EQ(m.num_misses(), 1);
  EXPECT_EQ(m.num_mismatches(), 1);
  EXPECT_EQ(m.num_matches(), 3);
  EXPECT_EQ(m.num_objects_gt(), 4);
  EXPECT_NEAR(m.matching_cost(), 0.2 * 3, 1e-6);
}

// TODO: Add NLZ and difficulty level related tests.
}  // namespace
}  // namespace open_dataset
}  // namespace waymo
