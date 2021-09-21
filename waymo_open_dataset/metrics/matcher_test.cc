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

#include "waymo_open_dataset/metrics/matcher.h"

#include <utility>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "waymo_open_dataset/label.pb.h"
#include "waymo_open_dataset/metrics/iou.h"
#include "waymo_open_dataset/metrics/test_utils.h"
#include "waymo_open_dataset/protos/metrics.pb.h"

namespace waymo {
namespace open_dataset {

// Build an object centered at the origin, with fixed width, height and heading.
Object BuildObject(float length, float score) {
  Label::Box box = BuildBox3d(0.0, 0.0, 0.0, length * 100, 1.0, 1.0, 0.0);
  Object object;
  *object.mutable_object()->mutable_box() = std::move(box);
  object.mutable_object()->set_type(Label::TYPE_VEHICLE);
  object.set_score(score);
  return object;
}

// Tests base class IoU implementation.
TEST(Matcher, MatcherIoU) {
  Config config = BuildDefaultConfig();
  config.set_matcher_type(MatcherProto::TYPE_HUNGARIAN);

  auto matcher = Matcher::Create(config);
  const std::vector<Object> pds{BuildObject(0.0, 1), BuildObject(0.2, 1),
                                BuildObject(0.6, 1), BuildObject(1.0, 1)};
  const std::vector<Object> gts{BuildObject(1.0, 1)};

  matcher->SetPredictions(pds);
  matcher->SetGroundTruths(gts);
  matcher->SetPredictionSubset({0, 1, 2, 3});
  matcher->SetGroundTruthSubset({0});

  EXPECT_EQ(0, matcher->QuantizedIoU(0, 0));
  EXPECT_EQ(0.2 * kMaxIoU, matcher->QuantizedIoU(1, 0));
  EXPECT_EQ(0.6 * kMaxIoU, matcher->QuantizedIoU(2, 0));
  EXPECT_EQ(kMaxIoU, matcher->QuantizedIoU(3, 0));
}

// A sample IOU calculation by using the smaller area of the input boxes.
double ComputeIoU2dMin(const Label::Box& b1, const Label::Box& b2) {
  constexpr double kMinBoxDim = 1e-2;
  constexpr double kEpsilon = 1e-6;
  if (b1.length() <= kMinBoxDim || b1.width() <= kMinBoxDim ||
      b2.length() <= kMinBoxDim || b2.width() <= kMinBoxDim) {
    LOG_EVERY_N(WARNING, 1000)
        << "Tiny box dim seen, return 0.0 IOU."
        << "\nb1: " << b1.DebugString() << "\nb2: " << b2.DebugString();
    return 0.0;
  }

  const Polygon2d p1 = ToPolygon2d(b1);
  const Polygon2d p2 = ToPolygon2d(b2);

  const double intersection_area = p1.ComputeIntersectionArea(p2);
  const double p1_area = b1.length() * b1.width();
  const double p2_area = b2.length() * b2.width();
  const double min_area = std::min(p1_area, p2_area);
  if (min_area <= kEpsilon) return 0.0;
  const double iom = intersection_area / min_area;
  CHECK(!std::isnan(iom)) << "b1: " << b1.DebugString()
                          << "\nb2: " << b2.DebugString();
  CHECK_GE(iom, -kEpsilon) << "b1: " << b1.DebugString()
                           << "\nb2: " << b2.DebugString();
  CHECK_LE(iom, 1.0 + kEpsilon)
      << "b1: " << b1.DebugString() << "\nb2: " << b2.DebugString();

  return std::max(std::min(iom, 1.0), 0.0);
}

// Tests matcher with a custom IoU calculation by passing in a callback.
TEST(Matcher, MatcherCustomIoU) {
  Config config = BuildDefaultConfig();
  config.set_matcher_type(MatcherProto::TYPE_HUNGARIAN);

  auto matcher = Matcher::Create(config);
  matcher->SetCustomIoUComputeFunc(ComputeIoU2dMin);
  const std::vector<Object> pds{BuildObject(0.0, 1), BuildObject(0.2, 1),
                                BuildObject(0.6, 1), BuildObject(1.0, 1)};
  const std::vector<Object> gts{BuildObject(1.0, 1)};

  matcher->SetPredictions(pds);
  matcher->SetGroundTruths(gts);
  matcher->SetPredictionSubset({0, 1, 2, 3});
  matcher->SetGroundTruthSubset({0});

  EXPECT_EQ(0, matcher->QuantizedIoU(0, 0));
  EXPECT_EQ(1.0 * kMaxIoU, matcher->QuantizedIoU(1, 0));
  EXPECT_EQ(1.0 * kMaxIoU, matcher->QuantizedIoU(2, 0));
  EXPECT_EQ(kMaxIoU, matcher->QuantizedIoU(3, 0));
}

namespace {
// Tests Hungarian match.
TEST(Matcher, HungarianMatch) {
  Config config = BuildDefaultConfig();
  auto matcher = Matcher::Create(config);
  std::vector<Object> gts(2);
  std::vector<Object> pds(4);
  for (auto& o : gts) {
    o.mutable_object()->set_type(Label::TYPE_PEDESTRIAN);
  }
  for (auto& o : pds) {
    o.mutable_object()->set_type(Label::TYPE_PEDESTRIAN);
  }
  matcher->SetGroundTruths(gts);
  matcher->SetPredictions(pds);
  matcher->SetGroundTruthSubset({0, 1});
  matcher->SetPredictionSubset({0, 1, 2, 3});

  // Override the IoU matrix. dim 0: prediction subset, dim 1: ground truth
  // subset.
  const std::vector<std::vector<float>> iou(
      {{0.60, 0}, {1.0, 0.70}, {0.65, 0.71}, {0, 0}});

  static_cast<TEST_HungarianMatcher*>(matcher.get())->SetIoU(iou);
  std::vector<int> pd_matches;
  std::vector<int> gt_matches;
  matcher->Match(&pd_matches, &gt_matches);
  EXPECT_THAT(pd_matches, testing::ElementsAre(-1, 0, 1, -1));
  EXPECT_THAT(gt_matches, testing::ElementsAre(1, 2));

  // Empty prediction.
  matcher->SetPredictionSubset({});
  matcher->Match(&pd_matches, &gt_matches);
  EXPECT_TRUE(pd_matches.empty());
  EXPECT_THAT(gt_matches, testing::ElementsAre(-1, -1));

  // Empty ground truth.
  matcher->SetPredictionSubset({});
  matcher->SetGroundTruthSubset({});
  matcher->Match(&pd_matches, &gt_matches);
  EXPECT_TRUE(pd_matches.empty());
  EXPECT_TRUE(gt_matches.empty());
}

// Tests for ScoreFirstMatcher.

TEST(Matcher, ScoreFirstMatchCornerCases) {
  Config config = BuildDefaultConfig();
  config.set_matcher_type(MatcherProto::TYPE_SCORE_FIRST);

  auto matcher = Matcher::Create(config);
  const std::vector<Object> pds{BuildObject(0.0, 1), BuildObject(0.2, 1),
                                BuildObject(0.6, 1), BuildObject(1.0, 1)};
  const std::vector<Object> gts{BuildObject(1.0, 1)};

  matcher->SetPredictions(pds);
  matcher->SetGroundTruths(gts);

  std::vector<int> pd_matches;
  std::vector<int> gt_matches;
  // Empty groundtruth.
  matcher->SetPredictionSubset({0, 1, 2, 3});
  matcher->SetGroundTruthSubset({});

  matcher->Match(&pd_matches, &gt_matches);
  EXPECT_TRUE(gt_matches.empty());
  EXPECT_THAT(pd_matches, testing::ElementsAre(-1, -1, -1, -1));

  // Empty ground truth and predictions.
  matcher->SetPredictionSubset({});
  matcher->SetGroundTruthSubset({});
  matcher->Match(&pd_matches, &gt_matches);
  EXPECT_TRUE(pd_matches.empty());
  EXPECT_TRUE(gt_matches.empty());
}

TEST(Matcher, ScoreFirstMatchSimpleCase) {
  Config config = BuildDefaultConfig();
  config.set_matcher_type(MatcherProto::TYPE_SCORE_FIRST);

  auto matcher = Matcher::Create(config);
  // Two gt objects of length 0.1 and 0.9, respectively.
  // For each gt, create two pd bboxes.
  const std::vector<Object> gts{BuildObject(0.1, 1.0), BuildObject(0.9, 1.0)};
  const std::vector<Object> pds{BuildObject(0.0, 0.9), BuildObject(0.2, 0.3),
                                BuildObject(0.8, 1.0), BuildObject(0.9, 0.9)};
  matcher->SetPredictions(pds);
  matcher->SetGroundTruths(gts);
  matcher->SetPredictionSubset({0, 1, 2, 3});
  matcher->SetGroundTruthSubset({0, 1});

  std::vector<int> pd_matches;
  std::vector<int> gt_matches;

  matcher->Match(&pd_matches, &gt_matches);
  EXPECT_THAT(gt_matches, testing::ElementsAre(1, 2));
  EXPECT_THAT(pd_matches, testing::ElementsAre(-1, 0, 1, -1));
}

}  // namespace
}  // namespace open_dataset
}  // namespace waymo
