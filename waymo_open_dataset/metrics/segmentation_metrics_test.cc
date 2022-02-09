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

#include "waymo_open_dataset/metrics/segmentation_metrics.h"

#include <stdlib.h>

#include <vector>

#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "waymo_open_dataset/label.pb.h"

namespace waymo {
namespace open_dataset {
namespace {

TEST(SegmentationMetricsTest, MetricsMeanIOU) {
  std::vector<Segmentation::Type> segmentation_typs{
      Segmentation::TYPE_CAR, Segmentation::TYPE_PEDESTRIAN};
  MetricsMeanIOU mean_iou(segmentation_typs);
  {
    // If we do not have valid points it should return 0.
    float miou = mean_iou.ComputeMeanIOU();
    EXPECT_NEAR(miou, 0.0, 1e-4);
  }

  {  // miou should be 1.0 for perfect predictions.
    mean_iou.Reset();
    std::vector<Segmentation::Type> ground_truth = {
        Segmentation::TYPE_CAR, Segmentation::TYPE_CAR, Segmentation::TYPE_CAR,
        Segmentation::TYPE_PEDESTRIAN};
    Status status = mean_iou.Update(ground_truth, ground_truth);
    EXPECT_TRUE(status.ok());
    float miou = mean_iou.ComputeMeanIOU();
    EXPECT_NEAR(miou, 1.0, 1e-4);
  }

  {
    mean_iou.Reset();
    std::vector<Segmentation::Type> ground_truth = {
        Segmentation::TYPE_CAR, Segmentation::TYPE_CAR, Segmentation::TYPE_CAR,
        Segmentation::TYPE_PEDESTRIAN};
    std::vector<Segmentation::Type> prediction = {
        Segmentation::TYPE_CAR, Segmentation::TYPE_PEDESTRIAN,
        Segmentation::TYPE_CAR, Segmentation::TYPE_PEDESTRIAN};
    Status status = mean_iou.Update(prediction, ground_truth);
    EXPECT_TRUE(status.ok());
    float miou = mean_iou.ComputeMeanIOU();
    EXPECT_NEAR(miou, 0.5833, 1e-4);
  }

  {  // Ground truth contains class that is not to be evaluated.
    mean_iou.Reset();
    std::vector<Segmentation::Type> ground_truth = {
        Segmentation::TYPE_CAR, Segmentation::TYPE_CAR, Segmentation::TYPE_CAR,
        Segmentation::TYPE_UNKNOWN};
    std::vector<Segmentation::Type> prediction = {
        Segmentation::TYPE_CAR, Segmentation::TYPE_CAR, Segmentation::TYPE_CAR,
        Segmentation::TYPE_PEDESTRIAN};
    Status status = mean_iou.Update(prediction, ground_truth);
    EXPECT_TRUE(status.ok());
    float miou = mean_iou.ComputeMeanIOU();
    EXPECT_NEAR(miou, 1.0, 1e-4);
  }
  {  // Prediction contains class that is not to be evaluated.
    mean_iou.Reset();
    std::vector<Segmentation::Type> ground_truth = {
        Segmentation::TYPE_CAR, Segmentation::TYPE_CAR, Segmentation::TYPE_CAR,
        Segmentation::TYPE_PEDESTRIAN};
    std::vector<Segmentation::Type> prediction = {
        Segmentation::TYPE_CAR, Segmentation::TYPE_CAR, Segmentation::TYPE_CAR,
        Segmentation::TYPE_UNKNOWN};
    Status status = mean_iou.Update(prediction, ground_truth);
    EXPECT_TRUE(status.ok());
    float miou = mean_iou.ComputeMeanIOU();
    EXPECT_NEAR(miou, 0.5, 1e-4);
  }
}

}  // namespace
}  // namespace open_dataset
}  // namespace waymo
