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
#include "waymo_open_dataset/protos/segmentation.pb.h"

namespace waymo {
namespace open_dataset {
namespace {

TEST(SegmentationMetricsTest, SegmentationMetricsIOU) {
  SegmentationMetricsConfig segmentation_metrics_config;
  segmentation_metrics_config.mutable_segmentation_types()->Add(
      Segmentation::TYPE_CAR);
  segmentation_metrics_config.mutable_segmentation_types()->Add(
      Segmentation::TYPE_PEDESTRIAN);
  SegmentationMetricsIOU mean_iou(segmentation_metrics_config);
  {
    // If we do not have any valid points, per class iou should be 1.0
    SegmentationMetrics segmentation_metrics = mean_iou.ComputeIOU();
    EXPECT_NEAR(segmentation_metrics.miou(), 1.0, 1e-4);
    auto iou_it = segmentation_metrics.per_class_iou().find(
        static_cast<int>(Segmentation::TYPE_CAR));
    EXPECT_NE(iou_it, segmentation_metrics.per_class_iou().end());
    EXPECT_NEAR(iou_it->second, 1.0, 1e-4);
    iou_it = segmentation_metrics.per_class_iou().find(
        static_cast<int>(Segmentation::TYPE_PEDESTRIAN));
    EXPECT_NE(iou_it, segmentation_metrics.per_class_iou().end());
    EXPECT_NEAR(iou_it->second, 1.0, 1e-4);
  }

  {  // miou should be 1.0 for perfect predictions.
    mean_iou.Reset();
    std::vector<Segmentation::Type> ground_truth = {
        Segmentation::TYPE_CAR, Segmentation::TYPE_CAR, Segmentation::TYPE_CAR,
        Segmentation::TYPE_PEDESTRIAN};
    Status status = mean_iou.Update(ground_truth, ground_truth);
    EXPECT_TRUE(status.ok());
    SegmentationMetrics segmentation_metrics = mean_iou.ComputeIOU();
    EXPECT_NEAR(segmentation_metrics.miou(), 1.0, 1e-4);
    auto iou_it = segmentation_metrics.per_class_iou().find(
        static_cast<int>(Segmentation::TYPE_CAR));
    EXPECT_NE(iou_it, segmentation_metrics.per_class_iou().end());
    EXPECT_NEAR(iou_it->second, 1.0, 1e-4);
    iou_it = segmentation_metrics.per_class_iou().find(
        static_cast<int>(Segmentation::TYPE_PEDESTRIAN));
    EXPECT_NE(iou_it, segmentation_metrics.per_class_iou().end());
    EXPECT_NEAR(iou_it->second, 1.0, 1e-4);
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
    SegmentationMetrics segmentation_metrics = mean_iou.ComputeIOU();
    EXPECT_NEAR(segmentation_metrics.miou(), 0.5833, 1e-4);
    auto iou_it = segmentation_metrics.per_class_iou().find(
        static_cast<int>(Segmentation::TYPE_CAR));
    EXPECT_NE(iou_it, segmentation_metrics.per_class_iou().end());
    EXPECT_NEAR(iou_it->second, 0.6666, 1e-4);
    iou_it = segmentation_metrics.per_class_iou().find(
        static_cast<int>(Segmentation::TYPE_PEDESTRIAN));
    EXPECT_NE(iou_it, segmentation_metrics.per_class_iou().end());
    EXPECT_NEAR(iou_it->second, 0.5, 1e-4);
  }

  {  // Ground truth contains class that is not to be evaluated.
    mean_iou.Reset();
    std::vector<Segmentation::Type> ground_truth = {
        Segmentation::TYPE_CAR, Segmentation::TYPE_CAR, Segmentation::TYPE_CAR,
        Segmentation::TYPE_UNDEFINED};
    std::vector<Segmentation::Type> prediction = {
        Segmentation::TYPE_CAR, Segmentation::TYPE_CAR, Segmentation::TYPE_CAR,
        Segmentation::TYPE_PEDESTRIAN};
    Status status = mean_iou.Update(prediction, ground_truth);
    EXPECT_TRUE(status.ok());
    SegmentationMetrics segmentation_metrics = mean_iou.ComputeIOU();
    EXPECT_NEAR(segmentation_metrics.miou(), 1.0, 1e-4);
    auto iou_it = segmentation_metrics.per_class_iou().find(
        static_cast<int>(Segmentation::TYPE_CAR));
    EXPECT_NE(iou_it, segmentation_metrics.per_class_iou().end());
    EXPECT_NEAR(iou_it->second, 1.0, 1e-4);
    iou_it = segmentation_metrics.per_class_iou().find(
        static_cast<int>(Segmentation::TYPE_PEDESTRIAN));
    EXPECT_NE(iou_it, segmentation_metrics.per_class_iou().end());
    EXPECT_NEAR(iou_it->second, 1.0, 1e-4);
  }
  {  // Prediction contains class that is not to be evaluated.
    mean_iou.Reset();
    std::vector<Segmentation::Type> ground_truth = {
        Segmentation::TYPE_CAR, Segmentation::TYPE_CAR, Segmentation::TYPE_CAR,
        Segmentation::TYPE_PEDESTRIAN};
    std::vector<Segmentation::Type> prediction = {
        Segmentation::TYPE_CAR, Segmentation::TYPE_CAR, Segmentation::TYPE_CAR,
        Segmentation::TYPE_UNDEFINED};
    Status status = mean_iou.Update(prediction, ground_truth);
    EXPECT_TRUE(status.ok());
    SegmentationMetrics segmentation_metrics = mean_iou.ComputeIOU();
    EXPECT_NEAR(segmentation_metrics.miou(), 0.5, 1e-4);
    auto iou_it = segmentation_metrics.per_class_iou().find(
        static_cast<int>(Segmentation::TYPE_CAR));
    EXPECT_NE(iou_it, segmentation_metrics.per_class_iou().end());
    EXPECT_NEAR(iou_it->second, 1.0, 1e-4);
    iou_it = segmentation_metrics.per_class_iou().find(
        static_cast<int>(Segmentation::TYPE_PEDESTRIAN));
    EXPECT_NE(iou_it, segmentation_metrics.per_class_iou().end());
    EXPECT_NEAR(iou_it->second, 0.0, 1e-4);
  }
}

}  // namespace
}  // namespace open_dataset
}  // namespace waymo
