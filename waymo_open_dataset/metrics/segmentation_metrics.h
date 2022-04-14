/* Copyright 2022 The Waymo Open Dataset Authors. All Rights Reserved.

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
#ifndef WAYMO_OPEN_DATASET_METRICS_SEGMENTATION_METRICS_H_
#define WAYMO_OPEN_DATASET_METRICS_SEGMENTATION_METRICS_H_

#include <vector>

#include "absl/container/flat_hash_map.h"
#include "waymo_open_dataset/common/status.h"
#include "waymo_open_dataset/protos/segmentation.pb.h"
#include "waymo_open_dataset/protos/segmentation_metrics.pb.h"

namespace waymo {
namespace open_dataset {

SegmentationMetrics ComputeIOU(SegmentationMeasurements measurements);

class SegmentationMetricsIOU {
  // A class for calculating mean intersection over union for 3D semantic
  // segmentation.
  // IOU = true_positive / (true_positive + false_positive +
  // false_negative)
  // Mean IOU is the average of all classes.
  // If a class does not ever appear in the groundtruth or prediction, aka both
  // intersection and union are zeros, its IOU will be counted as 1.0
 public:
  SegmentationMetricsIOU(
      const SegmentationMetricsConfig segmentation_metrics_config);

  // Update the metrics with predictions and groud_truth. The results will be
  // accumulated.
  Status Update(const std::vector<Segmentation::Type>& prediction,
                const std::vector<Segmentation::Type>& ground_truth);

  // Reset the metrics states and clear all previous updates.
  void Reset();

  // Compute and return per class iou and mean iou.
  // If there is no point for a given class in all frames, including both
  // prediction and ground truth, the IOU for that class will be 1.0.
  SegmentationMetrics ComputeIOU();

  // Merge results
  SegmentationMetrics MergeResults(
      std::vector<SegmentationMeasurements> results);

 private:
  // Classes to be evaluated. Prediction and groundtruth that are not in this
  // list will be ignored and will not impact calculation.
  const SegmentationMetricsConfig segmentation_metrics_config_;
  // Number of classes to be evaluated.
  int num_classes_;
  // Confusion matrix, accumulated with all predictions.
  std::vector<std::vector<int>> confusion_matrix_;
  // A mapper from class type to int.
  absl::flat_hash_map<Segmentation::Type, int> segmentation_type_mapper_;
};

}  // namespace open_dataset
}  // namespace waymo

#endif  // WAYMO_OPEN_DATASET_METRICS_SEGMENTATION_METRICS_H_
