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
#include "waymo_open_dataset/label.pb.h"

namespace waymo {
namespace open_dataset {

class MetricsMeanIOU {
  // A class for calculating mean intersection over union.
  // IOU = true_positive / (true_positive + false_positive + false_negative)
  // Mean IOU is the average of all classes.
  // If one class does not appear in the groundtruth, it will be ignored during
  // mean-iou calculation.
 public:
  MetricsMeanIOU(const std::vector<Segmentation::Type>& segmentation_types);

  // Update the metrics with predictions and groud_truth. The results will be
  // accumulated.
  Status Update(const std::vector<Segmentation::Type>& prediction,
                const std::vector<Segmentation::Type>& ground_truth);

  // Reset the metrics states and clear all previous updates.
  void Reset();

  // Compute and return mean-iou. If there is no class with valid iou, it will
  // return 0.0.
  float ComputeMeanIOU();

 private:
  // Classes to be evaluated. Prediction and groundtruth that are not in this
  // list will be ignored and will not impact calculation.
  std::vector<Segmentation::Type> segmentation_types_;
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
