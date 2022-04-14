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
#include "waymo_open_dataset/metrics/segmentation_metrics.h"

#include <algorithm>
#include <iterator>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <glog/logging.h>
#include "waymo_open_dataset/common/status.h"

namespace waymo {
namespace open_dataset {

SegmentationMetricsIOU::SegmentationMetricsIOU(
    const SegmentationMetricsConfig segmentation_metrics_config)
    : segmentation_metrics_config_(segmentation_metrics_config) {
  num_classes_ = segmentation_metrics_config_.segmentation_types().size();
  // We use index `num_classes` to store points with classes that are not
  // in the allowed segmentation_types.
  confusion_matrix_ = std::vector<std::vector<int>>(
      num_classes_ + 1, std::vector<int>(num_classes_ + 1, 0));
  for (int i = 0; i < num_classes_; ++i) {
    auto index = segmentation_type_mapper_.find(static_cast<Segmentation::Type>(
        segmentation_metrics_config_.segmentation_types()[i]));
    // Input list of segmentation typs should not have duplicates.
    CHECK(index == segmentation_type_mapper_.end());
    segmentation_type_mapper_.insert(
        {static_cast<Segmentation::Type>(
             segmentation_metrics_config_.segmentation_types()[i]),
         i});
  }
}

Status SegmentationMetricsIOU::Update(
    const std::vector<Segmentation::Type>& prediction,
    const std::vector<Segmentation::Type>& ground_truth) {
  if (prediction.size() != ground_truth.size()) {
    return InvalidArgumentError(
        "Prediction and Groudtruth length do not match.");
  }
  for (int i = 0; i < prediction.size(); ++i) {
    // We ignore the points with ground truth or prediction that are outside the
    // list of segmentation classes.
    auto pred_index = segmentation_type_mapper_.find(prediction[i]);
    auto gt_index = segmentation_type_mapper_.find(ground_truth[i]);
    int pd_idx = (pred_index == segmentation_type_mapper_.end())
                     ? num_classes_
                     : pred_index->second;
    int gt_idx = (gt_index == segmentation_type_mapper_.end())
                     ? num_classes_
                     : gt_index->second;
    ++confusion_matrix_[gt_idx][pd_idx];
  }
  return OkStatus();
}

void SegmentationMetricsIOU::Reset() {
  for (int i = 0; i < num_classes_ + 1; ++i) {
    for (int j = 0; j < num_classes_ + 1; ++j) {
      confusion_matrix_[i][j] = 0;
    }
  }
}

SegmentationMetrics SegmentationMetricsIOU::ComputeIOU() {
  SegmentationMetrics results;
  std::vector<float> ious(num_classes_, 1.0);
  for (int i = 0; i < num_classes_; ++i) {
    int num_intersection = confusion_matrix_[i][i];
    int num_union = -num_intersection;
    for (int j = 0; j < num_classes_; ++j) {
      num_union += confusion_matrix_[i][j];
      num_union += confusion_matrix_[j][i];
    }
    // When the ground truth is valid but prediction is not valid, it still
    // counts as false negative.
    num_union += confusion_matrix_[i][num_classes_];
    if (num_union > 0) {
      ious[i] = 1.0f * num_intersection / num_union;
    }
    results.mutable_per_class_iou()->insert(
        {segmentation_metrics_config_.segmentation_types()[i], ious[i]});
    results.mutable_segmentation_measurements()->add_intersections(
        num_intersection);
    results.mutable_segmentation_measurements()->add_unions(num_union);
  }
  float mean_iou = 0;
  for (const auto iou : ious) {
    mean_iou += iou;
  }
  mean_iou = mean_iou / ious.size();
  results.set_miou(mean_iou);
  return results;
}

SegmentationMetrics SegmentationMetricsIOU::MergeResults(
    std::vector<SegmentationMeasurements> results) {
  SegmentationMetrics merged;
  std::vector<int64> num_intersections(num_classes_, 0);
  std::vector<int64> num_unions(num_classes_, 0);
  for (const auto& result : results) {
    CHECK(result.unions_size() == num_classes_);
    CHECK(result.intersections_size() == num_classes_);
    for (int i = 0; i < num_classes_; ++i) {
      num_intersections[i] += result.intersections()[i];
      num_unions[i] += result.unions()[i];
    }
  }
  std::vector<float> ious(num_classes_, 1.0);
  for (int i = 0; i < num_classes_; ++i) {
    int num_intersection = num_intersections[i];
    int num_union = num_unions[i];
    if (num_union > 0) {
      ious[i] = 1.0f * num_intersection / num_union;
    }
    merged.mutable_per_class_iou()->insert(
        {segmentation_metrics_config_.segmentation_types()[i], ious[i]});
    merged.mutable_segmentation_measurements()->add_intersections(
        num_intersection);
    merged.mutable_segmentation_measurements()->add_unions(num_union);
  }
  float mean_iou = 0;
  for (const auto iou : ious) {
    mean_iou += iou;
  }
  mean_iou = mean_iou / ious.size();
  merged.set_miou(mean_iou);
  return merged;
}

}  // namespace open_dataset
}  // namespace waymo
