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

#include <string.h>

#include <algorithm>
#include <iterator>
#include <memory>
#include <numeric>

#include <glog/logging.h>
#include "absl/memory/memory.h"
#include "waymo_open_dataset/label.pb.h"
#include "waymo_open_dataset/metrics/hungarian.h"
#include "waymo_open_dataset/metrics/iou.h"
#include "waymo_open_dataset/metrics/matcher.h"
#include "waymo_open_dataset/protos/metrics.pb.h"

namespace waymo {
namespace open_dataset {

std::unique_ptr<Matcher> Matcher::Create(
    MatcherProto_Type matcher_type, const std::vector<float>& iou_thresholds,
    Label::Box::Type box_type) {
  CHECK_EQ(iou_thresholds.size(), Label::Type_MAX + 1)
      << "Must specifiy IoU thresholds for every label type.";
  CHECK_NE(box_type, Label::Box::TYPE_UNKNOWN);

  switch (matcher_type) {
    case MatcherProto::TYPE_HUNGARIAN:
      return absl::make_unique<HungarianMatcher>(iou_thresholds, box_type);
    case MatcherProto::TYPE_SCORE_FIRST:
      return absl::make_unique<ScoreFirstMatcher>(iou_thresholds, box_type);
    case MatcherProto::TYPE_HUNGARIAN_TEST_ONLY:
      return absl::make_unique<TEST_HungarianMatcher>(iou_thresholds, box_type);
    case MatcherProto::TYPE_UNKNOWN:
    default:
      LOG(FATAL) << "Unknown matcher type.";
  }
}

float Matcher::IoU(int prediction_index, int ground_truth_index) const {
  ValidPredictionIndex(prediction_index);
  ValidGroundTruthIndex(ground_truth_index);

  if (iou_caches_.empty()) {
    iou_caches_.resize(predictions().size(),
                       std::vector<float>(ground_truths().size(), -1.0));
  }
  if (predictions()[prediction_index].object().type() !=
      ground_truths()[ground_truth_index].object().type()) {
    iou_caches_[prediction_index][ground_truth_index] = 0;
    return 0.0;
  }

  if (iou_caches_[prediction_index][ground_truth_index] < 0.0) {
    float iou =
        (custom_iou_func_ == nullptr)
            ? ComputeIoU(predictions()[prediction_index].object().box(),
                         ground_truths()[ground_truth_index].object().box(),
                         box_type_)
            : custom_iou_func_(
                  predictions()[prediction_index].object().box(),
                  ground_truths()[ground_truth_index].object().box());

    CHECK_GE(iou, 0.0) << "prediction_index: " << prediction_index
                       << ", ground_truth_index: " << ground_truth_index;
    CHECK_LE(iou, 1.0);
    iou_caches_[prediction_index][ground_truth_index] = iou;
  }

  return iou_caches_[prediction_index][ground_truth_index];
}

bool Matcher::CanMatch(int prediction_index, int ground_truth_index) const {
  const Label::Type object_type =
      predictions()[prediction_index].object().type();
  CHECK_NE(object_type, Label::TYPE_UNKNOWN);
  const float iou = IoU(prediction_index, ground_truth_index);
  if (iou < iou_thresholds_[object_type]) {
    return false;
  }
  return true;
}

// This implements the Hungarian based matching which maximizes the IOU
// overlaps for all predictions.
void HungarianMatcher::Match(std::vector<int>* prediction_matches,
                             std::vector<int>* ground_truth_matches) {
  if (prediction_matches == nullptr && ground_truth_matches == nullptr) return;

  const int prediction_subset_size = prediction_subset().size();
  const int ground_truth_subset_size = ground_truth_subset().size();
  if (prediction_matches) {
    prediction_matches->clear();
    prediction_matches->resize(prediction_subset_size, -1);
  }
  if (ground_truth_matches) {
    ground_truth_matches->clear();
    ground_truth_matches->resize(ground_truth_subset_size, -1);
  }
  if (prediction_subset_size <= 0 || ground_truth_subset_size <= 0) return;

  const int num_vertices =
      std::max(prediction_subset_size, ground_truth_subset_size);
  std::unique_ptr<int[]> edge(new int[num_vertices * num_vertices]);
  memset(edge.get(), 0, num_vertices * num_vertices * sizeof(int));
  std::unique_ptr<int[]> matching(new int[num_vertices]);
  memset(matching.get(), 0, num_vertices * sizeof(int));

  // Set edges.
  for (int i = 0; i < prediction_subset_size; ++i) {
    for (int j = 0; j < ground_truth_subset_size; ++j) {
      edge[i * num_vertices + j] =
          CanMatch(prediction_subset()[i], ground_truth_subset()[j])
              ? QuantizedIoU(prediction_subset()[i], ground_truth_subset()[j])
              : 0;
    }
  }

  Hungarian(num_vertices, edge.get(), matching.get());

  for (int i = 0; i < prediction_subset_size; ++i) {
    const int prediction_subset_index = i;
    int ground_truth_subset_index = matching[i];
    CHECK_GE(ground_truth_subset_index, 0);
    CHECK_LT(ground_truth_subset_index, num_vertices);
    bool can_match = false;
    if (ground_truth_subset_index >= ground_truth_subset_size) {
      ground_truth_subset_index = -1;
    } else {
      // This deals with cases where we have to match pd with gt that has iou <
      // threshold as there are no other better matches. Imagine all pairs have
      // IoU = 0.
      can_match = CanMatch(prediction_subset()[prediction_subset_index],
                           ground_truth_subset()[ground_truth_subset_index]);
    }
    if (prediction_matches) {
      (*prediction_matches)[i] = can_match ? ground_truth_subset_index : -1;
    }
    if (ground_truth_matches && ground_truth_subset_index >= 0) {
      (*ground_truth_matches)[ground_truth_subset_index] =
          can_match ? prediction_subset_index : -1;
    }
  }
}

void ScoreFirstMatcher::Match(std::vector<int>* prediction_matches,
                              std::vector<int>* ground_truth_matches) {
  CHECK(prediction_matches);
  CHECK(ground_truth_matches);

  const int prediction_subset_size = prediction_subset().size();
  const int ground_truth_subset_size = ground_truth_subset().size();
  prediction_matches->resize(prediction_subset_size, -1);
  ground_truth_matches->resize(ground_truth_subset_size, -1);

  std::vector<int> sorted_pred_subset_idx(prediction_subset_size);
  // Sort predictions by score in descending order.
  std::iota(sorted_pred_subset_idx.begin(), sorted_pred_subset_idx.end(), 0);
  std::sort(sorted_pred_subset_idx.begin(), sorted_pred_subset_idx.end(),
            [this](const int a, const int b) {
              return predictions()[prediction_subset()[a]].score() >
                     predictions()[prediction_subset()[b]].score();
            });

  for (const int pd_subset_idx : sorted_pred_subset_idx) {
    int best_gt_match = -1;
    float best_gt_match_iou = -1;
    for (int gt_subset_idx = 0; gt_subset_idx < ground_truth_subset_size;
         ++gt_subset_idx) {
      if ((*ground_truth_matches)[gt_subset_idx] != -1) {
        // Skip already matched groundtruth.
        continue;
      }
      const int pd_idx = prediction_subset()[pd_subset_idx];
      const int gt_idx = ground_truth_subset()[gt_subset_idx];
      if (CanMatch(pd_idx, gt_idx)) {
        if (IoU(pd_subset_idx, gt_subset_idx) > best_gt_match_iou) {
          best_gt_match_iou = IoU(pd_subset_idx, gt_subset_idx);
          best_gt_match = gt_subset_idx;
        }
      }
    }
    if (best_gt_match != -1) {
      (*ground_truth_matches)[best_gt_match] = pd_subset_idx;
      (*prediction_matches)[pd_subset_idx] = best_gt_match;
    }
  }
}

}  // namespace open_dataset
}  // namespace waymo
