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

#ifndef WAYMO_OPEN_DATASET_METRICS_METRICS_UTILS_H_
#define WAYMO_OPEN_DATASET_METRICS_METRICS_UTILS_H_

// Some internal helper functions to implement detection and tracking metrics.
// Refer to detection_metrics.h and tracking_metrics.h for public APIs.

#include <vector>

#include "waymo_open_dataset/label.pb.h"
#include "waymo_open_dataset/metrics/matcher.h"
#include "waymo_open_dataset/protos/breakdown.pb.h"
#include "waymo_open_dataset/protos/metrics.pb.h"

namespace waymo {
namespace open_dataset {
namespace internal {

// Returns true if the i-th entry in the pd_matches is a true positive.
// Requires: i >= 0, i < pd_matches.size().
bool IsTP(const std::vector<int>& pd_matches, int i);

// Returns true if the i-th entry in the pd_matches is a false positive.
// 'matcher' is needed to examine whether the prediction overlaps any NLZs.
// Requires: pd_matches is a valid matching result from 'matcher'
// Requires: i >= 0, i < pd_matches.size().
bool IsFP(const Matcher& matcher, const std::vector<int>& pd_matches, int i);

// Returns true if the i-th entry in the gt_matches is a false positive for
// detection at the given difficulty level.
// Detection and tracking are considered differently as they have a different
// definition of difficulty.
// Requires: gt_matches is a valid matching result from 'matcher'.
// Requires: i >= 0, i < gt_matches.size().
bool IsDetectionFN(const Matcher& matcher, const std::vector<int>& gt_matches,
                   int i, Label::DifficultyLevel level);

// Returns true if the i-th entry in the gt_matches is a false positive for
// tracking at the given difficulty level.
// Detection and tracking are considered differently as they have different
// definition of difficulty.
// Requires: gt_matches is a valid matching result from 'matcher'.
// Requires: i >= 0, i < gt_matches.size().
bool IsTrackingFN(const Matcher& matcher, const std::vector<int>& gt_matches,
                  int i, Label::DifficultyLevel level);

// Returns the heading accuracy for the given prediction and ground truth.
// The heading accuracy is within [0.0, 1.0]. The higher, the better.
// Requires: prediction_index and ground_truth_index are valid indices from
// 'matcher'.
float ComputeHeadingAccuracy(const Matcher& matcher, int prediction_index,
                             int ground_truth_index);

// A convenient struct that wraps object subsets for all score cutoffs.
// It is used to provide inputs for Matcher::Set{Prediction,GroundTruth}Subset.
struct BreakdownShardSubset {
  // The subsets ordered the same as scores in the config.
  // The size of this is set to 1 if the struct is used for ground truths as all
  // score cutoffs share the ground truth same subset.
  std::vector<std::vector<int>> indices;

  // The breakdown ID index of Config::breakdown_generator_ids associated with
  // this subset. This is provided as downstream consumers of this struct need
  // to compute metrics for different difficulty levels which is specified per
  // breakdown generator.
  int breakdown_generator_id_index = -1;
  // The breakdown shard this subset is computed for.
  int breakdown_shard = -1;
};

// Builds all breakdown subsets for the given objects (either prediction or
// ground truths) based on the config.
// Set is_gt if 'objects' are ground truths.
// Set is_detection if this function is called when computing detection metrics.
// Output ordering:
// [{generator_i_shard_j}].
// i \in [0, num_breakdown_generators).
// j \in [0, num_shards for the i-th breakdown generator).
std::vector<BreakdownShardSubset> BuildSubsets(
    const Config& config, const std::vector<Object>& objects, bool is_gt,
    bool is_detection);

// Returns a vector of difficulty levels for the given breakdown generator ID
// index of Config::breakdown_generator_ids based on the config.
std::vector<Label::DifficultyLevel> GetDifficultyLevels(
    const Config& config, int breakdown_generator_id_index);

// Decides score cutoffs that evenly distributes scores into
// `num_desired_cutoffs` buckets.
// Requires: scores are sorted in ascending order. Every element in scores is
//   within [0.0, 1.0].
std::vector<float> DecideScoreCutoffs(const std::vector<float>& scores,
                                      int num_desired_cutoffs);

// Computes mean average precision from pairs of precision and recalls.
// Notes about the implementation details:
// 1. We follow COCO by updating p(r) to be p(r) = max_{r' >= r}p(r').
// 2. If the gap between consecutive recalls are bigger than a desired threshold
//    (e.g. 0.05), p/r points are explicitly added with conservative precisions.
//    Example: p(r): p(0) = 1.0, p(1) = 0.0, delta = 0.05. We add p(0.95) = 0.0,
//    p(0.90) = 0.0, ..., p(0.05) = 0.0. The mAP = 0.05 after this augmentation.
//    This avoids producing an over-estimated mAP with very sparse p/r curve
//    sampling.
//    To disable this, set recall_delta to 1.0.
// Note: the precisions and recalls passed here are not necessarily to be sorted
// but need to have 1:1 correspondence.
float ComputeMeanAveragePrecision(const std::vector<float>& precisions,
                                  const std::vector<float>& recalls,
                                  float max_recall_delta);

// Estimates predicted object speed by looking for the nearest groundtruth
// object.
std::vector<Object> EstimateObjectSpeed(const std::vector<Object>& pds,
                                        const std::vector<Object>& gts);

// Similar as above but takes multiple frames prediction as input.
std::vector<std::vector<Object>> EstimateObjectSpeed(
    const std::vector<std::vector<Object>>& pds,
    const std::vector<std::vector<Object>>& gts);

// Returns true if velocity breakdown is enabled.
bool HasVelocityBreakdown(const Config& config);

// Finds the ground truth that has the largest IoU with the given prediction
// represented by its subset ID within the given matcher.
int FindGTWithLargestIoU(const Matcher& matcher, int pd_subset_id,
                         double iou_threshold);

// Returns true if the object is in the given breakdown.
bool IsInBreakdown(const Object& object, const Breakdown& breakdown);

// Returns true if the breakdown is a ground truth only breakdown.
bool IsGroundTruthOnlyBreakdown(const Breakdown& breakdown);

}  // namespace internal
}  // namespace open_dataset
}  // namespace waymo

#endif  // WAYMO_OPEN_DATASET_METRICS_METRICS_UTILS_H_
