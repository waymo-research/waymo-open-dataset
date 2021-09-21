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

#ifndef WAYMO_OPEN_DATASET_METRICS_DETECTION_METRICS_H_
#define WAYMO_OPEN_DATASET_METRICS_DETECTION_METRICS_H_

#include <vector>

#include "waymo_open_dataset/metrics/matcher.h"
#include "waymo_open_dataset/protos/metrics.pb.h"

namespace waymo {
namespace open_dataset {

// Computes detection measurements for a single frame.
// pds: the predicted objects.
// gts: the ground truths.
// custom_iou_func: an optional callback for customizing IoU calculation.
// The output vector is ordered as:
// [{generator_i_shard_j_difficulty_level_k}].
// i \in [0, num_breakdown_generators).
// j \in [0, num_shards for the i-th breakdown generator).
// k \in [0, num_difficulty_levels for each shard in the  i-th breakdown
//   generator).
//
//   Requires: config.score_thresholds() is populated.
std::vector<DetectionMeasurements> ComputeDetectionMeasurements(
    const Config& config, const std::vector<Object>& pds,
    const std::vector<Object>& gts, ComputeIoUFunc custom_iou_func = nullptr);

// Computes detection metrics from measurements.
// Each element of `measurements` is an output of ComputeDetectionMeasurements.
// The output vector is ordered as:
// [{generator_i_shard_j_difficulty_level_k}].
// i \in [0, num_breakdown_generators).
// j \in [0, num_shards for the i-th breakdown generator).
// k \in [0, num_difficulty_levels for each shard in the  i-th breakdown
//   generator).
//
// Requires: Every element of `measurements` is computed with the same
// configuration.
std::vector<DetectionMetrics> ComputeDetectionMetrics(
    const Config& config,
    const std::vector<std::vector<DetectionMeasurements>>& measurements);

// Computes detection metrics for multiple frames.
// pds: the predicted objects.
// gts: the ground truths.
// custom_iou_func: an optional callback for customizing IoU calculation.
// The output vector is ordered as:
// [{generator_i_shard_j_difficulty_level_k}].
// i \in [0, num_breakdown_generators).
// j \in [0, num_shards for the i-th breakdown generator).
// k \in [0, num_difficulty_levels for each shard in the i-th breakdown
//   generator).
//
// Requires: 'pds', 'gts' have the same dim-0 size.
std::vector<DetectionMetrics> ComputeDetectionMetrics(
    const Config& config, const std::vector<std::vector<Object>>& pds,
    const std::vector<std::vector<Object>>& gts,
    ComputeIoUFunc custom_iou_func = nullptr);

// Estimates the score cutoffs that evenly sample the P/R curve.
// pds: the predicted objects.
// gts: the ground truths.
// Returns a Config that has Config::scores populated.
// Requires: config.scores is not populated.
Config EstimateScoreCutoffs(const Config& config,
                            const std::vector<std::vector<Object>>& pds,
                            const std::vector<std::vector<Object>>& gts);

}  // namespace open_dataset
}  // namespace waymo

#endif  // WAYMO_OPEN_DATASET_METRICS_DETECTION_METRICS_H_
