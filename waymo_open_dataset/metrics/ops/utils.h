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

#ifndef WAYMO_OPEN_DATASET_METRICS_OPS_UTILS_H_
#define WAYMO_OPEN_DATASET_METRICS_OPS_UTILS_H_

#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/types/optional.h"
#include "tensorflow/core/framework/tensor.h"
#include "waymo_open_dataset/common/integral_types.h"
#include "waymo_open_dataset/label.pb.h"
#include "waymo_open_dataset/protos/metrics.pb.h"
#include "waymo_open_dataset/protos/motion_metrics.pb.h"
#include "waymo_open_dataset/protos/motion_submission.pb.h"
#include "waymo_open_dataset/protos/scenario.pb.h"

namespace waymo {
namespace open_dataset {

// Returns the desired number of box degrees of freedom given the config.
// 3d -> 7, 2d -> 5, 2d axis aligned -> 4.
int GetDesiredBoxDOF(Label::Box::Type box_type);

// Parses objects from tensors.
// bbox: [N, D] float tensor encodes the boxes.
// type: [N] uint8 tensor encodes the object types.
// frame_id: [N] int64 tensor tells which frame the box belongs to.
// score: [N] float tensor encodes the confidence scrore of each box.
// overlap_nlz: [N] boolean tensor tells whether a box overlaps with any no
//   label zone.
// detection_dificulty: [N] uint8 tensor tells the difficulty level of each box
//   for detection problems.
// tracking_dificulty: [N] uint8 tensor tells the difficulty level of each box
//   for tracking problems.
// Returns objects parsed from tensors grouped by frame IDs.
absl::flat_hash_map<int64, std::vector<Object>> ParseObjectFromTensors(
    const tensorflow::Tensor& bbox, const tensorflow::Tensor& type,
    const tensorflow::Tensor& frame_id,
    const absl::optional<const tensorflow::Tensor>& score,
    const absl::optional<const tensorflow::Tensor>& overlap_nlz,
    const absl::optional<const tensorflow::Tensor>& detection_difficulty,
    const absl::optional<const tensorflow::Tensor>& tracking_difficulty,
    const absl::optional<const tensorflow::Tensor>& object_speed);

// Parses objects from tensors such that they are grouped by sequence_id and
// then frame_id.
// bbox: [N, D] float tensor encodes the boxes.
// type: [N] uint8 tensor encodes the object types.
// frame_id: [N] int64 tensor tells which frame the box belongs to.
// sequence_id: [N] string tensor tells which frame the box belongs to.
// object_id: [N] int64 tensor tells which frame the box belongs to.
// score: [N] float tensor encodes the confidence scrore of each box.
// overlap_nlz: [N] boolean tensor tells whether a box overlaps with any no
//   label zone.
// detection_dificulty: [N] uint8 tensor tells the difficulty level of each box
//   for detection problems.
// tracking_dificulty: [N] uint8 tensor tells the difficulty level of each box
//   for tracking problems.
// object_speed: [N, 2] float tensor that tells the speed in xy of the object.
absl::flat_hash_map<std::string,
                    absl::flat_hash_map<int64, std::vector<Object>>>
ParseObjectGroupedBySequenceFromTensors(
    const tensorflow::Tensor& bbox, const tensorflow::Tensor& type,
    const tensorflow::Tensor& frame_id, const tensorflow::Tensor& sequence_id,
    const tensorflow::Tensor& object_id,
    const absl::optional<const tensorflow::Tensor>& score,
    const absl::optional<const tensorflow::Tensor>& overlap_nlz,
    const absl::optional<const tensorflow::Tensor>& detection_difficulty,
    const absl::optional<const tensorflow::Tensor>& tracking_difficulty,
    const absl::optional<const tensorflow::Tensor>& object_speed);

// Parse Scenario and ScenarioPredictions protos from tensors.
//
// - B: batch size. Each batch should contain 1 scenario.
// - M: Number of joint prediction groups to predict per scenario.
// - K: top_K predictions per joint prediction.
// - N: number of agents in a joint prediction. 1 if mutual independence is
//     assumed between agents.
// - A: number of agents in the groundtruth.
// - TP: number of steps to evaluate on. Matches len(config.step_measurement).
// - TG: number of steps in the groundtruth track. Matches
//     config.track_history_samples + 1 + config.future_history_samples.
//
// For the marginal (mutually-independent) prediction, M is the number of agents
// to predict for, N is 1.
//
// For joint predictions of multiple agents, M is the number of joint
// predictions, and N is the number of agents per joint prediction.
//
// pred_trajectory: [B, M, K, N, TP, 2]. Predicted trajectories.
//   The inner-most dimensions are [x, y].
// pred_score: [B, M, K]. Scores per joint prediction.
// gt_trajectory: [B, A, TG, 7]. Groundtruth trajectories.
//   The inner-most dimensions are [x, y, length, width, heading, velocity_x,
//   velocity_y].
// gt_is_valid: [B, A, TG]. Indicates whether a time stamp is valid
//   per agent. If all timestamps for a trajectory are invalid, the
//   trajectory is assumed invalid.
// pred_gt_indices: [B, M, N]. Indices to gather the predictions
//   of shape [B, M, ?, N] from the groundtruth of shape [B, A], values must be
//   between [0, A).
// pred_gt_indices_mask: [B, M, N]. A validity mask for
//   `prediction_ground_truth_indices`.
// object_type: [B, A] Object type per trajectory.
// object_id: [B, A]. Object IDs per trajectory.
// scenario_id: [B]. Scenario IDs of all groundtruth trajectories.
absl::flat_hash_map<std::string, std::pair<Scenario, ScenarioPredictions>>
ParseScenarioAndPredictonsFromTensors(
    const tensorflow::Tensor& pred_trajectory,
    const tensorflow::Tensor& pred_score,
    const tensorflow::Tensor& gt_trajectory,
    const tensorflow::Tensor& gt_is_valid,
    const tensorflow::Tensor& pred_gt_indices,
    const tensorflow::Tensor& pred_gt_indices_mask,
    const tensorflow::Tensor& object_type, const tensorflow::Tensor& object_id,
    const tensorflow::Tensor& scenario_id);

}  // namespace open_dataset
}  // namespace waymo

#endif  // WAYMO_OPEN_DATASET_METRICS_OPS_UTILS_H_
