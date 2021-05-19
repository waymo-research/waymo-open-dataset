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

#include "absl/strings/escaping.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "waymo_open_dataset/metrics/config_util.h"
#include "waymo_open_dataset/metrics/ops/utils.h"
#include "waymo_open_dataset/protos/metrics.pb.h"
#include "waymo_open_dataset/protos/motion_metrics.pb.h"

namespace tensorflow {
namespace {

using ::tensorflow::shape_inference::InferenceContext;

REGISTER_OP("DetectionMetrics")
    .Input("prediction_bbox: float")
    .Input("prediction_type: uint8")
    .Input("prediction_score: float")
    .Input("prediction_frame_id: int64")
    .Input("prediction_overlap_nlz: bool")
    .Input("ground_truth_bbox: float")
    .Input("ground_truth_type: uint8")
    .Input("ground_truth_frame_id: int64")
    .Input("ground_truth_difficulty: uint8")
    .Input("ground_truth_speed: float")
    .Output("average_precision: float")
    .Output("average_precision_ha_weighted: float")
    .Output("precision_recall: float")
    .Output("precision_recall_ha_weighted: float")
    .Output("breakdown: uint8")
    .Attr("config: string")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      return Status::OK();
    })
    .Doc(R"doc(
Computes detection metrics.

prediction_bbox: [N, D]. N predicted bounding boxes of D (4, 5 or 7)
  dimensions. It is OK to pass boxes with higher D than necessary. For example,
  you can pass boxes with D = 7 while evaluating BEV metrics by setting
  box_type in the config as TYPE_2D.
prediction_type: [N]. Predicted object type of each bounding box.
prediction_score: [N]. N prediction scores for each predicted box.
prediction_frame_id: [N]. N frame IDs for each predicted box.
prediction_overlap_nlz: [N]. Whether each predicted box overlaps with any no
  label zone.
ground_truth_bbox: [M, D]. M ground truth bounding boxes of D (4, 5 or 7)
  dimensions. It is OK to pass boxes with higher D than necessary. For example,
  you can pass boxes with D = 7 while evaluating BEV metrics by setting
  box_type in the config as TYPE_2D.
ground_truth_type: [M]. ground truth object type of each bounding box.
ground_truth_frame_id: [M]. M frame IDs for each ground truth box.
ground_truth_difficulty: [M] Difficulty level (1 or 2) for each ground truth
  box.
ground_truth_speed: [M, 2] M ground truth objects and their corresponding speed
  label to use for VELOCITY breakdown.
average_precision: [B]. average precision for each breakdown.
average_precision_ha_weighted: [B]. average precision with heading accuracy
  weighted for each breakdown.
precision_recall: [B, S, 2]. precision and recall pairs for each breakdown.
  S is the number of score cutoffs.
precision_recall_ha_weighted: [B, S, 2]. precision and recall with heading
  accuracy weighted pairs for each breakdown.
breakdown: [B, 3]. [generator_id, shard, difficulty] uint8 tuple for each
  breakdown.
config: a string serialized proto of metrics configuration protobuf.
)doc");

REGISTER_OP("MotionMetrics")
    .Input("prediction_trajectory: float")
    .Input("prediction_score: float")
    .Input("ground_truth_trajectory: float")
    .Input("ground_truth_is_valid: bool")
    .Input("prediction_ground_truth_indices: int64")
    .Input("prediction_ground_truth_indices_mask: bool")
    .Input("object_type: int64")
    .Input("object_id: int64")
    .Input("scenario_id: string")
    .Output("min_ade: float")
    .Output("min_fde: float")
    .Output("miss_rate: float")
    .Output("overlap_rate: float")
    .Output("mean_average_precision: float")
    .Attr("config: string")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      return Status::OK();
    })
    .Doc(R"doc(
Computes motion metrics.

- Notations:
- B: batch size. Each batch should contain exactly 1 scenario.
- M: Number of joint prediction groups to predict per scenario.
- K: top_K predictions per joint prediction.
- N: number of agents in a joint prediction. 1 if mutual independence is
    assumed between agents.
- A: number of agents in the groundtruth.
- TP: number of steps to evaluate on. Matches len(config.step_measurement).
- TG: number of steps in the groundtruth track. Matches
    config.track_history_samples + 1 + config.future_history_samples.
- BR: number of breakdowns.

For the marginal (mutually-independent) prediction, M is the number of agents to
predict for, N is 1.

For joint predictions of multiple agents, M is the number of joint predictions,
and N is the number of agents per joint prediction.

prediction_trajectory: [B, M, K, N, TP, 2]. Predicted trajectories.
  The inner-most dimensions are [x, y].
prediction_score: [B, M, K]. Scores per joint prediction.
ground_truth_trajectory: [B, A, TG, 7]. Groundtruth trajectories.
  The inner-most dimensions are [x, y, length, width, heading, velocity_x,
  velocity_y].
ground_truth_is_valid: [B, A, TG]. Indicates whether a time stamp is valid per
  agent. If all timestamps for a trajectory are invalid, the trajectory is
  assumed invalid.
prediction_ground_truth_indices: [B, M, N]. Indices to gather the predictions
  of shape [B, M, ?, N] from the groundtruth of shape [B, A], values must be
  between [0, A).
prediction_ground_truth_indices_mask: [B, M, N]. A validity mask for
  `prediction_ground_truth_indices`.
object_type: [B, A] Object type per trajectory.
object_id: [B, A]. Object IDs per trajectory.
scenario_id: [B]. Scenario IDs of all groundtruth trajectories.
min_ade: [BR]. Minimum average distance error among K proposals.
min_fde: [BR]. Minimum final distance error among K proposals.
miss_rate: [BR]. Miss rate given K guesses.
overlap_rate: [BR]. Overlap rate for each breakdown.
mean_average_precision: [BR]. Average precision for each breakdown.
config: a string serialized proto of metrics MotionConfig protobuf.
)doc");

REGISTER_OP("TrackingMetrics")
    .Input("prediction_bbox: float")
    .Input("prediction_type: uint8")
    .Input("prediction_score: float")
    .Input("prediction_frame_id: int64")
    .Input("prediction_sequence_id: string")
    .Input("prediction_object_id: int64")
    .Input("prediction_overlap_nlz: bool")
    .Input("ground_truth_bbox: float")
    .Input("ground_truth_type: uint8")
    .Input("ground_truth_frame_id: int64")
    .Input("ground_truth_sequence_id: string")
    .Input("ground_truth_object_id: int64")
    .Input("ground_truth_difficulty: uint8")
    .Input("ground_truth_speed: float")
    .Output("mota: float")
    .Output("motp: float")
    .Output("miss: float")
    .Output("mismatch: float")
    .Output("fp: float")
    .Output("score_cutoff: float")
    .Output("breakdown: uint8")
    .Attr("config: string")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      return Status::OK();
    })
    .Doc(R"doc(
Computes tracking metrics.

prediction_bbox: [N, D]. N predicted bounding boxes of D (4, 5 or 7)
  dimensions. It is OK to pass boxes with higher D than necessary. For example,
  you can pass boxes with D = 7 while evaluating BEV metrics by setting
  box_type in the config as TYPE_2D.
prediction_type: [N]. Predicted object type of each bounding box.
prediction_score: [N]. N prediction scores for each predicted box.
prediction_frame_id: [N]. N frame IDs for each predicted box.
prediction_overlap_nlz: [N]. Whether each predicted box overlaps with any no
  label zone.
prediction_sequence_id: [N]. sequence IDs for each predicted box, e.g.
  20171001_234201_C00847_2767_300_2787_300-FRONT for video detection/tracking
  and 20171001_234201_C00847_2767_300_2787_300 for point cloud 3D
  detection/tracking.
prediction_object_id: [N]. object IDs for each predicted box.

ground_truth_bbox: [M, D]. M ground truth bounding boxes of D (4, 5 or 7)
  dimensions. It is OK to pass boxes with higher D than necessary. For example,
  you can pass boxes with D = 7 while evaluating BEV metrics by setting
  box_type in the config as TYPE_2D.
ground_truth_type: [M]. ground truth object type of each bounding box.
ground_truth_frame_id: [M]. M frame IDs for each ground truth box.
ground_truth_difficulty: [M] Difficulty level (1 or 2) for each ground truth
  box.
ground_truth_speed: [M, 2] M ground truth objects and their corresponding speed
  label to use for VELOCITY breakdown.
ground_truth_sequence_id: [M]. sequence IDs for each predicted box.
ground_truth_object_id: [M]. object IDs for each ground truth box.

mota: [B]. Multiple object tracking accuracy (sum of miss, mismatch and fp).
motp: [B]. Multiple object tracking precision (matching_cost / num_matches).
miss: [B]. Miss ratio (num_misses / num_objects_gt).
mismatch: [B]. Mismatch ratio (num_mismatches / num_objects_gt).
fp: [B]. False positive ratio (num_fps / num_objects_gt).
score_cutoff: [B]. score cutoff for TrackingMetrics.
breakdown: [B, 3]. [generator_id, shard, difficulty] uint8 tuple for each
  breakdown.
config: a string serialized proto of metrics configuration protobuf.
)doc");

}  // namespace
}  // namespace tensorflow
