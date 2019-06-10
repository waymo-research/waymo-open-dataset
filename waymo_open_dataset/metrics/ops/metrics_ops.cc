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

namespace tensorflow {
namespace {

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

}  // namespace
}  // namespace tensorflow
