# Copyright 2023 The Waymo Open Dataset Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Tools for computing box matching using Open Dataset criteria."""
import dataclasses
from typing import Optional

import tensorflow as tf

from waymo_open_dataset import label_pb2
from waymo_open_dataset.metrics.ops import py_metrics_ops
from waymo_open_dataset.protos import breakdown_pb2
from waymo_open_dataset.protos import metrics_pb2

TensorLike = tf.types.experimental.TensorLike
Type = label_pb2.Label.Box.Type
LongitudinalErrorTolerantConfig = (
    metrics_pb2.Config.LongitudinalErrorTolerantConfig
)


@dataclasses.dataclass
class MatchResult:
  """Object holding matching result.

  A single i-th element in each tensor describes a match between predicted and
  groundtruth box, the associated IOU (2D IOU, 3D IOU, or the LET variant), and
  the index in the batch where the batch was made.
  """

  # Indices into the prediction boxes, as if boxes were reshaped as [-1, dim].
  prediction_ids: TensorLike  # [N]
  # Indices into the groundtruth boxes, as if boxes were reshaped as [-1, dim].
  groundtruth_ids: TensorLike  # [N]
  # Matching quality (IOU variant depends on the matching config).
  ious: TensorLike  # [N]
  # Longitudinal affinity for a given match.
  longitudinal_affinities: TensorLike  # [N]


def match(
    prediction_boxes: TensorLike,
    groundtruth_boxes: TensorLike,
    iou: float,
    box_type: 'label_pb2.Label.Box.Type',
    matcher_type: metrics_pb2.MatcherProto.Type = metrics_pb2.MatcherProto.TYPE_HUNGARIAN,
    let_metric_config: Optional[LongitudinalErrorTolerantConfig] = None,
) -> MatchResult:
  """Returns a matching between predicted and groundtruth boxes.

  Matching criteria and thresholds are specified through the config. Boxes are
  represented as [center_x, center_y, bottom_z, length, width, height, heading].
  The function treats "zeros(D)" as an invalid box and will not try to match it.

  Args:
    prediction_boxes: [B, N, D] or [N, D] tensor.
    groundtruth_boxes: [B, M, D] or [N, D] tensor.
    iou: IOU threshold to use for matching.
    box_type: whether to perform matching in 2D or 3D.
    matcher_type: the matching algorithm for the matcher. Default to Hungarian.
    let_metric_config: Optional config describing how LET matching should be
      done.

  Returns:
    A match result struct with flattened tensors where i-th element describes a
    match between predicted and groundtruth box, the associated IOU
    (AA_2D, 2D IOU, 3D IOU, or the LET variant), and the index in the batch
    where the match was made:
      - indices of predicted boxes [Q]
      - corresponding indices of groundtruth boxes [Q]
      - IOUs for each match. [Q]
      - index in the input batch [Q] with values in {0, ..., B-1}.
  """
  with tf.name_scope('open_dataset_matcher/match'):
    config = _create_config(iou, box_type, matcher_type, let_metric_config)
    if tf.rank(prediction_boxes) == 2:
      prediction_boxes = tf.expand_dims(prediction_boxes, 0)
    if tf.rank(groundtruth_boxes) == 2:
      groundtruth_boxes = tf.expand_dims(groundtruth_boxes, 0)
    tf.debugging.assert_shapes([
        (prediction_boxes, ('b', 'n', 'd')),
        (groundtruth_boxes, ('b', 'm', 'd')),
    ])
    pred_ids, gt_ids, ious, la = py_metrics_ops.match(
        prediction_boxes, groundtruth_boxes, config=config.SerializeToString()
    )
    return MatchResult(
        prediction_ids=pred_ids,
        groundtruth_ids=gt_ids,
        ious=ious,
        longitudinal_affinities=la,
    )


def _create_config(
    iou: float,
    box_type: 'label_pb2.Label.Box.Type',
    matcher_type: metrics_pb2.MatcherProto.Type,
    let_metric_config: Optional[LongitudinalErrorTolerantConfig] = None,
) -> metrics_pb2.Config:
  return metrics_pb2.Config(
      score_cutoffs=[0.0],
      box_type=box_type,
      difficulties=[metrics_pb2.Difficulty(levels=[])],
      breakdown_generator_ids=[breakdown_pb2.Breakdown.ONE_SHARD],
      matcher_type=matcher_type,
      iou_thresholds=[0, iou, iou, iou, iou],
      let_metric_config=let_metric_config,
  )
