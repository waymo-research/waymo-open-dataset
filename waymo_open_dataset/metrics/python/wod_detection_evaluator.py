# Copyright 2019 The Waymo Open Dataset Authors. All Rights Reserved.
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
"""The WOD (Waymo Open Dataset) evaluator.

The following snippet demonstrates the use of interfaces:

  evaluator = WODDetectionEvaluator(...)
  for _ in range(num_evals):
    for _ in range(num_batches_per_eval):
      predictions, groundtruth = predictor.predict(...)  # pop a batch.
      evaluator.update_state(groundtruths, predictions)
    evaluator.result()  # finish one full eval and reset states.
"""

import six
import tensorflow as tf

from waymo_open_dataset import label_pb2
from waymo_open_dataset.metrics.ops import py_metrics_ops
from waymo_open_dataset.metrics.python import config_util_py as config_util
from waymo_open_dataset.protos import breakdown_pb2
from waymo_open_dataset.protos import metrics_pb2


class WODDetectionEvaluator(object):
  """WOD detection evaluation metric class."""

  def __init__(self, config=None):
    """Constructs WOD detection evaluation class.

    Args:
      config: The metrics config defined in protos/metrics.proto.
    """
    if config is None:
      config = self._get_default_config()
    self._config = config

    # These are the keys in the metric_dict returned by evaluate.
    self._metric_names = [
        'average_precision',
        'average_precision_ha_weighted',
        'precision_recall',
        'precision_recall_ha_weighted',
        'breakdown',
    ]
    self._breakdown_names = config_util.get_breakdown_names_from_config(config)

    self._required_prediction_fields = [
        'prediction_frame_id',
        'prediction_bbox',
        'prediction_type',
        'prediction_score',
        'prediction_overlap_nlz',
    ]

    self._required_groundtruth_fields = [
        'ground_truth_frame_id',
        'ground_truth_bbox',
        'ground_truth_type',
        'ground_truth_difficulty',
    ]
    self.reset_states()

  @property
  def name(self):
    return 'wod_metric'

  def reset_states(self):
    """Resets internal states for a fresh run."""
    self._predictions = {}
    self._groundtruths = {}

  def result(self):
    """Evaluates detection results, and reset_states."""
    metric_dict = self.evaluate()
    # Cleans up the internal variables in order for a fresh eval next time.
    self.reset_states()
    return metric_dict

  def evaluate(self):
    """Evaluates with detections from all images with WOD API.

    Returns:
      metric_dict: dictionary to float numpy array representing the wod
      evaluation metrics. Keys in metric dictionary:
        - average_precision
        - average_precision_ha_weighted
        - precision_recall
        - precision_recall_ha_weighted
        - breakdown
    """
    metric_dict = py_metrics_ops.detection_metrics(
        prediction_bbox=tf.concat(self._predictions['prediction_bbox'], axis=0),
        prediction_type=tf.concat(self._predictions['prediction_type'], axis=0),
        prediction_score=tf.concat(
            self._predictions['prediction_score'], axis=0),
        prediction_frame_id=tf.concat(
            self._predictions['prediction_frame_id'], axis=0),
        prediction_overlap_nlz=tf.concat(
            self._predictions['prediction_overlap_nlz'], axis=0),
        ground_truth_bbox=tf.concat(
            self._groundtruths['ground_truth_bbox'], axis=0),
        ground_truth_type=tf.concat(
            self._groundtruths['ground_truth_type'], axis=0),
        ground_truth_frame_id=tf.concat(
            self._groundtruths['ground_truth_frame_id'], axis=0),
        ground_truth_difficulty=tf.concat(
            self._groundtruths['ground_truth_difficulty'], axis=0),
        config=self._config.SerializeToString(),
        ground_truth_speed=(tf.concat(
            self._groundtruths['ground_truth_speed'],
            axis=0) if 'ground_truth_speed' in self._groundtruths else None),
    )

    return metric_dict

  def update_state(self, groundtruths, predictions):
    """Update and aggregate detection results and groundtruth data.

    Notation:
      * M: number of predicted boxes.
      * D: number of box dimensions. The number of box dimensions can be one of
           the following:
             4: Used for boxes with type TYPE_AA_2D (center_x, center_y, length,
                width)
             5: Used for boxes with type TYPE_2D (center_x, center_y, length,
                width, heading).
             7: Used for boxes with type TYPE_3D (center_x, center_y, center_z,
                length, width, height, heading).
      * N: number of ground truth boxes.

    Args:
      groundtruths: a dictionary of Tensors including the fields below.
        Required fields:
          - ground_truth_frame_id: [N] int64 tensor that identifies frame for
            each ground truth.
          - ground_truth_bbox: [N, D] tensor encoding the ground truth bounding
            boxes.
          - ground_truth_type: [N] tensor encoding the object type of each
            ground truth.
          - ground_truth_difficulty: [N] tensor encoding the difficulty level of
            each ground truth.
        Optional fields:
          - ground_truth_speed: [N, 2] tensor with the vx, vy velocity for each
            object.
          - recall_at_precision: a float within [0,1]. If set, returns a 3rd
            metric that reports the recall at the given precision.
      predictions: a dictionary of tensors including the fields below.
        Required fields:
          - prediction_frame_id: [M] int64 tensor that identifies frame for each
            prediction.
          - prediction_bbox: [M, D] tensor encoding the predicted bounding
            boxes.
          - prediction_type: [M] tensor encoding the object type of each
            prediction.
          - prediction_score: [M] tensor encoding the score of each prediciton.
          - prediction_overlap_nlz: [M] tensor encoding whether each prediciton
            overlaps with any no label zone.

    Raises:
      ValueError: if the required prediction or groundtruth fields are not
        present in the incoming `predictions` or `groundtruths`.
    """
    # Append predictions.
    for k in self._required_prediction_fields:
      if k not in predictions:
        raise ValueError(
            'Missing the required key `{}` in predictions!'.format(k))
    for k, v in six.iteritems(predictions):
      if k not in self._predictions:
        self._predictions[k] = [v]
      else:
        self._predictions[k].append(v)

    # Append groundtruths.
    for k in self._required_groundtruth_fields:
      if k not in groundtruths:
        raise ValueError(
            'Missing the required key `{}` in groundtruths!'.format(k))
    for k, v in six.iteritems(groundtruths):
      if k not in self._groundtruths:
        self._groundtruths[k] = [v]
      else:
        self._groundtruths[k].append(v)

  def _get_default_config(self):
    """Returns the default Config proto for detection.

    This is the python version of the GetConfig() function in
    metrics/tools/compute_detection_metrics_main.cc
    """
    config = metrics_pb2.Config()

    config.breakdown_generator_ids.append(breakdown_pb2.Breakdown.OBJECT_TYPE)
    difficulty = config.difficulties.add()
    difficulty.levels.append(label_pb2.Label.LEVEL_1)
    difficulty.levels.append(label_pb2.Label.LEVEL_2)
    config.breakdown_generator_ids.append(breakdown_pb2.Breakdown.RANGE)
    difficulty = config.difficulties.add()
    difficulty.levels.append(label_pb2.Label.LEVEL_1)
    difficulty.levels.append(label_pb2.Label.LEVEL_2)

    config.matcher_type = metrics_pb2.MatcherProto.TYPE_HUNGARIAN
    config.iou_thresholds.append(0.0)
    config.iou_thresholds.append(0.7)
    config.iou_thresholds.append(0.5)
    config.iou_thresholds.append(0.5)
    config.iou_thresholds.append(0.5)
    config.box_type = label_pb2.Label.Box.TYPE_3D

    for i in range(100):
      config.score_cutoffs.append(i * 0.01)
    config.score_cutoffs.append(1.0)

    return config
