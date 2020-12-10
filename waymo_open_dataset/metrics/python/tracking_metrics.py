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
"""tf.metrics implementation for tracking metrics."""

import tensorflow as tf

from waymo_open_dataset import label_pb2
from waymo_open_dataset.metrics.ops import py_metrics_ops
from waymo_open_dataset.metrics.python import config_util_py as config_util


def _update(name, update, init_shape, dtype):
  """Updates variable 'name' by concatenating 'update'.

  Args:
    name: Variable name.
    update: tensor to be be concatenated to the existing value of the variable.
    init_shape: The initial shape of the variable.
    dtype: the type of the variable.

  Returns:
    v: the variable ref.
    v_assign: tensor that hold the new value of the variable after the update.
  """
  with tf.compat.v1.variable_scope(
      'tracking_metrics', reuse=tf.compat.v1.AUTO_REUSE):
    initializer = lambda: tf.constant([], shape=init_shape, dtype=dtype)
    v = tf.compat.v1.get_local_variable(
        name,
        dtype=dtype,
        collections=[
            tf.compat.v1.GraphKeys.LOCAL_VARIABLES,
            tf.compat.v1.GraphKeys.METRIC_VARIABLES
        ],
        # init_shape is required to pass the shape inference check.
        initializer=initializer,
        validate_shape=False)
    shape = tf.concat([[-1], tf.shape(input=update)[1:]], axis=0)
    v_reshape = tf.reshape(v.value(), shape)
    v_assign = tf.compat.v1.assign(
        v, tf.concat([v_reshape, update], axis=0), validate_shape=False)
  return v, v_assign


def _get_box_dof(box_type):
  """Gets the desired number of box degree of freedom for a box type.

  Args:
    box_type: The type of the box.

  Returns:
    The desired degrees of freedom for the box type.
  """
  if box_type == label_pb2.Label.Box.Type.Value('TYPE_3D'):
    return 7
  if box_type == label_pb2.Label.Box.Type.Value('TYPE_2D'):
    return 5
  if box_type == label_pb2.Label.Box.Type.Value('TYPE_AA_2D'):
    return 4
  return -1


def get_tracking_metric_ops(config,
                            prediction_bbox,
                            prediction_type,
                            prediction_score,
                            prediction_frame_id,
                            prediction_sequence_id,
                            prediction_object_id,
                            ground_truth_bbox,
                            ground_truth_type,
                            ground_truth_frame_id,
                            ground_truth_sequence_id,
                            ground_truth_object_id,
                            ground_truth_difficulty,
                            prediction_overlap_nlz=None,
                            ground_truth_speed=None):
  """Returns dict of metric name to tuples of `(value_op, update_op)`.

  Each update_op accumulates the prediction and ground truth tensors to its
  corresponding tf variables. Each value_op computes tracking metrics on all
  prediction and ground truth seen so far. This works similar as `tf.metrics`
  code.

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
    config: The metrics config defined in protos/metrics.proto.
    prediction_bbox: [M, D] tensor encoding the predicted bounding boxes.
    prediction_type: [M] tensor encoding the object type of each prediction.
    prediction_score: [M] tensor encoding the score of each prediction.
    prediction_frame_id: [M] int64 tensor that identifies frame for each
      prediction.
    prediction_sequence_id: [M] int64 tensor that identifies sequence for each
      prediction.
    prediction_object_id: [M] int64 tensor that identifies object for each
      prediction.
    ground_truth_bbox: [N, D] tensor encoding the ground truth bounding boxes.
    ground_truth_type: [N] tensor encoding the object type of each ground truth.
    ground_truth_frame_id: [N] int64 tensor that identifies frame for each
      ground truth.
    ground_truth_sequence_id: [M] int64 tensor that identifies sequence for each
      ground truth.
    ground_truth_object_id: [M] int64 tensor that identifies object for each
      ground truth.
    ground_truth_difficulty: [N] tensor encoding the difficulty level of each
      ground truth.
    prediction_overlap_nlz: [M] tensor encoding whether each prediction overlaps
      with any no label zone.
    ground_truth_speed: [N, 2] tensor with the vx, vy velocity for each object.

  Returns:
    A dictionary of metric names to tuple of value_op and update_op.
  """
  if ground_truth_speed is None:
    num_gt_boxes = tf.shape(ground_truth_bbox)[0]
    ground_truth_speed = tf.zeros((num_gt_boxes, 2), tf.float32)

  if prediction_overlap_nlz is None:
    prediction_overlap_nlz = tf.zeros_like(prediction_frame_id, dtype=tf.bool)

  eval_dict = {
      'prediction_bbox':
          (prediction_bbox, [0, _get_box_dof(config.box_type)], tf.float32),
      'prediction_type': (prediction_type, [0], tf.uint8),
      'prediction_score': (prediction_score, [0], tf.float32),
      'prediction_frame_id': (prediction_frame_id, [0], tf.int64),
      'prediction_sequence_id': (prediction_sequence_id, [0], tf.string),
      'prediction_object_id': (prediction_object_id, [0], tf.int64),
      'ground_truth_bbox':
          (ground_truth_bbox, [0, _get_box_dof(config.box_type)], tf.float32),
      'ground_truth_type': (ground_truth_type, [0], tf.uint8),
      'ground_truth_frame_id': (ground_truth_frame_id, [0], tf.int64),
      'ground_truth_sequence_id': (ground_truth_sequence_id, [0], tf.string),
      'ground_truth_object_id': (ground_truth_object_id, [0], tf.int64),
      'ground_truth_difficulty': (ground_truth_difficulty, [0], tf.uint8),
      'prediction_overlap_nlz': (prediction_overlap_nlz, [0], tf.bool),
      'ground_truth_speed': (ground_truth_speed, [0, 2], tf.float32),
  }

  variable_and_update_ops = {
      name: _update(name, update, init_shape, dtype)
      for name, (update, init_shape, dtype) in eval_dict.items()
  }

  update_ops = [value[1] for value in variable_and_update_ops.values()]
  update_op = tf.group(update_ops)
  variable_map = {
      name: value[0] for name, value in variable_and_update_ops.items()
  }

  config_str = config.SerializeToString()
  mota, motp, miss, mismatch, fp, _, _ = py_metrics_ops.tracking_metrics(
      config=config_str, **variable_map)
  breakdown_names = config_util.get_breakdown_names_from_config(config)
  metric_ops = {}
  for i, name in enumerate(breakdown_names):
    if i == 0:
      metric_ops['{}/MOTA'.format(name)] = (mota[i], update_op)
    else:
      # Set update_op to be an no-op just in case if anyone runs update_ops in
      # multiple session.run()s.
      metric_ops['{}/MOTA'.format(name)] = (mota[i], tf.constant([]))

    metric_ops['{}/MOTP'.format(name)] = (motp[i], tf.constant([]))
    metric_ops['{}/MISS'.format(name)] = (miss[i], tf.constant([]))
    metric_ops['{}/MISMATCH'.format(name)] = (mismatch[i], tf.constant([]))
    metric_ops['{}/FP'.format(name)] = (fp[i], tf.constant([]))
  return metric_ops
