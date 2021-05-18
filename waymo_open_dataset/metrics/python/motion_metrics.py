# Copyright 2021 The Waymo Open Dataset Authors. All Rights Reserved.
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
"""tf.metrics implementation for motion metrics."""

import tensorflow as tf

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
      'motion_metrics', reuse=tf.compat.v1.AUTO_REUSE):
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


def get_motion_metric_ops(config,
                          prediction_trajectory,
                          prediction_score,
                          ground_truth_trajectory,
                          ground_truth_is_valid,
                          prediction_ground_truth_indices,
                          prediction_ground_truth_indices_mask,
                          object_type,
                          object_id=None,
                          scenario_id=None):
  """Returns dict of metric name to tuples of `(value_op, update_op)`.

  Each update_op accumulates the prediction and ground truth tensors to its
  corresponding tf variables. Each value_op computes tracking metrics on all
  prediction and ground truth seen so far. This works similar as `tf.metrics`
  code.

  All shapes except for the batch size must be statically defined.

  - Notations:
    - B: batch size. Each batch should contain 1 scenario.
    - M: Number of joint prediction groups to predict per scenario.
    - N: number of agents in a joint prediction. 1 if mutual independence is
        assumed between agents.
    - K: top_K predictions per joint prediction.
    - A: number of agents in the groundtruth.
    - TP: number of steps to evaluate on. Matches len(config.step_measurement).
    - TG: number of steps in the groundtruth track. Matches
        config.track_history_samples + 1 + config.future_history_samples.
    - BR: number of breakdowns.

  Args:
    config: The metrics config defined in protos/metrics.proto.
    prediction_trajectory: [B, M, K, N, TP, 2]. Predicted trajectories. The
      inner-most dimensions are [x, y].
    prediction_score: [B, M, K]. Scores per joint prediction.
    ground_truth_trajectory: [B, A, TG, 7]. Groundtruth trajectories. The
      inner-most dimensions are [x, y, length, width, heading, velocity_x,
      velocity_y].
    ground_truth_is_valid: [B, A, TG]. Indicates whether a time stamp is valid
      per trajectory. If all timestamps for a trajectory are invalid, the
      trajectory is assumed invalid.
    prediction_ground_truth_indices: [B, M, N]. Indices to gather the
      predictions of shape [B, M, ?, N] from the groundtruth of shape [B, A],
      values must be between [0, A).
    prediction_ground_truth_indices_mask: [B, M, N]. A validity mask for
      `prediction_ground_truth_indices`.
    object_type: [B, A] Object type per trajectory.
    object_id: [B, A]. Object IDs per trajectory.
    scenario_id: [B]. Scenario IDs of all groundtruth trajectories.

  Returns:
    A dictionary of metric names to tuple of value_op and update_op.
  """
  (_, num_groups, top_k, num_agents, num_pred_steps,
   _) = prediction_trajectory.shape.as_list()
  _, _, num_gt_steps, _ = ground_truth_trajectory.shape.as_list()

  assert prediction_trajectory.shape.as_list()[5] == 2
  assert ground_truth_trajectory.shape.as_list()[1] == num_agents
  assert ground_truth_trajectory.shape.as_list()[3] == 7
  assert ground_truth_is_valid.shape.as_list()[1] == num_agents
  assert ground_truth_is_valid.shape.as_list()[2] == num_gt_steps
  assert prediction_ground_truth_indices.shape.as_list()[1] == num_groups
  assert prediction_ground_truth_indices.shape.as_list()[2] == num_agents
  assert prediction_ground_truth_indices_mask.shape.as_list()[1] == num_groups
  assert prediction_ground_truth_indices_mask.shape.as_list()[2] == num_agents
  assert prediction_score.shape.as_list()[1] == num_groups
  assert prediction_score.shape.as_list()[2] == top_k
  assert object_type.shape.as_list()[1] == num_agents

  if object_id is None:
    batch_size = tf.shape(prediction_trajectory)[0]
    object_id = tf.tile(
        tf.range(num_agents, dtype=tf.int64)[tf.newaxis], (batch_size, 1))
  else:
    assert object_id.shape.as_list()[1] == num_agents

  if scenario_id is None:
    batch_size = tf.shape(prediction_trajectory)[0]
    scenario_id = tf.strings.as_string(tf.range(batch_size))

  eval_dict = {
      'ground_truth_trajectory':
          (ground_truth_trajectory, [0, num_agents, num_gt_steps,
                                     7], tf.float32),
      'ground_truth_is_valid':
          (ground_truth_is_valid, [0, num_agents, num_gt_steps], tf.bool),
      'prediction_trajectory':
          (prediction_trajectory,
           [0, num_groups, top_k, num_agents, num_pred_steps, 2], tf.float32),
      'prediction_score': (prediction_score, [0, num_groups,
                                              top_k], tf.float32),
      'prediction_ground_truth_indices':
          (prediction_ground_truth_indices, [0, num_groups,
                                             num_agents], tf.int64),
      'prediction_ground_truth_indices_mask':
          (prediction_ground_truth_indices_mask, [0, num_groups,
                                                  num_agents], tf.bool),
      'object_type': (object_type, [0, num_agents], tf.int64),
      'object_id': (object_id, [0, num_agents], tf.int64),
      'scenario_id': (scenario_id, [0], tf.string)
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
  (min_ade, min_fde, miss_rate, overlap_rate,
   mean_average_precision) = py_metrics_ops.motion_metrics(
       config=config_str, **variable_map)

  breakdown_names = config_util.get_breakdown_names_from_motion_config(config)
  metric_ops = {}
  for i, name in enumerate(breakdown_names):
    if i == 0:
      metric_ops['{}/minADE'.format(name)] = (min_ade[i], update_op)
    else:
      # Set update_op to be an no-op just in case if anyone runs update_ops in
      # multiple session.run()s.
      metric_ops['{}/minADE'.format(name)] = (min_ade[i], tf.constant([]))

    metric_ops['{}/minFDE'.format(name)] = (min_fde[i], tf.constant([]))
    metric_ops['{}/MissRate'.format(name)] = (miss_rate[i], tf.constant([]))
    metric_ops['{}/OverlapRate'.format(name)] = (overlap_rate[i],
                                                 tf.constant([]))
    metric_ops['{}/mAP'.format(name)] = (mean_average_precision[i],
                                         tf.constant([]))
  return metric_ops
