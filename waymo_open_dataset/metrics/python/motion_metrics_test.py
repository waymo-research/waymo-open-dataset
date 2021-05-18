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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from google.protobuf import text_format
from waymo_open_dataset.metrics.python import motion_metrics
from waymo_open_dataset.protos import motion_metrics_pb2


class MotionMetricsEstimatorTest(tf.test.TestCase):
  """Unit tests for the motion metrics estimator."""

  def _BuildConfig(self, additional_config_str=''):
    """Builds a metrics config."""
    config = motion_metrics_pb2.MotionMetricsConfig()
    config_text = """
    track_steps_per_second: 10
    prediction_steps_per_second: 10
    track_history_samples: 0
    track_future_samples: 4
    step_configurations {
      measurement_step: 3
      lateral_miss_threshold: 1.0
      longitudinal_miss_threshold: 2.0
    }
    max_predictions: 6
    speed_scale_lower: 1.0
    speed_scale_upper: 1.0
    speed_lower_bound: 1.4
    speed_upper_bound: 11.0
    """ + additional_config_str
    text_format.Parse(config_text, config)
    return config

  def _CreateTestScenario(self, batch_size):
    gt_scenario_id = ['test']
    gt_object_id = [[1, 2]]
    gt_object_type = [[1, 1]]
    gt_is_valid = np.ones([1, 2, 5], dtype=np.bool)
    gt_trajectory = np.reshape([[[2, 2, 1, 1, 0.78539816, 20.0, 20.0],
                                 [4, 4, 1, 1, 0.78539816, 20.0, 20.0],
                                 [6, 6, 1, 1, 0.78539816, 20.0, 20.0],
                                 [8, 8, 1, 1, 0.78539816, 20.0, 20.0],
                                 [10, 10, 1, 1, 0.78539816, 20.0, 20.0]],
                                [[-1, 0, 1, 1, 3.14159, -10.0, 0.0],
                                 [-2, 0, 1, 1, 3.14159, -10.0, 0.0],
                                 [-3, 0, 1, 1, 3.14159, -10.0, 0.0],
                                 [-4, 0, 1, 1, 3.14159, -10.0, 0.0],
                                 [-5, 0, 1, 1, 3.14159, -10.0, 0.0]]],
                               [1, 2, 5, 7])

    pred_gt_indices = np.reshape([0, 1], (1, 1, 2))
    pred_gt_indices_mask = np.ones((1, 1, 2)) > 0.0

    return {
        'scenario_id':
            [[gt_scenario_id[0] + '%s' % i] for i in range(batch_size)],
        'object_id': [gt_object_id for _ in range(batch_size)],
        'object_type': [gt_object_type for _ in range(batch_size)],
        'gt_is_valid': [gt_is_valid for _ in range(batch_size)],
        'gt_trajectory': [gt_trajectory for _ in range(batch_size)],
        'pred_gt_indices': [pred_gt_indices for _ in range(batch_size)],
        'pred_gt_indices_mask': [
            pred_gt_indices_mask for _ in range(batch_size)
        ],
    }

  def setUp(self):
    super(MotionMetricsEstimatorTest, self).setUp()
    self._config = self._BuildConfig()

  def _BuildGraph(self,
                  graph,
                  config,
                  num_agent_groups=1,
                  num_agents=2,
                  top_k=2,
                  num_pred_steps=4,
                  num_gt_steps=5,
                  set_scenario_and_object_id=True):
    with graph.as_default():
      self._gt_trajectory = tf.compat.v1.placeholder(
          dtype=tf.float32, shape=(None, num_agents, num_gt_steps, 7))
      self._gt_is_valid = tf.compat.v1.placeholder(
          dtype=tf.bool, shape=[None, num_agents, num_gt_steps])
      self._object_type = tf.compat.v1.placeholder(
          dtype=tf.int64, shape=[None, num_agents])
      self._pd_trajectory = tf.compat.v1.placeholder(
          dtype=tf.float32,
          shape=(None, num_agent_groups, top_k, num_agents, num_pred_steps, 2))
      self._pd_score = tf.compat.v1.placeholder(
          dtype=tf.float32, shape=[None, num_agent_groups, top_k])
      self._pred_gt_indices = tf.compat.v1.placeholder(
          dtype=tf.int64, shape=(None, num_agent_groups, num_agents))
      self._pred_gt_indices_mask = tf.compat.v1.placeholder(
          dtype=tf.bool, shape=(None, num_agent_groups, num_agents))
      self._object_id = tf.compat.v1.placeholder(
          dtype=tf.int64, shape=[None, num_agents])
      self._scenario_id = tf.compat.v1.placeholder(
          dtype=tf.string, shape=[None])

      if set_scenario_and_object_id:
        scenario_id = self._scenario_id
        object_id = self._object_id
      else:
        scenario_id = None
        object_id = None

      metrics = motion_metrics.get_motion_metric_ops(
          config=config,
          prediction_trajectory=self._pd_trajectory,
          prediction_score=self._pd_score,
          ground_truth_trajectory=self._gt_trajectory,
          ground_truth_is_valid=self._gt_is_valid,
          prediction_ground_truth_indices=self._pred_gt_indices,
          prediction_ground_truth_indices_mask=self._pred_gt_indices_mask,
          object_type=self._object_type,
          object_id=object_id,
          scenario_id=scenario_id,
      )
      return metrics

  def _EvalValueOps(self, sess, graph, metrics):
    return sess.run({item[0]: item[1][0] for item in metrics.items()})

  def _RunEval(self,
               pred_score,
               pred_trajectory,
               gt,
               config=None,
               set_scenario_and_object_id=True):
    if not config:
      config = self._config

    num_updates = len(pred_score)
    assert num_updates == len(pred_trajectory)
    assert num_updates == len(gt['scenario_id'])
    assert num_updates == len(gt['object_id'])
    assert num_updates == len(gt['object_type'])
    assert num_updates == len(gt['gt_is_valid'])
    assert num_updates == len(gt['gt_trajectory'])
    assert num_updates == len(gt['pred_gt_indices'])
    assert num_updates == len(gt['pred_gt_indices_mask'])

    graph = tf.Graph()
    metrics = self._BuildGraph(
        graph, config, set_scenario_and_object_id=set_scenario_and_object_id)
    with self.test_session(graph=graph) as sess:
      sess.run(tf.compat.v1.initializers.local_variables())

      for i in range(num_updates):
        sess.run(
            [tf.group([value[1] for value in metrics.values()])],
            feed_dict={
                self._scenario_id: gt['scenario_id'][i],
                self._object_id: gt['object_id'][i],
                self._object_type: gt['object_type'][i],
                self._gt_is_valid: gt['gt_is_valid'][i],
                self._gt_trajectory: gt['gt_trajectory'][i],
                self._pred_gt_indices: gt['pred_gt_indices'][i],
                self._pred_gt_indices_mask: gt['pred_gt_indices_mask'][i],
                self._pd_score: pred_score[i],
                self._pd_trajectory: pred_trajectory[i],
            })
      metric_dict = self._EvalValueOps(sess, graph, metrics)
      return metric_dict

  def testComputeMinADE(self):
    pred_score = np.reshape([0.5, 0.5], (1, 1, 2))
    pred_trajectory = np.reshape(
        [[[[4, 0], [6, 0], [8, 0], [10, 0]], [[0, 2], [0, 3], [0, 4], [0, 5]]],
         [[[14, 0], [16, 0], [18, 0], [20, 0]],
          [[0, 22], [0, 23], [0, 24], [0, 25]]]], (1, 1, 2, 2, 4, 2))
    gt = self._CreateTestScenario(1)
    metric_dict = self._RunEval([pred_score], [pred_trajectory], gt)
    # ADE of Vehicle.
    self.assertAlmostEqual(
        metric_dict['TYPE_VEHICLE_3/minADE'], 5.97487, delta=1e-4)
    # FDE of Vehicle.
    self.assertAlmostEqual(
        metric_dict['TYPE_VEHICLE_3/minFDE'], 8.53553, delta=1e-4)

  def testComputeMinADEBatch2(self):
    pred_score = np.reshape([0.5, 0.5], (1, 1, 2))
    pred_trajectory = np.reshape(
        [[[[4, 0], [6, 0], [8, 0], [10, 0]], [[0, 2], [0, 3], [0, 4], [0, 5]]],
         [[[14, 0], [16, 0], [18, 0], [20, 0]],
          [[0, 22], [0, 23], [0, 24], [0, 25]]]], (1, 1, 2, 2, 4, 2))
    pred_trajectory_2 = np.reshape(
        [[[[4, 0], [6, 0], [8, 0], [10, 0]], [[0, 2], [0, 3], [0, 5], [0, 5]]],
         [[[14, 0], [16, 0], [18, 0], [20, 0]],
          [[0, 22], [0, 23], [0, 24], [0, 25]]]], (1, 1, 2, 2, 4, 2))
    gt = self._CreateTestScenario(2)
    metric_dict = self._RunEval([pred_score, pred_score],
                                [pred_trajectory, pred_trajectory_2], gt)
    # ADE of Vehicle.
    self.assertAlmostEqual(
        metric_dict['TYPE_VEHICLE_3/minADE'], 6.021516, delta=1e-4)
    # FDE of Vehicle.
    self.assertAlmostEqual(
        metric_dict['TYPE_VEHICLE_3/minFDE'], 8.53553, delta=1e-4)

  def testComputeMinADEDefaultIds(self):
    pred_score = np.reshape([0.5, 0.5], (1, 1, 2))
    pred_trajectory = np.reshape(
        [[[[4, 0], [6, 0], [8, 0], [10, 0]], [[0, 2], [0, 3], [0, 4], [0, 5]]],
         [[[14, 0], [16, 0], [18, 0], [20, 0]],
          [[0, 22], [0, 23], [0, 24], [0, 25]]]], (1, 1, 2, 2, 4, 2))
    gt = self._CreateTestScenario(1)
    metric_dict = self._RunEval([pred_score], [pred_trajectory],
                                gt,
                                set_scenario_and_object_id=False)
    # ADE of Vehicle.
    self.assertAlmostEqual(
        metric_dict['TYPE_VEHICLE_3/minADE'], 5.97487, delta=1e-4)
    # FDE of Vehicle.
    self.assertAlmostEqual(
        metric_dict['TYPE_VEHICLE_3/minFDE'], 8.53553, delta=1e-4)


if __name__ == '__main__':
  tf.compat.v1.disable_eager_execution()
  tf.test.main()
