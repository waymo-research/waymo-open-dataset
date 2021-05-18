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

import copy
import numpy as np
import tensorflow as tf

from google.protobuf import text_format
from waymo_open_dataset.metrics.ops import py_metrics_ops
from waymo_open_dataset.protos import motion_metrics_pb2


class MotionMetricsOpsTest(tf.test.TestCase):
  """Unit tests for motion metrics."""

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

  def _CreateTestScenario(self):
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
        'scenario_id': gt_scenario_id,
        'object_id': gt_object_id,
        'object_type': gt_object_type,
        'gt_is_valid': gt_is_valid,
        'gt_trajectory': gt_trajectory,
        'pred_gt_indices': pred_gt_indices,
        'pred_gt_indices_mask': pred_gt_indices_mask,
    }

  def setUp(self):
    super(MotionMetricsOpsTest, self).setUp()
    self._config = self._BuildConfig()
    self._gt = self._CreateTestScenario()

  def _RunEval(self, pred_score, pred_trajectory, gt=None, config=None):
    if not gt:
      gt = self._gt
    if not config:
      config = self._config

    g = tf.Graph()
    with g.as_default():
      (min_ade, min_fde, miss_rate, overlap_rate,
       mean_ap) = py_metrics_ops.motion_metrics(
           config=config.SerializeToString(),
           prediction_trajectory=pred_trajectory,
           prediction_score=pred_score,
           ground_truth_trajectory=gt['gt_trajectory'],
           ground_truth_is_valid=gt['gt_is_valid'],
           prediction_ground_truth_indices=gt['pred_gt_indices'],
           prediction_ground_truth_indices_mask=gt['pred_gt_indices_mask'],
           object_type=gt['object_type'],
           object_id=gt['object_id'],
           scenario_id=gt['scenario_id'])

    with self.test_session(graph=g) as sess:
      return sess.run([min_ade, min_fde, miss_rate, overlap_rate, mean_ap])

  def testComputeMissRateNoMisses(self):
    pred_score = np.reshape([0.5], (1, 1, 1))
    pred_trajectory = np.reshape([[[4, 4], [6, 6], [8, 8], [10, 10]],
                                  [[-2, 0], [-3, 0], [-4, 0], [-5, 0]]],
                                 (1, 1, 1, 2, 4, 2))

    val = self._RunEval(pred_score, pred_trajectory)
    # miss_rate of Vehicle.
    self.assertEqual(val[2][0], 0.0)
    # mean_ap of Vehicle.
    self.assertEqual(val[4][0], 1.0)

  def testComputeMissRateNoMisses2(self):
    pred_score = np.reshape([0.5], (1, 1, 1))
    pred_trajectory = np.reshape([[[-2, 0], [-3, 0], [-4, 0], [-5, 0]],
                                  [[4, 4], [6, 6], [8, 8], [10, 10]]],
                                 (1, 1, 1, 2, 4, 2))

    gt = copy.deepcopy(self._gt)
    gt['pred_gt_indices'] = np.reshape([1, 0], (1, 1, 2))

    val = self._RunEval(pred_score, pred_trajectory, gt=gt)
    # miss_rate of Vehicle.
    self.assertEqual(val[2][0], 0.0)
    # mean_ap of Vehicle.
    self.assertEqual(val[4][0], 1.0)

  def testComputeMissRateLateral_2(self):
    pred_score = np.reshape([0.5], (1, 1, 1))
    pred_trajectory = np.reshape(
        [[[4, 4], [6, 6], [8, 8], [10, 10]],
         [[-2, 1.01], [-3, 1.01], [-4, 1.01], [-5, 1.01]]], (1, 1, 1, 2, 4, 2))
    val = self._RunEval(pred_score, pred_trajectory)
    # miss_rate of Vehicle.
    self.assertEqual(val[2][0], 1.0)
    # mean_ap of Vehicle.
    self.assertEqual(val[4][0], 0.0)

  def testComputeMissRateLateral_1(self):
    pred_score = np.reshape([0.5], (1, 1, 1))
    pred_trajectory = np.reshape([[[4, 4], [6, 6], [8, 8], [9.292, 10.708]],
                                  [[-2, 0], [-3, 0], [-4, 0], [-5, 0]]],
                                 (1, 1, 1, 2, 4, 2))
    val = self._RunEval(pred_score, pred_trajectory)
    # miss_rate of Vehicle.
    self.assertEqual(val[2][0], 1.0)
    # mean_ap of Vehicle.
    self.assertEqual(val[4][0], 0.0)

  def testComputeMissRateLongitudinal_2(self):
    pred_score = np.reshape([0.5], (1, 1, 1))
    pred_trajectory = np.reshape([[[4, 4], [6, 6], [8, 8], [10, 10]],
                                  [[-2, 0], [-3, 0], [-4, 0], [-7.01, 0]]],
                                 (1, 1, 1, 2, 4, 2))
    val = self._RunEval(pred_score, pred_trajectory)
    # miss_rate of Vehicle.
    self.assertEqual(val[2][0], 1.0)
    # mean_ap of Vehicle.
    self.assertEqual(val[4][0], 0.0)

  def testComputeMissRateLongitudinal_1(self):
    pred_score = np.reshape([0.5], (1, 1, 1))
    pred_trajectory = np.reshape([[[4, 4], [6, 6], [8, 8], [11.415, 11.415]],
                                  [[-2, 0], [-3, 0], [-4, 0], [-5, 0]]],
                                 (1, 1, 1, 2, 4, 2))
    val = self._RunEval(pred_score, pred_trajectory)
    # miss_rate of Vehicle.
    self.assertEqual(val[2][0], 1.0)
    # mean_ap of Vehicle.
    self.assertEqual(val[4][0], 0.0)

  def testComputeNoMissLongitudinal_1(self):
    pred_score = np.reshape([0.5], (1, 1, 1))
    pred_trajectory = np.reshape([[[4, 4], [6, 6], [8, 8], [11.414, 11.414]],
                                  [[-2, 0], [-3, 0], [-4, 0], [-5, 0]]],
                                 (1, 1, 1, 2, 4, 2))
    val = self._RunEval(pred_score, pred_trajectory)
    # miss_rate of Vehicle.
    self.assertEqual(val[2][0], 0.0)
    # mean_ap of Vehicle.
    self.assertEqual(val[4][0], 1.0)

  def testComputeVelocityScalingLatitudinal(self):
    pred_score = np.reshape([0.5], (1, 1, 1))
    pred_trajectory = np.reshape([[[4, 4], [6, 6], [8, 8], [10, 10]],
                                  [[-2, 0], [-3, 0], [-4, 0], [-5, 0.75]]],
                                 (1, 1, 1, 2, 4, 2))

    config = motion_metrics_pb2.MotionMetricsConfig()
    config.CopyFrom(self._config)
    config.speed_scale_lower = 0.5
    config.speed_scale_upper = 1.0
    config.speed_lower_bound = 1.0
    config.speed_upper_bound = 3.0

    val = self._RunEval(pred_score, pred_trajectory, config=config)
    # miss_rate of Vehicle.
    self.assertEqual(val[2][0], 0.0)
    # mean_ap of Vehicle.
    self.assertEqual(val[4][0], 1.0)

    # Decrease the velocity below the speed lower bound.
    gt = copy.deepcopy(self._gt)
    gt['gt_trajectory'][0, 1, :, 5:7] = 0.0
    val = self._RunEval(pred_score, pred_trajectory, config=config, gt=gt)
    # miss_rate of Vehicle.
    self.assertEqual(val[2][0], 1.0)

    # Set the velocity to just below the speed required for object2 to fit.
    gt = copy.deepcopy(self._gt)
    gt['gt_trajectory'][0, 1, :, 5] = 1.999
    val = self._RunEval(pred_score, pred_trajectory, config=config, gt=gt)
    # miss_rate of Vehicle.
    self.assertEqual(val[2][0], 1.0)

    # Set the velocity to just above the speed required for object2 to fit.
    gt = copy.deepcopy(self._gt)
    gt['gt_trajectory'][0, 1, :, 5] = 2.001
    val = self._RunEval(pred_score, pred_trajectory, config=config, gt=gt)
    # miss_rate of Vehicle.
    self.assertEqual(val[2][0], 0.0)

  def testComputeVelocityScalingLongitudinal(self):
    pred_score = np.reshape([0.5], (1, 1, 1))
    pred_trajectory = np.reshape([[[4, 4], [6, 6], [8, 8], [10, 10]],
                                  [[-2, 0], [-3, 0], [-4, 0], [-6.5, 0]]],
                                 (1, 1, 1, 2, 4, 2))

    config = motion_metrics_pb2.MotionMetricsConfig()
    config.CopyFrom(self._config)
    config.speed_scale_lower = 0.5
    config.speed_scale_upper = 1.0
    config.speed_lower_bound = 1.0
    config.speed_upper_bound = 3.0

    val = self._RunEval(pred_score, pred_trajectory, config=config)
    # miss_rate of Vehicle.
    self.assertEqual(val[2][0], 0.0)
    # mean_ap of Vehicle.
    self.assertEqual(val[4][0], 1.0)

    # Decrease the velocity below the speed lower bound.
    gt = copy.deepcopy(self._gt)
    gt['gt_trajectory'][0, 1, :, 5:7] = 0.0
    val = self._RunEval(pred_score, pred_trajectory, config=config, gt=gt)
    # miss_rate of Vehicle.
    self.assertEqual(val[2][0], 1.0)

    # Set the velocity to just below the speed required for object2 to fit.
    gt = copy.deepcopy(self._gt)
    gt['gt_trajectory'][0, 1, :, 5] = 1.999
    val = self._RunEval(pred_score, pred_trajectory, config=config, gt=gt)
    # miss_rate of Vehicle.
    self.assertEqual(val[2][0], 1.0)

    # Set the velocity to just above the speed required for object2 to fit.
    gt = copy.deepcopy(self._gt)
    gt['gt_trajectory'][0, 1, :, 5] = 2.001
    val = self._RunEval(pred_score, pred_trajectory, config=config, gt=gt)
    # miss_rate of Vehicle.
    self.assertEqual(val[2][0], 0.0)

  def testComputeNoMissLateral_2(self):
    pred_score = np.reshape([0.8, 0.5], (1, 1, 2))
    pred_trajectory = np.reshape([[[[4, 4], [6, 6], [8, 8], [9.294, 10.706]],
                                   [[-2, 0], [-3, 0], [-4, 0], [-5, 0]]],
                                  [[[4, 4], [6, 6], [8, 8], [10, 10]],
                                   [[-2, 0], [-3, 0], [-4, 0], [-5, 0]]]],
                                 (1, 1, 2, 2, 4, 2))
    val = self._RunEval(pred_score, pred_trajectory)
    # miss_rate of Vehicle.
    self.assertEqual(val[2][0], 0.0)
    # mean_ap of Vehicle.
    self.assertEqual(val[4][0], 1.0)

  def testTwoJointPredictionsNoMiss(self):
    pred_score = np.reshape([0.8, 0.5], (1, 1, 2))
    pred_trajectory = np.reshape([[[[4, 4], [6, 6], [8, 8], [10, 10]],
                                   [[-2, 0], [-3, 0], [-4, 0], [-7.01, 0]]],
                                  [[[4, 4], [6, 6], [8, 8], [10, 10]],
                                   [[-2, 0], [-3, 0], [-4, 0], [-5, 0]]]],
                                 (1, 1, 2, 2, 4, 2))
    val = self._RunEval(pred_score, pred_trajectory)
    # miss_rate of Vehicle.
    self.assertEqual(val[2][0], 0.0)
    # mean_ap of Vehicle.
    self.assertEqual(val[4][0], 0.5)

  def testTwoJointPredictionsMiss(self):
    pred_score = np.reshape([0.8, 0.5], (1, 1, 2))
    pred_trajectory = np.reshape([[[[4, 4], [6, 6], [8, 8], [10, 10]],
                                   [[-2, 0], [-3, 0], [-4, 0], [-7.01, 0]]],
                                  [[[4, 4], [6, 6], [8, 8], [14, 14]],
                                   [[-2, 0], [-3, 0], [-4, 0], [-5, 0]]]],
                                 (1, 1, 2, 2, 4, 2))
    val = self._RunEval(pred_score, pred_trajectory)
    # miss_rate of Vehicle.
    self.assertEqual(val[2][0], 1.0)
    # mean_ap of Vehicle.
    self.assertEqual(val[4][0], 0.0)

  def testComputeMinADE(self):
    pred_score = np.reshape([0.5, 0.5], (1, 1, 2))
    pred_trajectory = np.reshape(
        [[[[4, 0], [6, 0], [8, 0], [10, 0]], [[0, 2], [0, 3], [0, 4], [0, 5]]],
         [[[14, 0], [16, 0], [18, 0], [20, 0]],
          [[0, 22], [0, 23], [0, 24], [0, 25]]]], (1, 1, 2, 2, 4, 2))
    val = self._RunEval(pred_score, pred_trajectory)
    # 5 metrics.
    self.assertEqual(len(val), 5)
    # 3 steps.
    self.assertEqual(len(val[0]), 3)
    # ADE of Vehicle.
    self.assertAlmostEqual(val[0][0], 5.97487, delta=1e-4)
    # FDE of Vehicle.
    self.assertAlmostEqual(val[1][0], 8.53553, delta=1e-4)


if __name__ == '__main__':
  tf.compat.v1.disable_eager_execution()
  tf.test.main()
