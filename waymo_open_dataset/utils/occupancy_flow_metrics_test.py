# Copyright 2022 The Waymo Open Dataset Authors. All Rights Reserved.
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
# =============================================================================
"""Tests for occupancy_flow_metrics."""

import numpy as np
import tensorflow as tf

from waymo_open_dataset.utils import occupancy_flow_grids
from waymo_open_dataset.utils import occupancy_flow_metrics
from waymo_open_dataset.utils import occupancy_flow_test_util


class OccupancyFlowMetricsTest(tf.test.TestCase):

  def test_compute_occupancy_flow_metrics(self):
    config = occupancy_flow_test_util.make_test_config()
    # Two boxes with 25% overlap.
    true_occupancy = np.zeros([4, 256, 256, 1], dtype=np.float32)
    true_occupancy[:, 100:120, 100:120, :] = 1.0
    true_occupancy = tf.convert_to_tensor(true_occupancy)
    pred_occupancy = np.zeros([4, 256, 256, 1], dtype=np.float32)
    pred_occupancy[:, 110:130, 110:130, :] = 1.0
    pred_occupancy = tf.convert_to_tensor(pred_occupancy)
    base_flow = tf.ones([4, 256, 256, 2])

    true_waypoints = occupancy_flow_grids.WaypointGrids()
    pred_waypoints = occupancy_flow_grids.WaypointGrids()
    true_waypoints.vehicles.observed_occupancy = [true_occupancy] * 8
    pred_waypoints.vehicles.observed_occupancy = [pred_occupancy] * 8
    true_waypoints.vehicles.occluded_occupancy = [true_occupancy * 0.2] * 8
    pred_waypoints.vehicles.occluded_occupancy = [pred_occupancy * 0.2] * 8
    true_waypoints.vehicles.flow_origin_occupancy = [true_occupancy] * 8
    true_waypoints.vehicles.flow = [base_flow * 0.5] * 8
    pred_waypoints.vehicles.flow = [base_flow * 3.5] * 8

    metrics = occupancy_flow_metrics.compute_occupancy_flow_metrics(
        config=config,
        true_waypoints=true_waypoints,
        pred_waypoints=pred_waypoints,
    )

    self.assertNear(metrics.vehicles_observed_auc, 0.07, err=0.01)
    self.assertNear(metrics.vehicles_occluded_auc, 0.07, err=0.01)
    self.assertNear(metrics.vehicles_observed_iou, 0.14, err=0.01)
    self.assertNear(metrics.vehicles_occluded_iou, 0.02, err=0.01)
    self.assertNear(metrics.vehicles_flow_epe, 4.24, err=0.01)
    self.assertNear(metrics.vehicles_flow_warped_occupancy_auc, 0.12, err=0.01)
    self.assertNear(metrics.vehicles_flow_warped_occupancy_iou, 0.10, err=0.01)

  def test_compute_occupancy_soft_iou(self):
    # Both zeros.
    true_occupancy = tf.zeros([2, 100, 100, 1])
    pred_occupancy = tf.zeros([2, 100, 100, 1])
    soft_iou = occupancy_flow_metrics._compute_occupancy_soft_iou(
        true_occupancy=true_occupancy,
        pred_occupancy=pred_occupancy,
    )
    self.assertNear(soft_iou, 0.0, err=1e-3)

    # Both ones.
    true_occupancy = tf.ones([2, 100, 100, 1])
    pred_occupancy = tf.ones([2, 100, 100, 1])
    soft_iou = occupancy_flow_metrics._compute_occupancy_soft_iou(
        true_occupancy=true_occupancy,
        pred_occupancy=pred_occupancy,
    )
    self.assertNear(soft_iou, 1.0, err=1e-3)

    # Ground-truth = 0, predictions = 1.
    true_occupancy = tf.zeros([2, 100, 100, 1])
    pred_occupancy = tf.ones([2, 100, 100, 1])
    soft_iou = occupancy_flow_metrics._compute_occupancy_soft_iou(
        true_occupancy=true_occupancy,
        pred_occupancy=pred_occupancy,
    )
    # Score is explicitly zero if ground-truth is all zeros.
    self.assertNear(soft_iou, 0.0, err=1e-3)

    # Ground-truth has a 10x10 patch of 1s, predictions = 0.
    true_occupancy = np.zeros([2, 100, 100, 1], dtype=np.float32)
    true_occupancy[:, 10:20, 10:20, :] = 1.0
    true_occupancy = tf.convert_to_tensor(true_occupancy)
    pred_occupancy = tf.zeros([2, 100, 100, 1])
    soft_iou = occupancy_flow_metrics._compute_occupancy_soft_iou(
        true_occupancy=true_occupancy,
        pred_occupancy=pred_occupancy,
    )
    self.assertNear(soft_iou, 0.0, err=1e-3)

    # Ground-truth and predictions have a 10x10 patch of 1s at different places.
    true_occupancy = np.zeros([2, 100, 100, 1], dtype=np.float32)
    true_occupancy[:, 10:20, 10:20, :] = 1.0
    true_occupancy = tf.convert_to_tensor(true_occupancy)
    pred_occupancy = np.zeros([2, 100, 100, 1], dtype=np.float32)
    pred_occupancy[:, 30:40, 30:40, :] = 1.0
    pred_occupancy = tf.convert_to_tensor(pred_occupancy)
    soft_iou = occupancy_flow_metrics._compute_occupancy_soft_iou(
        true_occupancy=true_occupancy,
        pred_occupancy=pred_occupancy,
    )
    self.assertNear(soft_iou, 0.0, err=1e-3)

    # Ground-truth and predictions have a 10x10 patch of 1s with 50% overlap.
    true_occupancy = np.zeros([2, 100, 100, 1], dtype=np.float32)
    true_occupancy[:, 10:20, 10:20, :] = 1.0
    pred_occupancy = np.zeros([2, 100, 100, 1], dtype=np.float32)
    pred_occupancy[:, 15:25, 10:20, :] = 1.0
    soft_iou = occupancy_flow_metrics._compute_occupancy_soft_iou(
        true_occupancy=true_occupancy,
        pred_occupancy=pred_occupancy,
    )
    self.assertNear(soft_iou, 0.333, err=1e-3)

    # Predictions are a linear transformation of ground-truth.
    true_occupancy = np.zeros([2, 100, 100, 1], dtype=np.float32)
    true_occupancy[:, 10:20, 10:20, :] = 1.0
    true_occupancy = tf.convert_to_tensor(true_occupancy)
    pred_occupancy = true_occupancy / 2 + 0.25
    soft_iou = occupancy_flow_metrics._compute_occupancy_soft_iou(
        true_occupancy=true_occupancy,
        pred_occupancy=pred_occupancy,
    )
    self.assertNear(soft_iou, 0.029, err=1e-3)

  def test_compute_occupancy_auc(self):
    # Both zeros.
    true_occupancy = tf.zeros([2, 100, 100, 1])
    pred_occupancy = tf.zeros([2, 100, 100, 1])
    auc = occupancy_flow_metrics._compute_occupancy_auc(
        true_occupancy=true_occupancy,
        pred_occupancy=pred_occupancy,
    )
    self.assertNear(auc, 0.0, err=1e-3)

    # Both ones.
    true_occupancy = tf.ones([2, 100, 100, 1])
    pred_occupancy = tf.ones([2, 100, 100, 1])
    auc = occupancy_flow_metrics._compute_occupancy_auc(
        true_occupancy=true_occupancy,
        pred_occupancy=pred_occupancy,
    )
    self.assertNear(auc, 1.0, err=1e-3)

    # Ground-truth = 0, predictions = 1.
    true_occupancy = tf.zeros([2, 100, 100, 1])
    pred_occupancy = tf.ones([2, 100, 100, 1])
    auc = occupancy_flow_metrics._compute_occupancy_auc(
        true_occupancy=true_occupancy,
        pred_occupancy=pred_occupancy,
    )
    self.assertNear(auc, 0.0, err=1e-3)

    # Ground-truth has a 10x10 patch of 1s, predictions = 0.
    true_occupancy = np.zeros([2, 100, 100, 1], dtype=np.float32)
    true_occupancy[:, 10:20, 10:20, :] = 1.0
    true_occupancy = tf.convert_to_tensor(true_occupancy)
    pred_occupancy = tf.zeros([2, 100, 100, 1])
    auc = occupancy_flow_metrics._compute_occupancy_auc(
        true_occupancy=true_occupancy,
        pred_occupancy=pred_occupancy,
    )
    self.assertNear(auc, 0.01, err=1e-3)

    # Ground-truth and predictions have a 10x10 patch of 1s at different places.
    true_occupancy = np.zeros([2, 100, 100, 1], dtype=np.float32)
    true_occupancy[:, 10:20, 10:20, :] = 1.0
    true_occupancy = tf.convert_to_tensor(true_occupancy)
    pred_occupancy = np.zeros([2, 100, 100, 1], dtype=np.float32)
    pred_occupancy[:, 30:40, 30:40, :] = 1.0
    pred_occupancy = tf.convert_to_tensor(pred_occupancy)
    auc = occupancy_flow_metrics._compute_occupancy_auc(
        true_occupancy=true_occupancy,
        pred_occupancy=pred_occupancy,
    )
    self.assertNear(auc, 0.009, err=1e-3)

    # Ground-truth and predictions have a 10x10 patch of 1s with 50% overlap.
    true_occupancy = np.zeros([2, 100, 100, 1], dtype=np.float32)
    true_occupancy[:, 10:20, 10:20, :] = 1.0
    pred_occupancy = np.zeros([2, 100, 100, 1], dtype=np.float32)
    pred_occupancy[:, 15:25, 10:20, :] = 1.0
    auc = occupancy_flow_metrics._compute_occupancy_auc(
        true_occupancy=true_occupancy,
        pred_occupancy=pred_occupancy,
    )
    self.assertNear(auc, 0.26404, err=1e-3)

    # Predictions are a linear transformation of ground-truth.
    true_occupancy = np.zeros([2, 100, 100, 1], dtype=np.float32)
    true_occupancy[:, 10:20, 10:20, :] = 1.0
    true_occupancy = tf.convert_to_tensor(true_occupancy)
    pred_occupancy = true_occupancy / 2 + 0.25
    auc = occupancy_flow_metrics._compute_occupancy_auc(
        true_occupancy=true_occupancy,
        pred_occupancy=pred_occupancy,
    )
    self.assertNear(auc, 1.0, err=1e-3)

  def test_compute_flow_epe(self):
    # Both zeros.
    true_flow = tf.zeros([4, 100, 100, 2])
    pred_flow = tf.zeros([4, 100, 100, 2])
    epe = occupancy_flow_metrics._compute_flow_epe(
        true_flow=true_flow,
        pred_flow=pred_flow,
    )
    self.assertNear(epe, 0.0, err=1e-3)

    # Both ones.
    true_flow = tf.ones([4, 100, 100, 2])
    pred_flow = tf.ones([4, 100, 100, 2])
    epe = occupancy_flow_metrics._compute_flow_epe(
        true_flow=true_flow,
        pred_flow=pred_flow,
    )
    self.assertNear(epe, 0.0, err=1e-3)

    # 3**2 + 4**2 = 5**2
    true_flow = tf.ones([4, 10, 10, 2]) * tf.constant([3.0, -4.0])
    pred_flow = tf.zeros([4, 10, 10, 2])
    epe = occupancy_flow_metrics._compute_flow_epe(
        true_flow=true_flow,
        pred_flow=pred_flow,
    )
    self.assertNear(epe, 5.0, err=1e-3)


if __name__ == '__main__':
  tf.test.main()
