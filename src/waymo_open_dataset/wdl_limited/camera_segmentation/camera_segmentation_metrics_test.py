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
"""Tests for camera_segmentation_metrics."""

import numpy as np
import tensorflow as tf

from waymo_open_dataset.wdl_limited.camera_segmentation import camera_segmentation_metrics


class CameraSegmentationMetricsTest(tf.test.TestCase):

  def setUp(self):
    super().setUp()
    self._true_panoptic_labels = [
        np.array([[1, 2, 3, 4]], dtype=np.int32),
        np.array([[2, 3, 4, 5]], dtype=np.int32)]
    self._pred_panoptic_labels = [
        np.array([[4, 2, 3, 1]], dtype=np.int32),
        np.array([[0, 0, 0, 0]], dtype=np.int32)]
    self._num_cameras_covered = [
        np.array([[1, 1, 1, 1]], dtype=np.int32),
        np.array([[2, 2, 2, 2]], dtype=np.int32),
    ]
    self._is_tracked_masks = [
        np.array([[1, 1, 1, 1]], dtype=bool),
        np.array([[0, 0, 0, 0]], dtype=bool),
    ]

  def test_get_eval_config(self):
    eval_config = camera_segmentation_metrics.get_eval_config()
    self.assertIsNotNone(eval_config)
    self.assertEqual(eval_config.panoptic_label_divisor, 100000)

  def test_get_metric_object_by_sequence_basic(self):
    metric_obj = camera_segmentation_metrics.get_metric_object_by_sequence(
        self._true_panoptic_labels,
        self._pred_panoptic_labels,
        self._num_cameras_covered,
        self._is_tracked_masks,
    )
    self.assertIsNotNone(metric_obj)

  def test_compute_metrics_basic(self):
    metric_obj = camera_segmentation_metrics.get_metric_object_by_sequence(
        self._true_panoptic_labels,
        self._pred_panoptic_labels,
        self._num_cameras_covered,
        self._is_tracked_masks,
    )
    metrics = camera_segmentation_metrics.aggregate_metrics([metric_obj])
    self.assertGreaterEqual(metrics.wstq, 0.0)
    self.assertLessEqual(metrics.wstq, 1.0)
    self.assertGreaterEqual(metrics.waq, 0.0)
    self.assertLessEqual(metrics.waq, 1.0)
    self.assertGreaterEqual(metrics.miou, 0.0)
    self.assertLessEqual(metrics.miou, 1.0)

  def test_compute_metrics_with_zero_tracked_masks(self):
    metric_obj = camera_segmentation_metrics.get_metric_object_by_sequence(
        self._true_panoptic_labels,
        self._pred_panoptic_labels,
        self._num_cameras_covered,
        [is_tracked * 0 for is_tracked in self._is_tracked_masks],
    )
    metrics = camera_segmentation_metrics.aggregate_metrics([metric_obj])
    self.assertGreaterEqual(metrics.wstq, 0.0)
    self.assertLessEqual(metrics.wstq, 1.0)
    self.assertGreaterEqual(metrics.waq, 0.0)
    self.assertLessEqual(metrics.waq, 1.0)
    self.assertGreaterEqual(metrics.miou, 0.0)
    self.assertLessEqual(metrics.miou, 1.0)


if __name__ == '__main__':
  tf.test.main()
