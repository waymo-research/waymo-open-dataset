# Copyright (c) 2024 Waymo LLC. All rights reserved.

# This is licensed under a BSD+Patent license.
# Please see LICENSE and PATENTS text files.
# ==============================================================================
"""Tests for camera_segmentation_metrics."""

import numpy as np
import tensorflow as tf

from waymo_open_dataset.wdl_limited.camera_segmentation import camera_segmentation_metrics


class CameraSegmentationMetricsTest(tf.test.TestCase):

  def setUp(self):
    super().setUp()
    self._panoptic_label_divisor = 100000
    self._true_panoptic_labels = [
        np.array([[1, 2, 3, 4]], dtype=np.int32) * self._panoptic_label_divisor,
        np.array([[2, 3, 4, 5]], dtype=np.int32) * self._panoptic_label_divisor]
    self._pred_panoptic_labels = [
        np.array([[4, 2, 3, 1]], dtype=np.int32) * self._panoptic_label_divisor,
        np.array([[0, 0, 0, 0]], dtype=np.int32) * self._panoptic_label_divisor]
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
    self.assertEqual(eval_config.panoptic_label_divisor,
                     self._panoptic_label_divisor)

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
