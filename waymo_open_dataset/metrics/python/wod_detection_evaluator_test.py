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
"""Tests for waymo_open_dataset.metrics.python.wod_detection_evaluator."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from google.protobuf import text_format
from waymo_open_dataset.metrics.python import wod_detection_evaluator
from waymo_open_dataset.protos import metrics_pb2


class WODDetectionEvaluatorTest(tf.test.TestCase):

  def _GenerateRandomBBoxes(self, num_frames, num_bboxes):
    # TODO(chsin): Should WODDetectionEvaluator do the casting? If so, remove
    # the astype calls here.
    center_xyz = np.random.uniform(
        low=-1.0, high=1.0, size=(num_bboxes, 3)).astype(np.float32)
    dimension = np.random.uniform(
        low=0.1, high=1.0, size=(num_bboxes, 3)).astype(np.float32)
    rotation = np.random.uniform(
        low=-np.pi, high=np.pi, size=(num_bboxes, 1)).astype(np.float32)
    bboxes = np.concatenate([center_xyz, dimension, rotation], axis=-1)
    types = np.random.randint(1, 5, size=[num_bboxes]).astype(np.uint8)
    frame_ids = np.random.randint(0, num_frames, size=[num_bboxes])
    scores = np.random.uniform(size=[num_bboxes]).astype(np.float32)
    return bboxes, types, frame_ids, scores

  def _BuildConfig(self):
    config = metrics_pb2.Config()
    config_text = """
    num_desired_score_cutoffs: 11
    breakdown_generator_ids: OBJECT_TYPE
    difficulties {
    }
    matcher_type: TYPE_HUNGARIAN
    iou_thresholds: 0.5
    iou_thresholds: 0.5
    iou_thresholds: 0.5
    iou_thresholds: 0.5
    iou_thresholds: 0.5
    box_type: TYPE_3D
    """
    text_format.Merge(config_text, config)
    return config

  def testBasic(self):
    num_frames, gt_num_bboxes, pd_num_bboxes = 10, 10, 2000
    pd_bbox, pd_type, pd_frameid, pd_score = self._GenerateRandomBBoxes(
        num_frames, pd_num_bboxes)
    pd_overlap_nlz = np.zeros_like(pd_frameid, dtype=np.bool)
    gt_bbox, gt_type, gt_frameid, _ = self._GenerateRandomBBoxes(
        num_frames, gt_num_bboxes)
    config = self._BuildConfig()

    evaluator = wod_detection_evaluator.WODDetectionEvaluator(config)
    num_breakdowns = len(evaluator._breakdown_names)
    self.assertEqual(num_breakdowns, 4)

    predictions = {
        'prediction_frame_id': pd_frameid,
        'prediction_bbox': pd_bbox,
        'prediction_type': pd_type,
        'prediction_score': pd_score,
        'prediction_overlap_nlz': pd_overlap_nlz,
    }

    # Expect ap and aph metrics is all zeros.
    groundtruths = {
        'ground_truth_frame_id': gt_frameid,
        'ground_truth_bbox': gt_bbox,
        'ground_truth_type': gt_type,
        'ground_truth_difficulty': np.ones_like(gt_frameid, dtype=np.uint8),
        'ground_truth_speed': np.zeros((gt_num_bboxes, 2), dtype=np.float32),
    }

    # Expect ap and aph metrics is all ones.
    groundtruths_as_predictions = {
        'ground_truth_frame_id': pd_frameid,
        'ground_truth_bbox': pd_bbox,
        'ground_truth_type': pd_type,
        'ground_truth_difficulty': np.ones_like(pd_frameid, dtype=np.uint8),
        'ground_truth_speed': np.zeros((pd_num_bboxes, 2), dtype=np.float32),
    }

    evaluator.update_state(groundtruths, predictions)
    metric_dict = evaluator.result()
    self.assertEqual(len(metric_dict.average_precision), num_breakdowns)
    self.assertAllClose(metric_dict.average_precision, [0, 0, 0, 0])
    self.assertAllClose(metric_dict.average_precision_ha_weighted, [0, 0, 0, 0])

    evaluator.update_state(groundtruths_as_predictions, predictions)
    metric_dict = evaluator.result()
    self.assertAllClose(metric_dict.average_precision, [1, 1, 1, 1])
    self.assertAllClose(metric_dict.average_precision_ha_weighted, [1, 1, 1, 1])

    evaluator.update_state(groundtruths, predictions)
    evaluator.update_state(groundtruths_as_predictions, predictions)
    metric_dict = evaluator.result()
    self.assertAllClose(
        metric_dict.average_precision, [0.5, 0.5, 0.5, 0.5], atol=0.01)
    self.assertAllClose(
        metric_dict.average_precision_ha_weighted, [0.5, 0.5, 0.5, 0.5],
        atol=0.01)

  def testDefaultConfig(self):
    num_frames, gt_num_bboxes, pd_num_bboxes = 10, 10, 2000
    pd_bbox, pd_type, pd_frameid, pd_score = self._GenerateRandomBBoxes(
        num_frames, pd_num_bboxes)
    pd_overlap_nlz = np.zeros_like(pd_frameid, dtype=np.bool)
    gt_bbox, gt_type, gt_frameid, _ = self._GenerateRandomBBoxes(
        num_frames, gt_num_bboxes)

    evaluator = wod_detection_evaluator.WODDetectionEvaluator()
    num_breakdowns = len(evaluator._breakdown_names)
    self.assertEqual(num_breakdowns, 32)

    predictions = {
        'prediction_frame_id': pd_frameid,
        'prediction_bbox': pd_bbox,
        'prediction_type': pd_type,
        'prediction_score': pd_score,
        'prediction_overlap_nlz': pd_overlap_nlz,
    }

    # Expect ap and aph metrics is all zeros.
    groundtruths = {
        'ground_truth_frame_id': gt_frameid,
        'ground_truth_bbox': gt_bbox,
        'ground_truth_type': gt_type,
        'ground_truth_difficulty': np.ones_like(gt_frameid, dtype=np.uint8),
        'ground_truth_speed': np.zeros((gt_num_bboxes, 2), dtype=np.float32),
    }

    # Expect ap and aph metrics is all ones.
    groundtruths_as_predictions = {
        'ground_truth_frame_id': pd_frameid,
        'ground_truth_bbox': pd_bbox,
        'ground_truth_type': pd_type,
        'ground_truth_difficulty': np.ones_like(pd_frameid, dtype=np.uint8),
        'ground_truth_speed': np.zeros((pd_num_bboxes, 2), dtype=np.float32),
    }

    # Only checking the first index, which is OBJECT_TYPE_TYPE_VEHICLE_LEVEL_1,
    # because the others are more complicated.
    evaluator.update_state(groundtruths, predictions)
    metric_dict = evaluator.result()
    self.assertEqual(len(metric_dict.average_precision), num_breakdowns)
    self.assertAllClose(metric_dict.average_precision[0], 0)
    self.assertAllClose(metric_dict.average_precision_ha_weighted[0], 0)

    evaluator.update_state(groundtruths_as_predictions, predictions)
    metric_dict = evaluator.result()
    self.assertAllClose(metric_dict.average_precision[0], 1)
    self.assertAllClose(metric_dict.average_precision_ha_weighted[0], 1)

    evaluator.update_state(groundtruths, predictions)
    evaluator.update_state(groundtruths_as_predictions, predictions)
    metric_dict = evaluator.result()
    self.assertAllClose(metric_dict.average_precision[0], 0.5, atol=0.01)
    self.assertAllClose(
        metric_dict.average_precision_ha_weighted[0], 0.5, atol=0.01)


if __name__ == '__main__':
  tf.test.main()
