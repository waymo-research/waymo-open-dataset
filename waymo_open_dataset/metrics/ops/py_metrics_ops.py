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
"""Waymo Open Dataset tensorflow ops python interface."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

detection_metrics_module = tf.load_op_library(
    tf.compat.v1.resource_loader.get_path_to_datafile(
        'detection_metrics_ops.so'))


def detection_metrics(prediction_bbox, prediction_type, prediction_score,
                      prediction_frame_id, prediction_overlap_nlz,
                      ground_truth_bbox, ground_truth_type,
                      ground_truth_frame_id, ground_truth_difficulty,
                      config, ground_truth_speed=None):
  """Wraps detection_metrics. See metrics_ops.cc for full documentation."""
  if ground_truth_speed is None:
    num_gt_boxes = tf.shape(ground_truth_bbox)[0]
    ground_truth_speed = tf.zeros((num_gt_boxes, 2), dtype=tf.float32)

  return detection_metrics_module.detection_metrics(
      prediction_bbox=prediction_bbox,
      prediction_type=prediction_type,
      prediction_score=prediction_score,
      prediction_frame_id=prediction_frame_id,
      prediction_overlap_nlz=prediction_overlap_nlz,
      ground_truth_bbox=ground_truth_bbox,
      ground_truth_type=ground_truth_type,
      ground_truth_frame_id=ground_truth_frame_id,
      ground_truth_difficulty=ground_truth_difficulty,
      ground_truth_speed=ground_truth_speed,
      config=config)
