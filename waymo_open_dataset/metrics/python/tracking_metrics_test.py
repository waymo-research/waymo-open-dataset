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
"""Tests for waymo_open_dataset.metrics.python.tracking_metrics."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from google.protobuf import text_format
from waymo_open_dataset.metrics.python import tracking_metrics
from waymo_open_dataset.protos import metrics_pb2

ERROR = 1e-6


class TrackinMetricsEstimatorTest(tf.test.TestCase):

  def _GenerateRandomBBoxes(self, num_sequences, num_frames, num_bboxes):
    center_xyz = np.random.uniform(low=-1.0, high=1.0, size=(num_bboxes, 3))
    dimension = np.random.uniform(low=0.1, high=1.0, size=(num_bboxes, 3))
    rotation = np.random.uniform(low=-np.pi, high=np.pi, size=(num_bboxes, 1))
    bboxes = np.concatenate([center_xyz, dimension, rotation], axis=-1)
    # Make sure all types are used.
    self.assertGreaterEqual(num_bboxes, 4)
    types = [1, 2, 3, 4] + np.random.randint(
        1, 4, size=[num_bboxes - 4]).tolist()

    sequence_ids = [
        str(x) for x in np.random.randint(0, num_sequences, size=[num_bboxes])
    ]
    frame_ids = np.random.randint(0, num_frames, size=[num_bboxes])

    object_ids = []
    # Ensure object ids aren't repeated for a (sequence id, frame id) pair.
    sequence_id_frame_id_to_next_object_id = {}
    for i in range(num_bboxes):
      key = (sequence_ids[i], frame_ids[i])
      if sequence_id_frame_id_to_next_object_id.get(key, None) is None:
        sequence_id_frame_id_to_next_object_id[key] = 0
      object_ids.append(sequence_id_frame_id_to_next_object_id[key])
      sequence_id_frame_id_to_next_object_id[key] += 1

    scores = np.random.uniform(size=[num_bboxes])
    speed = np.random.uniform(size=[num_bboxes, 2])

    return bboxes, types, frame_ids, sequence_ids, object_ids, scores, speed

  def _BuildConfig(self, additional_config_str=''):
    """Builds a metrics config."""
    config = metrics_pb2.Config()

    # OBJECT_TYPE adds 4 breakdowns
    # RANGE adds 12
    config_text = """
    num_desired_score_cutoffs: 2
    breakdown_generator_ids: OBJECT_TYPE
    difficulties {
      levels: LEVEL_1
    }
    breakdown_generator_ids: RANGE
    difficulties {
      levels: LEVEL_1
    }
    matcher_type: TYPE_HUNGARIAN
    iou_thresholds: 0.5
    iou_thresholds: 0.5
    iou_thresholds: 0.5
    iou_thresholds: 0.5
    iou_thresholds: 0.5
    box_type: TYPE_3D
    score_cutoffs: 0.5
    score_cutoffs: 0.9
    """ + additional_config_str
    text_format.Merge(config_text, config)
    return config

  def _BuildGraph(self, graph):
    with graph.as_default():

      self._prediction_bbox = tf.compat.v1.placeholder(dtype=tf.float32)
      self._prediction_type = tf.compat.v1.placeholder(dtype=tf.uint8)
      self._prediction_score = tf.compat.v1.placeholder(dtype=tf.float32)
      self._prediction_frame_id = tf.compat.v1.placeholder(dtype=tf.int64)
      self._prediction_sequence_id = tf.compat.v1.placeholder(dtype=tf.string)
      self._prediction_object_id = tf.compat.v1.placeholder(dtype=tf.int64)
      self._ground_truth_bbox = tf.compat.v1.placeholder(dtype=tf.float32)
      self._ground_truth_type = tf.compat.v1.placeholder(dtype=tf.uint8)
      self._ground_truth_frame_id = tf.compat.v1.placeholder(dtype=tf.int64)
      self._ground_truth_sequence_id = tf.compat.v1.placeholder(dtype=tf.string)
      self._ground_truth_object_id = tf.compat.v1.placeholder(dtype=tf.int64)
      self._ground_truth_difficulty = tf.compat.v1.placeholder(dtype=tf.uint8)
      self._prediction_overlap_nlz = tf.compat.v1.placeholder(dtype=tf.bool)
      self._ground_truth_speed = tf.compat.v1.placeholder(dtype=tf.float32)

      metrics = tracking_metrics.get_tracking_metric_ops(
          config=self._BuildConfig(),
          prediction_bbox=self._prediction_bbox,
          prediction_type=self._prediction_type,
          prediction_score=self._prediction_score,
          prediction_frame_id=self._prediction_frame_id,
          prediction_sequence_id=self._prediction_sequence_id,
          prediction_object_id=self._prediction_object_id,
          ground_truth_bbox=self._ground_truth_bbox,
          ground_truth_type=self._ground_truth_type,
          ground_truth_frame_id=self._ground_truth_frame_id,
          ground_truth_sequence_id=self._ground_truth_sequence_id,
          ground_truth_object_id=self._ground_truth_object_id,
          ground_truth_difficulty=tf.ones_like(
              self._ground_truth_frame_id, dtype=tf.uint8),
          prediction_overlap_nlz=tf.zeros_like(
              self._prediction_frame_id, dtype=tf.bool),
          ground_truth_speed=self._ground_truth_speed)
      return metrics

  def _EvalUpdateOps(
      self,
      sess,
      graph,
      metrics,
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
      ground_truth_speed,
  ):
    sess.run(
        [tf.group([value[1] for value in metrics.values()])],
        feed_dict={
            self._prediction_bbox: prediction_bbox,
            self._prediction_type: prediction_type,
            self._prediction_score: prediction_score,
            self._prediction_frame_id: prediction_frame_id,
            self._prediction_sequence_id: prediction_sequence_id,
            self._prediction_object_id: prediction_object_id,
            self._ground_truth_bbox: ground_truth_bbox,
            self._ground_truth_type: ground_truth_type,
            self._ground_truth_frame_id: ground_truth_frame_id,
            self._ground_truth_sequence_id: ground_truth_sequence_id,
            self._ground_truth_object_id: ground_truth_object_id,
            self._ground_truth_speed: ground_truth_speed,
        })

  def _EvalValueOps(self, sess, graph, metrics):
    # Get value_op from metrics dictionary.
    return {key: sess.run([value_op]) for key, (value_op, _) in metrics.items()}

  def testMetricsBasic(self):
    num_breakdowns = 16
    num_sequences, num_frames, n, m = 5, 10, 100, 200
    pd_bbox, pd_type, pd_frame_id, pd_sequence_id, pd_object_id, pd_score, _ = self._GenerateRandomBBoxes(
        num_sequences, num_frames, m)
    gt_bbox, gt_type, gt_frame_id, gt_sequence_id, gt_object_id, _, gt_speed = self._GenerateRandomBBoxes(
        num_sequences, num_frames, n)
    # Need to split tensors because need uniqueness to run twice since inputs
    # are concatenated. The split is used for both pd (len n) and gt (len m).
    split = min(n, m) // 2

    graph = tf.Graph()
    metrics = self._BuildGraph(graph)
    with self.test_session(graph=graph) as sess:
      sess.run(tf.compat.v1.initializers.local_variables())
      self._EvalUpdateOps(sess, graph, metrics, pd_bbox[:split],
                          pd_type[:split], pd_score[:split],
                          pd_frame_id[:split], pd_sequence_id[:split],
                          pd_object_id[:split], gt_bbox[:split],
                          gt_type[:split], gt_frame_id[:split],
                          gt_sequence_id[:split], gt_object_id[:split],
                          gt_speed[:split])
      self._EvalUpdateOps(sess, graph, metrics, pd_bbox[split:],
                          pd_type[split:], pd_score[split:],
                          pd_frame_id[split:], pd_sequence_id[split:],
                          pd_object_id[split:], gt_bbox[split:],
                          gt_type[split:], gt_frame_id[split:],
                          gt_sequence_id[split:], gt_object_id[split:],
                          gt_speed[split:])
      with tf.compat.v1.variable_scope('tracking_metrics', reuse=True):
        # Looking up an exisitng var to check that data is accumulated properly
        # in the variable.
        pd_frame_id_accumulated_var = tf.compat.v1.get_variable(
            'prediction_frame_id', dtype=tf.int64)
      pd_frame_id_accumulated = sess.run([pd_frame_id_accumulated_var])
      self.assertEqual(len(pd_frame_id_accumulated[0]), m)

      # For each breakdown, there are 5 metrics (MOTA/MOTP/FP/MISS/MISMATCH).
      mot_metrics_len = num_breakdowns * 5
      mot_metrics = self._EvalValueOps(sess, graph, metrics)
      self.assertEqual(len(mot_metrics), mot_metrics_len)
      for i in range(0, mot_metrics_len):
        self.assertTrue(-ERROR <= list(mot_metrics.values())[i][0] and
                        list(mot_metrics.values())[i][0] <= 1.0 + ERROR)

    # Test with prediction scores as ones and with groundtruth as prediction.
    with self.test_session(graph=graph) as sess:
      sess.run(tf.compat.v1.initializers.local_variables())
      pd_score_ones = np.ones_like(pd_frame_id)
      pd_speed_zeroes = np.zeros(shape=(len(pd_bbox), 2))
      self._EvalUpdateOps(sess, graph, metrics, pd_bbox[:split],
                          pd_type[:split], pd_score_ones[:split],
                          pd_frame_id[:split], pd_sequence_id[:split],
                          pd_object_id[:split], pd_bbox[:split],
                          pd_type[:split], pd_frame_id[:split],
                          pd_sequence_id[:split], pd_object_id[:split],
                          pd_speed_zeroes[:split])
      self._EvalUpdateOps(sess, graph, metrics, pd_bbox[split:],
                          pd_type[split:], pd_score_ones[split:],
                          pd_frame_id[split:], pd_sequence_id[split:],
                          pd_object_id[split:], pd_bbox[split:],
                          pd_type[split:], pd_frame_id[split:],
                          pd_sequence_id[split:], pd_object_id[split:],
                          pd_speed_zeroes[split:])
      with tf.compat.v1.variable_scope('tracking_metrics', reuse=True):
        # Looking up an exisitng var to check that data is accumulated properly
        # in the variable.
        pd_frame_id_accumulated_var = tf.compat.v1.get_variable(
            'prediction_frame_id', dtype=tf.int64)
      pd_frame_id_accumulated = sess.run([pd_frame_id_accumulated_var])
      self.assertEqual(len(pd_frame_id_accumulated[0]), m)

      # For each breakdown, there are 5 metrics (MOTA/MOTP/FP/MISS/MISMATCH).
      mot_metrics_len = num_breakdowns * 5
      mot_metrics = self._EvalValueOps(sess, graph, metrics)
      self.assertEqual(len(mot_metrics), mot_metrics_len)
      # Need to have boxes generated for every object type.
      self.assertAlmostEqual(list(mot_metrics.values())[0][0], 1.0, places=5)
      self.assertAlmostEqual(list(mot_metrics.values())[1][0], 0.0, places=5)


if __name__ == '__main__':
  tf.compat.v1.disable_eager_execution()
  tf.test.main()
