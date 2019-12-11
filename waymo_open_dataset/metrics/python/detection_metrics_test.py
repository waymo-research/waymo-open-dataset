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
"""Tests for waymo_open_dataset.metrics.python.detection_metrics."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from google.protobuf import text_format
from waymo_open_dataset.metrics.python import detection_metrics
from waymo_open_dataset.protos import metrics_pb2

ERROR = 1e-6


class DetectionMetricsEstimatorTest(tf.test.TestCase):

  def _GenerateRandomBBoxes(self, num_frames, num_bboxes):
    center_xyz = np.random.uniform(low=-1.0, high=1.0, size=(num_bboxes, 3))
    dimension = np.random.uniform(low=0.1, high=1.0, size=(num_bboxes, 3))
    rotation = np.random.uniform(low=-np.pi, high=np.pi, size=(num_bboxes, 1))
    bboxes = np.concatenate([center_xyz, dimension, rotation], axis=-1)
    types = np.random.randint(1, 5, size=[num_bboxes])
    frame_ids = np.random.randint(0, num_frames, size=[num_bboxes])
    scores = np.random.uniform(size=[num_bboxes])
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

  def _BuildGraph(self, graph):
    with graph.as_default():
      self._pd_frame_id = tf.compat.v1.placeholder(dtype=tf.int64)
      self._pd_bbox = tf.compat.v1.placeholder(dtype=tf.float32)
      self._pd_type = tf.compat.v1.placeholder(dtype=tf.uint8)
      self._pd_score = tf.compat.v1.placeholder(dtype=tf.float32)
      self._gt_frame_id = tf.compat.v1.placeholder(dtype=tf.int64)
      self._gt_bbox = tf.compat.v1.placeholder(dtype=tf.float32)
      self._gt_type = tf.compat.v1.placeholder(dtype=tf.uint8)

      metrics = detection_metrics.get_detection_metric_ops(
          config=self._BuildConfig(),
          prediction_frame_id=self._pd_frame_id,
          prediction_bbox=self._pd_bbox,
          prediction_type=self._pd_type,
          prediction_score=self._pd_score,
          prediction_overlap_nlz=tf.zeros_like(
              self._pd_frame_id, dtype=tf.bool),
          ground_truth_bbox=self._gt_bbox,
          ground_truth_type=self._gt_type,
          ground_truth_frame_id=self._gt_frame_id,
          ground_truth_difficulty=tf.ones_like(
              self._gt_frame_id, dtype=tf.uint8),
          recall_at_precision=0.95,
      )
      return metrics

  def _EvalUpdateOps(
      self,
      sess,
      graph,
      metrics,
      prediction_frame_id,
      prediction_bbox,
      prediction_type,
      prediction_score,
      ground_truth_frame_id,
      ground_truth_bbox,
      ground_truth_type,
  ):
    sess.run(
        [tf.group([value[1] for value in metrics.values()])],
        feed_dict={
            self._pd_bbox: prediction_bbox,
            self._pd_frame_id: prediction_frame_id,
            self._pd_type: prediction_type,
            self._pd_score: prediction_score,
            self._gt_bbox: ground_truth_bbox,
            self._gt_type: ground_truth_type,
            self._gt_frame_id: ground_truth_frame_id,
        })

  def _EvalValueOps(self, sess, graph, metrics):
    return {item[0]: sess.run([item[1][0]]) for item in metrics.items()}

  def testAPBasic(self):
    k, n, m = 10, 10, 2000
    pd_bbox, pd_type, pd_frameid, pd_score = self._GenerateRandomBBoxes(k, m)
    gt_bbox, gt_type, gt_frameid, _ = self._GenerateRandomBBoxes(k, n)

    graph = tf.Graph()
    metrics = self._BuildGraph(graph)
    with self.test_session(graph=graph) as sess:
      sess.run(tf.compat.v1.initializers.local_variables())
      self._EvalUpdateOps(sess, graph, metrics, pd_frameid, pd_bbox, pd_type,
                          pd_score, gt_frameid, gt_bbox, gt_type)
      self._EvalUpdateOps(sess, graph, metrics, pd_frameid, pd_bbox, pd_type,
                          pd_score, gt_frameid, gt_bbox, gt_type)
      with tf.compat.v1.variable_scope('detection_metrics', reuse=True):
        # Looking up an exisitng var to check that data is accumulated properly
        # in the variable.
        pd_frame_id_accumulated_var = tf.compat.v1.get_variable(
            'prediction_frame_id', dtype=tf.int64)
      pd_frame_id_accumulated = sess.run([pd_frame_id_accumulated_var])
      self.assertEqual(len(pd_frame_id_accumulated[0]), m * 2)

      aps = self._EvalValueOps(sess, graph, metrics)
      self.assertEqual(len(aps), 12)
      for i in range(0, 12):
        self.assertTrue(-ERROR <= list(aps.values())[i][0] and
                        list(aps.values())[i][0] <= 1.0 + ERROR)

    with self.test_session(graph=graph) as sess:
      sess.run(tf.compat.v1.initializers.local_variables())
      self._EvalUpdateOps(sess, graph, metrics, pd_frameid, pd_bbox, pd_type,
                          np.ones_like(pd_frameid), pd_frameid, pd_bbox,
                          pd_type)
      self._EvalUpdateOps(sess, graph, metrics, pd_frameid, pd_bbox, pd_type,
                          np.ones_like(pd_frameid), pd_frameid, pd_bbox,
                          pd_type)
      with tf.compat.v1.variable_scope('detection_metrics', reuse=True):
        # Looking up an exisitng var to check that data is accumulated properly
        # in the variable.
        pd_frame_id_accumulated_var = tf.compat.v1.get_variable(
            'prediction_frame_id', dtype=tf.int64)
      pd_frame_id_accumulated = sess.run([pd_frame_id_accumulated_var])
      self.assertEqual(len(pd_frame_id_accumulated[0]), m * 2)

      aps = self._EvalValueOps(sess, graph, metrics)
      self.assertEqual(len(aps), 12)
      for i in range(0, 12):
        # Note: 'm' (num_boxes) needs to be large enough such that we have boxes
        # generated for every object type.
        self.assertAlmostEqual(list(aps.values())[i][0], 1.0, places=5)


if __name__ == '__main__':
  tf.compat.v1.disable_eager_execution()
  tf.test.main()
