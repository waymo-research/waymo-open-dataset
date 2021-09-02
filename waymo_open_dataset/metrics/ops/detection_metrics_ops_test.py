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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import tensorflow as tf

from google.protobuf import text_format
from waymo_open_dataset import label_pb2
from waymo_open_dataset.metrics.ops import py_metrics_ops
from waymo_open_dataset.protos import breakdown_pb2
from waymo_open_dataset.protos import metrics_pb2

# Numerical errors allowed when checking float values.
EPSILON = 1e-6


class DetectionMetricsOpsTest(tf.test.TestCase):

  def _GenerateRandomBBoxes(self, num_frames, num_bboxes):
    center_xyz = np.random.uniform(low=-1.0, high=1.0, size=(num_bboxes, 3))
    dimension = np.random.uniform(low=0.1, high=1.0, size=(num_bboxes, 3))
    rotation = np.random.uniform(low=-np.pi, high=np.pi, size=(num_bboxes, 1))
    bboxes = np.concatenate([center_xyz, dimension, rotation], axis=-1)
    types = np.random.randint(1, 4, size=[num_bboxes])
    frame_ids = np.random.randint(0, num_frames, size=[num_bboxes])
    scores = np.random.uniform(size=[num_bboxes])
    speed = np.random.uniform(size=[num_bboxes, 2])
    return bboxes, types, frame_ids, scores, speed

  def _BuildConfig(self, additional_config_str=''):
    """ Builds a metrics config. """
    config = metrics_pb2.Config()
    config_text = """
    num_desired_score_cutoffs: 11
    breakdown_generator_ids: ONE_SHARD
    difficulties {
    }
    matcher_type: TYPE_HUNGARIAN
    iou_thresholds: 0.5
    iou_thresholds: 0.5
    iou_thresholds: 0.5
    iou_thresholds: 0.5
    iou_thresholds: 0.5
    box_type: TYPE_3D
    """ + additional_config_str
    text_format.Merge(config_text, config)
    return config

  def _GetAP(self, pd_bbox, pd_types, pd_frameid, pd_score, gt_bbox, gt_types,
             gt_frameid, gt_speed, additional_config_str=''):
    """ Calls detection metrics op to compute detection metrics. """
    g = tf.Graph()
    with g.as_default():
      ap, aph, pr, prh, breakdown = py_metrics_ops.detection_metrics(
          prediction_bbox=pd_bbox,
          prediction_type=pd_types,
          prediction_score=pd_score,
          prediction_frame_id=pd_frameid,
          prediction_overlap_nlz=tf.zeros_like(pd_frameid, dtype=tf.bool),
          ground_truth_bbox=gt_bbox,
          ground_truth_type=gt_types,
          ground_truth_frame_id=gt_frameid,
          ground_truth_difficulty=tf.ones_like(gt_frameid, dtype=tf.uint8),
          ground_truth_speed=gt_speed,
          config=self._BuildConfig(additional_config_str).SerializeToString())
    with self.test_session(graph=g) as sess:
      val = sess.run([ap, aph, pr, prh, breakdown])
    return val

  def testAPBasic(self):
    k, n, m = 10, 100, 200
    pd_bbox, pd_type, pd_frameid, pd_score, _ = self._GenerateRandomBBoxes(k, m)
    gt_bbox, gt_type, gt_frameid, _, _ = self._GenerateRandomBBoxes(k, n)

    ap, aph, pr, prh, breakdown = self._GetAP(pd_bbox, pd_type, pd_frameid,
                                              pd_score, gt_bbox, gt_type,
                                              gt_frameid, gt_speed=None)
    self.assertEqual(pr.shape, (1, 11, 2))
    self.assertEqual(prh.shape, (1, 11, 2))
    self.assertAllEqual(
        breakdown[0, :],
        [breakdown_pb2.Breakdown.GeneratorId.Value('ONE_SHARD'), 0, 2])
    self.assertTrue(-EPSILON <= ap and ap <= 1.0 + EPSILON)
    self.assertTrue(-EPSILON <= aph and aph <= 1.0 + EPSILON)

    ap, aph, pr, prh, breakdown = self._GetAP(gt_bbox, gt_type, gt_frameid,
                                              np.ones(n), gt_bbox, gt_type,
                                              gt_frameid, gt_speed=None)
    self.assertAlmostEqual(ap[0], 1.0, places=5)
    self.assertAlmostEqual(aph[0], 1.0, places=5)
    self.assertEqual(pr.shape, (1, 11, 2))
    self.assertEqual(prh.shape, (1, 11, 2))
    self.assertAllEqual(
        breakdown[0, :],
        [breakdown_pb2.Breakdown.GeneratorId.Value('ONE_SHARD'), 0, 2])

    ap, aph, pr, prh, breakdown = self._GetAP(gt_bbox, gt_type, gt_frameid,
                                              np.ones(n), gt_bbox, gt_type,
                                              gt_frameid + n, gt_speed=None)
    self.assertAlmostEqual(ap, 0.0, places=5)
    self.assertAlmostEqual(aph, 0.0, places=5)
    self.assertEqual(pr.shape, (1, 11, 2))
    self.assertEqual(prh.shape, (1, 11, 2))
    self.assertAllEqual(
        breakdown[0, :],
        [breakdown_pb2.Breakdown.GeneratorId.Value('ONE_SHARD'), 0, 2])

  def testAllZeroValue(self):
    k, n, m = 10, 100, 20
    pd_bbox, pd_type, pd_frameid, pd_score, _ = self._GenerateRandomBBoxes(k, m)
    gt_bbox, gt_type, gt_frameid, _, gt_speed = self._GenerateRandomBBoxes(k, n)
    ap, aph, pr, prh, breakdown = self._GetAP(pd_bbox * 0, pd_type, pd_frameid,
                                              pd_score * 0, gt_bbox * 0,
                                              gt_type, gt_frameid * 0,
                                              gt_speed * 0)

    self.assertEqual(0, ap)
    self.assertAllEqual(pr.shape, (1, 11, 2))
    # IoU for 2 boxes with all zeros params is 0.0.
    self.assertAllEqual(pr[0, 0], [1.0, 0.0])
    self.assertAllEqual(pr[0, 1], [1.0, 0.0])

  def testEmpty(self):
    # We're generating 0 prediction and 0 ground truth boxes here
    k, n, m = 10, 0, 0
    pd_bbox, pd_type, pd_frameid, pd_score, _ = self._GenerateRandomBBoxes(k, m)
    gt_bbox, gt_type, gt_frameid, _, gt_speed = self._GenerateRandomBBoxes(k, n)
    ap, aph, pr, prh, breakdown = self._GetAP(pd_bbox, pd_type, pd_frameid,
                                              pd_score, gt_bbox, gt_type,
                                              gt_frameid, gt_speed)

    self.assertEqual(0, ap)
    self.assertEqual(0, aph)
    self.assertAllEqual(pr.shape, (1, 11, 2))
    self.assertAllEqual(prh.shape, (1, 11, 2))
    self.assertAllEqual(len(breakdown), 1)

  def testVelocityBreakdown(self):
    k, n, m = 10, 100, 200
    pd_bbox, pd_type, pd_frameid, pd_score, _ = self._GenerateRandomBBoxes(k, m)
    gt_bbox, gt_type, gt_frameid, _, gt_speed = self._GenerateRandomBBoxes(k, n)
    additional_config_str = """
    breakdown_generator_ids: VELOCITY
    difficulties {
    }
    """
    ap, aph, pr, prh, breakdown = self._GetAP(pd_bbox, pd_type, pd_frameid,
                                              pd_score, gt_bbox,
                                              gt_type, gt_frameid,
                                              gt_speed,
                                              additional_config_str)
    self.assertEqual(pr.shape, (21, 11, 2))
    self.assertEqual(prh.shape, (21, 11, 2))
    self.assertAllEqual(
        breakdown[0, :],
        [breakdown_pb2.Breakdown.GeneratorId.Value('ONE_SHARD'), 0, 2])
    for shard_idx in range(20):
      self.assertAllEqual(
          breakdown[shard_idx + 1, :],
          [breakdown_pb2.Breakdown.GeneratorId.Value('VELOCITY'), shard_idx, 2])
    self.assertTrue(np.all(ap >= -EPSILON))
    self.assertTrue(np.all(ap <= 1.0 + EPSILON))
    self.assertTrue(np.all(aph >= -EPSILON))
    self.assertTrue(np.all(aph <= 1.0 + EPSILON))

if __name__ == '__main__':
  tf.compat.v1.disable_eager_execution()
  tf.test.main()
