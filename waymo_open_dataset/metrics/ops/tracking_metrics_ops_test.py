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
import collections
import numpy as np
import tensorflow as tf

from google.protobuf import text_format
from waymo_open_dataset.metrics.ops import py_metrics_ops
from waymo_open_dataset.protos import metrics_pb2

# Numerical errors allowed when checking float values.
EPSILON = 1e-6

BBoxData = collections.namedtuple('BBoxData', [
    'bboxes', 'types', 'frame_ids', 'sequence_ids', 'object_ids', 'scores',
    'speed'
])


class TrackingMetricsOpsTest(tf.test.TestCase):

  def _GenerateRandomBBoxes(self, num_sequences, num_frames, num_bboxes):
    center_xyz = np.random.uniform(low=-1.0, high=1.0, size=(num_bboxes, 3))
    dimension = np.random.uniform(low=0.1, high=1.0, size=(num_bboxes, 3))
    rotation = np.random.uniform(low=-np.pi, high=np.pi, size=(num_bboxes, 1))
    bboxes = np.concatenate([center_xyz, dimension, rotation], axis=-1)
    types = np.random.randint(1, 4, size=[num_bboxes])
    frame_ids = np.random.randint(0, num_frames, size=[num_bboxes])
    sequence_ids = [
        str(x) for x in np.random.randint(0, num_sequences, size=[num_bboxes])
    ]

    # Choose object id so that they are not repeated bc this will cause errors
    # in tracking metrics' MOT computation.
    frame_ids_to_next_object_id = {x: 0 for x in range(num_frames)}
    object_ids = []
    for i in range(num_bboxes):
      frame_id = frame_ids[i]
      object_ids.append(frame_ids_to_next_object_id[frame_id])
      frame_ids_to_next_object_id[frame_id] += 1

    scores = np.random.uniform(size=[num_bboxes])
    speed = np.random.uniform(size=[num_bboxes, 2])
    return BBoxData(bboxes, types, frame_ids, sequence_ids, object_ids, scores,
                    speed)

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

  def _GetMetrics(self,
                  pd_bbox,
                  pd_types,
                  pd_frameid,
                  pd_sequenceid,
                  pd_objectid,
                  pd_score,
                  gt_bbox,
                  gt_types,
                  gt_frameid,
                  gt_sequenceid,
                  gt_objectid,
                  gt_speed=None,
                  additional_config_str=''):
    """Calls tracking metrics op to compute tracking metrics."""
    g = tf.Graph()
    with g.as_default():
      mota, motp, miss, mismatch, fp, score_cutoff, breakdown = py_metrics_ops.tracking_metrics(
          prediction_bbox=pd_bbox,
          prediction_type=pd_types,
          prediction_score=pd_score,
          prediction_frame_id=pd_frameid,
          prediction_sequence_id=pd_sequenceid,
          prediction_object_id=pd_objectid,
          ground_truth_bbox=gt_bbox,
          ground_truth_type=gt_types,
          ground_truth_frame_id=gt_frameid,
          ground_truth_sequence_id=gt_sequenceid,
          ground_truth_object_id=gt_objectid,
          ground_truth_difficulty=tf.ones_like(gt_frameid, dtype=tf.uint8),
          ground_truth_speed=gt_speed,
          config=self._BuildConfig(additional_config_str).SerializeToString())
    with self.test_session(graph=g) as sess:
      val = sess.run([mota, motp, miss, mismatch, fp, score_cutoff, breakdown])
    return val

  def testMetricsBasic(self):
    num_breakdowns = 16
    v, k, n, m = 5, 10, 100, 200
    pd = self._GenerateRandomBBoxes(v, k, m)
    gt = self._GenerateRandomBBoxes(v, k, n)

    mota, motp, miss, mismatch, fp, score_cutoff, breakdown = self._GetMetrics(
        pd.bboxes, pd.types, pd.frame_ids, pd.sequence_ids, pd.object_ids,
        pd.scores, gt.bboxes, gt.types, gt.frame_ids, gt.sequence_ids,
        gt.object_ids)
    # check shape
    for metric_val in [mota, motp, miss, mismatch, fp, score_cutoff]:
      self.assertEqual(metric_val.shape, (num_breakdowns,))
    self.assertEqual(breakdown.shape, (num_breakdowns, 3))
    # check values
    self.assertTrue((-EPSILON <= mota).all() and (mota <= 1.0 + EPSILON).all())
    self.assertTrue((-EPSILON <= motp).all() and (motp <= 1.0 + EPSILON).all())
    self.assertTrue((-EPSILON <= miss).all() and (miss <= 1.0 + EPSILON).all())
    self.assertTrue((-EPSILON <= mismatch).all() and
                    (mismatch <= 1.0 + EPSILON).all())
    self.assertTrue((-EPSILON <= fp).all() and (fp <= 1.0 + EPSILON).all())

    # Check metrics when gt is compared to gt.
    mota, motp, miss, mismatch, fp, score_cutoff, breakdown = self._GetMetrics(
        gt.bboxes, gt.types, gt.frame_ids, gt.sequence_ids, gt.object_ids,
        np.ones(n), gt.bboxes, gt.types, gt.frame_ids, gt.sequence_ids,
        gt.object_ids)
    # check shape
    for metric_val in [mota, motp, miss, mismatch, fp, score_cutoff]:
      self.assertEqual(metric_val.shape, (num_breakdowns,))
    self.assertEqual(breakdown.shape, (num_breakdowns, 3))
    # check values
    self.assertAlmostEqual(mota[1], 1.0, places=5)

    # Check metrics when gt is compared to gt where frame ids are all off.
    mota, motp, miss, mismatch, fp, score_cutoff, breakdown = self._GetMetrics(
        gt.bboxes, gt.types, gt.frame_ids, gt.sequence_ids, gt.object_ids,
        np.ones(n), gt.bboxes, gt.types, gt.frame_ids + n, gt.sequence_ids,
        gt.object_ids)
    # check shape
    for metric_val in [mota, motp, miss, mismatch, fp, score_cutoff]:
      self.assertEqual(metric_val.shape, (num_breakdowns,))
    self.assertEqual(breakdown.shape, (num_breakdowns, 3))
    # check values
    np.testing.assert_almost_equal(mota, 0.0, decimal=5)

  def testAllZeroValue(self):
    num_breakdowns = 16
    v, k, n, m = 5, 10, 100, 20
    pd = self._GenerateRandomBBoxes(v, k, m)
    gt = self._GenerateRandomBBoxes(v, k, n)
    mota, motp, miss, mismatch, fp, score_cutoff, breakdown = self._GetMetrics(
        pd.bboxes, pd.types, pd.frame_ids, pd.sequence_ids, pd.object_ids,
        pd.scores, gt.bboxes, gt.types, gt.frame_ids, gt.sequence_ids,
        gt.object_ids)
    # check shape
    for metric_val in [mota, motp, miss, mismatch, fp, score_cutoff]:
      self.assertEqual(metric_val.shape, (num_breakdowns,))
    self.assertEqual(breakdown.shape, (num_breakdowns, 3))
    # check values
    np.testing.assert_almost_equal(mota, 0.0, decimal=5)


if __name__ == '__main__':
  tf.compat.v1.disable_eager_execution()
  tf.test.main()
