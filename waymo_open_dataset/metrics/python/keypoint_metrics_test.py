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
# ==============================================================================
"""Tests for keypoint_metrics."""
import json
import math

import tensorflow as tf

# copybara removed file resource import
from waymo_open_dataset.metrics.python import keypoint_metrics as _lib

# Values copied from `computeOks` from pycocotools package:
# https://github.com/matteorr/coco-analyze/blob/9eb8a0a9e57ad1e592661efc2b8964864c0e6f28/pycocotools/cocoeval.py#L216
# The order of keypoints is the same as in the testdata:
#   nose, left_eye, right_eye, left_ear, right_ear, left_shoulder,
#   right_shoulder, left_elbow, right_elbow, left_wrist, right_wrist, left_hip,
#   right_hip, left_knee, right_knee, left_ankle, right_ankle
# https://github.com/matteorr/coco-analyze/blob/9eb8a0a9e57ad1e592661efc2b8964864c0e6f28/pycocotools/cocoanalyze.py#L928
# Note we hardcode 5x values to match the `computeOks` constant, to get actual
# scales use `_coco_scales`.
_COCO_KEYPOINT_SCALES_5X = (.26, .25, .25, .35, .35, .79, .79, .72, .72, .62,
                            .62, 1.07, 1.07, .87, .87, .89, .89)


def _coco_scales():
  # We multiply all sigmas from the referenced implementation by 2x to make it
  # consistent with the definition of per keypoint scales.
  return [s / 5.0 for s in _COCO_KEYPOINT_SCALES_5X]


def _create_keypoints_tensors(testdata) -> _lib.KeypointsTensors:
  location_and_visibility = tf.reshape(tf.constant(testdata), (-1, 3))
  return _lib.KeypointsTensors(
      location=location_and_visibility[:, 0:2],
      visibility=location_and_visibility[:, 2])


def _create_bbox_tensors(testdata) -> _lib.KeypointsTensors:
  bbox = tf.constant(testdata)
  top_left = bbox[:2]
  size = bbox[2:]
  return _lib.BoundingBoxTensors(center=top_left + size / 2, size=size)


def _convert_sample(sample):
  # See the `testdata/README.md` for details about testdata format.
  gt = _create_keypoints_tensors(sample['ground_truth']['keypoints'])
  box = _create_bbox_tensors(sample['ground_truth']['bbox'])
  pr = _create_keypoints_tensors(sample['prediction']['keypoints'])
  return gt, pr, box, sample['oks']


def _load_testdata():
  """Loads all ground truth and predicted keypoints from testdata."""
  # pylint: disable=line-too-long
  # pyformat: disable
  testdata_path = '{pyglib_resource}waymo_open_dataset/metrics/python/testdata/oks_testdata.json'.format(pyglib_resource='')
  # pyformat: enable
  # pylint: enable=line-too-long
  with open(testdata_path) as f:
    return json.load(f)


def _convert_testdata(testdata):
  """Converts test data into keypoints."""
  converted = [_convert_sample(s) for s in testdata]
  gts, prs, boxes, scores = zip(*converted)
  return (_lib.stack_keypoints(gts), _lib.stack_keypoints(prs),
          _lib.stack_boxes(boxes), scores)


class MiscTest(tf.test.TestCase):

  def test_box_displacement_returns_expected_shifts_for_outside_points_2d(self):
    # batch_size=2, num_points=4
    location = tf.constant(
        [[[0, 0], [2, 0], [4, 0], [0, 2]], [[4, 2], [0, 5], [2, 5], [4, 5]]],
        dtype=tf.float32)
    box = _lib.BoundingBoxTensors(
        center=tf.constant([[2, 2.5], [2, 2.5]], dtype=tf.float32),
        size=tf.constant([[2, 3], [2, 3]], dtype=tf.float32))

    shift = _lib.box_displacement(location, box)

    self.assertAllClose(shift, [[[1, 1], [0, 1], [-1, 1], [1, 0]],
                                [[-1, 0], [1, -1], [0, -1], [-1, -1]]])

  def test_box_displacement_returns_expected_shifts_for_inside_points_2d(self):
    # batch_size=2, num_points=1
    location = tf.constant([[[2, 2]], [[2, 3]]], dtype=tf.float32)
    box = _lib.BoundingBoxTensors(
        center=tf.constant([[2, 2.5], [2, 2.5]], dtype=tf.float32),
        size=tf.constant([[2, 3], [2, 3]], dtype=tf.float32))

    shift = _lib.box_displacement(location, box)

    self.assertAllClose(shift, [[[0, 0]], [[0, 0]]])


class KeypointMetricsTest(tf.test.TestCase):

  def test_oks_returns_exactly_same_values_as_coco_eval(self):
    testdata = _load_testdata()
    gt, pr, gt_box, expected_scores = _convert_testdata(testdata)

    scores = _lib.object_keypoint_similarity(gt, pr, gt_box, _coco_scales())

    self.assertAllClose(scores, expected_scores)

  def test_oks_supports_unbatched_tensors(self):
    testdata = _load_testdata()
    gt, pr, gt_box, expected_score = _convert_sample(testdata[1])

    score = _lib.object_keypoint_similarity(gt, pr, gt_box, _coco_scales())

    self.assertAllClose(score, expected_score)

  def test_average_precision_returns_precision_at_all_thresholds(self):
    thresholds = [0.5, 0.9]
    # We assume that OKS computation is tested directly, so for checking the
    # metric computation we need just one keypoint with a simple scale.
    # batch_size=3, num_keypoints=1
    # For a single visible keypoint OKS = exp(- d^2 / (2*scale^2*k^2)), where
    # scale=sqrt(area) and k is a scale for a specific type of the keypoint.
    # We set per keypoint scale and object area in a way to make the denominator
    # equal to 1.0, so OKS = exp(-d^2). This way for a selected threshold h we
    # can find d = sqrt(-ln(h))
    per_type_scales = [1.0 / math.sqrt(2)]
    gt = _lib.KeypointsTensors(
        location=tf.constant([[[1.0, 1.0]], [[2.0, 2.0]], [[3.0, 3.0]]]),
        visibility=tf.constant([[2.0], [2.0], [2.0]]))
    d = lambda h: math.sqrt(-math.log(h))
    # Shift predictions by a delta to get required threshold value.
    pr = _lib.KeypointsTensors(
        location=gt.location +
        tf.constant([[[d(0.25), 0.0]], [[d(0.7), 0.0]], [[d(0.95), 0.0]]]),
        visibility=tf.constant([[2.0], [2.0], [2.0]]))
    box = _lib.BoundingBoxTensors(
        center=gt.location[:, 0, :],
        size=tf.constant([[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]]))

    ap = _lib.AveragePrecisionAtOKS(
        thresholds,
        per_type_scales,
        name='All Points',
        precision_format='{} P @ OKS>={}',
        average_precision_format='{} AP')

    with self.subTest(name='correct_mean_values'):
      ap.reset_state()
      ap.update_state([gt, pr, box])
      metrics = ap.result()

      self.assertIsInstance(metrics, dict)
      self.assertEqual(
          metrics.keys(),
          set([
              'All Points P @ OKS>=0.5', 'All Points P @ OKS>=0.9',
              'All Points AP'
          ]))
      self.assertNear(metrics['All Points P @ OKS>=0.5'], 2.0 / 3, err=1e-5)
      self.assertNear(metrics['All Points P @ OKS>=0.9'], 1.0 / 3, err=1e-5)
      self.assertNear(metrics['All Points AP'], 1.0 / 2, err=1e-5)

    with self.subTest(name='respects_sample_weights'):
      # Weights for each keypoint.
      sample_weight = tf.constant([[0.0], [1.0], [1.0]])
      ap.reset_state()
      ap.update_state([gt, pr, box], sample_weight=sample_weight)
      metrics = ap.result()

      self.assertNear(metrics['All Points P @ OKS>=0.5'], 1.0, err=1e-5)
      self.assertNear(metrics['All Points P @ OKS>=0.9'], 0.5, err=1e-5)
      self.assertNear(metrics['All Points AP'], 0.75, err=1e-5)


if __name__ == '__main__':
  tf.test.main()
