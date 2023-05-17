# Copyright 2022 The Waymo Open Dataset Authors.
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

from absl.testing import parameterized
import numpy as np
import tensorflow as tf

# copybara removed file resource import
from waymo_open_dataset.metrics.python import keypoint_metrics as _lib
from waymo_open_dataset.utils import keypoint_data as _data


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

_LEFT_SHOULDER = _lib.KeypointType.KEYPOINT_TYPE_LEFT_SHOULDER
_RIGHT_SHOULDER = _lib.KeypointType.KEYPOINT_TYPE_RIGHT_SHOULDER


def _coco_scales():
  # We multiply all sigmas from the referenced implementation by 2x to make it
  # consistent with the definition of per keypoint scales.
  return [s / 5.0 for s in _COCO_KEYPOINT_SCALES_5X]


def _create_keypoints_tensors(testdata) -> _data.KeypointsTensors:
  location_and_visibility = tf.reshape(tf.constant(testdata), (-1, 3))
  return _data.KeypointsTensors(
      location=location_and_visibility[:, 0:2],
      visibility=location_and_visibility[:, 2])


def _create_bbox_tensors(testdata) -> _data.KeypointsTensors:
  bbox = tf.constant(testdata)
  top_left = bbox[:2]
  size = bbox[2:]
  return _data.BoundingBoxTensors(center=top_left + size / 2, size=size)


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
  return (_data.stack_keypoints(gts), _data.stack_keypoints(prs),
          _data.stack_boxes(boxes), scores)


class MiscTest(tf.test.TestCase):

  def test_box_displacement_returns_expected_shifts_for_outside_points_2d(self):
    # batch_size=2, num_points=4
    location = tf.constant(
        [[[0, 0], [2, 0], [4, 0], [0, 2]], [[4, 2], [0, 5], [2, 5], [4, 5]]],
        dtype=tf.float32)
    box = _data.BoundingBoxTensors(
        center=tf.constant([[2, 2.5], [2, 2.5]], dtype=tf.float32),
        size=tf.constant([[2, 3], [2, 3]], dtype=tf.float32))

    shift = _lib.box_displacement(location, box)

    self.assertAllClose(shift, [[[1, 1], [0, 1], [-1, 1], [1, 0]],
                                [[-1, 0], [1, -1], [0, -1], [-1, -1]]])

  def test_box_displacement_returns_expected_shifts_for_inside_points_2d(self):
    # batch_size=2, num_points=1
    location = tf.constant([[[2, 2]], [[2, 3]]], dtype=tf.float32)
    box = _data.BoundingBoxTensors(
        center=tf.constant([[2, 2.5], [2, 2.5]], dtype=tf.float32),
        size=tf.constant([[2, 3], [2, 3]], dtype=tf.float32))

    shift = _lib.box_displacement(location, box)

    self.assertAllClose(shift, [[[0, 0]], [[0, 0]]])

  def test_box_displacement_takes_into_account_heading(self):
    # batch_size=3, num_points=1
    location = tf.constant(
        [[[2, 0, 0]], [[2, 0, 0]], [[2, 0, 0]]], dtype=tf.float32
    )
    # All boxes have the same size and center, the only difference - heading.
    box = _data.BoundingBoxTensors(
        center=tf.constant([[0, 0, 0], [0, 0, 0], [0, 0, 0]], dtype=tf.float32),
        size=tf.constant([[5, 2, 1], [5, 2, 1], [5, 2, 1]], dtype=tf.float32),
        heading=tf.constant([0, math.pi / 2, math.pi / 4], dtype=np.float32),
    )

    shift = _lib.box_displacement(location, box)

    d = 1 - 1 / math.sqrt(2)
    self.assertAllClose(shift, [[[0, 0, 0]], [[-1, 0, 0]], [[-d, -d, 0]]])

  def test_box_displacement_works_correctly_for_boxes_shifted_along_ox_oy(self):
    # batch_size=3, num_points=1
    location = tf.constant(
        [[[3, 1, 0]], [[3, 1, 0]], [[3, 1, 0]]], dtype=tf.float32
    )
    # All boxes have the same size and center, the only difference - heading.
    box = _data.BoundingBoxTensors(
        center=tf.constant([[1, 1, 0], [1, 1, 0], [1, 1, 0]], dtype=tf.float32),
        size=tf.constant([[5, 2, 1], [5, 2, 1], [5, 2, 1]], dtype=tf.float32),
        heading=tf.constant([0, math.pi / 2, math.pi / 4], dtype=np.float32),
    )

    shift = _lib.box_displacement(location, box)

    d = 1 - 1 / math.sqrt(2)
    self.assertAllClose(shift, [[[0, 0, 0]], [[-1, 0, 0]], [[-d, -d, 0]]])


class AveragePrecisionAtOKSTest(tf.test.TestCase):

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

  def test_ignores_masked_predictions(self):
    # sinle object with two 2D keypoints.
    gt = _data.KeypointsTensors(
        location=tf.constant([[[1, 1], [2, 2]]], dtype=tf.float32),
        visibility=tf.constant([[2, 2]], dtype=tf.float32),
    )
    # sinle object with two 2D keypoints: one visible, another - missing.
    pr = _data.KeypointsTensors(
        location=tf.constant([[[1, 1], [0, 0]]], dtype=tf.float32),
        visibility=tf.constant([[2, 0]], dtype=tf.float32),
    )
    gt_box = _data.BoundingBoxTensors(
        center=tf.constant([[0, 0]], dtype=tf.float32),
        size=tf.constant([[3, 3]], dtype=tf.float32),
    )
    # Scales for keypoint types #0 and #1.
    per_type_scales = [0.5, 0.5]

    score = _lib.object_keypoint_similarity(gt, pr, gt_box, per_type_scales)

    self.assertAllClose(score, [1.0])

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
    gt = _data.KeypointsTensors(
        location=tf.constant([[[1.0, 1.0]], [[2.0, 2.0]], [[3.0, 3.0]]]),
        visibility=tf.constant([[2], [2], [2]]))
    d = lambda h: math.sqrt(-math.log(h))
    # Shift predictions by a delta to get required threshold value.
    pr = _data.KeypointsTensors(
        location=gt.location +
        tf.constant([[[d(0.25), 0.0]], [[d(0.7), 0.0]], [[d(0.95), 0.0]]]),
        visibility=tf.constant([[2.0], [2.0], [2.0]]))
    box = _data.BoundingBoxTensors(
        center=gt.location[:, 0, :],
        size=tf.constant([[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]]))

    ap = _lib.AveragePrecisionAtOKS(
        per_type_scales,
        thresholds,
        name='All Points',
        precision_format='{name} P @ {threshold:.1f}',
        average_precision_format='{name} AP')

    with self.subTest(name='correct_mean_values'):
      ap.reset_state()
      ap.update_state([gt, pr, box])
      metrics = ap.result()

      self.assertIsInstance(metrics, dict)
      self.assertEqual(
          metrics.keys(),
          set(['All Points P @ 0.5', 'All Points P @ 0.9', 'All Points AP']))
      self.assertNear(metrics['All Points P @ 0.5'], 2.0 / 3, err=1e-5)
      self.assertNear(metrics['All Points P @ 0.9'], 1.0 / 3, err=1e-5)
      self.assertNear(
          metrics['All Points AP'], (2.0 / 3 + 1.0 / 3) / 2, err=1e-5
      )

    with self.subTest(name='respects_sample_zero_weights'):
      # Weights for each keypoint.
      sample_weight = tf.constant([[0.0], [1.0], [1.0]])
      ap.reset_state()
      ap.update_state([gt, pr, box], sample_weight=sample_weight)
      metrics = ap.result()

      self.assertNear(metrics['All Points P @ 0.5'], 2.0 / 2, err=1e-5)
      self.assertNear(metrics['All Points P @ 0.9'], 1.0 / 2, err=1e-5)
      self.assertNear(
          metrics['All Points AP'], (2.0 / 2 + 1.0 / 2) / 2, err=1e-5
      )

    with self.subTest(name='respects_sample_nonzero_weights'):
      # Weights for each keypoint.
      sample_weight = tf.constant([[0.5], [1.0], [1.0]])
      ap.reset_state()
      ap.update_state([gt, pr, box], sample_weight=sample_weight)
      metrics = ap.result()

      # First sample has OKS = 0.25, which is below both thresholds.
      # So its weight (0.5) is accounted only in the denominator.
      self.assertNear(metrics['All Points P @ 0.5'], 2.0 / 2.5, err=1e-5)
      self.assertNear(metrics['All Points P @ 0.9'], 1.0 / 2.5, err=1e-5)
      self.assertNear(
          metrics['All Points AP'], (2.0 / 2.5 + 1.0 / 2.5) / 2, err=1e-5
      )

  def test_do_not_use_box_displacement_for_padded_predictions(self):
    # Two GT objects: actual, padded
    gt = _data.KeypointsTensors(
        location=tf.constant([[[1, 1]], [[0, 0]]], dtype=tf.float32),
        visibility=tf.constant([[2], [0]], dtype=tf.float32),
    )
    gt_box = _data.BoundingBoxTensors(
        center=tf.constant([[1, 1], [0, 0]], dtype=tf.float32),
        size=tf.constant([[1, 1], [0, 0]], dtype=tf.float32),
    )
    # Two PR objects: padded, actual
    pr = _data.KeypointsTensors(
        location=tf.constant([[[0, 0]], [[2, 2]]], dtype=tf.float32),
        visibility=tf.constant([[0], [2]], dtype=tf.float32),
    )
    # The threshold and scale are selected to not match padded keypoints
    # unintentionally.
    per_type_scales = [0.5]
    thresholds = [0.5]

    ap = _lib.AveragePrecisionAtOKS(
        per_type_scales,
        thresholds,
        name='All Points',
        precision_format='{name} P @ {threshold:.1f}',
        average_precision_format='{name} AP',
    )
    ap.update_state([gt, pr, gt_box])
    metrics = ap.result()

    # No objects should match, so metric should be zero.
    self.assertNear(metrics['All Points P @ 0.5'], 0.0, err=1e-5)


class MpjpeTest(tf.test.TestCase):

  def test_returns_mean_square_error_for_all_keypoints(self):
    # batch_size = 3, num_points = 2
    gt = _data.KeypointsTensors(
        location=tf.constant([[[1.0, 1.0], [-1.0, -1.0]],
                              [[2.0, 2.0], [-2.0, -2.0]],
                              [[3.0, 3.0], [-3.0, -3.0]]]),
        visibility=tf.constant([[2, 2], [2, 2], [2, 2]]))
    # Predicted points are [1, 2, 3, 4, 5, 6] pixels away from ground truth.
    pr = _data.KeypointsTensors(
        location=tf.constant([[[1.0, 0.0], [1.0, -1.0]],
                              [[2.0, -1.0], [2.0, -2.0]],
                              [[3.0, -2.0], [-3.0, 3.0]]]),
        visibility=tf.constant([[2, 2], [2, 2], [2, 2]]))
    box = None  # is not used by the metric

    mpjpe = _lib.MeanPerJointPositionError(name='MPJPE')
    mpjpe.update_state([gt, pr, box])
    metrics = mpjpe.result()

    self.assertNear(metrics['MPJPE'], (1 + 2 + 3 + 4 + 5 + 6) / 6, err=1e-5)

  def test_takes_into_account_keypoint_visibility(self):
    # batch_size = 3, num_points = 2
    gt = _data.KeypointsTensors(
        location=tf.constant([[[1.0, 1.0], [-1.0, -1.0]],
                              [[2.0, 2.0], [-2.0, -2.0]],
                              [[3.0, 3.0], [-3.0, -3.0]]]),
        visibility=tf.constant([[1.0, 2.0], [0.0, 2.0], [2.0, 0.0]]))
    # Predicted points are [1, 2, 3, 4, 5, 6] pixels away from ground truth.
    pr = _data.KeypointsTensors(
        location=tf.constant([[[1.0, 0.0], [1.0, -1.0]],
                              [[2.0, -1.0], [2.0, -2.0]],
                              [[3.0, -2.0], [-3.0, 3.0]]]),
        visibility=tf.constant([[2, 2], [2, 2], [2, 2]]))
    box = None  # is not used by the metric

    mpjpe = _lib.MeanPerJointPositionError(name='MPJPE')
    mpjpe.update_state([gt, pr, box])
    metrics = mpjpe.result()

    self.assertNear(metrics['MPJPE'], (1 + 2 + 0 + 4 + 5 + 0) / 4, err=1e-5)

  def test_respects_sample_weights(self):
    # batch_size = 3, num_points = 2
    gt = _data.KeypointsTensors(
        location=tf.constant([[[1.0, 1.0], [-1.0, -1.0]],
                              [[2.0, 2.0], [-2.0, -2.0]],
                              [[3.0, 3.0], [-3.0, -3.0]]]),
        visibility=tf.constant([[2, 2], [0, 2], [2, 0]]))
    # Predicted points are [1, 2, 3, 4, 5, 6] pixels away from ground truth.
    pr = _data.KeypointsTensors(
        location=tf.constant([[[1.0, 0.0], [1.0, -1.0]],
                              [[2.0, -1.0], [2.0, -2.0]],
                              [[3.0, -2.0], [-3.0, 3.0]]]),
        visibility=tf.constant([[2, 2], [2, 2], [2, 2]]))
    box = None  # is not used by the metric
    sample_weight = tf.constant([0.0, 0.5, 1.0])

    mpjpe = _lib.MeanPerJointPositionError(name='MPJPE')
    mpjpe.update_state([gt, pr, box], sample_weight=sample_weight)
    metrics = mpjpe.result()

    self.assertNear(
        metrics['MPJPE'], (0 + 0 + 0 + 4 * 0.5 + 5 * 1.0 + 0) / (0.5 + 1.0),
        err=1e-5)

  def test_ignores_masked_predictions(self):
    # sinle object with two 2D keypoints.
    gt = _data.KeypointsTensors(
        location=tf.constant([[[1, 1], [2, 2]]], dtype=tf.float32),
        visibility=tf.constant([[2, 2]], dtype=tf.float32),
    )
    # sinle object with two 2D keypoints: one visible, another - missing.
    pr = _data.KeypointsTensors(
        location=tf.constant([[[1, 1], [0, 0]]], dtype=tf.float32),
        visibility=tf.constant([[2, 0]], dtype=tf.float32),
    )
    gt_box = _data.BoundingBoxTensors(
        center=tf.constant([[0, 0]], dtype=tf.float32),
        size=tf.constant([[3, 3]], dtype=tf.float32),
    )

    mpjpe = _lib.MeanPerJointPositionError(name='MPJPE')
    mpjpe.update_state([gt, pr, gt_box])
    metrics = mpjpe.result()

    self.assertAllClose(metrics['MPJPE'], 0.0)


class PckTest(tf.test.TestCase):

  def test_returns_correct_result_for_a_large_abs_threshold_value(self):
    # batch_size = 3, num_points = 2
    gt = _data.KeypointsTensors(
        location=tf.constant([[[1.0, 1.0], [-1.0, -1.0]],
                              [[2.0, 2.0], [-2.0, -2.0]],
                              [[3.0, 3.0], [-3.0, -3.0]]]),
        visibility=tf.constant([[2, 2], [2, 2], [2, 2]]))
    # Predicted points are [1, 2, 3, 4, 5, 6] pixels away from ground truth.
    pr = _data.KeypointsTensors(
        location=tf.constant([[[1.0, 0.0], [1.0, -1.0]],
                              [[2.0, -1.0], [2.0, -2.0]],
                              [[3.0, -2.0], [-3.0, 3.0]]]),
        visibility=tf.constant([[2, 2], [2, 2], [2, 2]]))
    box = None  # is not used by the metric

    pck = _lib.PercentageOfCorrectKeypoints(name='PCK', thresholds=[10])
    pck.update_state([gt, pr, box])
    metrics = pck.result()

    self.assertNear(metrics['PCK @ 10.00'], 1.0, err=1e-5)

  def test_returns_correct_results_for_multiple_thresholds(self):
    # batch_size = 3, num_points = 2
    gt = _data.KeypointsTensors(
        location=tf.constant([[[1.0, 1.0], [-1.0, -1.0]],
                              [[2.0, 2.0], [-2.0, -2.0]],
                              [[3.0, 3.0], [-3.0, -3.0]]]),
        visibility=tf.constant([[2, 2], [2, 2], [2, 2]]))
    # Predicted points are [1, 2, 3, 4, 5, 6] pixels away from ground truth.
    pr = _data.KeypointsTensors(
        location=tf.constant([[[1.0, 0.0], [1.0, -1.0]],
                              [[2.0, -1.0], [2.0, -2.0]],
                              [[3.0, -2.0], [-3.0, 3.0]]]),
        visibility=tf.constant([[2, 2], [2, 2], [2, 2]]))
    box = None  # is not used by the metric

    pck = _lib.PercentageOfCorrectKeypoints(name='PCK', thresholds=[2, 5])
    pck.update_state([gt, pr, box])
    metrics = pck.result()

    self.assertCountEqual(metrics.keys(), ['PCK @ 2.00', 'PCK @ 5.00'])
    self.assertNear(metrics['PCK @ 2.00'], 2 / 6, err=1e-5)
    self.assertNear(metrics['PCK @ 5.00'], 5 / 6, err=1e-5)

  def test_takes_into_account_keypoint_visibility_abs_threshold(self):
    # batch_size = 3, num_points = 2
    gt = _data.KeypointsTensors(
        location=tf.constant([[[1.0, 1.0], [-1.0, -1.0]],
                              [[2.0, 2.0], [-2.0, -2.0]],
                              [[3.0, 3.0], [-3.0, -3.0]]]),
        visibility=tf.constant([[2, 2], [2, 2], [2, 0]]))
    # Predicted points are [1, 2, 3, 4, 5, 6] pixels away from ground truth.
    pr = _data.KeypointsTensors(
        location=tf.constant([[[1.0, 0.0], [1.0, -1.0]],
                              [[2.0, -1.0], [2.0, -2.0]],
                              [[3.0, -2.0], [-3.0, 3.0]]]),
        visibility=tf.constant([[1, 2], [2, 2], [2, 2]]))
    box = None  # is not used by the metric

    pck = _lib.PercentageOfCorrectKeypoints(thresholds=[4], name='PCK')
    pck.update_state([gt, pr, box])
    metrics = pck.result()

    self.assertNear(metrics['PCK @ 4.00'], 0.8, err=1e-5)

  def test_takes_respects_sample_weights_abs_threshold(self):
    # batch_size = 3, num_points = 2
    gt = _data.KeypointsTensors(
        location=tf.constant([[[1.0, 1.0], [-1.0, -1.0]],
                              [[2.0, 2.0], [-2.0, -2.0]],
                              [[3.0, 3.0], [-3.0, -3.0]]]),
        visibility=tf.constant([[2, 2], [0, 2], [2, 0]]))
    # Predicted points are [1, 2, 3, 4, 5, 6] pixels away from ground truth.
    pr = _data.KeypointsTensors(
        location=tf.constant([[[1.0, 0.0], [1.0, -1.0]],
                              [[2.0, -1.0], [2.0, -2.0]],
                              [[3.0, -2.0], [-3.0, 3.0]]]),
        visibility=tf.constant([[2, 2], [2, 2], [2, 2]]))
    box = None  # is not used by the metric
    sample_weight = tf.constant([0.0, 1.0, 1.0])

    pck = _lib.PercentageOfCorrectKeypoints(thresholds=[4.0], name='PCK')
    pck.update_state([gt, pr, box], sample_weight=sample_weight)
    metrics = pck.result()

    # The weights exclude 2 keypoints from the sample #0, visibility mask in
    # samples #1 and #2 excludes 2 more. Only one out of two keypoints left is
    # closer than 4 pixels to the ground truth.
    self.assertNear(metrics['PCK @ 4.00'], 1 / 2, err=1e-5)

  def test_returns_correct_result_when_using_object_scale(self):
    # batch_size = 3, num_points = 2
    gt = _data.KeypointsTensors(
        location=tf.constant([[[1.0, 1.0], [-1.0, -1.0]],
                              [[2.0, 2.0], [-2.0, -2.0]],
                              [[3.0, 3.0], [-3.0, -3.0]]]),
        visibility=tf.constant([[2, 2], [2, 2], [2, 2]]))
    # Predicted points are [[1, 2], [3, 4], [5, 6]] px away from ground truth.
    pr = _data.KeypointsTensors(
        location=tf.constant([[[1.0, 0.0], [1.0, -1.0]],
                              [[2.0, -1.0], [2.0, -2.0]],
                              [[3.0, -2.0], [-3.0, 3.0]]]),
        visibility=tf.constant([[2, 2], [2, 2], [2, 2]]))
    # Objects scales are: 1, 10, 20
    box = _data.BoundingBoxTensors(
        center=tf.constant([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]),
        size=tf.constant([[1.0, 1.0], [10.0, 10.0], [20.0, 20.0]]))

    pck = _lib.PercentageOfCorrectKeypoints(
        name='PCK', thresholds=[0.5], use_object_scale=True)
    pck.update_state([gt, pr, box])
    metrics = pck.result()

    self.assertNear(
        metrics['PCK @ 0.50'], (0 + 0 + 1 + 1 + 1 + 1) / 6, err=1e-5)

  def test_returns_correct_result_when_using_keypoint_scale(self):
    # batch_size = 3, num_points = 2
    gt = _data.KeypointsTensors(
        location=tf.constant([[[1.0, 1.0], [-1.0, -1.0]],
                              [[2.0, 2.0], [-2.0, -2.0]],
                              [[3.0, 3.0], [-3.0, -3.0]]]),
        visibility=tf.constant([[2, 2], [2, 2], [2, 2]]))
    # Predicted points are [[1, 2], [3, 4], [5, 6]] px away from ground truth.
    pr = _data.KeypointsTensors(
        location=tf.constant([[[1.0, 0.0], [1.0, -1.0]],
                              [[2.0, -1.0], [2.0, -2.0]],
                              [[3.0, -2.0], [-3.0, 3.0]]]),
        visibility=tf.constant([[2, 2], [2, 2], [2, 2]]))
    # Objects scales are: 1, 10, 20
    box = _data.BoundingBoxTensors(
        center=tf.constant([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]),
        size=tf.constant([[1.0, 1.0], [10.0, 10.0], [20.0, 20.0]]))
    per_type_scales = [1.0, 0.5]

    pck = _lib.PercentageOfCorrectKeypoints(
        name='PCK',
        thresholds=[0.5],
        per_type_scales=per_type_scales,
        use_object_scale=True)
    pck.update_state([gt, pr, box])
    metrics = pck.result()

    # NOTE: Effective absolute thresholds for all keypoints will be:
    #   [[0.5 * 1.0 * 1.0 = 0.5, 0.5 * 1.0 * 0.5 = 0.25],
    #    [0.5 * 10.0 * 1.0 = 5.0, 0.5 * 10.0 * 0.5 = 2.5],
    #    [0.5 * 20.0 * 1.0 = 10., 0.5 * 20.0 * 0.5 = 5]]
    # So the second keypoints in all samples are not OK now, compared to the
    # `test_returns_correct_result_when_using_object_scale` test case.
    self.assertNear(
        metrics['PCK @ 0.50'], (0 + 0 + 1 + 0 + 1 + 0) / 6, err=1e-5)

  def test_returns_correct_result_when_using_absolute_keypoint_scales(self):
    # batch_size = 3, num_points = 2
    gt = _data.KeypointsTensors(
        location=tf.constant([[[1.0, 1.0], [-1.0, -1.0]],
                              [[2.0, 2.0], [-2.0, -2.0]],
                              [[3.0, 3.0], [-3.0, -3.0]]]),
        visibility=tf.constant([[2, 2], [2, 2], [2, 2]]))
    # Predicted points are [[1, 2], [3, 4], [5, 6]] px away from ground truth.
    pr = _data.KeypointsTensors(
        location=tf.constant([[[1.0, 0.0], [1.0, -1.0]],
                              [[2.0, -1.0], [2.0, -2.0]],
                              [[3.0, -2.0], [-3.0, 3.0]]]),
        visibility=tf.constant([[2, 2], [2, 2], [2, 2]]))
    box = None  # is not used by the metric
    per_type_scales = [4, 6]

    pck = _lib.PercentageOfCorrectKeypoints(
        name='PCK',
        thresholds=[0.5],
        per_type_scales=per_type_scales,
        use_object_scale=False)
    pck.update_state([gt, pr, box])
    metrics = pck.result()

    # NOTE: Effective absolute thresholds for keypoint types will be:
    #   [0.5 * 4 = 2, 0.5 * 6 = 3] in all samples in the batch.
    self.assertNear(
        metrics['PCK @ 0.50'], (1 + 1 + 0 + 0 + 0 + 0) / 6, err=1e-5)

  def test_ignores_masked_predictions(self):
    # sinle object with two 2D keypoints.
    gt = _data.KeypointsTensors(
        location=tf.constant([[[1, 1], [2, 2]]], dtype=tf.float32),
        visibility=tf.constant([[2, 2]], dtype=tf.float32),
    )
    # sinle object with two 2D keypoints: one visible, another - missing.
    pr = _data.KeypointsTensors(
        location=tf.constant([[[1, 1], [0, 0]]], dtype=tf.float32),
        visibility=tf.constant([[2, 0]], dtype=tf.float32),
    )
    gt_box = _data.BoundingBoxTensors(
        center=tf.constant([[0, 0]], dtype=tf.float32),
        size=tf.constant([[3, 3]], dtype=tf.float32),
    )

    pck = _lib.PercentageOfCorrectKeypoints(
        name='PCK', thresholds=[0.5], use_object_scale=False
    )
    pck.update_state([gt, pr, gt_box])
    metrics = pck.result()

    self.assertAllClose(metrics['PCK @ 0.50'], 1.0)


class MetricForSubsetsTest(tf.test.TestCase):

  def test_returns_results_for_all_subsets(self):
    # batch_size = 3, num_points = 2
    gt = _data.KeypointsTensors(
        location=tf.constant([[[1.0, 1.0], [-1.0, -1.0]],
                              [[2.0, 2.0], [-2.0, -2.0]],
                              [[3.0, 3.0], [-3.0, -3.0]]]),
        visibility=tf.constant([[2, 2], [2, 2], [2, 2]]))
    # Predicted points are [1, 2, 3, 4, 5, 6] pixels away from ground truth.
    pr = _data.KeypointsTensors(
        location=tf.constant([[[1.0, 0.0], [1.0, -1.0]],
                              [[2.0, -1.0], [2.0, -2.0]],
                              [[3.0, -2.0], [-3.0, 3.0]]]),
        visibility=tf.constant([[2, 2], [2, 2], [2, 2]]))
    box = None  # is not used by the metric
    src_order = (_LEFT_SHOULDER, _RIGHT_SHOULDER)
    subsets = {
        'LEFT': (_LEFT_SHOULDER,),
        'RIGHT': (_RIGHT_SHOULDER,),
    }
    create_mpjpe = lambda n: _lib.MeanPerJointPositionError(name=f'MPJPE/{n}')

    metric = _lib.MetricForSubsets(
        src_order=src_order, subsets=subsets, create_metric_fn=create_mpjpe)
    metric.update_state([gt, pr, box])
    result = metric.result()

    self.assertCountEqual(result.keys(), ['MPJPE/LEFT', 'MPJPE/RIGHT'])
    self.assertNear(result['MPJPE/LEFT'], (1 + 3 + 5) / 3, err=1e-5)
    self.assertNear(result['MPJPE/RIGHT'], (2 + 4 + 6) / 3, err=1e-5)


class CombinedMetricTest(tf.test.TestCase):

  def test_returns_results_from_all_metrics(self):
    gt = _data.KeypointsTensors(
        location=tf.constant([[[1.0, 1.0], [-1.0, -1.0]],
                              [[2.0, 2.0], [-2.0, -2.0]],
                              [[3.0, 3.0], [-3.0, -3.0]]]),
        visibility=tf.constant([[2, 2], [2, 2], [2, 2]]))
    # Predicted points are [1, 2, 3, 4, 5, 6] pixels away from ground truth.
    pr = _data.KeypointsTensors(
        location=tf.constant([[[1.0, 0.0], [1.0, -1.0]],
                              [[2.0, -1.0], [2.0, -2.0]],
                              [[3.0, -2.0], [-3.0, 3.0]]]),
        visibility=tf.constant([[2, 2], [2, 2], [2, 2]]))
    box = None  # is not used by the metric
    mpjpe = _lib.MeanPerJointPositionError(name='MPJPE')
    pck = _lib.PercentageOfCorrectKeypoints(name='PCK', thresholds=[10])

    metric = _lib.CombinedMetric([mpjpe, pck])
    metric.update_state([gt, pr, box])
    result = metric.result()

    self.assertCountEqual(result.keys(), ['MPJPE', 'PCK @ 10.00'])


def _random_keypoints(batch_size, num_points, dims=2):
  return _data.KeypointsTensors(
      location=tf.random.uniform((batch_size, num_points, dims)),
      visibility=tf.random.uniform((batch_size, num_points),
                                   minval=0,
                                   maxval=2,
                                   dtype=tf.int32))


def _random_box(batch_size, dims=2):
  return _data.BoundingBoxTensors(
      center=tf.random.uniform((batch_size, dims)),
      size=tf.random.uniform((batch_size, dims)))


def _random_inputs(batch_size, num_points, dims=2):
  gt = _random_keypoints(batch_size, num_points, dims=dims)
  pr = _random_keypoints(batch_size, num_points, dims=dims)
  box = _random_box(batch_size, dims=dims)
  return gt, pr, box


_ALL_METRIC_NAMES = (
    'MPJPE/SHOULDERS', 'MPJPE/ELBOWS', 'MPJPE/WRISTS', 'MPJPE/HIPS',
    'MPJPE/KNEES', 'MPJPE/ANKLES', 'MPJPE/ALL', 'MPJPE/HEAD',
    'PCK/SHOULDERS @ 0.05', 'PCK/SHOULDERS @ 0.10', 'PCK/SHOULDERS @ 0.20',
    'PCK/SHOULDERS @ 0.30', 'PCK/SHOULDERS @ 0.40', 'PCK/SHOULDERS @ 0.50',
    'PCK/ELBOWS @ 0.05', 'PCK/ELBOWS @ 0.10', 'PCK/ELBOWS @ 0.20',
    'PCK/ELBOWS @ 0.30', 'PCK/ELBOWS @ 0.40', 'PCK/ELBOWS @ 0.50',
    'PCK/WRISTS @ 0.05', 'PCK/WRISTS @ 0.10', 'PCK/WRISTS @ 0.20',
    'PCK/WRISTS @ 0.30', 'PCK/WRISTS @ 0.40', 'PCK/WRISTS @ 0.50',
    'PCK/HIPS @ 0.05', 'PCK/HIPS @ 0.10', 'PCK/HIPS @ 0.20', 'PCK/HIPS @ 0.30',
    'PCK/HIPS @ 0.40', 'PCK/HIPS @ 0.50', 'PCK/KNEES @ 0.05',
    'PCK/KNEES @ 0.10', 'PCK/KNEES @ 0.20', 'PCK/KNEES @ 0.30',
    'PCK/KNEES @ 0.40', 'PCK/KNEES @ 0.50', 'PCK/ANKLES @ 0.05',
    'PCK/ANKLES @ 0.10', 'PCK/ANKLES @ 0.20', 'PCK/ANKLES @ 0.30',
    'PCK/ANKLES @ 0.40', 'PCK/ANKLES @ 0.50', 'PCK/ALL @ 0.05',
    'PCK/ALL @ 0.10', 'PCK/ALL @ 0.20', 'PCK/ALL @ 0.30', 'PCK/ALL @ 0.40',
    'PCK/ALL @ 0.50', 'PCK/HEAD @ 0.05', 'PCK/HEAD @ 0.10', 'PCK/HEAD @ 0.20',
    'PCK/HEAD @ 0.30', 'PCK/HEAD @ 0.40', 'PCK/HEAD @ 0.50',
    'OKS/SHOULDERS P @ 0.50', 'OKS/SHOULDERS P @ 0.55',
    'OKS/SHOULDERS P @ 0.60', 'OKS/SHOULDERS P @ 0.65',
    'OKS/SHOULDERS P @ 0.70', 'OKS/SHOULDERS P @ 0.75',
    'OKS/SHOULDERS P @ 0.80', 'OKS/SHOULDERS P @ 0.85',
    'OKS/SHOULDERS P @ 0.90', 'OKS/SHOULDERS P @ 0.95', 'OKS/SHOULDERS AP',
    'OKS/ELBOWS P @ 0.50', 'OKS/ELBOWS P @ 0.55', 'OKS/ELBOWS P @ 0.60',
    'OKS/ELBOWS P @ 0.65', 'OKS/ELBOWS P @ 0.70', 'OKS/ELBOWS P @ 0.75',
    'OKS/ELBOWS P @ 0.80', 'OKS/ELBOWS P @ 0.85', 'OKS/ELBOWS P @ 0.90',
    'OKS/ELBOWS P @ 0.95', 'OKS/ELBOWS AP', 'OKS/WRISTS P @ 0.50',
    'OKS/WRISTS P @ 0.55', 'OKS/WRISTS P @ 0.60', 'OKS/WRISTS P @ 0.65',
    'OKS/WRISTS P @ 0.70', 'OKS/WRISTS P @ 0.75', 'OKS/WRISTS P @ 0.80',
    'OKS/WRISTS P @ 0.85', 'OKS/WRISTS P @ 0.90', 'OKS/WRISTS P @ 0.95',
    'OKS/WRISTS AP', 'OKS/HIPS P @ 0.50', 'OKS/HIPS P @ 0.55',
    'OKS/HIPS P @ 0.60', 'OKS/HIPS P @ 0.65', 'OKS/HIPS P @ 0.70',
    'OKS/HIPS P @ 0.75', 'OKS/HIPS P @ 0.80', 'OKS/HIPS P @ 0.85',
    'OKS/HIPS P @ 0.90', 'OKS/HIPS P @ 0.95', 'OKS/HIPS AP',
    'OKS/KNEES P @ 0.50', 'OKS/KNEES P @ 0.55', 'OKS/KNEES P @ 0.60',
    'OKS/KNEES P @ 0.65', 'OKS/KNEES P @ 0.70', 'OKS/KNEES P @ 0.75',
    'OKS/KNEES P @ 0.80', 'OKS/KNEES P @ 0.85', 'OKS/KNEES P @ 0.90',
    'OKS/KNEES P @ 0.95', 'OKS/KNEES AP', 'OKS/ANKLES P @ 0.50',
    'OKS/ANKLES P @ 0.55', 'OKS/ANKLES P @ 0.60', 'OKS/ANKLES P @ 0.65',
    'OKS/ANKLES P @ 0.70', 'OKS/ANKLES P @ 0.75', 'OKS/ANKLES P @ 0.80',
    'OKS/ANKLES P @ 0.85', 'OKS/ANKLES P @ 0.90', 'OKS/ANKLES P @ 0.95',
    'OKS/ANKLES AP', 'OKS/ALL P @ 0.50', 'OKS/ALL P @ 0.55', 'OKS/ALL P @ 0.60',
    'OKS/ALL P @ 0.65', 'OKS/ALL P @ 0.70', 'OKS/ALL P @ 0.75',
    'OKS/ALL P @ 0.80', 'OKS/ALL P @ 0.85', 'OKS/ALL P @ 0.90',
    'OKS/ALL P @ 0.95', 'OKS/ALL AP', 'OKS/HEAD P @ 0.50', 'OKS/HEAD P @ 0.55',
    'OKS/HEAD P @ 0.60', 'OKS/HEAD P @ 0.65', 'OKS/HEAD P @ 0.70',
    'OKS/HEAD P @ 0.75', 'OKS/HEAD P @ 0.80', 'OKS/HEAD P @ 0.85',
    'OKS/HEAD P @ 0.90', 'OKS/HEAD P @ 0.95', 'OKS/HEAD AP')


class AllMetricsTest(tf.test.TestCase):
  maxDiff = 10000

  def test_all_metrics_for_camera_returns_results_with_expected_names(self):
    gt, pr, box = _random_inputs(batch_size=3, num_points=14)

    metric = _lib.create_combined_metric(_lib.DEFAULT_CONFIG_CAMERA)
    metric.update_state([gt, pr, box])
    result = metric.result()

    self.assertCountEqual(result.keys(), _ALL_METRIC_NAMES)

  def test_all_metrics_for_laser_returns_expected_number_of_results(self):
    gt, pr, box = _random_inputs(batch_size=3, num_points=14)

    metric = _lib.create_combined_metric(_lib.DEFAULT_CONFIG_LASER)
    metric.update_state([gt, pr, box])
    result = metric.result()

    self.assertCountEqual(result.keys(), _ALL_METRIC_NAMES)

  def test_all_metrics_for_all_returns_expected_number_of_results(self):
    gt, pr, box = _random_inputs(batch_size=3, num_points=15)

    metric = _lib.create_combined_metric(_lib.DEFAULT_CONFIG_ALL)
    metric.update_state([gt, pr, box])
    result = metric.result()

    self.assertCountEqual(result.keys(), _ALL_METRIC_NAMES)


class MissingIdsTest(tf.test.TestCase):

  def test_missing_ids_empty_if_nothing_is_missing(self):
    ids = tf.constant([0, 1, 2], dtype=tf.int32)
    missing = _lib.missing_ids(ids, count=3)
    self.assertAllEqual(missing, [])

  def test_missing_ids_is_correct(self):
    ids = tf.constant([1, 3, 5], dtype=tf.int32)
    missing = _lib.missing_ids(ids, count=6)
    self.assertAllEqual(missing, [0, 2, 4])


class ClosestKeypointDistanceTest(tf.test.TestCase):

  def test_distance_to_points_inside_box_is_zero(self):
    # 2 objects with 2 keypoints each:
    #   point p00 is inside box b1
    #   point p01 is inside box b2
    #   point p10 is outside any box
    #   point p11 is inside box b0
    kp = _data.KeypointsTensors(
        location=tf.constant(
            [
                [[1, 0, 0], [3, 0, 0]],
                [[5, 0, 0], [7, 0, 0]],
            ],
            dtype=tf.float32,
        ),
        visibility=tf.constant([[2, 2], [2, 2]]),
    )
    # 3 boxes: b0 contains p11, b1 contains p00, b2 contains p01
    box = _data.BoundingBoxTensors(
        center=tf.constant([[7, 0, 0], [1, 0, 0], [3, 0, 0]], dtype=tf.float32),
        size=tf.constant([[2, 1, 1], [2, 1, 1], [2, 1, 1]], dtype=tf.float32),
        heading=tf.constant([0, 0, 0], dtype=tf.float32),
    )

    dist = _lib.closest_keypoint_distance(box, kp)

    # dist[i, j] is a distance between box[i] and object[j]
    self.assertAllClose(dist, [[3, 0], [0, 3], [0, 1]])

  def test_distance_takes_into_account_visibility_of_points(self):
    # 2 objects with 2 keypoints each:
    #   point p00 is inside box b1
    #   point p01 is inside box b2 (but it is invisible)
    #   point p10 is outside any box
    #   point p11 is inside box b0
    kp = _data.KeypointsTensors(
        location=tf.constant(
            [
                [[1, 0, 0], [3, 0, 0]],
                [[5, 0, 0], [7, 0, 0]],
            ],
            dtype=tf.float32,
        ),
        visibility=tf.constant([[2, 0], [2, 2]]),
    )
    # 3 boxes: b0 contains p11, b1 contains p00, b2 contains p01
    box = _data.BoundingBoxTensors(
        center=tf.constant([[7, 0, 0], [1, 0, 0], [3, 0, 0]], dtype=tf.float32),
        size=tf.constant([[2, 1, 1], [2, 1, 1], [2, 1, 1]], dtype=tf.float32),
        heading=tf.constant([0, 0, 0], dtype=tf.float32),
    )

    dist = _lib.closest_keypoint_distance(box, kp)

    # dist[i, j] is a distance between box[i] and object[j]
    self.assertAllClose(dist, [[5, 0], [0, 3], [1, 1]])


def _keypoint_coordinate_pairs(
    m: _lib.PoseEstimationPair, keypoint_index: int = 0, coord_index: int = 0
):
  gt = m.gt.keypoints.location[:, keypoint_index, coord_index]
  pr = m.pr.keypoints.location[:, keypoint_index, coord_index]
  return [(g, p) for g, p in zip(gt.numpy().tolist(), pr.numpy().tolist())]


class MeanErrorMatchingFnTest(tf.test.TestCase):

  def test_works_correctly_for_the_trivial_unambiguous_case(self):
    # A match mask for 3 ground truth and 3 predicted boxes.
    candidates = tf.constant([[0, 0, 1], [0, 1, 0], [1, 0, 0]]) > 0
    # All objects have 2 keypoints
    gt = _data.KeypointsTensors(
        location=tf.constant(
            [
                [[1, 0, 0], [3, 0, 0]],
                [[5, 0, 0], [7, 0, 0]],
                [[9, 0, 0], [11, 0, 0]],
            ],
            dtype=tf.float32,
        ),
        visibility=tf.constant([[2, 2], [2, 2], [2, 2]]),
    )
    # All prediction points are shifted relative to the corresponding ground
    # truth by [0,1,0].
    pr = _data.KeypointsTensors(
        location=tf.constant(
            [
                [[9, 1, 0], [11, 1, 0]],
                [[5, 1, 0], [7, 1, 0]],
                [[1, 1, 0], [3, 1, 0]],
            ],
            dtype=tf.float32,
        ),
        visibility=tf.constant([[2, 2], [2, 2], [2, 2]]),
    )
    params = dict(mismatch_penalty=2.0)

    m = _lib.mean_error_matching(gt, pr, candidates, **params)

    self.assertAllEqual(m.gt, [0, 1, 2])
    self.assertAllEqual(m.pr, [2, 1, 0])

  def test_resolves_ambiguous_cases_using_mean_distance_between_keypoints(self):
    # A match mask for 3 ground truth and 3 predicted boxes, each having two
    # potential matching objects.
    mask = tf.constant([[1, 1, 0], [0, 1, 1], [1, 0, 1]]) > 0
    # All objects have 2 keypoints
    gt = _data.KeypointsTensors(
        location=tf.constant(
            [
                [[1, 0, 0], [3, 0, 0]],
                [[5, 0, 0], [7, 0, 0]],
                [[9, 0, 0], [11, 0, 0]],
            ],
            dtype=tf.float32,
        ),
        visibility=tf.constant([[2, 2], [2, 2], [2, 2]]),
    )
    # Prediction points are shifted slightly to resolve ambiguity.
    pr = _data.KeypointsTensors(
        location=tf.constant(
            [
                [[1, 0.1, 0], [3, 0.1, 0]],
                [[5, 0.1, 0], [7, 0.1, 0]],
                [[9, 0.1, 0], [11, 0.1, 0]],
            ],
            dtype=tf.float32,
        ),
        visibility=tf.constant([[2, 2], [2, 2], [2, 2]]),
    )
    params = dict(mismatch_penalty=2.0)

    m = _lib.mean_error_matching(gt, pr, mask, **params)

    self.assertAllEqual(m.gt, [0, 1, 2])
    self.assertAllEqual(m.pr, [0, 1, 2])

  def test_clips_distance_to_resolve_ambiguity(self):
    # A match mask for 3 ground truth and 3 predicted boxes, each having two
    # potential matching objects.
    mask = tf.constant([[1, 1, 0], [0, 1, 1], [1, 0, 1]]) > 0
    # All objects have 2 keypoints
    gt = _data.KeypointsTensors(
        location=tf.constant(
            [
                [[1, 0, 0], [3, 0, 0]],
                [[5, 0, 0], [7, 0, 0]],
                [[9, 0, 0], [11, 0, 0]],
            ],
            dtype=tf.float32,
        ),
        visibility=tf.constant([[2, 2], [2, 2], [2, 2]]),
    )
    # One predicted keypoint shifted by a lot to break the matching without
    # clipping.
    pr = _data.KeypointsTensors(
        location=tf.constant(
            [
                [[1, 1.5, 0], [3, 1.5, 0]],
                [[5, 0.1, 0], [7, 0.1, 0]],
                [[9, 0.1, 0], [11, 100, 0]],
            ],
            dtype=tf.float32,
        ),
        visibility=tf.constant([[2, 2], [2, 2], [2, 2]]),
    )
    params = dict(mismatch_penalty=2.0)

    m = _lib.mean_error_matching(gt, pr, mask, **params)

    self.assertAllEqual(m.gt, [0, 1, 2])
    self.assertAllEqual(m.pr, [0, 1, 2])

  def test_takes_into_account_visibility_of_individual_keypoints(self):
    # A match mask for 3 ground truth and 3 predicted boxes, each having two
    # potential matching objects.
    mask = tf.constant([[1, 1, 1], [1, 1, 1], [1, 1, 1]]) > 0
    # All objects have 2 keypoints
    gt = _data.KeypointsTensors(
        location=tf.constant(
            [
                [[1, 0, 0], [3, 0, 0]],
                [[5, 0, 0], [7, 0, 0]],
                [[9, 0, 0], [11, 0, 0]],
            ],
            dtype=tf.float32,
        ),
        visibility=tf.constant([[2, 0], [2, 2], [0, 2]]),
    )
    # Keypoints with mismatched visibility should incure max_distance penalty.
    pr = _data.KeypointsTensors(
        location=tf.constant(
            [
                [[1, 2.0, 0], [3, 0, 0]],
                [[5, 0.1, 0], [7, 0.1, 0]],
                [[1, 0, 0], [11, 0.1, 0]],
            ],
            dtype=tf.float32,
        ),
        visibility=tf.constant([[0, 2], [2, 2], [0, 2]]),
    )
    params = dict(mismatch_penalty=2.0)

    m = _lib.mean_error_matching(gt, pr, mask, **params)

    # Ground truth object #0 didn't match any objects because visibility
    # mismatch.
    self.assertAllEqual(m.gt, [1, 2])
    self.assertAllEqual(m.pr, [1, 2])

  def test_omits_objects_without_matches(self):
    # A match mask for 3 ground truth and 3 predicted boxes, each having two
    # potential matching objects.
    mask = tf.constant([[0, 0, 0], [1, 0, 0], [0, 1, 0]]) > 0
    # All objects have 2 keypoints
    gt = _data.KeypointsTensors(
        location=tf.constant(
            [
                [[1, 0, 0], [3, 0, 0]],
                [[5, 0, 0], [7, 0, 0]],
                [[9, 0, 0], [11, 0, 0]],
            ],
            dtype=tf.float32,
        ),
        visibility=tf.constant([[2, 2], [2, 2], [2, 2]]),
    )
    # Keypoints with mismatched visibility should incure max_distance penalty.
    pr = _data.KeypointsTensors(
        location=tf.constant(
            [
                [[5, 0, 0], [7, 0, 0]],
                [[9, 0, 0], [11, 0, 0]],
                [[1, 0, 0], [3, 0, 0]],
            ],
            dtype=tf.float32,
        ),
        visibility=tf.constant([[2, 2], [2, 2], [2, 2]]),
    )
    params = dict(mismatch_penalty=2.0)

    m = _lib.mean_error_matching(gt, pr, mask, **params)

    self.assertAllEqual(m.gt, [1, 2])
    self.assertAllEqual(m.pr, [0, 1])

  def test_a_prediction_can_be_assigned_only_to_a_single_ground_truth(self):
    mask = tf.constant([[1], [1]]) > 0
    # Two objects with 1 keypoint each.
    gt = _data.KeypointsTensors(
        location=tf.constant(
            [[[1, 0, 0]], [[2, 0, 0]]],
            dtype=tf.float32,
        ),
        visibility=tf.constant([[2], [2]]),
    )
    # One objects with 1 keypoint.
    pr = _data.KeypointsTensors(
        location=tf.constant([[[1.9, 0, 0]]], dtype=tf.float32),
        visibility=tf.constant([[2]]),
    )
    params = dict(mismatch_penalty=2.0)

    m = _lib.mean_error_matching(gt, pr, mask, **params)

    self.assertAllEqual(m.gt, [1])
    self.assertAllEqual(m.pr, [0])


class MatchersTest(parameterized.TestCase, tf.test.TestCase):

  @parameterized.named_parameters(
      ('mean_matcher', _lib.MeanErrorMatcher()),
      ('cpp_matcher', _lib.CppMatcher()),
  )
  def test_keeps_gt_as_is(self, matcher: _lib.BaseMatcher):
    # 3 objects with 4 keypoints each:
    gt_kp = _data.KeypointsTensors(
        location=tf.constant(
            [
                [[4, 3, 0], [3, 5, 0], [6, 5, 0], [4, 8, 0]],
                [[15, 3, 0], [15, 9, 0], [17, 6, 0], [13, 6, 0]],
                [[20, 20, 0], [16, 18, 0], [19, 17, 0], [16, 21, 0]],
            ],
            dtype=tf.float32,
        ),
        visibility=tf.constant([[2, 2, 2, 2], [2, 2, 2, 2], [2, 2, 2, 2]]),
    )
    gt_box = _data.BoundingBoxTensors(
        center=tf.constant(
            [[4.5, 5.5, 0], [15, 6, 0], [18, 19, 0]], dtype=tf.float32
        ),
        size=tf.constant([[5, 7, 1], [6, 8, 1], [6, 6, 1]], dtype=tf.float32),
        heading=tf.constant([0, 0, 0], dtype=tf.float32),
    )
    gt = _data.PoseEstimationTensors(keypoints=gt_kp, box=gt_box)

    # Matching GT to itself should return exactly the same set of keypoints and
    # boxes.
    m = matcher.reorder(_lib.PoseEstimationPair(gt, gt))

    # Verify that we preserved the order of ground truth and "predictions",
    # which are the same as the ground truth in this case.
    self.assertAllClose(m.gt.keypoints.location, gt.keypoints.location)
    self.assertAllClose(m.gt.keypoints.visibility, gt.keypoints.visibility)
    self.assertAllClose(m.gt.box.center, gt.box.center)
    self.assertAllClose(m.gt.box.size, gt.box.size)
    self.assertAllClose(m.gt.box.heading, gt.box.heading)
    self.assertAllClose(m.gt.keypoints.location, m.pr.keypoints.location)
    self.assertAllClose(m.gt.keypoints.visibility, m.pr.keypoints.visibility)
    self.assertAllClose(m.gt.box.center, m.pr.box.center)
    self.assertAllClose(m.gt.box.size, m.pr.box.size)
    self.assertAllClose(m.gt.box.heading, m.pr.box.heading)

  @parameterized.named_parameters(
      ('mean_matcher', _lib.MeanErrorMatcher()),
      ('cpp_matcher', _lib.CppMatcher()),
  )
  def test_works_when_there_no_predictions(self, matcher: _lib.BaseMatcher):
    # This kind of dummy prediction is used when there no prected objects in a
    # frame, while there is a ground truth for it.
    # Ground truth: one object with one keypoint.
    gt_kp = _data.KeypointsTensors(
        location=tf.constant(
            [[[1, 1, 1]]],
            dtype=tf.float32,
        ),
        visibility=tf.constant([[2]]),
    )
    gt_box = _data.BoundingBoxTensors(
        center=tf.constant([[1, 1, 1]], dtype=tf.float32),
        size=tf.constant([[1, 1, 1]], dtype=tf.float32),
        heading=tf.constant([0.0], dtype=tf.float32),
    )
    gt = _data.PoseEstimationTensors(keypoints=gt_kp, box=gt_box)
    # Predicted keypoints have dim_0 = 0
    pr = _data.PoseEstimationTensors(
        keypoints=_data.KeypointsTensors(
            location=tf.zeros([0, 1, 3], dtype=tf.float32),
            visibility=tf.zeros([0, 1], dtype=tf.int32),
        ),
        box=_data.BoundingBoxTensors(
            center=tf.zeros([0, 3], dtype=tf.float32),
            size=tf.zeros([0, 3], dtype=tf.float32),
            heading=tf.zeros([0], dtype=tf.float32),
        ),
    )

    m = matcher.reorder(_lib.PoseEstimationPair(gt, pr))

    # Spot checking coordinates to assert the order of objects.
    self.assertAllEqual(m.gt.keypoints.location, [[[1, 1, 1]]])
    self.assertAllEqual(m.gt.box.center, [[1, 1, 1]])
    self.assertAllEqual(m.pr.keypoints.location, [[[0, 0, 0]]])
    self.assertAllEqual(m.pr.box.center, [[0, 0, 0]])

  @parameterized.named_parameters(
      ('mean_matcher', _lib.MeanErrorMatcher()),
      ('cpp_matcher', _lib.CppMatcher()),
  )
  def test_no_penalty_for_not_labeled_objects(self, matcher: _lib.BaseMatcher):
    # If all keypoints for a GT object are invisible it means that no keypoints
    # were labeled. If a prediction keypoint is close enough to the bounding
    # box of such object, the prediction should not be taken into account for
    # metric computation and will be absent in the matching results.
    # In this example we have two ground truth objects - one without labeled
    # keypoints and one with labeled keypoints.
    gt_kp = _data.KeypointsTensors(
        location=tf.constant(
            [[[0, 0, 0], [0, 0, 0]], [[1, 2, 3], [3, 2, 1]]],
            dtype=tf.float32,
        ),
        visibility=tf.constant([[0, 0], [2, 2]]),
    )
    gt_box = _data.BoundingBoxTensors(
        center=tf.constant([[0, 0, 0], [2, 2, 2]], dtype=tf.float32),
        size=tf.constant([[1, 1, 1], [2, 2, 2]], dtype=tf.float32),
        heading=tf.constant([0.0, 0.0], dtype=tf.float32),
    )
    gt = _data.PoseEstimationTensors(keypoints=gt_kp, box=gt_box)
    # Two predicted objects with two points each:
    #  obj0: first keypoint is inside gt box #0, second outside
    #  obj1: matches labeled gt obj #1.
    pr = _data.PoseEstimationTensors(
        keypoints=_data.KeypointsTensors(
            location=tf.constant(
                [
                    [[0.4, 0.3, 0], [1.5, 1.0, 0.0]],
                    [[1.1, 2.0, 2.9], [2.9, 2.0, 1.1]],
                ],
                dtype=tf.float32,
            ),
            visibility=tf.constant([[2, 2], [2, 2]], dtype=tf.int32),
        ),
        # pr.box is used only by the cpp matcher.
        box=gt_box,
    )

    m = matcher.reorder(_lib.PoseEstimationPair(gt, pr))

    # Ground truth keypoints were copied from the prediction.
    self.assertAllClose(
        m.gt.keypoints.location, [[[1.0, 2.0, 3.0], [3.0, 2.0, 1.0]]]
    )
    self.assertAllClose(
        m.pr.keypoints.location, [[[1.1, 2.0, 2.9], [2.9, 2.0, 1.1]]]
    )
    self.assertAllEqual(m.gt.box.center, [[2, 2, 2]])

  def test_assosiates_unlabeled_gt_boxes_with_pr_with_most_points_inside(self):
    # Specific to the MeanErrorMatcher.
    # Two ground truth boxes:
    #  0 - labeled, but contains only one predicted keypoint.
    #  1 - unlabled, contains all predicted keypoints.
    gt_kp = _data.KeypointsTensors(
        location=tf.constant(
            [[[1, 1, 0], [2, 2, 0]], [[0, 0, 0], [0, 0, 0]]],
            dtype=tf.float32,
        ),
        visibility=tf.constant([[2, 2], [0, 0]]),
    )
    gt_box = _data.BoundingBoxTensors(
        center=tf.constant([[1.5, 1.5, 0], [3.5, 3.5, 0]], dtype=tf.float32),
        size=tf.constant([[1.1, 1.1, 1.1], [3.1, 3.1, 1.1]], dtype=tf.float32),
        heading=tf.constant([0.0, 0.0], dtype=tf.float32),
    )
    gt = _data.PoseEstimationTensors(keypoints=gt_kp, box=gt_box)
    # Predicted one object.
    pr = _data.PoseEstimationTensors(
        keypoints=_data.KeypointsTensors(
            location=tf.constant([[[2, 2, 0], [4, 4, 0]]], dtype=tf.float32),
            visibility=tf.constant([[2, 2]], dtype=tf.int32),
        )
    )

    matcher = _lib.MeanErrorMatcher()
    m = matcher.reorder(_lib.PoseEstimationPair(gt, pr))

    # The first ground truth object is a false negative, the second (unlabeled)
    # is associated with the prediction.
    self.assertAllClose(m.gt.keypoints.location, [[[1, 1, 0], [2, 2, 0]]])
    self.assertAllEqual(m.gt.box.center, [[1.5, 1.5, 0]])
    self.assertAllClose(m.pr.keypoints.location, [[[0, 0, 0], [0, 0, 0]]])
    self.assertIsNone(m.pr.box)

  @parameterized.named_parameters(
      (
          'mean_matcher',
          _lib.MeanErrorMatcher(
              _lib.MeanErrorMatcherConfig(max_closest_keypoint_distance=5.0)
          ),
      ),
      (
          'cpp_matcher',
          _lib.CppMatcher(_lib.CppMatcherConfig(iou_threshold=0.1)),
      ),
  )
  def test_matches_prediction_to_ground_truth_correctly(self, matcher):
    # 3 objects with 4 keypoints each:
    gt_kp = _data.KeypointsTensors(
        location=tf.constant(
            [
                [[4, 3, 0], [3, 5, 0], [6, 5, 0], [4, 8, 0]],
                [[15, 3, 0], [15, 9, 0], [17, 6, 0], [13, 6, 0]],
                [[20, 20, 0], [16, 18, 0], [19, 17, 0], [16, 21, 0]],
            ],
            dtype=tf.float32,
        ),
        visibility=tf.constant([[2, 2, 2, 2], [2, 2, 2, 2], [2, 2, 2, 2]]),
    )
    gt_box = _data.BoundingBoxTensors(
        center=tf.constant(
            [[4.5, 5.5, 0], [15, 6, 0], [18, 19, 0]], dtype=tf.float32
        ),
        size=tf.constant([[5, 7, 1], [6, 8, 1], [6, 6, 1]], dtype=tf.float32),
        heading=tf.constant([0, 0, 0], dtype=tf.float32),
    )
    gt = _data.PoseEstimationTensors(keypoints=gt_kp, box=gt_box)
    # Only 2 out of 3 predicted objects match ground truth:
    #   pr0: close enough to gt0 and gt1, but error is smaller for gt1
    #   pr1: no match
    #   pr2: close enough to gt0 with max_distance=5
    pr_kp = _data.KeypointsTensors(
        location=tf.constant(
            [
                [[10, 5, 0], [14, 8, 0], [12, 3, 0], [11, 8, 0]],
                [[3, 22, 0], [6, 21, 0], [3, 19, 0], [5, 18, 0]],
                [[6, 6, 0], [3, 6, 0], [5, 3, 0], [5, 8, 0]],
            ],
            dtype=tf.float32,
        ),
        visibility=tf.constant([[2, 2, 2, 2], [2, 2, 2, 2], [2, 2, 2, 2]]),
    )
    # Should not be used by the matcher, but we specify boxes consistent with
    # keypoints anyway.
    pr_box = _data.BoundingBoxTensors(
        center=tf.constant(
            [[12.0, 5.5, 0], [4.5, 19.5, 0], [4, 6, 0]], dtype=tf.float32
        ),
        size=tf.constant([[6, 7, 1], [5, 7, 1], [8, 6, 1]], dtype=tf.float32),
        heading=tf.constant([0, 0, 0], dtype=tf.float32),
    )
    pr = _data.PoseEstimationTensors(keypoints=pr_kp, box=pr_box)

    m = matcher.reorder(_lib.PoseEstimationPair(gt, pr))

    # Spot check first coordinate of location if they were reordered correctly.
    # Reorder changes only order of objects. In this example objects have unique
    # x coordinate of the first keypoint.
    tp_02 = (4.0, 6.0)  # gt[0] = pr[2]
    tp_10 = (15.0, 10.0)  # gt[1] = pr[0]
    fn_2 = (20.0, 0.0)  # gt[2]
    fp_1 = (0.0, 3.0)  # pr[1]
    self.assertCountEqual(
        _keypoint_coordinate_pairs(m, keypoint_index=0, coord_index=0),
        [tp_02, tp_10, fn_2, fp_1],
    )

  @parameterized.named_parameters(
      ('mean_matcher', _lib.MeanErrorMatcher()),
      ('cpp_matcher', _lib.CppMatcher()),
  )
  def test_reorders_pr_to_match_gr(self, matcher: _lib.BaseMatcher):
    # 3 objects with 1 keypoint each:
    gt_kp = _data.KeypointsTensors(
        location=tf.constant(
            [[[1, 1, 1]], [[4, 4, 4]], [[7, 7, 7]]],
            dtype=tf.float32,
        ),
        visibility=tf.constant([[2], [2], [2]]),
    )
    gt_box = _data.BoundingBoxTensors(
        center=gt_kp.location[:, 0, :],
        size=tf.constant([[1.1] * 3, [1.2] * 3, [1.3] * 3]),
        heading=tf.constant([0.1, 0.2, 0.3], dtype=tf.float32),
    )
    gt = _data.PoseEstimationTensors(keypoints=gt_kp, box=gt_box)
    # Predictions are the same, but reordered: 2, 0, 1
    reorder = lambda t: tf.gather(t, [2, 0, 1])
    pr_kp = _data.KeypointsTensors(
        location=reorder(gt_kp.location), visibility=reorder(gt_kp.visibility)
    )
    pr_box = _data.BoundingBoxTensors(
        center=reorder(gt_box.center),
        size=reorder(gt_box.size),
        heading=reorder(gt_box.heading),
    )
    pr = _data.PoseEstimationTensors(keypoints=pr_kp, box=pr_box)

    m = matcher.reorder(_lib.PoseEstimationPair(gt, pr))

    self.assertCountEqual(
        m.gt.keypoints.location.numpy().flatten(),
        gt.keypoints.location.numpy().flatten(),
    )
    self.assertCountEqual(
        m.gt.keypoints.visibility.numpy().flatten(),
        gt.keypoints.visibility.numpy().flatten(),
    )
    self.assertCountEqual(
        m.gt.box.center.numpy().flatten(), gt.box.center.numpy().flatten()
    )
    self.assertCountEqual(
        m.gt.box.size.numpy().flatten(), gt.box.size.numpy().flatten()
    )
    self.assertCountEqual(
        m.gt.box.heading.numpy().flatten(), gt.box.heading.numpy().flatten()
    )
    self.assertAllClose(m.gt.keypoints.location, m.pr.keypoints.location)
    self.assertAllClose(m.gt.keypoints.visibility, m.pr.keypoints.visibility)
    self.assertAllClose(m.gt.box.center, m.pr.box.center)
    self.assertAllClose(m.gt.box.size, m.pr.box.size)
    self.assertAllClose(m.gt.box.heading, m.pr.box.heading)

  @parameterized.named_parameters(
      ('mean_matcher', _lib.MeanErrorMatcher()),
      ('cpp_matcher', _lib.CppMatcher()),
  )
  def test_inexact_pr_to_gr_match(self, matcher: _lib.BaseMatcher):
    # 3 objects with 1 keypoint each:
    gt_kp = _data.KeypointsTensors(
        location=tf.constant(
            [[[1, 1, 1]], [[4, 4, 4]], [[7, 7, 7]]],
            dtype=tf.float32,
        ),
        visibility=tf.constant([[2], [2], [2]]),
    )
    gt_box = _data.BoundingBoxTensors(
        center=gt_kp.location[:, 0, :],
        size=tf.constant([[1.1] * 3, [1.2] * 3, [1.3] * 3]),
        heading=tf.constant([0.1, 0.2, 0.3], dtype=tf.float32),
    )
    gt = _data.PoseEstimationTensors(keypoints=gt_kp, box=gt_box)
    # Predicted boxes are similar to the gt ordered: 2, 0, 1
    pr_kp = _data.KeypointsTensors(
        location=tf.constant(
            [[[7, 7, 7]], [[1, 1, 1]], [[4, 4, 4]]],
            dtype=tf.float32,
        ),
        visibility=tf.constant([[2], [2], [2]]),
    )
    pr_box = _data.BoundingBoxTensors(
        center=tf.constant([[7.1, 7.1, 7.1], [0.9, 1.1, 1.1], [3.9, 4, 4.1]]),
        size=tf.constant([[1.2] * 3, [1.2] * 3, [1.2] * 3], dtype=tf.float32),
        heading=tf.constant([0.1, 0.1, 0.1], dtype=tf.float32),
    )
    pr = _data.PoseEstimationTensors(keypoints=pr_kp, box=pr_box)

    m = matcher.reorder(_lib.PoseEstimationPair(gt, pr))

    # Spot checking coordinates to assert the order of objects.
    self.assertCountEqual(
        _keypoint_coordinate_pairs(m, keypoint_index=0, coord_index=0),
        [(1.0, 1.0), (4.0, 4.0), (7.0, 7.0)],
    )
    self.assertCountEqual(m.gt.box.center[:, 0].numpy(), [1, 4, 7])

    self.assertCountEqual(
        m.pr.box.center[:, 0].numpy(), np.asarray([0.9, 3.9, 7.1], np.float32)
    )
    self.assertAllClose(m.gt.keypoints.location, m.pr.keypoints.location)

  @parameterized.named_parameters(
      ('mean_matcher', _lib.MeanErrorMatcher()),
      ('cpp_matcher', _lib.CppMatcher()),
  )
  def test_appends_false_negatives_and_false_positives(
      self, matcher: _lib.BaseMatcher
  ):
    # 2 object with 1 keypoint each:
    gt = _data.PoseEstimationTensors(
        keypoints=_data.KeypointsTensors(
            location=tf.constant(
                [[[1.5, 1.5, 1.5]], [[3.5, 3.5, 3.5]]], dtype=tf.float32
            ),
            visibility=tf.constant([[2], [2]]),
        ),
        box=_data.BoundingBoxTensors(
            center=tf.constant([[1, 1, 1], [3, 3, 3]], dtype=tf.float32),
            size=tf.constant([[1, 1, 1], [1, 1, 1]], dtype=tf.float32),
            heading=tf.constant([0.1, 0.1], dtype=tf.float32),
        ),
    )
    # 2 predicted boxes: one far away and one match.
    pr = _data.PoseEstimationTensors(
        keypoints=_data.KeypointsTensors(
            location=tf.constant(
                [[[10.5, 10.5, 10.5]], [[3.4, 3.4, 3.4]]], dtype=tf.float32
            ),
            visibility=tf.constant([[2], [2]]),
        ),
        box=_data.BoundingBoxTensors(
            center=tf.constant(
                [[10, 10, 10], [2.9, 2.9, 2.9]], dtype=tf.float32
            ),
            size=tf.constant([[1, 1, 1], [1, 1, 1]], dtype=tf.float32),
            heading=tf.constant([0.1, 0.2], dtype=tf.float32),
        ),
    )

    m = matcher.reorder(_lib.PoseEstimationPair(gt, pr))

    # The order of objects: true positives, false negatives, false positives.
    self.assertAllClose(m.gt.keypoints.visibility, [[2], [2], [0]])
    self.assertAllClose(m.pr.keypoints.visibility, [[2], [0], [2]])
    # Check just the x coordinate to verify the order.
    self.assertAllClose(m.gt.box.center[:, 0], [3, 1, 0])
    self.assertAllClose(m.pr.box.center[:, 0], [2.9, 0, 10])

  @parameterized.named_parameters(
      ('mean_matcher', _lib.MeanErrorMatcher()),
      ('cpp_matcher', _lib.CppMatcher()),
  )
  def test_padds_correctly_even_if_pad_size_is_greater_than_orig_size(
      self, matcher: _lib.BaseMatcher
  ):
    # Single object
    gt = _data.PoseEstimationTensors(
        keypoints=_data.KeypointsTensors(
            location=tf.constant([[[1.5, 2.5, 3.5]]], dtype=tf.float32),
            visibility=tf.constant([[2]]),
        ),
        box=_data.BoundingBoxTensors(
            center=tf.constant([[1.0, 2.0, 3.0]], dtype=tf.float32),
            size=tf.constant([[1, 1, 1]], dtype=tf.float32),
            heading=tf.constant([0.1], dtype=tf.float32),
        ),
    )
    # Two objects with the same box in the same center as the ground truth (GT).
    # For CppMatcher none of the predicted (PR) objects should match the GT
    # because PR bounding box IoU with GT < 0.5.
    # For MeanMatcher - because the mean distance between keypoints > 0.25.
    pr = _data.PoseEstimationTensors(
        keypoints=_data.KeypointsTensors(
            location=tf.constant([[[2, 3, 4]], [[0, 1, 2]]], dtype=tf.float32),
            visibility=tf.constant([[2], [2]]),
        ),
        box=_data.BoundingBoxTensors(
            center=tf.constant([[1, 2, 3], [1, 2, 3]], dtype=tf.float32),
            size=tf.constant([[5, 4, 5], [5, 4, 6]], dtype=tf.float32),
            heading=tf.constant([0.001, 0.001], dtype=tf.float32),
        ),
    )

    m = matcher.reorder(_lib.PoseEstimationPair(gt, pr))

    # There is no match between gt and pr objects, so ground truth needs to be
    # padded with more objects that it had. Previously `_reorder` used
    # `tf.zeros_like(tensor[:num])`, which lead to incorrect shapes if
    # tensor.shape[0] < num.
    self.assertEqual(m.gt.box.center.shape, (3, 3))
    self.assertEqual(m.pr.box.center.shape, (3, 3))

  @parameterized.named_parameters(
      ('mean_matcher', _lib.MeanErrorMatcher()),
      ('cpp_matcher', _lib.CppMatcher()),
  )
  def test_case_with_occlusion_and_unlabled_box(
      self, matcher: _lib.BaseMatcher
  ):
    gt = _data.PoseEstimationTensors(
        keypoints=_data.KeypointsTensors(
            location=tf.constant(
                [[[1, 2, 3], [3, 2, 1]], [[0, 0, 0], [0, 0, 0]]],
                dtype=tf.float32,
            ),
            visibility=tf.constant([[0, 2], [0, 0]]),
        ),
        box=_data.BoundingBoxTensors(
            center=tf.constant([[2, 2, 2], [0, 0, 0]], dtype=tf.float32),
            size=tf.constant([[2, 2, 2], [1, 1, 1]], dtype=tf.float32),
            heading=tf.constant([0.0, 0.0], dtype=tf.float32),
        ),
    )
    # Predicted keypoints have dim_0 = 0
    pr = _data.PoseEstimationTensors(
        keypoints=_data.KeypointsTensors(
            location=tf.constant(
                [
                    [[1.1, 2.0, 2.9], [2.9, 2.0, 1.1]],
                ],
                dtype=tf.float32,
            ),
            visibility=tf.constant([[0, 2]], dtype=tf.int32),
        ),
        # Box used only by the CPP matcher.
        box=_data.BoundingBoxTensors(
            center=tf.constant([[2.1, 1.9, 1.9]], dtype=tf.float32),
            size=tf.constant([[1.9, 1.9, 2.1]], dtype=tf.float32),
            heading=tf.constant([0.0], dtype=tf.float32),
        ),
    )

    m = matcher.reorder(_lib.PoseEstimationPair(gt, pr))

    # Ground truth keypoints were copied from the prediction.
    self.assertAllClose(
        m.gt.keypoints.location, [[[1.0, 2.0, 3.0], [3.0, 2.0, 1.0]]]
    )
    self.assertAllClose(
        m.pr.keypoints.location, [[[1.1, 2.0, 2.9], [2.9, 2.0, 1.1]]]
    )
    self.assertAllEqual(m.gt.box.center, [[2, 2, 2]])

  def test_matches_pr_within_25cm_from_gt(self):
    # Two ground truth objects with one keypoint each. The keypoints are within
    # 1cm from the bounding box boundary.
    gt = _data.PoseEstimationTensors(
        keypoints=_data.KeypointsTensors(
            location=tf.constant([[[1, 1, 1]], [[2, 2, 2]]], dtype=tf.float32),
            visibility=tf.constant([[2], [2]]),
        ),
        box=_data.BoundingBoxTensors(
            center=tf.constant(
                [[0.5, 0.5, 0.5], [2.5, 2.5, 2.5]], dtype=tf.float32
            ),
            size=tf.constant(
                [[1.02, 1.02, 1.02], [1.02, 1.02, 1.02]], dtype=tf.float32
            ),
            heading=tf.constant([0.0, 0.0], dtype=tf.float32),
        ),
    )
    # Just to verify the boundaries of the box.
    self.assertAllClose(
        gt.box.min_corner, [[-0.01, -0.01, -0.01], [1.99, 1.99, 1.99]]
    )
    self.assertAllClose(
        gt.box.max_corner, [[1.01, 1.01, 1.01], [3.01, 3.01, 3.01]]
    )

    # Two predicted objects with one keypoint each:
    #   - the first object is <25cm from the corresponding gt box and <25cm from
    #     the corresponding keypoint.
    #   - the second object is 30cm from the closest gt box
    pr = _data.PoseEstimationTensors(
        keypoints=_data.KeypointsTensors(
            location=tf.constant(
                [
                    [[1.01 + 0.23, 1.01, 1.01]],
                    [[1.99 - 0.30, 1.99, 1.99]],
                ],
                dtype=tf.float32,
            ),
            visibility=tf.constant([[2], [2]], dtype=tf.int32),
        )
    )

    matcher = _lib.MeanErrorMatcher(
        _lib.MeanErrorMatcherConfig(max_closest_keypoint_distance=0.25)
    )
    m = matcher.reorder(_lib.PoseEstimationPair(gt, pr))

    # Order of matched objects: TP, FN, FP.
    # For ground truth FP - padded with zeros, for predictions FN - padded.
    self.assertAllClose(
        m.gt.keypoints.location, [[[1, 1, 1]], [[2, 2, 2]], [[0, 0, 0]]]
    )
    self.assertAllClose(
        m.pr.keypoints.location,
        [[[1.24, 1.01, 1.01]], [[0, 0, 0]], [[1.69, 1.99, 1.99]]],
    )

  def test_overlapping_labeled_and_unlabeled_gt_assigned_correctly(self):
    # Specific to the MeanErrorMatcher and using the `invisible_error` larger
    # than the prediction error.
    # Two ground truth objects:
    #  obj0 has keypoints
    #  obj1 has no keypoints, but has a box overlapping with obj0.
    gt = _data.PoseEstimationTensors(
        keypoints=_data.KeypointsTensors(
            location=tf.constant(
                [[[1.8, 2, 2.2], [2.2, 2, 1.8]], [[0, 0, 0], [0, 0, 0]]],
                dtype=tf.float32,
            ),
            visibility=tf.constant([[2, 2], [0, 0]]),
        ),
        box=_data.BoundingBoxTensors(
            center=tf.constant([[2, 2, 2], [2.1, 1.9, 2.0]], dtype=tf.float32),
            size=tf.constant([[2, 2, 2], [1.9, 2.0, 2.1]], dtype=tf.float32),
            heading=tf.constant([0.0, 0.0], dtype=tf.float32),
        ),
    )
    # Predicted keypoints matche the obj0, but also within the box of obj1.
    pr = _data.PoseEstimationTensors(
        keypoints=_data.KeypointsTensors(
            location=tf.constant(
                [
                    [[1.85, 2.05, 2.25], [2.25, 2.05, 1.85]],
                ],
                dtype=tf.float32,
            ),
            visibility=tf.constant([[2, 2]], dtype=tf.int32),
        ),
        # Box used only by the CPP matcher.
        box=_data.BoundingBoxTensors(
            center=tf.constant([[2.1, 1.9, 1.9]], dtype=tf.float32),
            size=tf.constant([[1.9, 1.9, 2.1]], dtype=tf.float32),
            heading=tf.constant([0.0], dtype=tf.float32),
        ),
    )

    matcher = _lib.MeanErrorMatcher()
    m = matcher.reorder(_lib.PoseEstimationPair(gt, pr))

    # Ground truth keypoints were copied from the prediction.
    self.assertAllClose(
        m.gt.keypoints.location, [[[1.8, 2, 2.2], [2.2, 2, 1.8]]]
    )
    self.assertAllClose(
        m.pr.keypoints.location, [[[1.85, 2.05, 2.25], [2.25, 2.05, 1.85]]]
    )
    self.assertAllClose(m.gt.box.center, [[2, 2, 2]])
    self.assertAllClose(m.pr.box.center, [[2.1, 1.9, 1.9]])

  @parameterized.named_parameters(
      ('mean_matcher', _lib.MeanErrorMatcher()),
      ('cpp_matcher', _lib.CppMatcher()),
  )
  def test_overlapping_gt_assigned_correctly_to_itself(
      self, matcher: _lib.BaseMatcher
  ):
    # Two ground truth objects have overlapping bounding boxes and each of them
    # contains only one keypoint.
    gt = _data.PoseEstimationTensors(
        keypoints=_data.KeypointsTensors(
            location=tf.constant(
                [
                    [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                    [[20.72, -3.96, 1.26], [20.55, -3.86, 1.12]],
                ],
                dtype=tf.float32,
            ),
            visibility=tf.constant([[0, 0], [1, 2]]),
        ),
        box=_data.BoundingBoxTensors(
            center=tf.constant(
                [[20.78, -3.71, 0.78], [20.64, -4.21, 0.64]], dtype=tf.float32
            ),
            size=tf.constant(
                [[1.52, 1.40, 1.98], [1.08, 0.98, 1.69]], dtype=tf.float32
            ),
            heading=tf.constant([0.07, 0.02], dtype=tf.float32),
        ),
    )

    m = matcher.reorder(_lib.PoseEstimationPair(gt, gt))

    self.assertAllEqual(m.gt.keypoints.has_visible, [True])
    self.assertAllClose(
        m.gt.keypoints.location, [[[20.72, -3.96, 1.26], [20.55, -3.86, 1.12]]]
    )
    self.assertAllEqual(m.pr.keypoints.has_visible, [True])
    self.assertAllClose(
        m.pr.keypoints.location, [[[20.72, -3.96, 1.26], [20.55, -3.86, 1.12]]]
    )

  @parameterized.named_parameters(
      ('mean_matcher', _lib.MeanErrorMatcher()),
      ('cpp_matcher', _lib.CppMatcher()),
  )
  def test_no_penality_for_gt_with_all_occluded_keypoints(
      self, matcher: _lib.BaseMatcher
  ):
    gt = _data.PoseEstimationTensors(
        keypoints=_data.KeypointsTensors(
            location=tf.constant(
                [[[1.8, 2, 2.2], [2.2, 2, 1.8]]],
                dtype=tf.float32,
            ),
            visibility=tf.constant([[1, 1]]),
        ),
        box=_data.BoundingBoxTensors(
            center=tf.constant([[2, 2, 2]], dtype=tf.float32),
            size=tf.constant([[2, 2, 2]], dtype=tf.float32),
            heading=tf.constant([0.0], dtype=tf.float32),
        ),
    )
    # Predicted keypoints matche the obj0, but also within the box of obj1
    pr = _data.PoseEstimationTensors(
        keypoints=_data.KeypointsTensors(
            location=tf.constant(
                [
                    [[1.9, 1.9, 2.1], [2.1, 2.1, 1.9]],
                ],
                dtype=tf.float32,
            ),
            visibility=tf.constant([[2, 2]], dtype=tf.int32),
        ),
        # Box used only by the CPP matcher.
        box=_data.BoundingBoxTensors(
            center=tf.constant([[2.1, 1.9, 1.9]], dtype=tf.float32),
            size=tf.constant([[1.9, 1.9, 2.1]], dtype=tf.float32),
            heading=tf.constant([0.0], dtype=tf.float32),
        ),
    )

    m = matcher.reorder(_lib.PoseEstimationPair(gt, pr))

    # Prediction should be matched with the fully the only ground truth with
    # all keypoints occluded, which should be treated as none of them were
    # labeled - matching set should be empty.
    self.assertEqual(m.gt.keypoints.location.shape, [0, 2, 3])
    self.assertEqual(m.pr.keypoints.location.shape, [0, 2, 3])


class PemTest(tf.test.TestCase):

  def test_returns_mean_square_error_for_all_visible_keypoints(self):
    # batch_size = 3, num_points = 2
    gt = _data.KeypointsTensors(
        location=tf.constant([
            [[1.0, 1.0], [-1.0, -1.0]],
            [[2.0, 2.0], [-2.0, -2.0]],
            [[3.0, 3.0], [-3.0, -3.0]],
        ]),
        visibility=tf.constant([[2, 2], [2, 2], [2, 2]]),
    )
    # Predicted points are [1, 2, 3, 4, 5, 6] pixels away from ground truth.
    pr = _data.KeypointsTensors(
        location=tf.constant([
            [[1.0, 0.0], [1.0, -1.0]],
            [[2.0, -1.0], [2.0, -2.0]],
            [[3.0, -2.0], [-3.0, 3.0]],
        ]),
        visibility=tf.constant([[2, 2], [2, 2], [2, 2]]),
    )
    box = None  # is not used by the metric

    pem = _lib.PoseEstimationMetric(name='PEM', mismatch_penalty=100.0)
    pem.update_state([gt, pr, box])
    metrics = pem.result()

    self.assertNear(metrics['PEM'], (1 + 2 + 3 + 4 + 5 + 6) / 6, err=1e-5)

  def test_adds_penalty_for_mismatches(self):
    # Three objects with two keypoints:
    #  0 - match
    #  1 - false negative
    #  2 - false positive
    # Mismatched keypoints have visibility=0
    gt = _data.KeypointsTensors(
        location=tf.constant([
            [[1.0, 1.0], [-1.0, -1.0]],
            [[2.0, 2.0], [-2.0, -2.0]],
            [[0.0, 0.0], [0.0, 0.0]],
        ]),
        visibility=tf.constant([[2, 2], [2, 2], [0, 0]]),
    )
    pr = _data.KeypointsTensors(
        location=tf.constant([
            [[1.0, 0.0], [1.0, -1.0]],
            [[0.0, 0.0], [0.0, 0.0]],
            [[3.0, -2.0], [-3.0, 3.0]],
        ]),
        visibility=tf.constant([[2, 2], [0, 0], [2, 2]]),
    )
    box = None  # is not used by the metric

    pem = _lib.PoseEstimationMetric(name='PEM', mismatch_penalty=100.0)
    pem.update_state([gt, pr, box])
    metrics = pem.result()

    self.assertNear(metrics['PEM'], (1 + 2 + 4 * 100) / 6, err=1e-5)

  def test_correctly_predicted_invisible_points_are_not_penalized(self):
    gt = _data.KeypointsTensors(
        # Values for keypoints with visibility 0 could be anything.
        location=tf.constant([
            [[1.0, 1.0], [-1.0, -1.0]],
            [[2.0, 2.0], [666.0, 666.0]],
            [[777.0, 777.0], [3.0, 3.0]],
        ]),
        visibility=tf.constant([[2, 2], [2, 0], [0, 2]]),
    )
    pr = _data.KeypointsTensors(
        location=tf.constant([
            [[1.0, 0.0], [1.0, -1.0]],
            [[2.0, -1.0], [0.0, 0.0]],
            [[3.0, -2.0], [-3.0, 3.0]],
        ]),
        visibility=tf.constant([[2, 2], [2, 0], [0, 2]]),
    )
    box = None  # is not used by the metric

    pem = _lib.PoseEstimationMetric(name='PEM', mismatch_penalty=100.0)
    pem.update_state([gt, pr, box])
    metrics = pem.result()

    # Expected errors per keypoint: 1, 2, 3, 0, 0, 6
    self.assertNear(metrics['PEM'], (1 + 2 + 3 + 6) / 4, err=1e-5)

  def test_respects_sample_weights(self):
    # batch_size = 3, num_points = 2
    gt = _data.KeypointsTensors(
        location=tf.constant([
            [[1.0, 1.0], [-1.0, -1.0]],
            [[2.0, 2.0], [-2.0, -2.0]],
            [[3.0, 3.0], [-3.0, -3.0]],
        ]),
        visibility=tf.constant([[2, 2], [0, 2], [2, 0]]),
    )
    # Predicted points are [1, 2, 3, 4, 5, 6] pixels away from ground truth.
    pr = _data.KeypointsTensors(
        location=tf.constant([
            [[1.0, 0.0], [1.0, -1.0]],
            [[2.0, -1.0], [2.0, -2.0]],
            [[3.0, -2.0], [-3.0, 3.0]],
        ]),
        visibility=tf.constant([[2, 2], [2, 2], [2, 2]]),
    )
    box = None  # is not used by the metric
    sample_weight = tf.constant([0.0, 0.5, 1.0])

    pem = _lib.PoseEstimationMetric(name='PEM', mismatch_penalty=100.0)
    pem.update_state([gt, pr, box], sample_weight=sample_weight)
    metrics = pem.result()

    self.assertNear(
        metrics['PEM'],
        (0 + 0 + (100.0 + 4) * 0.5 + (5 + 100.0) * 1.0)
        / (2 * 0 + 2 * 0.5 + 2 * 1.0),
        err=1e-5,
    )


class KeypointVisibilityPrecisionTest(tf.test.TestCase):

  def test_ratio_of_num_true_positives_to_predicted(self):
    # batch_size = 3, num_points = 2
    # Location is ignored by this metric
    gt = _data.KeypointsTensors(
        location=tf.constant(0.0, shape=[3, 2, 2]),
        visibility=tf.constant([[2, 2], [2, 0], [0, 0]]),
    )
    pr = _data.KeypointsTensors(
        location=tf.constant(0.0, shape=[3, 2, 2]),
        visibility=tf.constant([[2, 2], [0, 0], [2, 2]]),
    )
    box = None  # is not used by the metric

    precision = _lib.KeypointVisibilityPrecision(name='P')
    precision.update_state([gt, pr, box])
    metrics = precision.result()

    self.assertNear(metrics['P'], 2.0 / 4, err=1e-5)


class KeypointVisibilityRecallTest(tf.test.TestCase):

  def test_ratio_of_num_true_positives_to_ground_truth(self):
    # batch_size = 3, num_points = 2
    # Location is ignored by this metric
    gt = _data.KeypointsTensors(
        location=tf.constant(0.0, shape=[3, 2, 2]),
        visibility=tf.constant([[2, 2], [2, 0], [0, 0]]),
    )
    pr = _data.KeypointsTensors(
        location=tf.constant(0.0, shape=[3, 2, 2]),
        visibility=tf.constant([[2, 2], [0, 0], [2, 2]]),
    )
    box = None  # is not used by the metric

    recall = _lib.KeypointVisibilityRecall(name='R')
    recall.update_state([gt, pr, box])
    metrics = recall.result()

    self.assertNear(metrics['R'], 2.0 / 3, err=1e-5)


if __name__ == '__main__':
  tf.test.main()
