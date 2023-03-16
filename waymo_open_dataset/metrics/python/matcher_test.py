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
"""Tests for matcher.py."""

import dataclasses
from typing import Tuple

import numpy as np
import tensorflow as tf

from waymo_open_dataset.metrics.python import matcher
from waymo_open_dataset.protos import metrics_pb2


def get_let_metric_config() -> (
    metrics_pb2.Config.LongitudinalErrorTolerantConfig
):
  return metrics_pb2.Config.LongitudinalErrorTolerantConfig(
      enabled=True,
      sensor_location={'x': 0, 'y': 0, 'z': 0},
      longitudinal_tolerance_percentage=0.15,
      min_longitudinal_tolerance_meter=2.0,
  )


def create_shuffled_boxes(
    num_boxes: int, batch_size: int
) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
  boxes, shuffled_boxes, permutations = [], [], []
  for _ in range(batch_size):
    a = tf.random.uniform([1, num_boxes, 7], minval=0.1, maxval=1.0)
    permutation = tf.random.shuffle(tf.range(0, num_boxes))
    b = tf.gather(a, permutation, axis=1)
    boxes.append(a)
    shuffled_boxes.append(b)
    permutations.append(permutation)
  boxes = tf.concat(boxes, axis=0)
  shuffled_boxes = tf.concat(shuffled_boxes, axis=0)
  permutations = tf.stack(permutations, axis=0)
  return boxes, shuffled_boxes, permutations


def asdict(obj):
  return dataclasses.asdict(obj)


class OpenDatasetMatcherTest(tf.test.TestCase):

  def setUp(self):
    super().setUp()
    tf.random.set_seed(0)
    self.config_dict = {
        'iou': 0.3,
        'box_type': matcher.Type.TYPE_2D,
        'let_metric_config': get_let_metric_config(),
    }

  def test_aa_2d_boxes_match(self):
    box_type = matcher.Type.TYPE_AA_2D
    boxes1 = tf.reshape(np.array([2, 2, 10, 2], np.float32), [1, 1, 4])
    boxes2 = tf.reshape(np.array([2, 1, 10, 2], np.float32), [1, 1, 4])
    self.assertAllClose(
        asdict(matcher.MatchResult([0], [0], [1.0 / 3.0], [1])),
        asdict(matcher.match(boxes1, boxes2, 0.3, box_type)),
    )

  def test_pair_of_boxes_with_perfect_iou(self):
    box1 = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0])
    box2 = np.array([5.0, 5.0, 0.0, 1.0, 1.0, 1.0, 0.0])

    boxes1 = np.zeros([3, 2, 7], np.float32)
    boxes2 = np.zeros([3, 2, 7], np.float32)
    boxes1[1, :, :] = np.stack([box1, box2], axis=0)
    boxes2[1, :, :] = np.stack([box2, box1], axis=0)
    boxes1[2, :, :] = np.stack([box2, box1], axis=0)
    boxes2[2, :, :] = np.stack([box1, box2], axis=0)

    result = matcher.match(boxes1, boxes2, **self.config_dict)
    self.assertAllClose(tf.ones([4]), result.ious)
    self.assertAllClose(tf.ones([4]), result.longitudinal_affinities)
    self.assertAllEqual([2, 3, 4, 5], result.prediction_ids)
    self.assertAllEqual([3, 2, 5, 4], result.groundtruth_ids)

  def test_with_random_permutations_without_batching(self):
    boxes, boxes_shuffled, permutations = create_shuffled_boxes(
        num_boxes=10, batch_size=1
    )
    result = matcher.match(boxes, boxes_shuffled, **self.config_dict)
    self.assertAllClose(tf.ones([10]), result.ious)
    self.assertAllClose(tf.ones([10]), result.longitudinal_affinities)
    self.assertAllEqual(tf.range(0, 10), result.prediction_ids)
    self.assertAllEqual(
        tf.math.invert_permutation(permutations[0]), result.groundtruth_ids
    )

  def test_with_random_permutations_with_batching(self):
    num_boxes, batch_size = 10, 3
    boxes, boxes_shuffled, permutations = create_shuffled_boxes(
        num_boxes, batch_size
    )
    result = matcher.match(boxes, boxes_shuffled, **self.config_dict)
    self.assertAllClose(tf.ones([num_boxes * batch_size]), result.ious)
    self.assertAllEqual(
        tf.range(0, num_boxes * batch_size), result.prediction_ids
    )
    for i in range(batch_size):
      self.assertAllEqual(
          tf.math.invert_permutation(permutations[i]) + i * num_boxes,
          result.groundtruth_ids[i * num_boxes : (i + 1) * num_boxes],
      )

  def test_imperfect_iou_with_batching(self):
    box1 = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0])
    box2 = np.array([0.25, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0])
    boxes1, boxes2 = np.zeros([1, 2, 7]), np.zeros([1, 3, 7])
    boxes1[0, 0, :], boxes2[0, 2, :] = box1, box2

    result = matcher.match(
        boxes1, boxes2, iou=0.3, box_type=matcher.Type.TYPE_2D
    )
    self.assertAllClose([0.6], result.ious, 1e-2)

    result = matcher.match(boxes1, boxes2, **self.config_dict)
    self.assertAllClose([0.6], result.ious, 1e-2)
    self.assertAllClose([0.944], result.longitudinal_affinities, 1e-2)

  def test_invalid_box_handling(self):
    boxes = tf.reshape(np.array([0, 0, 0, 1, 1, 0, 0], np.float32), [1, 1, 7])
    no_match_result = matcher.MatchResult([], [], [], [])
    # A 'slip of paper' is considered invalid for 3D matching.
    self.assertAllClose(
        asdict(no_match_result),
        asdict(matcher.match(boxes, boxes, 0.3, matcher.Type.TYPE_3D)),
    )
    # ...but is valid for 2D AND AA_2D matching.
    box_type = matcher.Type.TYPE_2D
    self.assertAllClose(
        asdict(matcher.MatchResult([0], [0], [1], [1])),
        asdict(matcher.match(boxes, boxes, 0.3, box_type)),
    )
    box_type = matcher.Type.TYPE_AA_2D
    boxes_aa2d = tf.stack(
        [boxes[..., 0], boxes[..., 1], boxes[..., 3], boxes[..., 4]], axis=-1
    )
    self.assertAllClose(
        asdict(matcher.MatchResult([0], [0], [1], [1])),
        asdict((matcher.match(boxes_aa2d, boxes_aa2d, 0.3, box_type))),
    )

    # ...a 'line segment' is invalid in either.
    boxes = tf.reshape(np.array([0, 0, 0, 0, 1, 0, 0], np.float32), [1, 1, 7])
    for box_type in (
        matcher.Type.TYPE_2D,
        matcher.Type.TYPE_3D,
    ):
      self.assertAllClose(
          asdict(no_match_result),
          asdict(matcher.match(boxes, boxes, 0.3, box_type)),
      )
    box_type = matcher.Type.TYPE_AA_2D
    boxes_aa2d = tf.reshape(np.array([0, 0, 1, 0], np.float32), [1, 1, 4])
    self.assertAllClose(
        asdict(no_match_result),
        asdict(matcher.match(boxes_aa2d, boxes_aa2d, 0.3, box_type)),
    )

  def test_no_matching_possible(self):
    no_match_result = matcher.MatchResult([], [], [], [])
    boxes = tf.constant([[[0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0]]])
    self.assertAllClose(
        asdict(no_match_result),
        asdict(
            matcher.match(
                boxes, tf.zeros([1, 0, 7], tf.float32), **self.config_dict
            )
        ),
    )

    self.assertAllClose(
        asdict(no_match_result),
        asdict(
            matcher.match(
                tf.zeros([1, 0, 7], tf.float32), boxes, **self.config_dict
            )
        ),
    )

    self.assertAllClose(
        asdict(no_match_result),
        asdict(
            matcher.match(
                tf.zeros([1, 0, 7], tf.float32),
                tf.zeros([1, 0, 7], tf.float32),
                **self.config_dict,
            )
        ),
    )

    self.assertAllClose(
        asdict(no_match_result),
        asdict(
            matcher.match(
                tf.zeros([1, 1, 7], tf.float32),
                tf.zeros([1, 1, 7], tf.float32),
                **self.config_dict,
            )
        ),
    )

  def test_invalid_arguments(self):
    for a, b in [
        (tf.zeros([1, 1, 8]), tf.zeros([1, 1, 8])),
        (tf.zeros([1, 1, 7]), tf.zeros([1, 1, 7], dtype=tf.int32)),
        (tf.zeros([2, 1, 7]), tf.zeros([1, 1, 7])),
    ]:
      with self.assertRaises((ValueError, tf.errors.InvalidArgumentError)):
        matcher.match(a, b, **self.config_dict)


if __name__ == '__main__':
  tf.test.main()
