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
"""Tests for example_utils."""

import tensorflow as tf
from waymo_open_dataset.utils import example_utils
from absl.testing import absltest


def _make_example(
    sdc_current_timestamp_micros=None, state_current_timestamp_micros=None
):
  feature = {}
  if sdc_current_timestamp_micros is not None:
    feature['sdc/current/timestamp_micros'] = tf.train.Feature(
        int64_list=tf.train.Int64List(value=[sdc_current_timestamp_micros])
    )
  if state_current_timestamp_micros is not None:
    feature['state/current/timestamp_micros'] = tf.train.Feature(
        int64_list=tf.train.Int64List(value=state_current_timestamp_micros)
    )
  return tf.train.Example(features=tf.train.Features(feature=feature))


def _make_float_example(value_list: list[float]) -> tf.train.Example:
  feature = {}
  feature['test_feature'] = tf.train.Feature()
  feature['test_feature'].float_list.value.extend(value_list)
  return tf.train.Example(features=tf.train.Features(feature=feature))


class ExampleUtilsTest(absltest.TestCase):

  def test_examples_equal_int64(self):
    example_a = _make_example(
        sdc_current_timestamp_micros=1000,
        state_current_timestamp_micros=[2, 4, 5],
    )
    example_b = _make_example(
        sdc_current_timestamp_micros=1000,
        state_current_timestamp_micros=[2, 4, 5],
    )
    self.assertTrue(example_utils.examples_equal(example_a, example_b))

  def test_examples_equal_differ_int64(self):
    example_a = _make_example(
        sdc_current_timestamp_micros=1000,
        state_current_timestamp_micros=[2, 4, 5],
    )
    example_b = _make_example(
        sdc_current_timestamp_micros=100,
        state_current_timestamp_micros=[2, 4, 5],
    )
    self.assertFalse(example_utils.examples_equal(example_a, example_b))

  def test_examples_equal_missing_key(self):
    example_a = _make_example(
        sdc_current_timestamp_micros=1000,
        state_current_timestamp_micros=[2, 4, 5],
    )
    example_d = _make_example(state_current_timestamp_micros=[2, 4, 5])
    self.assertFalse(example_utils.examples_equal(example_a, example_d))
    self.assertFalse(example_utils.examples_equal(example_d, example_a))

  def test_examples_equal_float(self):
    example_a = _make_float_example([1.0, 2.0, 3.0])
    example_b = _make_float_example([1.0, 2.0, 3.0])
    self.assertTrue(example_utils.examples_equal(example_a, example_b))

  def test_examples_equal_float_unequal_length(self):
    example_a = _make_float_example([1.0, 2.0, 3.0])
    example_b = _make_float_example([1.0, 2.0])
    self.assertFalse(example_utils.examples_equal(example_a, example_b))

  def test_examples_equal_float_out_of_tolerance(self):
    example_a = _make_float_example([1.0, 2.0, 3.00001])
    example_b = _make_float_example([1.0, 2.0, 3.0])
    self.assertTrue(
        example_utils.examples_equal(example_a, example_b, atol=0.001)
    )
    self.assertFalse(
        example_utils.examples_equal(example_a, example_b, atol=0.000001)
    )


if __name__ == '__main__':
  absltest.main()
