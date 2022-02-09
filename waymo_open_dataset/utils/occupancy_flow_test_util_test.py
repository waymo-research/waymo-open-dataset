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
# =============================================================================
"""Tests for occupancy_flow_test_util."""

import tensorflow as tf

from waymo_open_dataset.utils import occupancy_flow_test_util


class OccupancyFlowTestUtilTest(tf.test.TestCase):

  def test_make_test_dataset(self):
    dataset = occupancy_flow_test_util.make_test_dataset(batch_size=8)
    it = iter(dataset)
    inputs = next(it)
    self.assertIn('state/current/x', inputs)
    self.assertEqual(inputs['state/current/x'].shape, (8, 128, 1))

  def test_make_one_data_batch(self):
    inputs = occupancy_flow_test_util.make_one_data_batch(batch_size=8)
    self.assertIn('state/current/x', inputs)
    self.assertEqual(inputs['state/current/x'].shape, (8, 128, 1))


if __name__ == '__main__':
  tf.test.main()
