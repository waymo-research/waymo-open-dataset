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
"""Tests for occupancy_flow_data."""


import tensorflow as tf

from waymo_open_dataset.utils import occupancy_flow_data
from waymo_open_dataset.utils import occupancy_flow_test_util


class OccupancyFlowDataTest(tf.test.TestCase):

  def test_parse_tf_example(self):
    dataset = tf.data.TFRecordDataset(occupancy_flow_test_util.test_data_path())
    dataset = dataset.repeat()
    it = iter(dataset)
    tf_example = next(it)
    parsed_example = occupancy_flow_data.parse_tf_example(tf_example)
    self.assertIn('state/current/x', parsed_example)
    self.assertEqual(parsed_example['state/current/x'].shape, (128, 1))

  def test_add_sdc_fields(self):
    inputs = occupancy_flow_test_util.make_one_data_batch(batch_size=8)
    self.assertNotIn('sdc/current/x', inputs)
    self.assertNotIn('sdc/current/y', inputs)
    self.assertNotIn('sdc/current/z', inputs)
    inputs_with_sdc = occupancy_flow_data.add_sdc_fields(inputs)
    self.assertIn('sdc/current/x', inputs_with_sdc)
    self.assertIn('sdc/current/y', inputs_with_sdc)
    self.assertIn('sdc/current/z', inputs_with_sdc)


if __name__ == '__main__':
  tf.test.main()
