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
"""Test utils used in other occupancy_flow_* tests."""

import tensorflow as tf

from google.protobuf import text_format
# copybara removed file resource import
from waymo_open_dataset.protos import occupancy_flow_metrics_pb2
from waymo_open_dataset.utils import occupancy_flow_data


def test_data_path():
  """Returns full path to the test data."""
  # pylint: disable=line-too-long
  # pyformat: disable
  return '{pyglib_resource}waymo_open_dataset/utils/testdata/motion_data_one_example.tfrecord'.format(pyglib_resource='')


def make_test_dataset(batch_size: int) -> tf.data.Dataset:
  """Makes a dataset containing the single tf example."""
  dataset = tf.data.TFRecordDataset(test_data_path())
  dataset = dataset.repeat()
  dataset = dataset.map(occupancy_flow_data.parse_tf_example)
  dataset = dataset.batch(batch_size)
  return dataset


def make_one_data_batch(batch_size: int) -> tf.Tensor:
  """Returns a sample batch of data."""
  dataset = make_test_dataset(batch_size)
  it = iter(dataset)
  inputs = next(it)
  return inputs


def make_test_config() -> occupancy_flow_metrics_pb2.OccupancyFlowTaskConfig:
  """Make nominal task config."""
  config = occupancy_flow_metrics_pb2.OccupancyFlowTaskConfig()
  config_text = """
      num_past_steps: 10
      num_future_steps: 80
      num_waypoints: 8
      cumulative_waypoints: true
      normalize_sdc_yaw: true
      grid_height_cells: 256
      grid_width_cells: 256
      sdc_y_in_grid: 192
      sdc_x_in_grid: 128
      pixels_per_meter: 3.2
      agent_points_per_side_length: 48
      agent_points_per_side_width: 16
  """
  text_format.Parse(config_text, config)
  return config
