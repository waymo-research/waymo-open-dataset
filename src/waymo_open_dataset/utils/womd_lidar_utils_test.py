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
# =============================================================================
"""Tests for waymo_open_dataset.utils.womd_lidar_utils."""

import numpy as np
import tensorflow as tf

# copybara removed file resource import
from waymo_open_dataset import dataset_pb2
from waymo_open_dataset.protos import compressed_lidar_pb2
from waymo_open_dataset.protos import scenario_pb2
from waymo_open_dataset.utils import womd_lidar_utils

# pylint: disable=line-too-long
# pyformat: disable
_WOMD_INPUT_SCENARIO_FILE = '{pyglib_resource}waymo_open_dataset/utils/testdata/womd_scenario_input.tfrecord'.format(pyglib_resource='')
_WOMD_LIDAR_DATA_FILE = '{pyglib_resource}waymo_open_dataset/utils/testdata/womd_lidar_and_camera_data.tfrecord'.format(pyglib_resource='')
# pyformat: enable
# pylint: enable=line-too-long


def _get_laser_calib(
    frame_lasers: compressed_lidar_pb2.CompressedFrameLaserData,
    laser_name: dataset_pb2.LaserName.Name,
):
  """Get the laser calibration information from the frame laser data."""
  for laser_calib in frame_lasers.laser_calibrations:
    if laser_calib.name == laser_name:
      return laser_calib
  return None


def _get_point_xyz_and_feature_from_laser(
    frame_lasers: compressed_lidar_pb2.CompressedFrameLaserData,
    extract_top_lidar: bool,
):
  """Get point cloud coordinates and features from frame laser data for test."""
  for laser in frame_lasers.lasers:
    if laser.name == dataset_pb2.LaserName.TOP and extract_top_lidar:
      frame_pose = np.reshape(np.array(frame_lasers.pose.transform), (4, 4))
      c = _get_laser_calib(frame_lasers, laser.name)
      return womd_lidar_utils.extract_top_lidar_points(laser, frame_pose, c)
    elif laser.name != dataset_pb2.LaserName.TOP and not extract_top_lidar:
      c = _get_laser_calib(frame_lasers, laser.name)
      return womd_lidar_utils.extract_side_lidar_points(laser, c)


def _load_scenario_data(tfrecord_file: str) -> scenario_pb2.Scenario:
  """Load a scenario proto from a tfrecord dataset file."""
  dataset = tf.data.TFRecordDataset(tfrecord_file, compression_type='')
  data = next(iter(dataset))
  return scenario_pb2.Scenario.FromString(data.numpy())


class WomdLidarUtilsTest(tf.test.TestCase):

  def test_augment_womd_scenario_with_lidar_points(self):
    """Test of augment_womd_scenario_with_lidar_points."""
    womd_original_scenario = _load_scenario_data(_WOMD_INPUT_SCENARIO_FILE)
    womd_lidar_scenario = _load_scenario_data(_WOMD_LIDAR_DATA_FILE)
    merged_scenario = womd_lidar_utils.augment_womd_scenario_with_lidar_points(
        womd_original_scenario, womd_lidar_scenario)
    self.assertLen(merged_scenario.compressed_frame_laser_data, 11)

  def test_extract_top_lidar_points(self):
    """Test of extract_top_lidar_points."""
    womd_original_scenario = _load_scenario_data(_WOMD_INPUT_SCENARIO_FILE)
    womd_lidar_scenario = _load_scenario_data(_WOMD_LIDAR_DATA_FILE)
    merged_scenario = womd_lidar_utils.augment_womd_scenario_with_lidar_points(
        womd_original_scenario, womd_lidar_scenario)
    frame_lasers = merged_scenario.compressed_frame_laser_data[0]
    points_xyz_ri1, points_feature_ri1, points_xyz_ri2, points_feature_ri2 = (
        _get_point_xyz_and_feature_from_laser(frame_lasers, True))
    self.assertAllEqual(points_xyz_ri1.shape, tf.TensorShape([138378, 3]))
    self.assertAllEqual(points_xyz_ri2.shape, tf.TensorShape([6532, 3]))
    self.assertAllEqual(points_xyz_ri1.shape, points_feature_ri1.shape)
    self.assertAllEqual(points_xyz_ri2.shape, points_feature_ri2.shape)

  def test_extract_side_lidar_points(self):
    """Test of extract_side_lidar_points."""
    womd_original_scenario = _load_scenario_data(_WOMD_INPUT_SCENARIO_FILE)
    womd_lidar_scenario = _load_scenario_data(_WOMD_LIDAR_DATA_FILE)
    merged_scenario = womd_lidar_utils.augment_womd_scenario_with_lidar_points(
        womd_original_scenario, womd_lidar_scenario)
    frame_lasers = merged_scenario.compressed_frame_laser_data[0]
    points_xyz_ri1, points_feature_ri1, points_xyz_ri2, points_feature_ri2 = (
        _get_point_xyz_and_feature_from_laser(frame_lasers, False))
    self.assertAllEqual(points_xyz_ri1.shape, tf.TensorShape([3911, 3]))
    self.assertAllEqual(points_xyz_ri2.shape, tf.TensorShape([60, 3]))
    self.assertAllEqual(points_xyz_ri1.shape, points_feature_ri1.shape)
    self.assertAllEqual(points_xyz_ri2.shape, points_feature_ri2.shape)

if __name__ == '__main__':
  tf.test.main()
