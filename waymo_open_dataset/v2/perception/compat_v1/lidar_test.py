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
"""Tests for waymo_open_dataset/v2/perception/compat_v1/lidar.py."""
import math

from absl.testing import absltest
from absl.testing import parameterized
import tensorflow as tf

# copybara removed file resource import
from waymo_open_dataset import dataset_pb2
from waymo_open_dataset.v2.perception import lidar as _v2_lidar
from waymo_open_dataset.v2.perception.compat_v1 import interfaces
from waymo_open_dataset.v2.perception.compat_v1 import lidar
from waymo_open_dataset.v2.perception.utils import lidar_utils

# pyformat: disable
_DATA_PATH = '{pyglib_resource}waymo_open_dataset/v2/perception/compat_v1/testdata/two_frame.tfrecord'.format(pyglib_resource='')  # pylint: disable=line-too-long
# pyformat: enable


class ExtractorTest(parameterized.TestCase, absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.frames = self._load_segment_data()

  def _load_segment_data(self):
    dataset = tf.data.TFRecordDataset(_DATA_PATH)
    dataset_iter = dataset.as_numpy_iterator()
    frames = []
    for _ in range(2):
      frames.append(next(dataset_iter))
    return frames

  @parameterized.named_parameters(
      ('top_lidar', _v2_lidar.LaserName.TOP, [64, 2650, 4]),
      ('front_lidar', _v2_lidar.LaserName.FRONT, [200, 600, 4]),
      ('side_left_lidar', _v2_lidar.LaserName.SIDE_LEFT, [200, 600, 4]),
      ('side_right_lidar', _v2_lidar.LaserName.SIDE_RIGHT, [200, 600, 4]),
      ('rear_lidar', _v2_lidar.LaserName.REAR, [200, 600, 4]),
  )
  def test_populates_lidar_range_image(self, laser_name, range_image_shape):
    frame = dataset_pb2.Frame.FromString(self.frames[0])
    src = interfaces.LiDARComponentSrc(
        frame=frame,
        laser=lidar_utils.get_laser_by_name(frame, laser_name),
    )
    extractor = lidar.LiDARComponentExtractor()
    component = next(extractor(src))

    self.assertEqual(
        component.key.segment_context_name,
        '10485926982439064520_4980_000_5000_000',
    )
    self.assertEqual(component.key.frame_timestamp_micros, 1543251275738460)
    self.assertEqual(component.key.laser_name, laser_name.value)
    self.assertLen(
        component.range_image_return1.values, math.prod(range_image_shape)
    )
    self.assertEqual(component.range_image_return1.shape, range_image_shape)
    self.assertLen(
        component.range_image_return2.values, math.prod(range_image_shape)
    )
    self.assertEqual(component.range_image_return2.shape, range_image_shape)

  def test_populates_lidar_pose_range_image(self):
    frame = dataset_pb2.Frame.FromString(self.frames[1])
    src = interfaces.LiDARComponentSrc(
        frame=frame,
        laser=frame.lasers[0],
    )
    extractor = lidar.LiDARPoseComponentExtractor()
    component = next(extractor(src))

    self.assertEqual(
        component.key.segment_context_name,
        '10485926982439064520_4980_000_5000_000',
    )
    self.assertEqual(component.key.frame_timestamp_micros, 1543251275838479)
    self.assertEqual(component.key.laser_name, 1)
    self.assertLen(component.range_image_return1.values, 64 * 2650 * 6)
    self.assertEqual(component.range_image_return1.shape, [64, 2650, 6])

  @parameterized.named_parameters(
      ('front_lidar', _v2_lidar.LaserName.FRONT),
      ('side_left_lidar', _v2_lidar.LaserName.SIDE_LEFT),
      ('side_right_lidar', _v2_lidar.LaserName.SIDE_RIGHT),
      ('rear_lidar', _v2_lidar.LaserName.REAR),
  )
  def test_returns_none_if_there_is_no_data(self, laser_name):
    frame = dataset_pb2.Frame.FromString(self.frames[0])
    src = interfaces.LiDARComponentSrc(
        frame=frame,
        laser=lidar_utils.get_laser_by_name(frame, laser_name),
    )
    extractor = lidar.LiDARPoseComponentExtractor()
    components = list(extractor(src))
    # The front, side left, side right, rear lidars don't have pose range image
    # populated.
    self.assertEmpty(components)

  @parameterized.named_parameters(
      ('top_lidar', _v2_lidar.LaserName.TOP, [64, 2650, 6]),
      ('front_lidar', _v2_lidar.LaserName.FRONT, [200, 600, 6]),
      ('side_left_lidar', _v2_lidar.LaserName.SIDE_LEFT, [200, 600, 6]),
      ('side_right_lidar', _v2_lidar.LaserName.SIDE_RIGHT, [200, 600, 6]),
      ('rear_lidar', _v2_lidar.LaserName.REAR, [200, 600, 6]),
  )
  def test_populates_lidar_camera_projection(
      self, laser_name, range_image_shape
  ):
    frame = dataset_pb2.Frame.FromString(self.frames[0])
    src = interfaces.LiDARComponentSrc(
        frame=frame,
        laser=lidar_utils.get_laser_by_name(frame, laser_name),
    )
    extractor = lidar.LiDARCameraProjectionComponentExtractor()
    component = next(extractor(src))

    self.assertEqual(
        component.key.segment_context_name,
        '10485926982439064520_4980_000_5000_000',
    )
    self.assertEqual(component.key.frame_timestamp_micros, 1543251275738460)
    self.assertEqual(component.key.laser_name, laser_name.value)
    self.assertLen(
        component.range_image_return1.values, math.prod(range_image_shape)
    )
    self.assertEqual(component.range_image_return1.shape, range_image_shape)
    self.assertLen(
        component.range_image_return2.values, math.prod(range_image_shape)
    )
    self.assertEqual(component.range_image_return2.shape, range_image_shape)


if __name__ == '__main__':
  absltest.main()
