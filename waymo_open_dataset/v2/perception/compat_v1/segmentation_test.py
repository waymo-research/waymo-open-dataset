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
"""Tests for waymo_open_dataset/v2/perception/compat_v1/segmentation.py."""

from absl.testing import absltest
from absl.testing import parameterized
import tensorflow as tf

# copybara removed file resource import
from waymo_open_dataset import dataset_pb2
from waymo_open_dataset.v2.perception import camera_image as _v2_camera_image
from waymo_open_dataset.v2.perception import lidar as _v2_lidar
from waymo_open_dataset.v2.perception.compat_v1 import interfaces
from waymo_open_dataset.v2.perception.compat_v1 import segmentation
from waymo_open_dataset.v2.perception.utils import lidar_utils

# pyformat: disable
_FRAME_WITH_CAMERA_SEGMENTATION_TEST_DATA_PATH = '{pyglib_resource}waymo_open_dataset/v2/perception/compat_v1/testdata/frame_with_camera_segmentation.tfrecord'.format(pyglib_resource='')
_FRAME_WITH_LIDAR_SEGMENTATION_TEST_DATA_PATH = '{pyglib_resource}waymo_open_dataset/v2/perception/compat_v1/testdata/frame_with_lidar_segmentation.tfrecord'.format(pyglib_resource='')
# pyformat: enable


def _get_camera_image_by_name(
    frame: dataset_pb2.Frame,
    camera_name: _v2_camera_image.CameraName,
) -> dataset_pb2.CameraImage:
  for image in frame.images:
    if image.name == camera_name.value:
      return image
  raise ValueError(f'{camera_name} not found.')


class CameraSegmentationExtractorTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.frame = self._load_segment_data()

  def _load_segment_data(self):
    dataset = tf.data.TFRecordDataset(
        _FRAME_WITH_CAMERA_SEGMENTATION_TEST_DATA_PATH
    )
    dataset_iter = dataset.as_numpy_iterator()
    return dataset_pb2.Frame.FromString(next(dataset_iter))

  @parameterized.named_parameters(
      ('front_camera', _v2_camera_image.CameraName.FRONT),
      ('front_left_camera', _v2_camera_image.CameraName.FRONT_LEFT),
      ('front_right_camera', _v2_camera_image.CameraName.FRONT_RIGHT),
      ('side_left_camera', _v2_camera_image.CameraName.SIDE_LEFT),
      ('side_right_camera', _v2_camera_image.CameraName.SIDE_RIGHT),
  )
  def test_populates_camera_segmentation_label(self, camera_name):
    src = interfaces.CameraImageComponentSrc(
        frame=self.frame,
        camera_image=_get_camera_image_by_name(self.frame, camera_name),
    )
    extractor = segmentation.CameraSegmentationLabelComponentExtractor()
    component = next(extractor(src))
    self.assertEqual(
        component.key.segment_context_name,
        '10485926982439064520_4980_000_5000_000',
    )
    self.assertEqual(component.key.frame_timestamp_micros, 1543251278438383)
    self.assertEqual(component.panoptic_label_divisor, 1000)
    self.assertNotEmpty(component.panoptic_label)
    self.assertNotEmpty(
        component.instance_id_to_global_id_mapping.local_instance_ids
    )
    self.assertNotEmpty(
        component.instance_id_to_global_id_mapping.global_instance_ids
    )
    self.assertNotEmpty(component.instance_id_to_global_id_mapping.is_tracked)
    self.assertEqual(component.sequence_id, '15692902915025018598')
    self.assertNotEmpty(component.num_cameras_covered)


class LiDARSegmentationExtractorTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.frame = self._load_segment_data()

  def _load_segment_data(self):
    dataset = tf.data.TFRecordDataset(
        _FRAME_WITH_LIDAR_SEGMENTATION_TEST_DATA_PATH
    )
    dataset_iter = dataset.as_numpy_iterator()
    return dataset_pb2.Frame.FromString(next(dataset_iter))

  def test_populates_lidar_segmentation_label(self):
    src = interfaces.LiDARComponentSrc(
        frame=self.frame,
        laser=lidar_utils.get_laser_by_name(
            self.frame, _v2_lidar.LaserName.TOP
        ),
    )
    extractor = segmentation.LiDARSegmentationLabelComponentExtractor()
    component = next(extractor(src))

    self.assertEqual(
        component.key.segment_context_name,
        '10485926982439064520_4980_000_5000_000',
    )
    self.assertEqual(component.key.frame_timestamp_micros, 1543251278138396)
    self.assertEqual(component.key.laser_name, _v2_lidar.LaserName.TOP.value)
    self.assertLen(component.range_image_return1.values, 64 * 2650 * 2)
    self.assertEqual(component.range_image_return1.shape, [64, 2650, 2])
    self.assertLen(component.range_image_return2.values, 64 * 2650 * 2)
    self.assertEqual(component.range_image_return2.shape, [64, 2650, 2])

  @parameterized.named_parameters(
      ('front_lidar', _v2_lidar.LaserName.FRONT),
      ('side_left_lidar', _v2_lidar.LaserName.SIDE_LEFT),
      ('side_right_lidar', _v2_lidar.LaserName.SIDE_RIGHT),
      ('rear_lidar', _v2_lidar.LaserName.REAR),
  )
  def test_returns_none_if_there_is_no_data(self, laser_name):
    src = interfaces.LiDARComponentSrc(
        frame=self.frame,
        laser=lidar_utils.get_laser_by_name(self.frame, laser_name),
    )
    extractor = segmentation.LiDARSegmentationLabelComponentExtractor()
    components = list(extractor(src))
    # The front, side left, side right, rear lidars don't have 3D semseg labels.
    self.assertEmpty(components)


if __name__ == '__main__':
  absltest.main()
