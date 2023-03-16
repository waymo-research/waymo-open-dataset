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
"""Tests for waymo_open_dataset/v2/perception/compat_v1/context.py."""

from absl.testing import absltest
from absl.testing import parameterized
import tensorflow as tf

# copybara removed file resource import
from waymo_open_dataset import dataset_pb2
from waymo_open_dataset.v2.perception import camera_image as _v2_camera_image
from waymo_open_dataset.v2.perception import lidar as _v2_lidar
from waymo_open_dataset.v2.perception.compat_v1 import context
from waymo_open_dataset.v2.perception.compat_v1 import interfaces


_TEST_DATA_PATH = '{pyglib_resource}waymo_open_dataset/v2/perception/compat_v1/testdata/two_frame.tfrecord'.format(pyglib_resource='')  # pylint: disable=line-too-long


def _get_camera_calibration_by_name(
    frame: dataset_pb2.Frame,
    camera_name: _v2_camera_image.CameraName,
) -> dataset_pb2.CameraImage:
  for calibration in frame.context.camera_calibrations:
    if calibration.name == camera_name.value:
      return calibration
  raise ValueError(f'{camera_name} not found.')


def _get_laser_calibration_by_name(
    frame: dataset_pb2.Frame,
    laser_name: _v2_lidar.LaserName
) -> dataset_pb2.Laser:
  for calibration in frame.context.laser_calibrations:
    if calibration.name == laser_name.value:
      return calibration
  raise ValueError(f'{laser_name} not found.')


class ExtractorTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.frames = self._load_segment_data()

  def _load_segment_data(self):
    dataset = tf.data.TFRecordDataset(_TEST_DATA_PATH)
    dataset_iter = dataset.as_numpy_iterator()
    frames = []
    for _ in range(2):
      frames.append(dataset_pb2.Frame.FromString(next(dataset_iter)))
    return frames

  @parameterized.named_parameters(
      ('front_camera', _v2_camera_image.CameraName.FRONT),
      ('front_left_camera', _v2_camera_image.CameraName.FRONT_LEFT),
      ('front_right_camera', _v2_camera_image.CameraName.FRONT_RIGHT),
      ('side_left_camera', _v2_camera_image.CameraName.SIDE_LEFT),
      ('side_right_camera', _v2_camera_image.CameraName.SIDE_RIGHT),
  )
  def test_populates_camera_calibration(self, camera_name):
    frame = self.frames[0]
    src = interfaces.CameraCalibrationComponentSrc(
        frame=frame,
        camera_calibration=_get_camera_calibration_by_name(frame, camera_name),
    )
    extractor = context.CameraCalibrationComponentExtractor()
    component = next(extractor(src))
    self.assertEqual(
        component.key.segment_context_name,
        '10485926982439064520_4980_000_5000_000',
    )
    self.assertEqual(component.key.camera_name, camera_name.value)
    self.assertLen(component.extrinsic.transform, 16)

  @parameterized.named_parameters(
      ('top_lidar', _v2_lidar.LaserName.TOP),
      ('front_lidar', _v2_lidar.LaserName.FRONT),
      ('side_left_lidar', _v2_lidar.LaserName.SIDE_LEFT),
      ('side_right_lidar', _v2_lidar.LaserName.SIDE_RIGHT),
      ('rear_lidar', _v2_lidar.LaserName.REAR),
  )
  def test_populates_lidar_calibration(self, laser_name):
    frame = self.frames[0]
    src = interfaces.LiDARCalibrationComponentSrc(
        frame=frame,
        lidar_calibration=_get_laser_calibration_by_name(frame, laser_name),
    )
    extractor = context.LiDARCalibrationComponentExtractor()
    component = next(extractor(src))
    self.assertEqual(
        component.key.segment_context_name,
        '10485926982439064520_4980_000_5000_000',
    )
    self.assertEqual(component.key.laser_name, laser_name.value)
    self.assertLen(component.extrinsic.transform, 16)
    if laser_name == _v2_lidar.LaserName.TOP:
      self.assertLen(component.beam_inclination.values, 64)
    else:
      self.assertIsNone(component.beam_inclination.values)

  def test_populates_stats(self):
    frame = self.frames[0]
    src = interfaces.FrameComponentSrc(frame=frame)
    extractor = context.StatsComponentExtractor()
    component = next(extractor(src))
    self.assertEqual(
        component.key.segment_context_name,
        '10485926982439064520_4980_000_5000_000',
    )
    self.assertEqual(
        component.key.frame_timestamp_micros,
        1543251275738460,
    )
    self.assertEqual(component.time_of_day, 'Day')
    self.assertEqual(component.location, 'location_sf')
    self.assertEqual(component.weather, 'sunny')
    self.assertCountEqual(
        component.lidar_object_counts.types,
        [1, 2, 3, 4],
    )
    self.assertCountEqual(
        component.lidar_object_counts.counts,
        [38, 98, 18, 3],
    )
    self.assertCountEqual(
        component.camera_object_counts.types,
        [1, 2, 4],
    )
    self.assertCountEqual(
        component.camera_object_counts.counts,
        [43, 99, 6],
    )

  def test_populates_none_if_no_count_information(self):
    frame = self.frames[0]
    frame.context.stats.ClearField('laser_object_counts')
    frame.context.stats.ClearField('camera_object_counts')
    src = interfaces.FrameComponentSrc(frame=frame)
    extractor = context.StatsComponentExtractor()
    component = next(extractor(src))
    self.assertEqual(
        component.key.segment_context_name,
        '10485926982439064520_4980_000_5000_000',
    )
    self.assertEqual(
        component.key.frame_timestamp_micros,
        1543251275738460,
    )
    self.assertEqual(component.time_of_day, 'Day')
    self.assertEqual(component.location, 'location_sf')
    self.assertEqual(component.weather, 'sunny')
    self.assertIsNone(component.lidar_object_counts)
    self.assertIsNone(component.camera_object_counts)


if __name__ == '__main__':
  absltest.main()
