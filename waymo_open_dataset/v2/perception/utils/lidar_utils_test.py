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
"""Tests for lidar_utils."""

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
import tensorflow as tf

# copybara removed file resource import
from waymo_open_dataset import dataset_pb2 as _v1_dataset_pb2
from waymo_open_dataset.v2 import column_types
from waymo_open_dataset.v2.perception import base as _v2_base
from waymo_open_dataset.v2.perception import context as _v2_context
from waymo_open_dataset.v2.perception import lidar as _v2_lidar
from waymo_open_dataset.v2.perception import pose as _v2_pose
from waymo_open_dataset.v2.perception import segmentation as _v2_segmentation
from waymo_open_dataset.v2.perception.compat_v1 import interfaces
from waymo_open_dataset.v2.perception.utils import lidar_utils as _lib

# pyformat: disable
_FRAME_WITH_LIDAR_SEGMENTATION_TEST_DATA_PATH = '{pyglib_resource}waymo_open_dataset/v2/perception/compat_v1/testdata/frame_with_lidar_segmentation.tfrecord'.format(pyglib_resource='')
# pyformat: enable


class LidarUtilsTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.range_image = _v2_lidar.RangeImage(
        values=[float(x) for x in list(range(-12, 12))],
        shape=[3, 2, 4],
    )
    self.lidar_pose_range_image = _v2_lidar.PoseRangeImage(
        values=np.tile(
            [np.pi / 2, np.pi / 2, np.pi / 2, 1, 2, 3], 3 * 2
        ).tolist(),
        shape=[3, 2, 6],
    )
    self.camera_projection = _v2_lidar.CameraProjectionRangeImage(
        values=list(range(3 * 2 * 6)),
        shape=[3, 2, 6],
    )
    self.frame_pose = _v2_pose.VehiclePoseComponent(
        key=_v2_base.FrameKey(
            segment_context_name='fake_context_name',
            frame_timestamp_micros=123456789,
        ),
        world_from_vehicle=column_types.Transform(
            transform=[1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
        ),
    )
    self.calibration = _v2_context.LiDARCalibrationComponent(
        key=_v2_base.SegmentLaserKey(
            segment_context_name='fake_context_name',
            laser_name=1,
        ),
        extrinsic=column_types.Transform(
            transform=[1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
        ),
        beam_inclination=_v2_context.BeamInclination(
            min=-np.pi/2,
            max=np.pi/2,
            values=[-np.pi/2, 0., np.pi/2],
        ),
    )
    self.loaded_frame = self._load_segment_data()

  def _load_segment_data(self):
    dataset = tf.data.TFRecordDataset(
        _FRAME_WITH_LIDAR_SEGMENTATION_TEST_DATA_PATH
    )
    dataset_iter = dataset.as_numpy_iterator()
    return _v1_dataset_pb2.Frame.FromString(next(dataset_iter))

  def test_convert_lidar_pose_range_image_to_transformation(self):
    transformation = _lib.convert_lidar_pose_range_image_to_transformation(
        self.lidar_pose_range_image,
    )
    np.testing.assert_allclose(
        transformation.numpy(),
        np.tile(
            np.array([[[
                [0, 0, 1, 1],
                [0, 1, 0, 2],
                [-1, 0, 0, 3],
                [0, 0, 0, 1]
            ]]]),
            (3, 2, 1, 1)
        ),
        atol=1e-7,
    )

  def test_convert_range_image_to_cartesian(self):
    range_image_cartesian = _lib.convert_range_image_to_cartesian(
        range_image=self.range_image,
        calibration=self.calibration,
        pixel_pose=self.lidar_pose_range_image,
        frame_pose=self.frame_pose,
        keep_polar_features=False,
    )
    np.testing.assert_allclose(
        range_image_cartesian.numpy(),
        np.asarray([
            [-11, 2, 3],
            [-7, 2, 3],
            [1, -2, 3],
            [1, 2, 3],
            [-3, 2, 3],
            [-7, 2, 3],
        ]).reshape(3, 2, 3),
        atol=1e-6,
    )

  def test_convert_range_image_to_point_cloud(self):
    points = _lib.convert_range_image_to_point_cloud(
        range_image=self.range_image,
        calibration=self.calibration,
        pixel_pose=self.lidar_pose_range_image,
        frame_pose=self.frame_pose,
        keep_polar_features=False,
    )
    np.testing.assert_allclose(
        points.numpy(),
        np.array([
            [-3, 2, 3],
            [-7, 2, 3],
        ]),
        atol=1e-6,
    )

  def test_extract_pointwise_camera_projection(self):
    pointwise_camera_projection_tensor = (
        _lib.extract_pointwise_camera_projection(
            range_image=self.range_image,
            camera_projection=self.camera_projection,
        )
    )
    np.testing.assert_allclose(
        pointwise_camera_projection_tensor.numpy(),
        np.array([
            [24, 25, 26, 27, 28, 29],
            [30, 31, 32, 33, 34, 35],
        ]),
    )

  def test_get_laser_key(self):
    frame = _v1_dataset_pb2.Frame(
        context=_v1_dataset_pb2.Context(
            name='fake_context_name',
        ),
        timestamp_micros=123456789,
    )
    laser = _v1_dataset_pb2.Laser(
        name=_v1_dataset_pb2.LaserName.Name.FRONT,
    )
    laser_key = _lib.get_laser_key(
        src=interfaces.LiDARComponentSrc(
            frame=frame,
            laser=laser,
        ),
    )
    self.assertEqual(laser_key.segment_context_name, 'fake_context_name')
    self.assertEqual(laser_key.frame_timestamp_micros, 123456789)
    self.assertEqual(laser_key.laser_name, 2)

  @parameterized.named_parameters(
      ('front', _v2_lidar.LaserName.FRONT, b'\x00\x01'),
      ('side_left', _v2_lidar.LaserName.SIDE_LEFT, b'\x00\x02'),
  )
  def test_get_laser_by_name(self, laser_name, ri):
    frame = _v1_dataset_pb2.Frame(
        lasers=[
            _v1_dataset_pb2.Laser(
                name=_v1_dataset_pb2.LaserName.Name.FRONT,
                ri_return1=_v1_dataset_pb2.RangeImage(
                    range_image_compressed=b'\x00\x01',
                )
            ),
            _v1_dataset_pb2.Laser(
                name=_v1_dataset_pb2.LaserName.Name.SIDE_LEFT,
                ri_return1=_v1_dataset_pb2.RangeImage(
                    range_image_compressed=b'\x00\x02',
                ),
            )
        ]
    )
    laser = _lib.get_laser_by_name(frame, laser_name)
    self.assertEqual(laser.ri_return1.range_image_compressed, ri)

  @parameterized.named_parameters(
      ('range_image', _v2_lidar.RangeImage, 'range_image_compressed'),
      (
          'camera_projection_range_image',
          _v2_lidar.CameraProjectionRangeImage,
          'camera_projection_compressed',
      ),
      (
          'pose_range_image',
          _v2_lidar.PoseRangeImage,
          'range_image_pose_compressed',
      ),
      (
          'lidar_segmentation_range_image',
          _v2_segmentation.LiDARSegmentationRangeImage,
          'segmentation_label_compressed',
      ),
  )
  def test_parse_range_image(self, range_image_class, field_name):
    laser = _lib.get_laser_by_name(self.loaded_frame, _v2_lidar.LaserName.TOP)
    parsed = _lib.parse_range_image(
        getattr(laser.ri_return1, field_name),
        range_image_class,
    )
    self.assertIsNotNone(parsed)


if __name__ == '__main__':
  absltest.main()
