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

from absl.testing import absltest
import numpy as np
import pyarrow as pa
import tensorflow as tf

from waymo_open_dataset.v2 import column_types
from waymo_open_dataset.v2.perception import base
from waymo_open_dataset.v2.perception import camera_image as _lib_camera
from waymo_open_dataset.v2.perception import object_asset as _lib


class ObjectAssetTest(absltest.TestCase):

  def test_creates_correct_keys_refined_pose_component(self):
    component = _lib.ObjectAssetRefinedPoseComponent(
        key=base.LaserLabelKey(
            segment_context_name='fake_context_name',
            frame_timestamp_micros=123456789,
            laser_object_id='fake_object_id',
        ),
        box_from_vehicle=column_types.Transform(transform=[1] * 16),
    )

    columns = component.to_flatten_dict()
    prefix = '[ObjectAssetRefinedPoseComponent]'

    self.assertCountEqual(
        [
            'key.segment_context_name',
            'key.frame_timestamp_micros',
            'key.laser_object_id',
            f'{prefix}.box_from_vehicle.transform',
        ],
        columns,
    )

  def test_creates_correct_schema_refined_pose_component(self):
    schema = _lib.ObjectAssetRefinedPoseComponent.schema()
    schema_dict = dict(zip(schema.names, schema.types))
    prefix = '[ObjectAssetRefinedPoseComponent]'
    self.assertEqual(
        {
            'key.segment_context_name': pa.string(),
            'key.frame_timestamp_micros': pa.int64(),
            'key.laser_object_id': pa.string(),
            f'{prefix}.box_from_vehicle.transform': pa.list_(pa.float64(), 16),
        },
        schema_dict,
    )

  def test_creates_correct_keys_lidar_sensor_component(self):
    component = _lib.ObjectAssetLiDARSensorComponent(
        key=base.LaserLabelKey(
            segment_context_name='fake_context_name',
            frame_timestamp_micros=123456789,
            laser_object_id='fake_object_id',
        ),
        points_xyz=_lib.PointCoordinates(
            values=[0, 1, 2, 3, 4, 5],
            shape=[2, 3],
        ),
    )

    columns = component.to_flatten_dict()
    prefix = '[ObjectAssetLiDARSensorComponent]'

    self.assertCountEqual(
        [
            'key.segment_context_name',
            'key.frame_timestamp_micros',
            'key.laser_object_id',
            f'{prefix}.points_xyz.values',
            f'{prefix}.points_xyz.shape',
        ],
        columns,
    )

  def test_lidar_sensor_component_points_xyz_numpy(self):
    component = _lib.ObjectAssetLiDARSensorComponent(
        key=base.LaserLabelKey(
            segment_context_name='fake_context_name',
            frame_timestamp_micros=123456789,
            laser_object_id='fake_object_id',
        ),
        points_xyz=_lib.PointCoordinates(
            values=[0, 1, 2, 3, 4, 5],
            shape=[2, 3],
        ),
    )

    self.assertNotEmpty(component.points_xyz.numpy)
    self.assertIsInstance(component.points_xyz.numpy, np.ndarray)
    self.assertTupleEqual(component.points_xyz.numpy.shape, (2, 3))

  def test_creates_correct_schema_lidar_sensor_component(self):
    schema = _lib.ObjectAssetLiDARSensorComponent.schema()
    schema_dict = dict(zip(schema.names, schema.types))
    prefix = '[ObjectAssetLiDARSensorComponent]'
    self.assertEqual(
        {
            'key.segment_context_name': pa.string(),
            'key.frame_timestamp_micros': pa.int64(),
            'key.laser_object_id': pa.string(),
            f'{prefix}.points_xyz.values': pa.list_(pa.float32()),
            f'{prefix}.points_xyz.shape': pa.list_(pa.int32(), 2),
        },
        schema_dict,
    )

  def test_creates_correct_keys_camera_sensor_component(self):
    component = _lib.ObjectAssetCameraSensorComponent(
        key=base.ObjectAssetKey(
            segment_context_name='fake_context_name',
            frame_timestamp_micros=123456789,
            laser_object_id='fake_object_id',
            camera_name=_lib_camera.CameraName.FRONT,
        ),
        camera_region=column_types.BoxAxisAligned2d(
            center=column_types.Vec2d(
                x=1,
                y=2,
            ),
            size=column_types.Vec2d(
                x=3,
                y=4,
            ),
        ),
        rgb_image=b'\x00\x01\x02\x03\x04\x05',
        proj_points_dist=_lib.LidarDistanceImage(
            values=[0, 0.1, 0, 0, 0, 0.2],
            shape=[2, 3],
        ),
        proj_points_mask=b'\x00\x01\x00\x00\x00\x01',
    )

    columns = component.to_flatten_dict()
    prefix = '[ObjectAssetCameraSensorComponent]'

    self.assertCountEqual(
        [
            'key.segment_context_name',
            'key.frame_timestamp_micros',
            'key.laser_object_id',
            'key.camera_name',
            f'{prefix}.camera_region.center.x',
            f'{prefix}.camera_region.center.y',
            f'{prefix}.camera_region.size.x',
            f'{prefix}.camera_region.size.y',
            f'{prefix}.rgb_image',
            f'{prefix}.proj_points_dist.values',
            f'{prefix}.proj_points_dist.shape',
            f'{prefix}.proj_points_mask',
        ],
        columns,
    )

  def test_camera_sensor_component_numpy_basic(self):
    rgb_image_encoded = tf.io.encode_png(
        np.zeros((2, 3, 3), dtype=np.uint8)
    ).numpy()
    proj_points_mask_encoded = tf.io.encode_png(
        np.zeros((2, 3, 1), dtype=np.uint8)
    ).numpy()
    component = _lib.ObjectAssetCameraSensorComponent(
        key=base.ObjectAssetKey(
            segment_context_name='fake_context_name',
            frame_timestamp_micros=123456789,
            laser_object_id='fake_object_id',
            camera_name=_lib_camera.CameraName.FRONT,
        ),
        camera_region=column_types.BoxAxisAligned2d(
            center=column_types.Vec2d(
                x=1,
                y=2,
            ),
            size=column_types.Vec2d(
                x=3,
                y=4,
            ),
        ),
        rgb_image=rgb_image_encoded,
        proj_points_dist=_lib.LidarDistanceImage(
            values=[0, 0.1, 0, 0, 0, 0.2],
            shape=[2, 3],
        ),
        proj_points_mask=proj_points_mask_encoded,
    )

    self.assertNotEmpty(component.rgb_image_numpy)
    self.assertIsInstance(component.rgb_image_numpy, np.ndarray)
    self.assertTupleEqual(component.rgb_image_numpy.shape, (2, 3, 3))
    self.assertEqual(component.rgb_image_numpy.dtype, np.uint8)

    self.assertNotEmpty(component.proj_points_mask_numpy)
    self.assertIsInstance(component.proj_points_mask_numpy, np.ndarray)
    self.assertTupleEqual(component.proj_points_mask_numpy.shape, (2, 3, 1))
    self.assertEqual(component.proj_points_mask_numpy.dtype, np.uint8)

    self.assertNotEmpty(component.proj_points_dist.numpy)
    self.assertIsInstance(component.proj_points_dist.numpy, np.ndarray)
    self.assertTupleEqual(component.proj_points_dist.numpy.shape, (2, 3))

  def test_creates_correct_schema_camera_sensor_component(self):
    schema = _lib.ObjectAssetCameraSensorComponent.schema()
    schema_dict = dict(zip(schema.names, schema.types))
    prefix = '[ObjectAssetCameraSensorComponent]'
    self.assertEqual(
        {
            'key.segment_context_name': pa.string(),
            'key.frame_timestamp_micros': pa.int64(),
            'key.laser_object_id': pa.string(),
            'key.camera_name': pa.int8(),
            f'{prefix}.camera_region.center.x': pa.float64(),
            f'{prefix}.camera_region.center.y': pa.float64(),
            f'{prefix}.camera_region.size.x': pa.float64(),
            f'{prefix}.camera_region.size.y': pa.float64(),
            f'{prefix}.rgb_image': pa.binary(),
            f'{prefix}.proj_points_dist.values': pa.list_(pa.float32()),
            f'{prefix}.proj_points_dist.shape': pa.list_(pa.int32(), 2),
            f'{prefix}.proj_points_mask': pa.binary(),
        },
        schema_dict,
    )

  def test_creates_correct_keys_auto_label_component(self):
    component = _lib.ObjectAssetAutoLabelComponent(
        key=base.ObjectAssetKey(
            segment_context_name='fake_context_name',
            frame_timestamp_micros=123456789,
            laser_object_id='fake_object_id',
            camera_name=_lib_camera.CameraName.FRONT,
        ),
        object_mask=b'\x00\x01\x00\x00\x00\x01',
        semantic_mask=b'\x00\x01\x00\x00\x00\x05',
        instance_mask=b'\x00\x01\x00\x00\x00\x05',
    )

    columns = component.to_flatten_dict()
    prefix = '[ObjectAssetAutoLabelComponent]'
    self.assertCountEqual(
        [
            'key.segment_context_name',
            'key.frame_timestamp_micros',
            'key.laser_object_id',
            'key.camera_name',
            f'{prefix}.object_mask',
            f'{prefix}.semantic_mask',
            f'{prefix}.instance_mask',
        ],
        columns,
    )

  def test_creates_correct_schema_auto_label_component(self):
    schema = _lib.ObjectAssetAutoLabelComponent.schema()
    schema_dict = dict(zip(schema.names, schema.types))
    prefix = '[ObjectAssetAutoLabelComponent]'

    self.assertEqual(
        {
            'key.segment_context_name': pa.string(),
            'key.frame_timestamp_micros': pa.int64(),
            'key.laser_object_id': pa.string(),
            'key.camera_name': pa.int8(),
            f'{prefix}.object_mask': pa.binary(),
            f'{prefix}.semantic_mask': pa.binary(),
            f'{prefix}.instance_mask': pa.binary(),
        },
        schema_dict,
    )

  def test_auto_label_component_numpy_basic(self):
    object_mask_encoded = tf.io.encode_png(
        np.zeros((2, 3, 1), dtype=np.uint8)
    ).numpy()
    semantic_mask_encoded = tf.io.encode_png(
        np.zeros((2, 3, 1), dtype=np.uint8)
    ).numpy()
    instance_mask_encoded = tf.io.encode_png(
        np.zeros((2, 3, 1), dtype=np.uint16)
    ).numpy()
    component = _lib.ObjectAssetAutoLabelComponent(
        key=base.ObjectAssetKey(
            segment_context_name='fake_context_name',
            frame_timestamp_micros=123456789,
            laser_object_id='fake_object_id',
            camera_name=_lib_camera.CameraName.FRONT,
        ),
        object_mask=object_mask_encoded,
        semantic_mask=semantic_mask_encoded,
        instance_mask=instance_mask_encoded,
    )
    self.assertNotEmpty(component.object_mask_numpy)
    self.assertIsInstance(component.object_mask_numpy, np.ndarray)
    self.assertTupleEqual(component.object_mask_numpy.shape, (2, 3, 1))
    self.assertEqual(component.object_mask_numpy.dtype, np.uint8)

    self.assertNotEmpty(component.semantic_mask_numpy)
    self.assertIsInstance(component.semantic_mask_numpy, np.ndarray)
    self.assertTupleEqual(component.semantic_mask_numpy.shape, (2, 3, 1))
    self.assertEqual(component.semantic_mask_numpy.dtype, np.uint8)

    self.assertNotEmpty(component.instance_mask_numpy)
    self.assertIsInstance(component.instance_mask_numpy, np.ndarray)
    self.assertTupleEqual(component.instance_mask_numpy.shape, (2, 3, 1))
    self.assertEqual(component.instance_mask_numpy.dtype, np.uint16)

  def test_creates_correct_keys_ray_component(self):
    component = _lib.ObjectAssetRayComponent(
        key=base.ObjectAssetKey(
            segment_context_name='fake_context_name',
            frame_timestamp_micros=123456789,
            laser_object_id='fake_object_id',
            camera_name=_lib_camera.CameraName.FRONT,
        ),
        ray_origin=_lib.CameraRayImage(
            values=[0, 1, 2, 3, 4, 5],
            shape=[1, 2, 3],
        ),
        ray_direction=_lib.CameraRayImage(
            values=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
            shape=[1, 2, 3],
        ),
    )

    columns = component.to_flatten_dict()
    prefix = '[ObjectAssetRayComponent]'
    self.assertCountEqual(
        {
            'key.segment_context_name',
            'key.frame_timestamp_micros',
            'key.laser_object_id',
            'key.camera_name',
            f'{prefix}.ray_origin.values',
            f'{prefix}.ray_origin.shape',
            f'{prefix}.ray_direction.values',
            f'{prefix}.ray_direction.shape',
        },
        columns,
    )

  def test_ray_component_numpy_basic(self):
    component = _lib.ObjectAssetRayComponent(
        key=base.ObjectAssetKey(
            segment_context_name='fake_context_name',
            frame_timestamp_micros=123456789,
            laser_object_id='fake_object_id',
            camera_name=_lib_camera.CameraName.FRONT,
        ),
        ray_origin=_lib.CameraRayImage(
            values=[0, 1, 2, 3, 4, 5],
            shape=[1, 2, 3],
        ),
        ray_direction=_lib.CameraRayImage(
            values=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
            shape=[1, 2, 3],
        ),
    )

    self.assertNotEmpty(component.ray_origin.numpy)
    self.assertIsInstance(component.ray_origin.numpy, np.ndarray)
    self.assertTupleEqual(component.ray_origin.numpy.shape, (1, 2, 3))

    self.assertNotEmpty(component.ray_direction.numpy)
    self.assertIsInstance(component.ray_direction.numpy, np.ndarray)
    self.assertTupleEqual(component.ray_direction.numpy.shape, (1, 2, 3))

  def test_creates_correct_schema_ray_component(self):
    schema = _lib.ObjectAssetRayComponent.schema()
    schema_dict = dict(zip(schema.names, schema.types))
    prefix = '[ObjectAssetRayComponent]'

    self.assertEqual(
        {
            'key.segment_context_name': pa.string(),
            'key.frame_timestamp_micros': pa.int64(),
            'key.laser_object_id': pa.string(),
            'key.camera_name': pa.int8(),
            f'{prefix}.ray_origin.values': pa.list_(pa.float32()),
            f'{prefix}.ray_origin.shape': pa.list_(pa.int32(), 3),
            f'{prefix}.ray_direction.values': pa.list_(pa.float32()),
            f'{prefix}.ray_direction.shape': pa.list_(pa.int32(), 3),
        },
        schema_dict,
    )

  def test_creates_correct_keys_ray_compressed_component(self):
    component = _lib.ObjectAssetRayCompressedComponent(
        key=base.ObjectAssetKey(
            segment_context_name='fake_context_name',
            frame_timestamp_micros=123456789,
            laser_object_id='fake_object_id',
            camera_name=_lib_camera.CameraName.FRONT,
        ),
        reference=[0, 1, 2, 3, 4, 5],
        quantized_values=[0, 1, 2, 3, 4, 5],
        shape=[1, 2, 3],
    )

    columns = component.to_flatten_dict()
    prefix = '[ObjectAssetRayCompressedComponent]'
    self.assertCountEqual(
        {
            'key.segment_context_name',
            'key.frame_timestamp_micros',
            'key.laser_object_id',
            'key.camera_name',
            f'{prefix}.reference',
            f'{prefix}.quantized_values',
            f'{prefix}.shape',
        },
        columns,
    )

  def test_creates_correct_schema_ray_compressed_component(self):
    schema = _lib.ObjectAssetRayCompressedComponent.schema()
    schema_dict = dict(zip(schema.names, schema.types))
    prefix = '[ObjectAssetRayCompressedComponent]'

    self.assertEqual(
        {
            'key.segment_context_name': pa.string(),
            'key.frame_timestamp_micros': pa.int64(),
            'key.laser_object_id': pa.string(),
            'key.camera_name': pa.int8(),
            f'{prefix}.reference': pa.list_(pa.float32()),
            f'{prefix}.quantized_values': pa.list_(pa.int32()),
            f'{prefix}.shape': pa.list_(pa.int32()),
        },
        schema_dict,
    )


if __name__ == '__main__':
  absltest.main()
