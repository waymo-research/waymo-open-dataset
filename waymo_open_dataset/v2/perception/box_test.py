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
import pyarrow as pa

from waymo_open_dataset.v2 import column_types
from waymo_open_dataset.v2.perception import base
from waymo_open_dataset.v2.perception import box as _lib


class BoxComponentTest(absltest.TestCase):

  def test_creates_correct_keys_lidar_box_component(self):
    component = _lib.LiDARBoxComponent(
        key=base.LaserLabelKey(
            segment_context_name='fake_context_name',
            frame_timestamp_micros=123456789,
            laser_object_id='fake_object_id',
        ),
        box=column_types.Box3d(
            center=column_types.Vec3d(
                x=1, y=2, z=3,
            ),
            size=column_types.Vec3d(
                x=1, y=2, z=3,
            ),
            heading=0,
        ),
        type=_lib.BoxType.TYPE_VEHICLE,
        speed=column_types.Vec3d(
            x=1, y=2, z=3,
        ),
        acceleration=column_types.Vec3d(
            x=4, y=5, z=6,
        ),
        num_lidar_points_in_box=100,
        num_top_lidar_points_in_box=80,
        difficulty_level=_lib.DifficultyLevel(
            detection=_lib.DifficultyLevelType.LEVEL_1,
            tracking=_lib.DifficultyLevelType.LEVEL_1,
        ),
    )

    columns = component.to_flatten_dict()
    prefix = '[LiDARBoxComponent]'
    self.assertCountEqual(
        [
            'key.segment_context_name',
            'key.frame_timestamp_micros',
            'key.laser_object_id',
            f'{prefix}.box.center.x',
            f'{prefix}.box.center.y',
            f'{prefix}.box.center.z',
            f'{prefix}.box.size.x',
            f'{prefix}.box.size.y',
            f'{prefix}.box.size.z',
            f'{prefix}.box.heading',
            f'{prefix}.type',
            f'{prefix}.speed.x',
            f'{prefix}.speed.y',
            f'{prefix}.speed.z',
            f'{prefix}.acceleration.x',
            f'{prefix}.acceleration.y',
            f'{prefix}.acceleration.z',
            f'{prefix}.num_lidar_points_in_box',
            f'{prefix}.num_top_lidar_points_in_box',
            f'{prefix}.difficulty_level.detection',
            f'{prefix}.difficulty_level.tracking',
        ],
        columns.keys(),
    )

  def test_creates_correct_schema_lidar_box_component(self):
    schema = _lib.LiDARBoxComponent.schema()
    schema_dict = dict(zip(schema.names, schema.types))
    prefix = '[LiDARBoxComponent]'
    self.assertEqual(
        {
            'key.segment_context_name': pa.string(),
            'key.frame_timestamp_micros': pa.int64(),
            'key.laser_object_id': pa.string(),
            f'{prefix}.box.center.x': pa.float64(),
            f'{prefix}.box.center.y': pa.float64(),
            f'{prefix}.box.center.z': pa.float64(),
            f'{prefix}.box.size.x': pa.float64(),
            f'{prefix}.box.size.y': pa.float64(),
            f'{prefix}.box.size.z': pa.float64(),
            f'{prefix}.box.heading': pa.float64(),
            f'{prefix}.type': pa.int8(),
            f'{prefix}.speed.x': pa.float64(),
            f'{prefix}.speed.y': pa.float64(),
            f'{prefix}.speed.z': pa.float64(),
            f'{prefix}.acceleration.x': pa.float64(),
            f'{prefix}.acceleration.y': pa.float64(),
            f'{prefix}.acceleration.z': pa.float64(),
            f'{prefix}.num_lidar_points_in_box': pa.int64(),
            f'{prefix}.num_top_lidar_points_in_box': pa.int64(),
            f'{prefix}.difficulty_level.detection': pa.int8(),
            f'{prefix}.difficulty_level.tracking': pa.int8(),
        },
        schema_dict,
    )

  def test_creates_correct_keys_lidar_camera_synced_box_component(self):
    component = _lib.LiDARCameraSyncedBoxComponent(
        key=base.LaserLabelKey(
            segment_context_name='fake_context_name',
            frame_timestamp_micros=123456789,
            laser_object_id='fake_object_id',
        ),
        most_visible_camera_name=1,
        camera_synced_box=column_types.Box3d(
            center=column_types.Vec3d(
                x=1, y=2, z=3,
            ),
            size=column_types.Vec3d(
                x=1, y=2, z=3,
            ),
            heading=1.5,
        )
    )
    columns = component.to_flatten_dict()
    prefix = '[LiDARCameraSyncedBoxComponent]'
    self.assertCountEqual(
        [
            'key.segment_context_name',
            'key.frame_timestamp_micros',
            'key.laser_object_id',
            f'{prefix}.most_visible_camera_name',
            f'{prefix}.camera_synced_box.center.x',
            f'{prefix}.camera_synced_box.center.y',
            f'{prefix}.camera_synced_box.center.z',
            f'{prefix}.camera_synced_box.size.x',
            f'{prefix}.camera_synced_box.size.y',
            f'{prefix}.camera_synced_box.size.z',
            f'{prefix}.camera_synced_box.heading',
        ],
        columns.keys(),
    )

  def test_creates_correct_schema_lidar_camera_synced_box_component(self):
    schema = _lib.LiDARCameraSyncedBoxComponent.schema()
    schema_dict = dict(zip(schema.names, schema.types))
    prefix = '[LiDARCameraSyncedBoxComponent]'
    self.assertEqual(
        {
            'key.segment_context_name': pa.string(),
            'key.frame_timestamp_micros': pa.int64(),
            'key.laser_object_id': pa.string(),
            f'{prefix}.most_visible_camera_name': pa.int8(),
            f'{prefix}.camera_synced_box.center.x': pa.float64(),
            f'{prefix}.camera_synced_box.center.y': pa.float64(),
            f'{prefix}.camera_synced_box.center.z': pa.float64(),
            f'{prefix}.camera_synced_box.size.x': pa.float64(),
            f'{prefix}.camera_synced_box.size.y': pa.float64(),
            f'{prefix}.camera_synced_box.size.z': pa.float64(),
            f'{prefix}.camera_synced_box.heading': pa.float64(),
        },
        schema_dict,
    )

  def test_creates_correct_keys_projected_lidar_box_component(self):
    component = _lib.ProjectedLiDARBoxComponent(
        key=_lib.ProjectedLaserLabelKey(
            segment_context_name='fake_context_name',
            frame_timestamp_micros=123456789,
            laser_object_id='fake_object_id',
            camera_name=1,
        ),
        box=column_types.BoxAxisAligned2d(
            center=column_types.Vec2d(
                x=1, y=2,
            ),
            size=column_types.Vec2d(
                x=3, y=4,
            ),
        ),
        type=_lib.BoxType.TYPE_PEDESTRIAN,
    )
    columns = component.to_flatten_dict()
    prefix = '[ProjectedLiDARBoxComponent]'
    self.assertCountEqual(
        [
            'key.segment_context_name',
            'key.frame_timestamp_micros',
            'key.laser_object_id',
            'key.camera_name',
            f'{prefix}.box.center.x',
            f'{prefix}.box.center.y',
            f'{prefix}.box.size.x',
            f'{prefix}.box.size.y',
            f'{prefix}.type',
        ],
        columns.keys(),
    )

  def test_creates_correct_schema_projected_lidar_box_component(self):
    schema = _lib.ProjectedLiDARBoxComponent.schema()
    schema_dict = dict(zip(schema.names, schema.types))
    prefix = '[ProjectedLiDARBoxComponent]'
    self.assertEqual(
        {
            'key.segment_context_name': pa.string(),
            'key.frame_timestamp_micros': pa.int64(),
            'key.laser_object_id': pa.string(),
            'key.camera_name': pa.int8(),
            f'{prefix}.box.center.x': pa.float64(),
            f'{prefix}.box.center.y': pa.float64(),
            f'{prefix}.box.size.x': pa.float64(),
            f'{prefix}.box.size.y': pa.float64(),
            f'{prefix}.type': pa.int8(),
        },
        schema_dict,
    )

  def test_creates_correct_keys_camera_box_component(self):
    component = _lib.CameraBoxComponent(
        key=base.CameraLabelKey(
            segment_context_name='fake_context_name',
            frame_timestamp_micros=123456789,
            camera_name=1,
            camera_object_id='fake_object_id',
        ),
        box=column_types.BoxAxisAligned2d(
            center=column_types.Vec2d(
                x=1, y=2,
            ),
            size=column_types.Vec2d(
                x=3, y=4,
            ),
        ),
        type=_lib.BoxType.TYPE_CYCLIST,
        difficulty_level=_lib.DifficultyLevel(
            detection=_lib.DifficultyLevelType.LEVEL_2,
            tracking=_lib.DifficultyLevelType.LEVEL_2,
        ),
    )
    columns = component.to_flatten_dict()
    prefix = '[CameraBoxComponent]'
    self.assertCountEqual(
        [
            'key.segment_context_name',
            'key.frame_timestamp_micros',
            'key.camera_name',
            'key.camera_object_id',
            f'{prefix}.box.center.x',
            f'{prefix}.box.center.y',
            f'{prefix}.box.size.x',
            f'{prefix}.box.size.y',
            f'{prefix}.type',
            f'{prefix}.difficulty_level.detection',
            f'{prefix}.difficulty_level.tracking',
        ],
        columns.keys(),
    )

  def test_creates_correct_schema_camera_box_component(self):
    schema = _lib.CameraBoxComponent.schema()
    schema_dict = dict(zip(schema.names, schema.types))
    prefix = '[CameraBoxComponent]'
    self.assertEqual(
        {
            'key.segment_context_name': pa.string(),
            'key.frame_timestamp_micros': pa.int64(),
            'key.camera_name': pa.int8(),
            'key.camera_object_id': pa.string(),
            f'{prefix}.box.center.x': pa.float64(),
            f'{prefix}.box.center.y': pa.float64(),
            f'{prefix}.box.size.x': pa.float64(),
            f'{prefix}.box.size.y': pa.float64(),
            f'{prefix}.type': pa.int8(),
            f'{prefix}.difficulty_level.detection': pa.int8(),
            f'{prefix}.difficulty_level.tracking': pa.int8(),
        },
        schema_dict,
    )

  def test_creates_correct_keys_camera_to_lidar_box_association_component(self):
    component = _lib.CameraToLiDARBoxAssociationComponent(
        key=_lib.CameraToLiDARBoxAssociationKey(
            segment_context_name='fake_context_name',
            frame_timestamp_micros=123456789,
            camera_name=1,
            camera_object_id='fake_object_id',
            laser_object_id='fake_laser_object_id',
        )
    )
    columns = component.to_flatten_dict()
    self.assertCountEqual(
        [
            'key.segment_context_name',
            'key.frame_timestamp_micros',
            'key.camera_name',
            'key.camera_object_id',
            'key.laser_object_id',
        ],
        columns.keys(),
    )

  def test_creates_correct_schema_camera_to_lidar_box_association_component(
      self,
  ):
    schema = _lib.CameraToLiDARBoxAssociationComponent.schema()
    schema_dict = dict(zip(schema.names, schema.types))
    self.assertEqual(
        {
            'key.segment_context_name': pa.string(),
            'key.frame_timestamp_micros': pa.int64(),
            'key.camera_name': pa.int8(),
            'key.camera_object_id': pa.string(),
            'key.laser_object_id': pa.string(),
        },
        schema_dict,
    )


if __name__ == '__main__':
  absltest.main()
