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
from waymo_open_dataset.v2.perception import keypoints as _lib


class CameraHumanKeypointsComponentTest(absltest.TestCase):

  def test_creates_correct_keys_camera_hkp_component(self):
    component = _lib.CameraHumanKeypointsComponent(
        key=base.CameraLabelKey(
            segment_context_name='fake_context_name',
            frame_timestamp_micros=123456789,
            camera_name=1,
            camera_object_id='fake_id',
        ),
        camera_keypoints=_lib.CameraHumanKeypoints(
            type=[1, 20],
            keypoint_2d=_lib.Keypoint2d(
                location_px=column_types.Vec2dList(x=[1.0, 3.0], y=[2.0, 4.0]),
                visibility=_lib.Visibility(is_occluded=[True, False]),
            ),
        ),
    )

    columns = component.to_flatten_dict()

    prefix = '[CameraHumanKeypointsComponent].camera_keypoints[*]'
    self.assertCountEqual(
        [
            'key.segment_context_name',
            'key.frame_timestamp_micros',
            'key.camera_name',
            'key.camera_object_id',
            f'{prefix}.type',
            f'{prefix}.keypoint_2d.location_px.x',
            f'{prefix}.keypoint_2d.location_px.y',
            f'{prefix}.keypoint_2d.visibility.is_occluded',
            f'{prefix}.keypoint_3d.location_m.x',
            f'{prefix}.keypoint_3d.location_m.y',
            f'{prefix}.keypoint_3d.location_m.z',
            f'{prefix}.keypoint_3d.visibility.is_occluded',
        ],
        columns.keys(),
    )
    self.assertIsNone(columns[f'{prefix}.keypoint_3d.location_m.x'])

  def test_returns_none_for_uninitialized_optional_columns(self):
    component = _lib.CameraHumanKeypointsComponent(
        key=base.CameraLabelKey(
            segment_context_name='fake_context_name',
            frame_timestamp_micros=123456789,
            camera_name=1,
            camera_object_id='fake_id',
        ),
        camera_keypoints=_lib.CameraHumanKeypoints(
            type=[1, 20],
            keypoint_2d=_lib.Keypoint2d(
                location_px=column_types.Vec2dList(x=[1.0, 3.0], y=[2.0, 4.0]),
                visibility=_lib.Visibility(is_occluded=[True, False]),
            ),
        ),
    )

    columns = component.to_flatten_dict()

    # keypoint_3d was not initialized.
    prefix = '[CameraHumanKeypointsComponent].camera_keypoints[*]'
    self.assertIsNone(columns[f'{prefix}.keypoint_3d.location_m.x'])
    self.assertIsNone(columns[f'{prefix}.keypoint_3d.location_m.y'])
    self.assertIsNone(columns[f'{prefix}.keypoint_3d.visibility.is_occluded'])

  def test_creates_correct_schema(self):
    schema = _lib.CameraHumanKeypointsComponent.schema()
    schema_dict = dict(zip(schema.names, schema.types))

    prefix = '[CameraHumanKeypointsComponent].camera_keypoints[*]'
    self.assertEqual(
        {
            'key.segment_context_name': pa.string(),
            'key.frame_timestamp_micros': pa.int64(),
            'key.camera_name': pa.int8(),
            'key.camera_object_id': pa.string(),
            f'{prefix}.type': pa.list_(pa.int8()),
            f'{prefix}.keypoint_2d.location_px.x': pa.list_(pa.float64()),
            f'{prefix}.keypoint_2d.location_px.y': pa.list_(pa.float64()),
            f'{prefix}.keypoint_2d.visibility.is_occluded': pa.list_(
                pa.bool_()
            ),
            f'{prefix}.keypoint_3d.location_m.x': pa.list_(pa.float64()),
            f'{prefix}.keypoint_3d.location_m.y': pa.list_(pa.float64()),
            f'{prefix}.keypoint_3d.location_m.z': pa.list_(pa.float64()),
            f'{prefix}.keypoint_3d.visibility.is_occluded': pa.list_(
                pa.bool_()
            ),
        },
        schema_dict,
    )


class LiDARHumanKeypointsComponentTest(absltest.TestCase):

  def test_creates_correct_keys_lidar_hkp_component(self):
    component = _lib.LiDARHumanKeypointsComponent(
        key=base.LaserLabelKey(
            segment_context_name='fake_context_name',
            frame_timestamp_micros=123456789,
            laser_object_id='fake_id',
        ),
        lidar_keypoints=_lib.LiDARHumanKeypoints(
            type=[1, 20],
            keypoint_3d=_lib.Keypoint3d(
                location_m=column_types.Vec3dList(
                    x=[1.0, 2.0], y=[3.0, 4.0], z=[5.0, 6.0]
                ),
                visibility=_lib.Visibility(is_occluded=[True, False, False]),
            ),
        )
    )

    columns = component.to_flatten_dict()

    prefix = '[LiDARHumanKeypointsComponent].lidar_keypoints[*]'
    self.assertCountEqual(
        [
            'key.segment_context_name',
            'key.frame_timestamp_micros',
            'key.laser_object_id',
            f'{prefix}.type',
            f'{prefix}.keypoint_3d.location_m.x',
            f'{prefix}.keypoint_3d.location_m.y',
            f'{prefix}.keypoint_3d.location_m.z',
            f'{prefix}.keypoint_3d.visibility.is_occluded',
        ],
        columns.keys(),
    )

  def test_creates_correct_schema(self):
    schema = _lib.LiDARHumanKeypointsComponent.schema()
    schema_dict = dict(zip(schema.names, schema.types))

    prefix = '[LiDARHumanKeypointsComponent].lidar_keypoints[*]'
    self.assertEqual(
        {
            'key.segment_context_name': pa.string(),
            'key.frame_timestamp_micros': pa.int64(),
            'key.laser_object_id': pa.string(),
            f'{prefix}.type': pa.list_(pa.int8()),
            f'{prefix}.keypoint_3d.location_m.x': pa.list_(pa.float64()),
            f'{prefix}.keypoint_3d.location_m.y': pa.list_(pa.float64()),
            f'{prefix}.keypoint_3d.location_m.z': pa.list_(pa.float64()),
            f'{prefix}.keypoint_3d.visibility.is_occluded': pa.list_(
                pa.bool_()
            ),
        },
        schema_dict,
    )

if __name__ == '__main__':
  absltest.main()
