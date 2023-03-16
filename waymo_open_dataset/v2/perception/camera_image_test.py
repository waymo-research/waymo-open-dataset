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
from waymo_open_dataset.v2.perception import camera_image as _lib


class CameraImageComponentTest(absltest.TestCase):

  def test_creates_correct_keys_camera_image_component(self):
    component = _lib.CameraImageComponent(
        key=base.CameraKey(
            segment_context_name='fake_context_name',
            frame_timestamp_micros=123456789,
            camera_name=1,
        ),
        image=b'\x00\x01\x02\x03\x04',
        pose=column_types.Transform(
            transform=[1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]
        ),
        velocity=_lib.Velocity(
            linear_velocity=column_types.Vec3s(
                x=0, y=1, z=2,
            ),
            angular_velocity=column_types.Vec3d(
                x=4, y=5, z=6,
            ),
        ),
        pose_timestamp=123456789.1234567,
        rolling_shutter_params=_lib.RollingShutterParams(
            shutter=0.01,
            camera_trigger_time=12345,
            camera_readout_done_time=12345.05,
        ),
    )

    columns = component.to_flatten_dict()
    prefix = '[CameraImageComponent]'
    self.assertCountEqual(
        [
            'key.segment_context_name',
            'key.frame_timestamp_micros',
            'key.camera_name',
            f'{prefix}.image',
            f'{prefix}.pose.transform',
            f'{prefix}.velocity.linear_velocity.x',
            f'{prefix}.velocity.linear_velocity.y',
            f'{prefix}.velocity.linear_velocity.z',
            f'{prefix}.velocity.angular_velocity.x',
            f'{prefix}.velocity.angular_velocity.y',
            f'{prefix}.velocity.angular_velocity.z',
            f'{prefix}.pose_timestamp',
            f'{prefix}.rolling_shutter_params.shutter',
            f'{prefix}.rolling_shutter_params.camera_trigger_time',
            f'{prefix}.rolling_shutter_params.camera_readout_done_time',
        ],
        columns.keys(),
    )

  def test_creates_correct_schema(self):
    schema = _lib.CameraImageComponent.schema()
    schema_dict = dict(zip(schema.names, schema.types))
    prefix = '[CameraImageComponent]'
    self.assertEqual(
        {
            'key.segment_context_name': pa.string(),
            'key.frame_timestamp_micros': pa.int64(),
            'key.camera_name': pa.int8(),
            f'{prefix}.image': pa.binary(),
            f'{prefix}.pose.transform': pa.list_(pa.float64(), 16),
            f'{prefix}.velocity.linear_velocity.x': pa.float32(),
            f'{prefix}.velocity.linear_velocity.y': pa.float32(),
            f'{prefix}.velocity.linear_velocity.z': pa.float32(),
            f'{prefix}.velocity.angular_velocity.x': pa.float64(),
            f'{prefix}.velocity.angular_velocity.y': pa.float64(),
            f'{prefix}.velocity.angular_velocity.z': pa.float64(),
            f'{prefix}.pose_timestamp': pa.float64(),
            f'{prefix}.rolling_shutter_params.shutter': pa.float64(),
            f'{prefix}.rolling_shutter_params.camera_trigger_time': (
                pa.float64()
            ),
            f'{prefix}.rolling_shutter_params.camera_readout_done_time': (
                pa.float64()
            ),
        },
        schema_dict,
    )


if __name__ == '__main__':
  absltest.main()
