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
from absl.testing import parameterized
import pyarrow as pa
import tensorflow as tf

from waymo_open_dataset.v2.perception import base
from waymo_open_dataset.v2.perception import lidar as _lib


class RangeImageTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ('RangeImage', _lib.RangeImage, 4),
      ('CameraProjectionRangeImage', _lib.CameraProjectionRangeImage, 6),
      ('PoseRangeImage', _lib.PoseRangeImage, 6),
      ('FlowRangeImage', _lib.FlowRangeImage, 4),
  )
  def test_tensor_values_and_shape(self, range_image_cls, inner_dim):
    height, width = 2, 3
    range_image = range_image_cls(
        values=[float(x) for x in range(height * width * inner_dim)],
        shape=[height, width, inner_dim],
    )
    range_image_tensor = range_image.tensor
    self.assertEqual(
        tf.reshape(range_image_tensor, -1).numpy().tolist(),
        range_image.values,
    )
    self.assertEqual(range_image_tensor.shape, [height, width, inner_dim])


class LiDARDataComponentTest(absltest.TestCase):

  def test_creates_correct_keys_lidar_component(self):
    component = _lib.LiDARComponent(
        key=base.LaserKey(
            segment_context_name='fake_context_name',
            frame_timestamp_micros=123456789,
            laser_name=1,
        ),
        range_image_return1=_lib.RangeImage(
            values=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
                    10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
            shape=[2, 2, 5],
        ),
        range_image_return2=_lib.RangeImage(
            values=[19, 18, 17, 16, 15, 14, 13, 12, 11, 10,
                    9, 8, 7, 6, 5, 4, 3, 2, 1, 0],
            shape=[2, 2, 5],
        ),
    )

    columns = component.to_flatten_dict()

    self.assertCountEqual(
        [
            'key.segment_context_name',
            'key.frame_timestamp_micros',
            'key.laser_name',
            '[LiDARComponent].range_image_return1.values',
            '[LiDARComponent].range_image_return1.shape',
            '[LiDARComponent].range_image_return2.values',
            '[LiDARComponent].range_image_return2.shape',
        ],
        columns.keys(),
    )

  def test_creates_correct_keys_lidar_camera_projection_component(self):
    component = _lib.LiDARCameraProjectionComponent(
        key=base.LaserKey(
            segment_context_name='fake_context_name',
            frame_timestamp_micros=123456789,
            laser_name=1,
        ),
        range_image_return1=_lib.CameraProjectionRangeImage(
            values=[0, 1, 2, 3, 4, 5,
                    10, 11, 12, 13, 14, 15],
            shape=[1, 2, 6],
        ),
        range_image_return2=_lib.CameraProjectionRangeImage(
            values=[15, 14, 13, 12, 11, 10,
                    5, 4, 3, 2, 1, 0],
            shape=[1, 2, 6],
        ),
    )

    columns = component.to_flatten_dict()

    self.assertCountEqual(
        [
            'key.segment_context_name',
            'key.frame_timestamp_micros',
            'key.laser_name',
            '[LiDARCameraProjectionComponent].range_image_return1.values',
            '[LiDARCameraProjectionComponent].range_image_return1.shape',
            '[LiDARCameraProjectionComponent].range_image_return2.values',
            '[LiDARCameraProjectionComponent].range_image_return2.shape',
        ],
        columns.keys(),
    )

  def test_creates_correct_keys_lidar_pose_component(self):
    component = _lib.LiDARPoseComponent(
        key=base.LaserKey(
            segment_context_name='fake_context_name',
            frame_timestamp_micros=123456789,
            laser_name=1,
        ),
        range_image_return1=_lib.PoseRangeImage(
            values=[0, 1, 2, 3, 4, 5,
                    10, 11, 12, 13, 14, 15],
            shape=[1, 2, 6],
        ),
    )

    columns = component.to_flatten_dict()

    self.assertCountEqual(
        [
            'key.segment_context_name',
            'key.frame_timestamp_micros',
            'key.laser_name',
            '[LiDARPoseComponent].range_image_return1.values',
            '[LiDARPoseComponent].range_image_return1.shape',
        ],
        columns.keys(),
    )

  def test_creates_correct_keys_lidar_flow_component(self):
    component = _lib.LiDARFlowComponent(
        key=base.LaserKey(
            segment_context_name='fake_context_name',
            frame_timestamp_micros=123456789,
            laser_name=1,
        ),
        range_image_return1=_lib.FlowRangeImage(
            values=[0, 1, 2, 3, 4, 5, 6, 7,
                    10, 11, 12, 13, 14, 15, 16, 17],
            shape=[2, 2, 4],
        ),
        range_image_return2=_lib.FlowRangeImage(
            values=[0, 1, 2, 3, 4, 5, 6, 7,
                    10, 11, 12, 13, 14, 15, 16, 17],
            shape=[2, 2, 4],
        ),
    )

    columns = component.to_flatten_dict()

    self.assertCountEqual(
        [
            'key.segment_context_name',
            'key.frame_timestamp_micros',
            'key.laser_name',
            '[LiDARFlowComponent].range_image_return1.values',
            '[LiDARFlowComponent].range_image_return1.shape',
            '[LiDARFlowComponent].range_image_return2.values',
            '[LiDARFlowComponent].range_image_return2.shape',
        ],
        columns.keys(),
    )

  def test_creates_correct_schema(self):
    schema = _lib.LiDARComponent.schema()
    schema_dict = dict(zip(schema.names, schema.types))
    self.assertEqual(
        {
            'key.segment_context_name': pa.string(),
            'key.frame_timestamp_micros': pa.int64(),
            'key.laser_name': pa.int8(),
            '[LiDARComponent].range_image_return1.values': pa.list_(
                pa.float32()
            ),
            '[LiDARComponent].range_image_return1.shape': pa.list_(
                pa.int32(), 3
            ),
            '[LiDARComponent].range_image_return2.values': pa.list_(
                pa.float32()
            ),
            '[LiDARComponent].range_image_return2.shape': pa.list_(
                pa.int32(), 3
            ),
        },
        schema_dict,
    )


if __name__ == '__main__':
  absltest.main()
