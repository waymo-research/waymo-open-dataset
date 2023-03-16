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

from waymo_open_dataset.v2.perception import base
from waymo_open_dataset.v2.perception import segmentation as _lib


class CameraSegmentationLabelTest(absltest.TestCase):

  def test_creates_correct_keys_camera_segmentation_label_component(self):
    component = _lib.CameraSegmentationLabelComponent(
        key=base.CameraKey(
            segment_context_name='fake_context_name',
            frame_timestamp_micros=123456789,
            camera_name=1,
        ),
        panoptic_label_divisor=1000,
        panoptic_label=b'\x00\x01\x02\x03\x04',
        instance_id_to_global_id_mapping=_lib.InstanceIdToGlobalIdMapping(
            local_instance_ids=[1, 2, 3],
            global_instance_ids=[10, 11, 13],
            is_tracked=[True, True, False],
        ),
        sequence_id='15692902915025018598',
        num_cameras_covered=b'\x00\x01\x02\x03\x04',
    )

    columns = component.to_flatten_dict()
    prefix = '[CameraSegmentationLabelComponent]'
    self.assertCountEqual(
        [
            'key.segment_context_name',
            'key.frame_timestamp_micros',
            'key.camera_name',
            f'{prefix}.panoptic_label_divisor',
            f'{prefix}.panoptic_label',
            f'{prefix}.instance_id_to_global_id_mapping.local_instance_ids',
            f'{prefix}.instance_id_to_global_id_mapping.global_instance_ids',
            f'{prefix}.instance_id_to_global_id_mapping.is_tracked',
            f'{prefix}.sequence_id',
            f'{prefix}.num_cameras_covered',
        ],
        columns.keys(),
    )

  def test_creates_correct_schema(self):
    schema = _lib.CameraSegmentationLabelComponent.schema()
    schema_dict = dict(zip(schema.names, schema.types))
    prefix = '[CameraSegmentationLabelComponent]'
    self.assertEqual(
        {
            'key.segment_context_name': pa.string(),
            'key.frame_timestamp_micros': pa.int64(),
            'key.camera_name': pa.int8(),
            f'{prefix}.panoptic_label_divisor': pa.int32(),
            f'{prefix}.panoptic_label': pa.binary(),
            f'{prefix}.instance_id_to_global_id_mapping.local_instance_ids': (
                pa.list_(pa.int32())
            ),
            f'{prefix}.instance_id_to_global_id_mapping.global_instance_ids': (
                pa.list_(pa.int32())
            ),
            f'{prefix}.instance_id_to_global_id_mapping.is_tracked': pa.list_(
                pa.bool_()
            ),
            f'{prefix}.sequence_id': pa.string(),
            f'{prefix}.num_cameras_covered': pa.binary(),
        },
        schema_dict,
    )


class LiDARSegmentationLabelTest(absltest.TestCase):

  def test_creates_correct_keys_lidar_segmentation_label_component(self):
    component = _lib.LiDARSegmentationLabelComponent(
        key=base.LaserKey(
            segment_context_name='fake_context_name',
            frame_timestamp_micros=123456789,
            laser_name=1,
        ),
        range_image_return1=_lib.LiDARSegmentationRangeImage(
            values=list(range(2 * 3 * 2)),
            shape=[2, 3, 2],
        ),
        range_image_return2=_lib.LiDARSegmentationRangeImage(
            values=list(range(2 * 3 * 2)),
            shape=[2, 3, 2],
        ),
    )

    columns = component.to_flatten_dict()
    prefix = '[LiDARSegmentationLabelComponent]'
    self.assertCountEqual(
        [
            'key.segment_context_name',
            'key.frame_timestamp_micros',
            'key.laser_name',
            f'{prefix}.range_image_return1.values',
            f'{prefix}.range_image_return1.shape',
            f'{prefix}.range_image_return2.values',
            f'{prefix}.range_image_return2.shape',
        ],
        columns.keys(),
    )

  def test_creates_correct_schema(self):
    schema = _lib.LiDARSegmentationLabelComponent.schema()
    schema_dict = dict(zip(schema.names, schema.types))
    prefix = '[LiDARSegmentationLabelComponent]'
    self.assertEqual(
        {
            'key.segment_context_name': pa.string(),
            'key.frame_timestamp_micros': pa.int64(),
            'key.laser_name': pa.int8(),
            f'{prefix}.range_image_return1.values': pa.list_(pa.int32()),
            f'{prefix}.range_image_return1.shape': pa.list_(pa.int32(), 3),
            f'{prefix}.range_image_return2.values': pa.list_(pa.int32()),
            f'{prefix}.range_image_return2.shape': pa.list_(pa.int32(), 3),
        },
        schema_dict,
    )


if __name__ == '__main__':
  absltest.main()
