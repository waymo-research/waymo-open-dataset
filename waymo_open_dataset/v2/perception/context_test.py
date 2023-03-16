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
from waymo_open_dataset.v2.perception import context as _lib


class ContextTest(absltest.TestCase):

  def test_creates_correct_keys_camera_calibration_component(self):
    component = _lib.CameraCalibrationComponent(
        key=base.SegmentCameraKey(
            segment_context_name='fake_context_name',
            camera_name=1,
        ),
        intrinsic=_lib.Intrinsic(
            f_u=2000,
            f_v=2000,
            c_u=1000,
            c_v=200,
            k1=0.03,
            k2=-0.30,
            p1=-0.0001,
            p2=0.001,
            k3=0.0,
        ),
        extrinsic=column_types.Transform(
            transform=[1] * 16,
        ),
        width=1920,
        height=1280,
        rolling_shutter_direction=_lib.RollingShutterReadOutDirection.LEFT_TO_RIGHT.value,
    )

    columns = component.to_flatten_dict()
    prefix = '[CameraCalibrationComponent]'
    self.assertCountEqual(
        [
            'key.segment_context_name',
            'key.camera_name',
            f'{prefix}.intrinsic.f_u',
            f'{prefix}.intrinsic.f_v',
            f'{prefix}.intrinsic.c_u',
            f'{prefix}.intrinsic.c_v',
            f'{prefix}.intrinsic.k1',
            f'{prefix}.intrinsic.k2',
            f'{prefix}.intrinsic.p1',
            f'{prefix}.intrinsic.p2',
            f'{prefix}.intrinsic.k3',
            f'{prefix}.extrinsic.transform',
            f'{prefix}.width',
            f'{prefix}.height',
            f'{prefix}.rolling_shutter_direction',
        ],
        columns.keys(),
    )

  def test_creates_correct_schema_camera_calibration_component(self):
    schema = _lib.CameraCalibrationComponent.schema()
    schema_dict = dict(zip(schema.names, schema.types))
    prefix = '[CameraCalibrationComponent]'
    self.assertEqual(
        {
            'key.segment_context_name': pa.string(),
            'key.camera_name': pa.int8(),
            f'{prefix}.intrinsic.f_u': pa.float64(),
            f'{prefix}.intrinsic.f_v': pa.float64(),
            f'{prefix}.intrinsic.c_u': pa.float64(),
            f'{prefix}.intrinsic.c_v': pa.float64(),
            f'{prefix}.intrinsic.k1': pa.float64(),
            f'{prefix}.intrinsic.k2': pa.float64(),
            f'{prefix}.intrinsic.p1': pa.float64(),
            f'{prefix}.intrinsic.p2': pa.float64(),
            f'{prefix}.intrinsic.k3': pa.float64(),
            f'{prefix}.extrinsic.transform': pa.list_(pa.float64(), 16),
            f'{prefix}.width': pa.int32(),
            f'{prefix}.height': pa.int32(),
            f'{prefix}.rolling_shutter_direction': pa.int8(),
        },
        schema_dict,
    )

  def test_creates_correct_keys_lidar_calibration_component(self):
    component = _lib.LiDARCalibrationComponent(
        key=base.SegmentLaserKey(
            segment_context_name='fake_context_name',
            laser_name=1,
        ),
        extrinsic=column_types.Transform(
            transform=[1] * 16,
        ),
        beam_inclination=_lib.BeamInclination(
            min=-0.31,
            max=0.04,
            values=[0.01] * 64,
        ),
    )

    columns = component.to_flatten_dict()
    prefix = '[LiDARCalibrationComponent]'
    self.assertCountEqual(
        [
            'key.segment_context_name',
            'key.laser_name',
            f'{prefix}.extrinsic.transform',
            f'{prefix}.beam_inclination.min',
            f'{prefix}.beam_inclination.max',
            f'{prefix}.beam_inclination.values',
        ],
        columns.keys(),
    )

  def test_creates_correct_schema_lidar_calibration_component(self):
    schema = _lib.LiDARCalibrationComponent.schema()
    schema_dict = dict(zip(schema.names, schema.types))
    prefix = '[LiDARCalibrationComponent]'
    self.assertEqual(
        {
            'key.segment_context_name': pa.string(),
            'key.laser_name': pa.int8(),
            f'{prefix}.extrinsic.transform': pa.list_(pa.float64(), 16),
            f'{prefix}.beam_inclination.min': pa.float64(),
            f'{prefix}.beam_inclination.max': pa.float64(),
            f'{prefix}.beam_inclination.values': pa.list_(pa.float64()),
        },
        schema_dict,
    )

  def test_returns_none_for_uninitialized_beam_inclination_values(self):
    component = _lib.LiDARCalibrationComponent(
        key=base.SegmentLaserKey(
            segment_context_name='fake_context_name',
            laser_name=1,
        ),
        extrinsic=column_types.Transform(
            transform=[1] * 16,
        ),
        beam_inclination=_lib.BeamInclination(
            min=-0.31,
            max=0.04,
        ),
    )

    columns = component.to_flatten_dict()
    prefix = '[LiDARCalibrationComponent]'
    self.assertIsNone(columns[f'{prefix}.beam_inclination.values'])

  def test_creates_correct_keys_stats_component(self):
    component = _lib.StatsComponent(
        key=base.FrameKey(
            segment_context_name='fake_context_name',
            frame_timestamp_micros=123456789,
        ),
        time_of_day='Day',
        location='location_moon',
        weather='sunny',
        lidar_object_counts=_lib.ObjectCounts(
            types=[1, 3, 2, 4],
            counts=[10, 20, 30, 40],
        ),
        camera_object_counts=_lib.ObjectCounts(
            types=[1, 2, 3],
            counts=[11, 22, 33],
        ),
    )

    columns = component.to_flatten_dict()
    prefix = '[StatsComponent]'
    self.assertCountEqual(
        [
            'key.segment_context_name',
            'key.frame_timestamp_micros',
            f'{prefix}.time_of_day',
            f'{prefix}.location',
            f'{prefix}.weather',
            f'{prefix}.lidar_object_counts.types',
            f'{prefix}.lidar_object_counts.counts',
            f'{prefix}.camera_object_counts.types',
            f'{prefix}.camera_object_counts.counts',
        ],
        columns.keys(),
    )

  def test_creates_correct_schema_stats_component(self):
    schema = _lib.StatsComponent.schema()
    schema_dict = dict(zip(schema.names, schema.types))
    prefix = '[StatsComponent]'
    self.assertEqual(
        {
            'key.segment_context_name': pa.string(),
            'key.frame_timestamp_micros': pa.int64(),
            f'{prefix}.time_of_day': pa.string(),
            f'{prefix}.location': pa.string(),
            f'{prefix}.weather': pa.string(),
            f'{prefix}.lidar_object_counts.types': pa.list_(pa.int8()),
            f'{prefix}.lidar_object_counts.counts': pa.list_(pa.int32()),
            f'{prefix}.camera_object_counts.types': pa.list_(pa.int8()),
            f'{prefix}.camera_object_counts.counts': pa.list_(pa.int32()),
        },
        schema_dict,
    )

  def test_returns_none_for_uninitialized_object_counts(self):
    component = _lib.StatsComponent(
        key=base.FrameKey(
            segment_context_name='fake_context_name',
            frame_timestamp_micros=123456789,
        ),
        time_of_day='Day',
        location='location_moon',
        weather='sunny',
    )

    columns = component.to_flatten_dict()
    prefix = '[StatsComponent]'
    self.assertIsNone(columns[f'{prefix}.lidar_object_counts.types'])
    self.assertIsNone(columns[f'{prefix}.lidar_object_counts.counts'])
    self.assertIsNone(columns[f'{prefix}.camera_object_counts.types'])
    self.assertIsNone(columns[f'{prefix}.camera_object_counts.counts'])


if __name__ == '__main__':
  absltest.main()
