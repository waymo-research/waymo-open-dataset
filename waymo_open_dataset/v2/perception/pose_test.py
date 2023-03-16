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
"""Tests for waymo_open_dataset.v2.perception.pose."""

from absl.testing import absltest
import pyarrow as pa

from waymo_open_dataset.v2 import column_types
from waymo_open_dataset.v2.perception import base
from waymo_open_dataset.v2.perception import pose as _lib


class PoseTest(absltest.TestCase):

  def test_creates_correct_keys(self):
    component = _lib.VehiclePoseComponent(
        key=base.FrameKey(
            segment_context_name='fake_context_name',
            frame_timestamp_micros=123456789,
        ),
        world_from_vehicle=column_types.Transform(transform=[1] * 16),
    )

    columns = component.to_flatten_dict()

    self.assertCountEqual(
        [
            'key.segment_context_name',
            'key.frame_timestamp_micros',
            '[VehiclePoseComponent].world_from_vehicle.transform',
        ],
        columns,
    )

  def test_creates_correct_schema(self):
    schema = _lib.VehiclePoseComponent.schema()
    schema_dict = dict(zip(schema.names, schema.types))
    self.assertEqual(
        {
            'key.segment_context_name': pa.string(),
            'key.frame_timestamp_micros': pa.int64(),
            '[VehiclePoseComponent].world_from_vehicle.transform': pa.list_(
                pa.float64(), 16
            ),
        },
        schema_dict,
    )


if __name__ == '__main__':
  absltest.main()
