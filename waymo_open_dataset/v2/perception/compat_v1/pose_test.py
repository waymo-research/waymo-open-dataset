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
"""Tests for waymo_open_dataset.v2.perception.compat_v1.pose."""

from absl.testing import absltest

from waymo_open_dataset import dataset_pb2 as _v1_dataset_pb2
from waymo_open_dataset.v2.perception import pose as _pose_component
from waymo_open_dataset.v2.perception.compat_v1 import interfaces
from waymo_open_dataset.v2.perception.compat_v1 import pose as _pose_extractor


class PoseTest(absltest.TestCase):

  def test_extractor_returns_same_schama(self):
    self.assertEqual(
        _pose_extractor.VehiclePoseFrameExtractor.schema(),
        _pose_component.VehiclePoseComponent.schema(),
    )

  def test_populates_all_proto_fields(self):
    src = interfaces.FrameComponentSrc(
        frame=_v1_dataset_pb2.Frame(
            context={'name': 'dummy_segment'},
            timestamp_micros=123456,
            pose={'transform': [1] * 16},
        )
    )

    extractor = _pose_extractor.VehiclePoseFrameExtractor()
    component = next(extractor(src))

    self.assertEqual(component.key.segment_context_name, 'dummy_segment')
    self.assertEqual(component.key.frame_timestamp_micros, 123456)
    self.assertEqual(component.world_from_vehicle.transform, [1] * 16)


if __name__ == '__main__':
  absltest.main()
