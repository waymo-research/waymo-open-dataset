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
"""Tools to convert vehicle pose v2 to and from v1 Frame protos."""
from typing import Iterable

from waymo_open_dataset.v2 import column_types
from waymo_open_dataset.v2.perception import base as _v2_base
from waymo_open_dataset.v2.perception import pose as _v2_pose
from waymo_open_dataset.v2.perception.compat_v1 import interfaces


class VehiclePoseFrameExtractor(interfaces.FrameComponentExtractor):
  """Extracts human keypoints from frame protos."""

  def __call__(
      self, src: interfaces.FrameComponentSrc
  ) -> Iterable[_v2_pose.VehiclePoseComponent]:
    yield _v2_pose.VehiclePoseComponent(
        key=_v2_base.FrameKey(
            segment_context_name=src.frame.context.name,
            frame_timestamp_micros=src.frame.timestamp_micros,
        ),
        world_from_vehicle=column_types.Transform(
            transform=list(src.frame.pose.transform)
        ),
    )
