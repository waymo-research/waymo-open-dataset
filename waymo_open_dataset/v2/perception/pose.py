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
"""Pose component."""

import dataclasses

from waymo_open_dataset.v2 import column_types
from waymo_open_dataset.v2.perception import base


@dataclasses.dataclass(frozen=True)
class VehiclePoseComponent(base.FrameComponent):
  """Vehicle pose component."""

  world_from_vehicle: column_types.Transform
