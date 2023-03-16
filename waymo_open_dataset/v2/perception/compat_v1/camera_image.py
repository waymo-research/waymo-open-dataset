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
"""Tools to convert camera image data v2 to and from v1 Frame protos."""
from typing import Iterable

from waymo_open_dataset.v2 import column_types
from waymo_open_dataset.v2.perception import base as _v2_base
from waymo_open_dataset.v2.perception import camera_image as _v2_camera_image
from waymo_open_dataset.v2.perception.compat_v1 import interfaces


class CameraImageComponentExtractor(interfaces.CameraImageComponentExtractor):
  """Extracts camera image data component captured by one camera."""

  def __call__(
      self, src: interfaces.CameraImageComponentSrc
  ) -> Iterable[_v2_camera_image.CameraImageComponent]:
    key = _v2_base.CameraKey(
        segment_context_name=src.frame.context.name,
        frame_timestamp_micros=src.frame.timestamp_micros,
        camera_name=src.camera_image.name,
    )
    pose = column_types.Transform(
        transform=list(src.camera_image.pose.transform),
    )
    velocity_v1 = src.camera_image.velocity
    velocity = _v2_camera_image.Velocity(
        linear_velocity=column_types.Vec3s(
            x=velocity_v1.v_x,
            y=velocity_v1.v_y,
            z=velocity_v1.v_z,
        ),
        angular_velocity=column_types.Vec3d(
            x=velocity_v1.w_x,
            y=velocity_v1.w_y,
            z=velocity_v1.w_z,
        ),
    )
    rolling_shutter_params = _v2_camera_image.RollingShutterParams(
        shutter=src.camera_image.shutter,
        camera_trigger_time=src.camera_image.camera_trigger_time,
        camera_readout_done_time=src.camera_image.camera_readout_done_time,
    )
    yield _v2_camera_image.CameraImageComponent(
        key=key,
        image=src.camera_image.image,
        pose=pose,
        velocity=velocity,
        pose_timestamp=src.camera_image.pose_timestamp,
        rolling_shutter_params=rolling_shutter_params,
    )
