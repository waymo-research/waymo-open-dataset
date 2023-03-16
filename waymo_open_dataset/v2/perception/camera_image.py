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
"""Camera image components."""

import dataclasses
import enum

import pyarrow as pa

from waymo_open_dataset.v2 import column_types
from waymo_open_dataset.v2 import component
from waymo_open_dataset.v2.perception import base


_column = component.create_column


@dataclasses.dataclass(frozen=True)
class Velocity:
  """A dataclass for 3D linear and augular velocity.

  Attributes:
    linear_velocity: Linear velocity in unit of m/s along x-, y-, and z-axes.
      It's up to the class user to determine which coordinate system to use.
    angular_velocity: Angular velocity in unit of rad/s around the x-, y-, and
      z-axes. It's up to the class user to determine which coordinate system to
      use.
  """
  linear_velocity: column_types.Vec3s = _column()
  angular_velocity: column_types.Vec3d = _column()


@dataclasses.dataclass(frozen=True)
class RollingShutterParams:
  """Rolling shutter parameters.

  The following explanation assumes left->right rolling shutter.

  Rolling shutter cameras expose and read the image column by column, offset
  by the read out time for each column. The desired timestamp for each column
  is the middle of the exposure of that column as outlined below for an image
  with 3 columns:
  ------time------>
  |---- exposure col 1----| read |
  -------|---- exposure col 2----| read |
  --------------|---- exposure col 3----| read |
  ^trigger time                                ^readout end time
              ^time for row 1 (= middle of exposure of row 1)
                     ^time image center (= middle of exposure of middle row)

  Attributes:
    shutter: Shutter duration in seconds. Exposure time per column.
    camera_trigger_time: Time when the sensor was triggered.
    camera_readout_done_time: Time when the last readout finished. The
      difference between trigger time and readout done time includes the
      exposure time and the actual sensor readout time.
  """
  shutter: float = _column(arrow_type=pa.float64())
  camera_trigger_time: float = _column(arrow_type=pa.float64())
  camera_readout_done_time: float = _column(arrow_type=pa.float64())


@dataclasses.dataclass(frozen=True)
class CameraImageComponent(base.CameraComponent):
  """A dataset component for camera image.

  Attributes:
    image: JPEG image.
    pose: SDC pose.
    velocity: SDC velocity at 'pose_timestamp' below. The velocity value is
      represented at *global* frame.
      With this velocity, the pose can be extrapolated.
      r(t+dt) = r(t) + dr/dt * dt where dr/dt = v_{x,y,z}.
      dR(t)/dt = W*R(t) where W = SkewSymmetric(w_{x,y,z})
      This differential equation solves to: R(t) = exp(Wt)*R(0) if W is
      constant.
      When dt is small: R(t+dt) = (I+W*dt)R(t)
      r(t) = (x(t), y(t), z(t)) is vehicle location at t in the global frame.
      R(t) = Rotation Matrix (3x3) from the body frame to the global frame at t.
      SkewSymmetric(x,y,z) is defined as the cross-product matrix in the
      following:
      https://en.wikipedia.org/wiki/Cross_product#Conversion_to_matrix_multiplication
    pose_timestamp: Timestamp of the `pose` above.
  """
  image: bytes = _column(arrow_type=pa.binary())
  pose: column_types.Transform = _column()
  velocity: Velocity = _column()
  pose_timestamp: float = _column(arrow_type=pa.float64())
  rolling_shutter_params: RollingShutterParams = _column()


class CameraName(enum.Enum):
  """Name of camera."""
  UNKNOWN = 0
  FRONT = 1
  FRONT_LEFT = 2
  FRONT_RIGHT = 3
  SIDE_LEFT = 4
  SIDE_RIGHT = 5

