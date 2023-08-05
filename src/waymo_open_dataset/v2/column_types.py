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
"""Definitions for columns shared between several components across datasets.

Some types below store lists of primitive types because they mean to represent a
collection, for example a list of 3D points. These collections are stored in a
columnar format where each column contains all values for a field from the
collection. See docstring of the `Component` class for more details.
"""
import dataclasses

import numpy as np
import pyarrow as pa

from waymo_open_dataset.v2 import component


_column = component.create_column


@dataclasses.dataclass(frozen=True)
class Vec2s:
  """A dataclass to store single-typed coordinates of a 2D vector."""

  x: float = _column(arrow_type=pa.float32())
  y: float = _column(arrow_type=pa.float32())


@dataclasses.dataclass(frozen=True)
class Vec2d:
  """A dataclass to store double-typed coordinates of a 2D vector."""

  x: float = _column(arrow_type=pa.float64())
  y: float = _column(arrow_type=pa.float64())


@dataclasses.dataclass(frozen=True)
class Vec3s:
  """A dataclass to store single-typed coordinates of a 3D vector."""

  x: float = _column(arrow_type=pa.float32())
  y: float = _column(arrow_type=pa.float32())
  z: float = _column(arrow_type=pa.float32())


@dataclasses.dataclass(frozen=True)
class Vec3d:
  """A dataclass to store double-typed coordinates of a 3D vector.

  Attributes:
    x: A float, indicating the X coordinate of the 3D vector.
    y: A float, indicating the Y coordinate of the 3D vector.
    z: A float, indicating the Z coordinate of the 3D vector.
    numpy: A numpy array, in the order of x, y, z.
  """

  x: float = _column(arrow_type=pa.float64())
  y: float = _column(arrow_type=pa.float64())
  z: float = _column(arrow_type=pa.float64())

  @property
  def numpy(self) -> np.ndarray:
    return np.asarray([self.x, self.y, self.z])


@dataclasses.dataclass(frozen=True)
class Vec2sList:
  """A dataclass to store single-typed coordinates of a list of 2D vectors."""

  x: list[float] = _column(arrow_type=pa.list_(pa.float32()))
  y: list[float] = _column(arrow_type=pa.list_(pa.float32()))


@dataclasses.dataclass(frozen=True)
class Vec2dList:
  """A dataclass to store double-typed coordinates of a list of 2D vectors."""

  x: list[float] = _column(arrow_type=pa.list_(pa.float64()))
  y: list[float] = _column(arrow_type=pa.list_(pa.float64()))


@dataclasses.dataclass(frozen=True)
class Vec3sList:
  """A dataclass to store single-typed coordinates of a list of 3D vectors."""

  x: list[float] = _column(arrow_type=pa.list_(pa.float32()))
  y: list[float] = _column(arrow_type=pa.list_(pa.float32()))
  z: list[float] = _column(arrow_type=pa.list_(pa.float32()))


@dataclasses.dataclass(frozen=True)
class Vec3dList:
  """A dataclass to store double-typed coordinates of a list of 3D vectors."""

  x: list[float] = _column(arrow_type=pa.list_(pa.float64()))
  y: list[float] = _column(arrow_type=pa.list_(pa.float64()))
  z: list[float] = _column(arrow_type=pa.list_(pa.float64()))


@dataclasses.dataclass(frozen=True)
class Transform:
  """A dataclass to store values of a flattened 4x4 transformation matrix."""

  transform: list[float] = _column(arrow_type=pa.list_(pa.float64(), 4 * 4))


@dataclasses.dataclass(frozen=True)
class Box3d:
  """A dataclass representing a 3D 7-DOF box (a.k.a. upright 3D box).

  Note that an upright 3D box has zero pitch and roll.

  Attributes:
    center: Box center coordinates.
    size: Dimensions of the box. Length: dim x. Width: dim y. Height: dim z.
    heading: The heading of the bounding box (in radians), a.k.a. yaw. The
      heading is the angle required to rotate +x to the surface normal of the
      box front face. It is normalized to [-pi, pi).
  """
  center: Vec3d = _column()
  size: Vec3d = _column()
  heading: float = _column(arrow_type=pa.float64())

  def numpy(self, dtype: type(np.dtype)) -> np.ndarray:
    """Return a numpy array, in the order of center, size, heading."""
    return np.asarray(
        [
            self.center.x,
            self.center.y,
            self.center.z,
            self.size.x,
            self.size.y,
            self.size.z,
            self.heading,
        ],
        dtype=dtype,
    )


@dataclasses.dataclass(frozen=True)
class BoxAxisAligned2d:
  """A dataclass representing a 2D box with edges aligned with image axes."""
  center: Vec2d = _column()
  size: Vec2d = _column()
