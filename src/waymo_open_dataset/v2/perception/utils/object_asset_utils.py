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
"""Utils to process Object Asset data components."""

from typing import Tuple
import numpy as np


RAY_DIR_MARGIN = 1e-10


def _sel_bounds(
    bounds: np.ndarray, ray_sign: np.ndarray, dim: int
) -> np.ndarray:
  """Helper function that selects ray bounds for ray-box intersections."""
  return (1 - ray_sign[:, dim]) * bounds[0, dim] + ray_sign[:, dim] * bounds[
      1, dim
  ]


def get_ray_box_intersects(
    ray_origin: np.ndarray, ray_dir: np.ndarray, box_dim: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
  """Computes box-ray intersections.

  Note: we assume point on the ray is represented by `x = origin + t * dir`,
    where `t` is the distance to the origin.
  Args:
    ray_origin: A (N, 3) float array, representing the origins of N rays in the
      centered box frame.
    ray_dir: A (N, 3) float array, representing the directions of N rays.
    box_dim: A float array of shape 3, representing the axis-aligned box
      dimensions centered at the origin.

  Returns:
    A tuple containing intersection flag, the distance to the first
      intersection, and the second intersection.
    - Intersection flag: A boolean array of size N, where k-th element indicates
      whether k-th ray intersects with the box.
    - First / second intersection: A float array of size N, where k-th element
      representing the distance to the ray origin. If the ray's origin is inside
      the box, the first intersection will take negative value, as it is behind
      the origin. If there is no ray-box intersection, both first and second
      ones will take negative values.
  """
  inv_ray_dir = 1 / (ray_dir + RAY_DIR_MARGIN * (ray_dir == 0))
  ray_sign = inv_ray_dir < 0

  box_bounds = np.stack([-0.5 * box_dim, 0.5 * box_dim], axis=0)

  tmin = (
      _sel_bounds(box_bounds, ray_sign, dim=0) - ray_origin[:, 0]
  ) * inv_ray_dir[:, 0]
  tmax = (
      _sel_bounds(box_bounds, 1 - ray_sign, dim=0) - ray_origin[:, 0]
  ) * inv_ray_dir[:, 0]
  tymin = (
      _sel_bounds(box_bounds, ray_sign, dim=1) - ray_origin[:, 1]
  ) * inv_ray_dir[:, 1]
  tymax = (
      _sel_bounds(box_bounds, 1 - ray_sign, dim=1) - ray_origin[:, 1]
  ) * inv_ray_dir[:, 1]

  isect_flag = np.ones(ray_dir.shape[0], dtype=bool)

  isect_flag[np.where(tmin > tymax)[0]] = False
  isect_flag[np.where(tymin > tmax)[0]] = False
  tmin = np.maximum(tmin, tymin)
  tmax = np.minimum(tmax, tymax)

  tzmin = (
      _sel_bounds(box_bounds, ray_sign, dim=2) - ray_origin[:, 2]
  ) * inv_ray_dir[:, 2]
  tzmax = (
      _sel_bounds(box_bounds, 1 - ray_sign, dim=2) - ray_origin[:, 2]
  ) * inv_ray_dir[:, 2]

  isect_flag[np.where(tmin > tzmax)[0]] = False
  isect_flag[np.where(tzmin > tmax)[0]] = False
  tmin = np.maximum(tmin, tzmin)
  tmax = np.minimum(tmax, tzmax)

  isect_flag[np.where(tmax <= 0)[0]] = False
  return isect_flag, tmin, tmax


def transform_points_to_frame(points: np.ndarray,
                              frame_pose_tfm: np.ndarray) -> np.ndarray:
  """Transforms point coordinates using the transformation matrix.

  Assuming we have points defined in A coordinate frame and the A's pose in B
    coordinate frame as a (4, 4) row-major matrix, this function outputs the
    points defined in B coordinate frame.

  Args:
    points: A float array of shape (N, 3), where N is the number of points.
    frame_pose_tfm: A (4, 4) array representing the pose.

  Returns:
    A float array of shape (N, 3) of the transformed point coordinates.
  """
  dtype = points.dtype
  frame_pose_rotation = frame_pose_tfm[0:3, 0:3].astype(np.float64)
  frame_pose_translation = frame_pose_tfm[0:3, 3].astype(np.float64)
  points_tfm = (
      np.matmul(
          frame_pose_rotation, points.astype(np.float64).transpose()
      ).transpose()
      + frame_pose_translation
  )
  return points_tfm.astype(dtype)


def transform_directions_to_frame(
    directions: np.ndarray, frame_pose_tfm: np.ndarray
) -> np.ndarray:
  """Transforms normalized directions using the transformation matrix.

  Assuming we have directions defined in A coordinate frame and the A's pose
    in B coordinate frame as a (4, 4) row-major matrix, this function outputs
    the directions defined in B coordinate frame.

  Args:
    directions: A float array of shape (N, 3), where N is the number of
      direction vectors.
    frame_pose_tfm: A (4, 4) array representing the pose.

  Returns:
    A float array of shape (N, 3) of the transformed point coordinates.
  """
  dtype = directions.dtype
  frame_pose_rotation = frame_pose_tfm[0:3, 0:3].astype(np.float64)
  directions_tfm = np.matmul(
      frame_pose_rotation, directions.astype(np.float64).transpose()
  ).transpose()
  return directions_tfm.astype(dtype)


def rotate_points_along_z(points, rot_angle):
  """Rotate a point cloud around +Z axis.

  Looking from +Z to the XY plane: rotate counter-clockwisely.

  Args:
    points: A numpy array of shape (N, 3), the first 3 channels are XYZ. X is
      facing forward, Y left ward, Z upward.
    rot_angle: A float scalar in radian.

  Returns:
    rotated_pc: A numpy array of shape (N, 3), representing the updated point
      cloud with XYZ rotated.
  """
  rotated_points = np.copy(points)
  cosval = np.cos(rot_angle)
  sinval = np.sin(rot_angle)
  rotmat = np.array([[cosval, -sinval], [sinval, cosval]])
  rotated_points[:, [0, 1]] = np.dot(points[:, [0, 1]], np.transpose(rotmat))
  return rotated_points


def transform_points_to_box_coord(point_cloud, box_3d):
  """Transform a point cloud to coordinates relative to the 3D box.

  The input point cloud and box are in the same coordinate system.
  The box coordinate means that the box center is the origin, the box
  heading direction is the +X. The +Z stays the same as the original coord.

  Args:
    point_cloud: (N, 3) numpy array with XYZ in the 3 channels.
    box_3d: A (7,) shape numpy array with cx,cy,cz,l,w,h,heading.

  Returns:
    output_pc: (N, 3) numpy array with transformed XYZ.
  """
  dtype = point_cloud.dtype
  output_pc = point_cloud - box_3d[None, 0:3]
  heading = box_3d[-1]
  output_pc = rotate_points_along_z(output_pc.astype(np.float64), -1 * heading)
  return output_pc.astype(dtype)


def transform_directions_to_box_coord(
    directions: np.ndarray, box_3d: np.ndarray
) -> np.ndarray:
  """Transforms normalized directions to coordinates relative to the 3D box.

  The input direction and box are in the same coordinate system.
  The box coordinate means that the box center is the origin, the box
  heading direction is the +X. The +Z stays the same as the original coord.

  Args:
    directions: A (N, 3) float array, representing directions (nX-nY-nZ) in the
      3 channels. Here, N is the number of points.
    box_3d: A float array of size 7, with cx,cy,cz,l,w,h,heading.

  Returns:
    A float array of shape (N, 3), indicating the transformed direction vectors.
  """
  dtype = directions.dtype
  output_dir = np.copy(directions).astype(np.float64)
  heading = box_3d[-1]
  output_dir = rotate_points_along_z(output_dir, -1 * heading)
  return output_dir.astype(dtype)


def transform_points_to_box_coord_reverse(
    point_cloud: np.ndarray, box_3d: np.ndarray) -> np.ndarray:
  """Transform a point cloud to the 3D box coordinate.

  The input point cloud and box are in the same coordinate system.
  The box coordinate means that the box center is the origin, the box
  heading direction is the +X. The +Z stays the same as the original coord.

  Args:
    point_cloud: (N, 3) numpy array with XYZ in the 3 channels.
    box_3d: A (7,) shape numpy array with cx,cy,cz,l,w,h,heading.

  Returns:
    output_pc: (N, 3) numpy array with transformed XYZ.
  """
  dtype = point_cloud.dtype
  heading = box_3d[-1]
  output_pc = rotate_points_along_z(point_cloud.astype(np.float64), heading)
  output_pc = output_pc + box_3d[None, 0:3]
  return output_pc.astype(dtype)
