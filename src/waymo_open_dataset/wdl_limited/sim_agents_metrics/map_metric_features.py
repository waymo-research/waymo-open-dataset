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
# =============================================================================
"""Map-based metric features for sim agents."""

from typing import Sequence

import tensorflow as tf

from waymo_open_dataset.protos import map_pb2
from waymo_open_dataset.utils import box_utils
from waymo_open_dataset.utils import geometry_utils

# Constant distance to apply when distances are invalid. This will avoid the
# propagation of nans and should be reduced out when taking the maximum anyway.
_EXTREMELY_LARGE_DISTANCE = 1e10
# Off-road threshold, i.e. smallest distance away from the road edge that is
# considered to be a off-road.
OFFROAD_DISTANCE_THRESHOLD = 0.0

# How close the start and end point of a map feature need to be for the feature
# to be considered cyclic, in m^2.
_CYCLIC_MAP_FEATURE_TOLERANCE_M2 = 1.0


_Polyline = Sequence[map_pb2.MapPoint]


def compute_distance_to_road_edge(
    *,
    center_x: tf.Tensor,
    center_y: tf.Tensor,
    center_z: tf.Tensor,
    length: tf.Tensor,
    width: tf.Tensor,
    height: tf.Tensor,
    heading: tf.Tensor,
    valid: tf.Tensor,
    evaluated_object_mask: tf.Tensor,
    road_edge_polylines: Sequence[_Polyline]
) -> tf.Tensor:
  """Computes the distance to the road edge for each of the evaluated objects.

  Args:
    center_x: A float Tensor of shape (num_objects, num_steps) containing the
      x-component of the object positions.
    center_y: A float Tensor of shape (num_objects, num_steps) containing the
      y-component of the object positions.
    center_z: A float Tensor of shape (num_objects, num_steps) containing the
      z-component of the object positions.
    length: A float Tensor of shape (num_objects, num_steps) containing the
      object lengths.
    width: A float Tensor of shape (num_objects, num_steps) containing the
      object widths.
    height: A float Tensor of shape (num_objects, num_steps) containing the
      object heights.
    heading: A float Tensor of shape (num_objects, num_steps) containing the
      object headings.
    valid: A boolean Tensor of shape (num_objects, num_steps) containing the
      validity of the objects over time.
    evaluated_object_mask: A boolean tensor of shape (num_objects), indicating
      whether each object should be considered part of the "evaluation set".
    road_edge_polylines: A sequence of polylines, each defined as a sequence of
      3d points with x, y, and z-coordinates. The polylines should be oriented
      such that port side is on-road and starboard side is off-road, a.k.a
      counterclockwise winding order.

  Returns:
    A tensor of shape (num_evaluated_objects, num_steps), containing the
    distance to the road edge, for each timestep and for all the objects
    to be evaluated, as specified by `evaluated_object_mask`.

  Raises:
    ValueError: When the `road_edge_polylines` is empty, i.e. there is no map
      information in the Scenario.
  """
  if not road_edge_polylines:
    raise ValueError('Missing road edges.')

  # Concatenate tensors to have the same convention as `box_utils`.
  boxes = tf.stack(
      [center_x, center_y, center_z, length, width, height, heading], axis=-1)
  num_objects, num_steps, num_features = boxes.shape
  boxes = tf.reshape(boxes, [num_objects * num_steps, num_features])
  # Compute box corners using `box_utils`, and take the xy coords of the bottom
  # corners, as we are only computing distances for 2D boxes.
  box_corners = box_utils.get_upright_3d_box_corners(boxes)[:, :4, :2]
  box_corners = tf.reshape(box_corners, (num_objects, num_steps, 4, 2))

  # Gather objects in the evaluation set
  # `eval_corners` shape: (num_evaluated_objects, num_steps, 4, 2).
  eval_corners = tf.gather(
      box_corners, tf.where(evaluated_object_mask)[:, 0], axis=0)
  num_eval_objects = eval_corners.shape[0]

  # Flatten query points.
  # `flat_eval_corners` shape: (num_query_points, 2).
  flat_eval_corners = tf.reshape(eval_corners, (-1, 2))

  # Tensorize road edges.
  polyline_tensors = []
  for polyline in road_edge_polylines:
    polyline_tensors.append(
        tf.constant([[map_point.x, map_point.y] for map_point in polyline]))

  # Compute distances for all query points.
  # `corner_distance_to_road_edge` shape: (num_query_points).
  corner_distance_to_road_edge = _compute_signed_distance_to_polylines(
      xys=flat_eval_corners, polylines=polyline_tensors
  )
  # `corner_distance_to_road_edge` shape: (num_evaluated_objects, num_steps, 4).
  corner_distance_to_road_edge = tf.reshape(
      corner_distance_to_road_edge, (num_eval_objects, num_steps, 4)
  )

  # Reduce to most off-road corner.
  # `signed_distances` shape: (num_evaluated_objects, num_steps).
  signed_distances = tf.math.reduce_max(corner_distance_to_road_edge, axis=-1)

  # Mask out invalid boxes.
  eval_validity = tf.gather(
      valid, tf.where(evaluated_object_mask)[:, 0], axis=0)
  return tf.where(eval_validity, signed_distances, -_EXTREMELY_LARGE_DISTANCE)


def _compute_signed_distance_to_polylines(
    xys: tf.Tensor,
    polylines: Sequence[tf.Tensor],
) -> tf.Tensor:
  """Computes the signed distance to the 2D boundary defined by polylines.

  Negative distances correspond to being inside the boundary (e.g. on the
  road), positive distances to being outside (e.g. off-road).

  The polylines should be oriented such that port side is inside the boundary
  and starboard is outside, a.k.a counterclockwise winding order.

  Note: degenerate segments (start == end) can cause undefined behaviour.

  Args:
    xys: A float Tensor of shape (num_points, 2) containing xy coordinates of
      query points.
    polylines: List of tensors of shape (num_segments+1, 2) containing sequences
      of xy coordinates representing start and end points of consecutive
      segments.

  Returns:
    A tensor of shape (num_points), containing the signed distance from queried
      points to the nearest polyline.
  """
  distances = []
  for polyline in polylines:
    # Skip degenerate polylines.
    if len(polyline) < 2:
      continue

    distances.append(_compute_signed_distance_to_polyline(xys, polyline))

  # `distances` shape: (num_points, num_nondegenerate_polylines).
  distances = tf.stack(distances, axis=-1)
  return tf.gather(
      distances, tf.argmin(tf.abs(distances), axis=-1), batch_dims=1
  )


def _compute_signed_distance_to_polyline(
    xys: tf.Tensor,
    polyline: tf.Tensor,
) -> tf.Tensor:
  """Computes the signed distance to the 2D boundary defined by a polyline.

  Negative distances correspond to being inside the boundary (e.g. on the
  road), positive distances to being outside (e.g. off-road).

  The polyline should be oriented such that port side is inside the boundary
  and starboard is outside, a.k.a counterclockwise winding order.

  Note: degenerate segments (start == end) can cause undefined behaviour.

  Args:
    xys: A float Tensor of shape (num_points, 2) containing xy coordinates of
      query points.
    polyline: A float Tensor of shape (num_segments+1, 2) containing sequences
      of xy coordinates representing start and end points of consecutive
      segments.

  Returns:
    A tensor of shape (num_points), containing the signed distance from queried
      points to the polyline.
  """
  is_cyclic = (
      tf.reduce_sum(tf.math.square(polyline[0] - polyline[-1]))
      < _CYCLIC_MAP_FEATURE_TOLERANCE_M2
  )
  # Get distance to each segment.
  # shape: (num_points, num_segments, 2)
  xy_starts = polyline[tf.newaxis, :-1, :2]
  xy_ends = polyline[tf.newaxis, 1:, :2]
  start_to_point = xys[:, tf.newaxis, :2] - xy_starts
  start_to_end = xy_ends - xy_starts

  # Relative coordinate of point projection on segment.
  # shape: (num_points, num_segments)
  rel_t = tf.math.divide_no_nan(
      geometry_utils.dot_product_2d(start_to_point, start_to_end),
      geometry_utils.dot_product_2d(start_to_end, start_to_end),
  )

  # Negative if point is on port side of segment, positive if point on
  # starboard side of segment.
  # shape: (num_points, num_segments)
  n = tf.sign(geometry_utils.cross_product_2d(start_to_point, start_to_end))
  # Absolute distance to segment.
  # shape: (n_points, n_segments)
  distance_to_segment = tf.linalg.norm(
      start_to_point
      - (start_to_end * tf.clip_by_value(rel_t, 0.0, 1.0)[..., tf.newaxis]),
      axis=-1,
  )

  # There are 3 cases:
  #   - if the point projection on the line falls within the segment, the sign
  #       of the distance is `n`.
  #   - if the point projection on the segment falls before the segment start,
  #       the sign of the distance depends on the convexity of the prior and
  #       nearest segments.
  #   - if the point projection on the segment falls after the segment end, the
  #       sign of the distance depends on the convexity of the nearest and next
  #       segments.

  # shape: (num_points, num_segments+2, 2)
  start_to_end_padded = tf.concat(
      [start_to_end[:, -1:], start_to_end, start_to_end[:, :1]], axis=1
  )
  # shape: (num_points, num_segments+1)
  is_locally_convex = tf.greater(
      geometry_utils.cross_product_2d(
          start_to_end_padded[:, :-1], start_to_end_padded[:, 1:]
      ),
      0.0,
  )

  # shape: (num_points, num_segments)
  n_prior = tf.concat(
      [tf.where(is_cyclic, n[:, -1:], n[:, :1]), n[:, :-1]], axis=-1
  )
  n_next = tf.concat(
      [n[:, 1:], tf.where(is_cyclic, n[:, :1], n[:, -1:])], axis=-1
  )

  # shape: (num_points, num_segments)
  sign_if_before = tf.where(
      is_locally_convex[:, :-1],
      tf.maximum(n, n_prior),
      tf.minimum(n, n_prior),
  )
  sign_if_after = tf.where(
      is_locally_convex[:, 1:], tf.maximum(n, n_next), tf.minimum(n, n_next)
  )

  # shape: (num_points, num_segments)
  sign_to_segment = tf.where(
      rel_t < 0.0, sign_if_before, tf.where(rel_t < 1.0, n, sign_if_after)
  )

  # shape: (num_points)
  distance_sign = tf.gather(
      sign_to_segment, tf.argmin(distance_to_segment, axis=-1), batch_dims=1
  )
  return distance_sign * tf.math.reduce_min(distance_to_segment, axis=-1)
