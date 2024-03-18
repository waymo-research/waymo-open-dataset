# Copyright (c) 2024 Waymo LLC. All rights reserved.

# This is licensed under a BSD+Patent license.
# Please see LICENSE and PATENTS text files.
# ==============================================================================
"""Map-based metric features for sim agents."""

from typing import Optional, Sequence

import tensorflow as tf

from waymo_open_dataset.protos import map_pb2
from waymo_open_dataset.utils import box_utils
from waymo_open_dataset.utils import geometry_utils

# Constant distance to apply when distances are invalid. This will avoid the
# propagation of nans and should be reduced out when taking the minimum anyway.
EXTREMELY_LARGE_DISTANCE = 1e10
# Off-road threshold, i.e. smallest distance away from the road edge that is
# considered to be a off-road.
OFFROAD_DISTANCE_THRESHOLD = 0.0

# How close the start and end point of a map feature need to be for the feature
# to be considered cyclic, in m^2.
_CYCLIC_MAP_FEATURE_TOLERANCE_M2 = 1.0
# Scaling factor for vertical distances used when finding the closest segment to
# a query point. This prevents wrong associations in cases with under- and
# over-passes.
_Z_STRETCH_FACTOR = 3.0


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
  # Compute box corners using `box_utils`, and take the xyz coords of the bottom
  # corners.
  box_corners = box_utils.get_upright_3d_box_corners(boxes)[:, :4]
  box_corners = tf.reshape(box_corners, (num_objects, num_steps, 4, 3))

  # Gather objects in the evaluation set
  # `eval_corners` shape: (num_evaluated_objects, num_steps, 4, 3).
  eval_corners = tf.gather(
      box_corners, tf.where(evaluated_object_mask)[:, 0], axis=0)
  num_eval_objects = eval_corners.shape[0]

  # Flatten query points.
  # `flat_eval_corners` shape: (num_query_points, 3).
  flat_eval_corners = tf.reshape(eval_corners, (-1, 3))

  # Tensorize road edges.
  polylines_tensor = _tensorize_polylines(road_edge_polylines)
  is_polyline_cyclic = _check_polyline_cycles(road_edge_polylines)

  # Compute distances for all query points.
  # `corner_distance_to_road_edge` shape: (num_query_points).
  corner_distance_to_road_edge = _compute_signed_distance_to_polylines(
      xyzs=flat_eval_corners, polylines=polylines_tensor,
      is_polyline_cyclic=is_polyline_cyclic, z_stretch=_Z_STRETCH_FACTOR
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
  return tf.where(eval_validity, signed_distances, -EXTREMELY_LARGE_DISTANCE)


def _tensorize_polylines(polylines: Sequence[_Polyline]) -> tf.Tensor:
  """Stacks a sequence of polylines into a tensor.

  Args:
    polylines: A sequence of Polyline objects.

  Returns:
    A float tensor with shape (num_polylines, max_length, 4) containing xyz
      coordinates and a validity flag for all points in the polylines. Polylines
      are padded with zeros up to the length of the longest one.
  """
  polyline_tensors = []
  max_length = 0
  for polyline in polylines:
    # Skip degenerate polylines.
    if len(polyline) < 2:
      continue
    max_length = max(max_length, len(polyline))
    polyline_tensors.append(
        # shape: (num_segments+1, 4: x,y,z,valid)
        tf.constant([
            [map_point.x, map_point.y, map_point.z, 1.0]
            for map_point in polyline
        ])
    )
  # shape: (num_polylines, max_length, 4)
  return tf.stack(
      [
          tf.concat([p, tf.zeros([max_length - p.shape[0], 4])], axis=0)
          for p in polyline_tensors
      ],
      axis=0,
  )


def _check_polyline_cycles(polylines: Sequence[_Polyline]) -> tf.Tensor:
  """Checks if given polylines are cyclic and returns the result as a tensor.

  Args:
    polylines: A sequence of Polyline objects.

  Returns:
    A bool tensor with shape (num_polylines) indicating whether each polyline is
    cyclic.
  """
  cycles = []
  for polyline in polylines:
    # Skip degenerate polylines.
    if len(polyline) < 2:
      continue
    first_point = tf.constant([polyline[0].x, polyline[0].y, polyline[0].z])
    last_point = tf.constant([polyline[-1].x, polyline[-1].y, polyline[-1].z])
    cycles.append(
        tf.reduce_sum(tf.math.square(first_point - last_point), axis=-1)
        < _CYCLIC_MAP_FEATURE_TOLERANCE_M2
    )
  # shape: (num_polylines)
  return tf.stack(cycles, axis=0)


def _compute_signed_distance_to_polylines(
    xyzs: tf.Tensor,
    polylines: tf.Tensor,
    is_polyline_cyclic: Optional[tf.Tensor] = None,
    z_stretch: float = 1.0,
) -> tf.Tensor:
  """Computes the signed distance to the 2D boundary defined by polylines.

  Negative distances correspond to being inside the boundary (e.g. on the
  road), positive distances to being outside (e.g. off-road).

  The polylines should be oriented such that port side is inside the boundary
  and starboard is outside, a.k.a counterclockwise winding order.

  The altitudes i.e. the z-coordinates of query points and polyline segments
  are used to pair each query point with the most relevant segment, that is
  closest and at the right altitude. The distances returned are 2D distances in
  the xy plane.

  Note: degenerate segments (start == end) can cause undefined behaviour.

  Args:
    xyzs: A float Tensor of shape (num_points, 3) containing xyz coordinates of
      query points.
    polylines: Tensor with shape (num_polylines, num_segments+1, 4) containing
      sequences of xyz coordinates and validity, representing start and end
      points of consecutive segments.
    is_polyline_cyclic: A boolean Tensor with shape (num_polylines) indicating
      whether each polyline is cyclic. If None, all polylines are considered
      non-cyclic.
    z_stretch: Factor by which to scale distances over the z axis. This can be
      done to ensure edge points from the wrong level (e.g. overpasses) are not
      selected. Defaults to 1.0 (no stretching).


  Returns:
    A tensor of shape (num_points), containing the signed 2D distance from
      queried points to the nearest polyline.
  """
  num_points = xyzs.shape[0]
  tf.ensure_shape(xyzs, [num_points, 3])
  num_polylines = polylines.shape[0]
  num_segments = polylines.shape[1] - 1
  tf.ensure_shape(polylines, [num_polylines, num_segments + 1, 4])

  # shape: (num_polylines, num_segments+1)
  is_point_valid = tf.cast(polylines[:, :, 3], dtype=tf.bool)
  # shape: (num_polylines, num_segments)
  is_segment_valid = tf.logical_and(
      is_point_valid[:, :-1], is_point_valid[:, 1:]
  )

  if is_polyline_cyclic is None:
    is_polyline_cyclic = tf.zeros([num_polylines], dtype=tf.bool)
  else:
    tf.ensure_shape(is_polyline_cyclic, [num_polylines])

  # Get distance to each segment.
  # shape: (num_points, num_polylines, num_segments, 3)
  xyz_starts = polylines[tf.newaxis, :, :-1, :3]
  xyz_ends = polylines[tf.newaxis, :, 1:, :3]
  start_to_point = xyzs[:, tf.newaxis, tf.newaxis, :3] - xyz_starts
  start_to_end = xyz_ends - xyz_starts

  # Relative coordinate of point projection on segment.
  # shape: (num_points, num_polylines, num_segments)
  rel_t = tf.math.divide_no_nan(
      geometry_utils.dot_product_2d(
          start_to_point[..., :2], start_to_end[..., :2]
      ),
      geometry_utils.dot_product_2d(
          start_to_end[..., :2], start_to_end[..., :2]
      ),
  )

  # Negative if point is on port side of segment, positive if point on
  # starboard side of segment.
  # shape: (num_points, num_polylines, num_segments)
  n = tf.sign(
      geometry_utils.cross_product_2d(
          start_to_point[..., :2], start_to_end[..., :2]
      )
  )

  # Compute the absolute 3d distance to segment.
  # The vertical component is scaled by `z-stretch` to increase the separation
  # between different road altitudes.
  # shape: (num_points, num_polylines, num_segments, 3)
  segment_to_point = start_to_point - (
      start_to_end * tf.clip_by_value(rel_t, 0.0, 1.0)[..., tf.newaxis]
  )
  # shape: (3)
  stretch_vector = tf.constant([1.0, 1.0, z_stretch], dtype=tf.float32)
  # shape: (num_points, num_polylines, num_segments)
  distance_to_segment_3d = tf.linalg.norm(
      segment_to_point * stretch_vector[tf.newaxis, tf.newaxis, tf.newaxis],
      axis=-1,
  )
  # Absolute planar distance to segment.
  # shape: (num_points, num_polylines, num_segments)
  distance_to_segment_2d = tf.linalg.norm(
      segment_to_point[..., :2],
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

  # shape: (num_points, num_polylines, num_segments+2, 2)
  start_to_end_padded = tf.concat(
      [
          start_to_end[:, :, -1:, :2],
          start_to_end[..., :2],
          start_to_end[:, :, :1, :2],
      ],
      axis=-2,
  )
  # shape: (num_points, num_polylines, num_segments+1)
  is_locally_convex = tf.greater(
      geometry_utils.cross_product_2d(
          start_to_end_padded[:, :, :-1], start_to_end_padded[:, :, 1:]
      ),
      0.0,
  )

  # Get shifted versions of `n` and `is_segment_valid`. If the polyline is
  # cyclic, the tensors are rolled, else they are padded with their edge value.
  # shape: (num_points, num_polylines, num_segments)
  n_prior = tf.concat(
      [
          tf.where(
              is_polyline_cyclic[tf.newaxis, :, tf.newaxis],
              n[:, :, -1:],
              n[:, :, :1],
          ),
          n[:, :, :-1],
      ],
      axis=-1,
  )
  n_next = tf.concat(
      [
          n[:, :, 1:],
          tf.where(
              is_polyline_cyclic[tf.newaxis, :, tf.newaxis],
              n[:, :, :1],
              n[:, :, -1:],
          ),
      ],
      axis=-1,
  )
  # shape: (num_polylines, num_segments)
  is_prior_segment_valid = tf.concat(
      [
          tf.where(
              is_polyline_cyclic[:, tf.newaxis],
              is_segment_valid[:, -1:],
              is_segment_valid[:, :1],
          ),
          is_segment_valid[:, :-1],
      ],
      axis=-1,
  )
  is_next_segment_valid = tf.concat(
      [
          is_segment_valid[:, 1:],
          tf.where(
              is_polyline_cyclic[:, tf.newaxis],
              is_segment_valid[:, :1],
              is_segment_valid[:, -1:],
          ),
      ],
      axis=-1,
  )

  # shape: (num_points, num_polylines, num_segments)
  sign_if_before = tf.where(
      is_locally_convex[:, :, :-1],
      tf.maximum(n, n_prior),
      tf.minimum(n, n_prior),
  )
  sign_if_after = tf.where(
      is_locally_convex[:, :, 1:], tf.maximum(n, n_next), tf.minimum(n, n_next)
  )

  # shape: (num_points, num_polylines, num_segments)
  sign_to_segment = tf.where(
      (rel_t < 0.0) & is_prior_segment_valid,
      sign_if_before,
      tf.where((rel_t > 1.0) & is_next_segment_valid, sign_if_after, n)
  )

  # Flatten polylines together.
  # shape: (num_points, all_segments)
  distance_to_segment_3d = tf.reshape(
      distance_to_segment_3d, (num_points, num_polylines * num_segments)
  )
  distance_to_segment_2d = tf.reshape(
      distance_to_segment_2d, (num_points, num_polylines * num_segments)
  )
  sign_to_segment = tf.reshape(
      sign_to_segment, (num_points, num_polylines * num_segments)
  )

  # Mask out invalid segments.
  # shape: (all_segments)
  is_segment_valid = tf.reshape(
      is_segment_valid, (num_polylines * num_segments)
  )
  # shape: (num_points, all_segments)
  distance_to_segment_3d = tf.where(
      is_segment_valid[tf.newaxis],
      distance_to_segment_3d,
      EXTREMELY_LARGE_DISTANCE,
  )
  distance_to_segment_2d = tf.where(
      is_segment_valid[tf.newaxis],
      distance_to_segment_2d,
      EXTREMELY_LARGE_DISTANCE,
  )

  # Get closest segment according to absolute 3D distance and return the
  # corresponding signed 2D distance.
  # shape: (num_points)
  closest_segment_index = tf.argmin(distance_to_segment_3d, axis=-1)
  distance_sign = tf.gather(
      sign_to_segment, closest_segment_index, batch_dims=1
  )
  distance_2d = tf.gather(
      distance_to_segment_2d, closest_segment_index, batch_dims=1
  )
  return distance_sign * distance_2d
