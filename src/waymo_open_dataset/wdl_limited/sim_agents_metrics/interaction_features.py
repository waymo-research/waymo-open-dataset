# Copyright (c) 2024 Waymo LLC. All rights reserved.

# This is licensed under a BSD+Patent license.
# Please see LICENSE and PATENTS text files.
# ==============================================================================
"""Interaction metric features for sim agents."""

import math

import tensorflow as tf

from waymo_open_dataset.utils import box_utils
from waymo_open_dataset.utils import geometry_utils
from waymo_open_dataset.wdl_limited.sim_agents_metrics import trajectory_features


# Constant distance to apply when distances between objects are invalid. This
# will avoid the propagation of nans and should be reduced out when taking the
# minimum anyway.
EXTREMELY_LARGE_DISTANCE = 1e10
# Collision threshold, i.e. largest distance between objects that is considered
# to be a collision.
COLLISION_DISTANCE_THRESHOLD = 0.0
# Rounding factor to apply to the corners of the object boxes in distance and
# collision computation. The rounding factor is between 0 and 1, where 0 yields
# rectangles with sharp corners (no rounding) and 1 yields capsule shapes.
# Default value of 0.7 conservately fits most vehicle contours.
CORNER_ROUNDING_FACTOR = 0.7

# Condition thresholds for filtering obstacles driving ahead of the ego pbject
# when computing the time-to-collision metric. This metric only considers
# collisions in lane-following a.k.a. tailgating situations.
# Maximum allowed difference in heading.
MAX_HEADING_DIFF = math.radians(75.0)  # radians.
# Maximum allowed difference in heading in case of small lateral overlap.
MAX_HEADING_DIFF_FOR_SMALL_OVERLAP = math.radians(10.0)  # radians.
# Lateral overlap threshold below which the tighter heading alignment condition
# `_MAX_HEADING_DIFF_FOR_SMALL_OVERLAP` is used.
SMALL_OVERLAP_THRESHOLD = 0.5  # meters.

# Maximum time-to-collision, in seconds, used to clip large values or in place
# of invalid values.
MAXIMUM_TIME_TO_COLLISION = 5.0





def compute_distance_to_nearest_object(
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
    corner_rounding_factor: float = CORNER_ROUNDING_FACTOR,
) -> tf.Tensor:
  """Computes the distance to nearest object for each of the evaluated objects.

  Objects are represented by 2D rectangles with rounded corners.

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
      object headings, in radians.
    valid: A boolean Tensor of shape (num_objects, num_steps) containing the
      validity of the objects over time.
    evaluated_object_mask: A boolean tensor of shape (num_objects), to index the
      objects identified by the tensors defined above. If True, the object is
      considered part of the "evaluation set", i.e. the object can actively
      collide into other objects. If False, the object can also be passively
      collided into.
    corner_rounding_factor: Rounding factor to apply to the corners of the
      object boxes, between 0 (no rounding) and 1 (capsule shape rounding).

  Returns:
    A tensor of shape (num_evaluated_objects, num_steps), containing the
    distance to the nearest object, for each timestep and for all the objects
    to be evaluated, as specified by `evaluated_object_mask`.
  """
  # Concatenate tensors to have the same convention as `box_utils`.
  boxes = tf.stack(
      [center_x, center_y, center_z, length, width, height, heading], axis=-1
  )
  num_objects, num_steps, num_features = boxes.shape

  # Shrink the bounding boxes to get their rectangular "core". The rounded
  # rectangles we want to process are distance isolines of the rectangle cores.

  # The shrinking distance is half of the minimal dimension between length and
  # width, multiplied by the rounding factor.
  # Shape: [num_objects, num_steps]
  shrinking_distance = (
      tf.minimum(boxes[:, :, 3], boxes[:, :, 4]) * corner_rounding_factor / 2.
  )
  # Box cores to use in distance computation below, after shrinking all sides
  # uniformly.
  boxes = tf.concat(
      [
          boxes[:, :, :3],
          boxes[:, :, 3:4] - 2.*shrinking_distance[..., tf.newaxis],
          boxes[:, :, 4:5] - 2.*shrinking_distance[..., tf.newaxis],
          boxes[:, :, 5:],
      ],
      axis=2,
  )

  boxes = tf.reshape(boxes, [num_objects * num_steps, num_features])

  # Compute box corners using `box_utils`, and take xy coordinates of the lower
  # 4 corners (lower in terms of z-coordinate), as we are only computing
  # distances for 2D boxes.
  box_corners = box_utils.get_upright_3d_box_corners(boxes)[:, :4, :2]
  box_corners = tf.reshape(box_corners, (num_objects, num_steps, 4, 2))

  # Rearrange the boxes based on `evaluated_object_mask`. We want two sets of
  # boxes: the first one including just the evaluated objects, the second one
  # with all the boxes, but having the evaluated objects as first (this is used
  # later to filter out self distances).
  # `eval_corners` shape: (num_evaluated_objects, num_steps, 4, 2).
  eval_corners = tf.gather(
      box_corners, tf.where(evaluated_object_mask)[:, 0], axis=0
  )
  num_eval_objects = eval_corners.shape[0]
  # `other_corners` shape: (num_objects-num_evaluated_objects, num_steps, 4, 2).
  other_corners = tf.gather(
      box_corners, tf.where(tf.logical_not(evaluated_object_mask))[:, 0], axis=0
  )
  # `all_corners` shape: (num_objects, num_steps, 4, 2).
  all_corners = tf.concat([eval_corners, other_corners], axis=0)
  # Broadcast both sets for pair-wise comparisons.
  eval_corners = tf.broadcast_to(
      eval_corners[:, tf.newaxis],
      [num_eval_objects, num_objects, num_steps, 4, 2],
  )
  all_corners = tf.broadcast_to(
      all_corners[tf.newaxis], [num_eval_objects, num_objects, num_steps, 4, 2]
  )
  # Flatten the first 3 dimensions to one single batch dimensions, as required
  # by `minkowski_sum_of_box_and_box_points()` and
  # `signed_distance_from_point_to_convex_polygon()`. We can reshape back
  # afterwards.
  eval_corners = tf.reshape(
      eval_corners, [num_eval_objects * num_objects * num_steps, 4, 2]
  )
  all_corners = tf.reshape(
      all_corners, [num_eval_objects * num_objects * num_steps, 4, 2]
  )
  # The signed distance between two polygons A and B is equal to the distance
  # between the origin and the Minkowski sum A + (-B), where we generate -B by a
  # reflection. See for example:
  # https://www2.cs.duke.edu/courses/spring07/cps296.2/course_projects/shashi_proj.pdf
  neg_all_corners = -1.0 * all_corners
  minkowski_sum = geometry_utils.minkowski_sum_of_box_and_box_points(
      box1_points=eval_corners, box2_points=neg_all_corners
  )
  # Shape: (num_evaluated_objects * num_objects * num_steps, 8, 2).
  minkowski_sum.shape.assert_is_compatible_with([None, 8, 2])
  # If the two convex shapes intersect, the Minkowski subtraction polygon will
  # containing the origin.
  signed_distances_flat = (
      geometry_utils.signed_distance_from_point_to_convex_polygon(
          query_points=tf.zeros_like(minkowski_sum[:, 0, :]),
          polygon_points=minkowski_sum,
      )
  )

  # Shape: (num_evaluated_objects, num_objects, num_steps).
  signed_distances = tf.reshape(
      signed_distances_flat, [num_eval_objects, num_objects, num_steps]
  )

  # Gather the shrinking distances for the evaluated objects and for all objects
  # after reordering.
  # `eval_shrinking_distance` shape: (num_evaluated_objects, num_steps).
  eval_shrinking_distance = tf.gather(
      shrinking_distance, tf.where(evaluated_object_mask)[:, 0], axis=0
  )
  other_shrinking_distance = tf.gather(
      shrinking_distance,
      tf.where(tf.logical_not(evaluated_object_mask))[:, 0], axis=0
  )
  # `all_shrinking_distance` shape: (num_objects, num_steps).
  all_shrinking_distance = tf.concat(
      [eval_shrinking_distance, other_shrinking_distance], axis=0
  )

  # Recover distances between rounded boxes from the distances between core
  # boxes by subtracting the shrinking distances. This is equivalent to
  # inflating the core boxes isotropically by the same amount they were shrunk.
  # Shape: (num_evaluated_objects, num_objects, num_steps).
  signed_distances -= eval_shrinking_distance[:, tf.newaxis, :]
  signed_distances -= all_shrinking_distance[tf.newaxis, :, :]

  # Mask out self-distances.
  self_mask = tf.eye(num_eval_objects, num_objects, dtype=tf.float32)[
      :, :, tf.newaxis
  ]
  signed_distances = signed_distances + self_mask * EXTREMELY_LARGE_DISTANCE

  # Mask out invalid boxes. As with box coordinates, the validity mask needs to
  # be reshuffled to have the same ordering. This is necessary because the
  # distance computations above have no validity mask. To correctly get the
  # *minimum* distance to other objects, we mask the invalid objects by
  # assigning a very large distance. This value could be returned by this
  # function only if there is at most one valid object in the scene.
  eval_validity = tf.gather(
      valid, tf.where(evaluated_object_mask)[:, 0], axis=0
  )
  other_validity = tf.gather(
      valid, tf.where(tf.logical_not(evaluated_object_mask))[:, 0], axis=0
  )
  all_validity = tf.concat([eval_validity, other_validity], axis=0)
  valid_mask = tf.logical_and(
      eval_validity[:, tf.newaxis, :], all_validity[tf.newaxis, :, :]
  )
  signed_distances = tf.where(
      valid_mask, signed_distances, EXTREMELY_LARGE_DISTANCE
  )
  # Aggregate over the "all objects" dimension.
  return tf.reduce_min(signed_distances, axis=1)


def compute_time_to_collision_with_object_in_front(
    *,
    center_x: tf.Tensor,
    center_y: tf.Tensor,
    length: tf.Tensor,
    width: tf.Tensor,
    heading: tf.Tensor,
    valid: tf.Tensor,
    evaluated_object_mask: tf.Tensor,
    seconds_per_step: float,
) -> tf.Tensor:
  """Computes the time-to-collision of the evaluated objects.

  The time-to-collision measures, in seconds, the time until an object collides
  with the object it is following, assuming constant speeds.

  If an object is not following any valid object, or if the time-to-collision is
  invalid or too large, `MAXIMUM_TIME_TO_COLLISION` is returned for that box.

  Args:
    center_x: A float Tensor of shape (num_objects, num_steps) containing the
      x-component of the object positions.
    center_y: A float Tensor of shape (num_objects, num_steps) containing the
      y-component of the object positions.
    length: A float Tensor of shape (num_objects, num_steps) containing the
      object lengths.
    width: A float Tensor of shape (num_objects, num_steps) containing the
      object widths.
    heading: A float Tensor of shape (num_objects, num_steps) containing the
      object headings.
    valid: A boolean Tensor of shape (num_objects, num_steps) containing the
      validity of the objects over time.
    evaluated_object_mask: A boolean tensor of shape (num_objects), indicating
      whether each object should be considered part of the "evaluation set".
    seconds_per_step: The duration (in seconds) of one step. This is used to
      scale speed and acceleration properly. This is always a positive value,
      usually `submission_specs.STEP_DURATION_SECONDS`.

  Returns:
    A tensor of shape (num_evaluated_objects, num_steps), containing the
    time-to-collision, for each timestep and for all the objects to be
    evaluated, as specified by `evaluated_object_mask`.
  """
  # `speed` shape: (num_objects, num_steps)
  speed = trajectory_features.compute_kinematic_features(
      x=center_x,
      y=center_y,
      z=tf.zeros_like(center_x),
      heading=heading,
      seconds_per_step=seconds_per_step,
  )[0]
  boxes = tf.stack([center_x, center_y, length, width, heading, speed], axis=-1)
  # `boxes` shape: (num_steps, num_objects, 6).
  boxes = tf.transpose(boxes, perm=[1, 0, 2])
  valid = tf.transpose(valid, perm=[1, 0])

  # Gather the boxes based on `evaluated_object_mask`.
  # `eval_boxes` shape: (num_steps, num_evaluated_objects, 6).
  eval_boxes = tf.gather(boxes, tf.where(evaluated_object_mask)[:, 0], axis=1)

  ego_xy, ego_sizes, ego_yaw, ego_speed = tf.split(
      eval_boxes, num_or_size_splits=[2, 2, 1, 1], axis=-1
  )
  other_xy, other_sizes, other_yaw, _ = tf.split(
      boxes, num_or_size_splits=[2, 2, 1, 1], axis=-1
  )

  # Absolute yaw difference between each ego box and every other box.
  # `yaw_diff` shape: (num_steps, num_evaluated_objects, num_objects, 1)
  yaw_diff = tf.math.abs(other_yaw[:, tf.newaxis] - ego_yaw[:, :, tf.newaxis])

  yaw_diff_cos = tf.math.cos(yaw_diff)
  yaw_diff_sin = tf.math.sin(yaw_diff)

  # Longitudinal and lateral offsets from the other box center to its corner(s)
  # closest to the ego box.
  # Note: The relevant corner can be different for the 2 directions. Taking the
  # absolute value of cos(yaw_diff), sin(yaw_diff) is equivalent to taking the
  # maximum over all 4 corners.
  # `other_long_offset` shape: (num_steps, num_evaluated_objects, num_objects)
  other_long_offset = geometry_utils.dot_product_2d(
      other_sizes[:, tf.newaxis] / 2.0,
      tf.math.abs(tf.concat([yaw_diff_cos, yaw_diff_sin], axis=-1)),
  )
  # `other_lat_offset` shape: (num_steps, num_evaluated_objects, num_objects)
  other_lat_offset = geometry_utils.dot_product_2d(
      other_sizes[:, tf.newaxis] / 2.0,
      tf.math.abs(tf.concat([yaw_diff_sin, yaw_diff_cos], axis=-1)),
  )

  # Coordinates of other box center relative to ego box.
  # `other_relative_xy` shape:
  #   (num_steps, num_evaluated_objects, num_objects, 2)
  other_relative_xy = geometry_utils.rotate_2d_points(
      (other_xy[:, tf.newaxis] - ego_xy[:, :, tf.newaxis]),
      -ego_yaw,
  )

  # Longitudinal distance from ego's front side to the most behind other box
  # corner, defined as positive ahead of the ego.
  # `long_distance` shape: (num_steps, num_evaluated_objects, num_objects)
  long_distance = (
      other_relative_xy[..., 0]
      - ego_sizes[:, :, tf.newaxis, 0] / 2.0
      - other_long_offset
  )

  # Lateral overlap of the other box corner onto the ego straight trail.
  # `lat_overlap` shape: (num_steps, num_evaluated_objects, num_objects)
  lat_overlap = (
      tf.math.abs(other_relative_xy[..., 1])
      - ego_sizes[:, :, tf.newaxis, 1] / 2.0
      - other_lat_offset
  )

  # Check that yaw difference, longitudinal distance and lateral overlap
  # satisfy "following" conditions.
  following_mask = _get_object_following_mask(
      long_distance,
      lat_overlap,
      yaw_diff[..., 0],
  )

  # Mask out boxes that are invalid or don't satisfy "following" conditions.
  valid_mask = tf.logical_and(valid[:, tf.newaxis], following_mask)
  # `masked_long_distance` shape: (num_steps, num_eval_objects, num_objects)
  masked_long_distance = (
      long_distance
      + (1.0 - tf.cast(valid_mask, tf.float32)) * EXTREMELY_LARGE_DISTANCE
  )

  # `box_ahead_index` shape: (num_steps, num_evaluated_objects)
  box_ahead_index = tf.math.argmin(masked_long_distance, axis=-1)
  # `distance_to_box_ahead` shape: (num_steps, num_evaluated_objects)
  distance_to_box_ahead = tf.gather(
      masked_long_distance, box_ahead_index, batch_dims=2
  )
  # `box_ahead_speed` shape: (num_steps, num_evaluated_objects)
  box_ahead_speed = tf.gather(
      tf.broadcast_to(
          tf.transpose(
              speed[
                  :,
                  tf.newaxis,
                  :,
              ]
          ),
          masked_long_distance.shape,
      ),
      box_ahead_index,
      batch_dims=2,
  )

  rel_speed = ego_speed[..., 0] - box_ahead_speed
  # `time_to_collision` shape: (num_steps, num_evaluated_objects)
  time_to_collision = tf.where(
      rel_speed > 0.0,
      tf.minimum(distance_to_box_ahead / rel_speed, MAXIMUM_TIME_TO_COLLISION),
      MAXIMUM_TIME_TO_COLLISION,
  )
  return tf.transpose(time_to_collision, [1, 0])


def _get_object_following_mask(
    longitudinal_distance: tf.Tensor,
    lateral_overlap: tf.Tensor,
    yaw_diff: tf.Tensor,
) -> tf.Tensor:
  """Checks whether objects satisfy criteria for following another object.

  An object on which the criteria are applied is called "ego object" in this
  function to disambiguate it from the other objects acting as obstacles.

  An "ego" object is considered to be following another object if they satisfy
  conditions on the longitudinal distance, lateral overlap, and yaw alignment
  between them.

  Args:
    longitudinal_distance: A float Tensor with shape (batch_dim, num_egos,
      num_others) containing longitudinal distances from the back side of each
      ego box to every other boxes.
    lateral_overlap: A float Tensor with shape (batch_dim, num_egos, num_others)
      containing lateral overlaps of other boxes over the trails of ego boxes.
    yaw_diff: A float Tensor with shape (batch_dim, num_egos, num_others)
      containing absolute yaw differences between egos and other boxes.

  Returns:
    A boolean Tensor with shape (batch_dim, num_egos, num_others) indicating for
    each ego box if it is following the other boxes.
  """
  # Check object is ahead of the ego box's front.
  valid_mask = longitudinal_distance > 0.0

  # Check alignment.
  valid_mask = tf.logical_and(valid_mask, yaw_diff <= MAX_HEADING_DIFF)

  # Check object is directly ahead of the ego box.
  valid_mask = tf.logical_and(valid_mask, lateral_overlap < 0.0)

  # Check strict alignment if the overlap is small.
  # `lateral_overlap` is a signed penetration distance: it is negative when the
  # boxes have an actual lateral overlap.
  return tf.logical_and(
      valid_mask,
      tf.logical_or(
          lateral_overlap < -SMALL_OVERLAP_THRESHOLD,
          yaw_diff <= MAX_HEADING_DIFF_FOR_SMALL_OVERLAP,
      ),
  )
