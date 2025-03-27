# Copyright (c) 2025 Waymo LLC. All rights reserved.

# This is licensed under a BSD+Patent license.
# Please see LICENSE and PATENTS text files.
# ==============================================================================
"""Traffic light metric features for sim agents."""

import numpy as np
import tensorflow as tf

from waymo_open_dataset.protos import map_pb2
from waymo_open_dataset.utils import geometry_utils
from waymo_open_dataset.wdl_limited.sim_agents_metrics import map_metric_features

_Polyline = list[map_pb2.MapPoint]

# Constant distance to apply when distances are invalid. This will avoid the
# propagation of nans and should be reduced out when taking the minimum anyway.
EXTREMELY_LARGE_DISTANCE = 1e10


def compute_red_light_violation(
    *,
    center_x: tf.Tensor,
    center_y: tf.Tensor,
    valid: tf.Tensor,
    evaluated_object_mask: tf.Tensor,
    lane_polylines: list[_Polyline],
    lane_ids: list[int],
    traffic_signals: list[list[map_pb2.TrafficSignalLaneState]],
) -> tf.Tensor:
  """Computes red light violations for each of the evaluated objects.

  A red light violation is triggered when:
    |     The lane the object is on has a traffic signal.
    | AND The traffic signal is in a STOP state.
    | AND object crosses the stop point of the traffic signal.

  Args:
    center_x: A float Tensor of shape (num_objects, num_steps) containing the
      x-component of the object positions.
    center_y: A float Tensor of shape (num_objects, num_steps) containing the
      y-component of the object positions.
    valid: A boolean Tensor of shape (num_objects, num_steps) containing the
      validity of the objects over time.
    evaluated_object_mask: A boolean tensor of shape (num_objects), indicating
      whether each object should be considered part of the "evaluation set".
    lane_polylines: A sequence of polylines, each defined as a sequence of 3d
      points with x, y, and z-coordinates.
    lane_ids: A sequence of integer lane ids for each polyline in
      `lane_polylines`.
    traffic_signals: A nested sequence of traffic signal dynamic states,
      per-time step and per-lane.

  Returns:
    A tensor of shape (num_evaluated_objects, num_steps), containing a boolean
    indicating whether the object is violating a red light, for each timestep
    and for all the objects to be evaluated, as specified by
    `evaluated_object_mask`.

  Raises:
    ValueError: When the `lanes` or `traffic_signals` is empty, i.e. there is no
      map information in the Scenario.
    ValueError: When the number of lane polylines and lane ids are inconsistent.
  """
  if not lane_polylines:
    raise ValueError('Missing lanes.')
  if not traffic_signals:
    raise ValueError('Missing traffic signals.')
  if len(lane_polylines) != len(lane_ids):
    raise ValueError('Inconsistent number of lane polylines and lane ids.')

  # Gather objects in the evaluation set.
  evaluated_object_indices = tf.where(evaluated_object_mask)[:, 0]
  # shape: (num_objects, num_steps, 2).
  xy = tf.gather(
      tf.stack([center_x, center_y], axis=-1), evaluated_object_indices, axis=0
  )
  # shape: (num_objects, num_steps).
  valid = tf.gather(valid, evaluated_object_indices, axis=0)

  num_objects, num_steps = valid.shape

  # shapes: (num_polylines, num_segments+1, 4), (num_lanes)
  lane_tensor, lane_ids_tensor = map_metric_features.tensorize_polylines(
      lane_polylines, lane_ids
  )

  # shape: (num_objects*num_steps)
  xy_flat = tf.reshape(xy, [-1, 2])

  # shape: (num_objects*num_steps, 2)
  nearest_lane_segment_index = _get_nearest_lane_segment_index(
      xy=xy_flat,
      lane_xyz_valid=lane_tensor[tf.newaxis],
  )
  # shape: (num_objects*num_steps)
  nearest_lane_index = nearest_lane_segment_index[:, 0]

  # shape: (num_objects*num_steps)
  current_lane_id_flat = tf.gather(lane_ids_tensor, nearest_lane_index)
  # shape: (num_objects, num_steps)
  current_lane_id = tf.reshape(current_lane_id_flat, [num_objects, num_steps])

  # shape: (num_steps, num_traffic_signals, )
  ts_lane_id, ts_state, ts_stop_point = _tensorize_traffic_signals(
      traffic_signals
  )
  num_traffic_signals = ts_lane_id.shape[1]

  # Find the traffic signals that match the current lane.
  # shape: (num_objects, num_steps, num_traffic_signals)
  ts_match = tf.equal(
      current_lane_id[:, :, tf.newaxis], ts_lane_id[tf.newaxis, :, :]
  )

  # Find the traffic signals that in the STOP state.
  # shape: (num_steps, num_traffic_signals)
  ts_is_stop = tf.equal(
      ts_state, map_pb2.TrafficSignalLaneState.State.LANE_STATE_ARROW_STOP
  ) | tf.equal(ts_state, map_pb2.TrafficSignalLaneState.State.LANE_STATE_STOP)

  # Find steps where the object crosses the stop point.

  # shape: (num_steps, num_traffic_signals, num_lanes)
  ts_match_lane = (
      lane_ids_tensor[tf.newaxis, tf.newaxis] == ts_lane_id[:, :, tf.newaxis]
  )
  # shape: (num_steps, num_traffic_signals)
  ts_lane_index = tf.argmax(ts_match_lane, axis=-1)
  ts_lane_valid = tf.reduce_any(ts_match_lane, axis=-1)

  # Segments in the Traffic Signal lane.
  # shape: (num_steps, num_traffic_signals, num_segments+1, 4)
  ts_segments = tf.gather(
      lane_tensor,
      ts_lane_index,
      axis=0,
  )
  # "Fence" index.
  # shape: (num_traffic_signals)
  ts_stop_point_segment_index_flat = _get_nearest_lane_segment_index(
      xy=tf.reshape(ts_stop_point, [num_steps * num_traffic_signals, 2]),
      lane_xyz_valid=tf.reshape(
          ts_segments, [num_steps * num_traffic_signals, -1, 4]
      )[:, tf.newaxis],
  )[:, 1]
  ts_stop_point_segment_index = tf.reshape(
      ts_stop_point_segment_index_flat, [num_steps, num_traffic_signals]
  )
  # "Fence post" indices.
  # shape: (num_steps, num_traffic_signals, 2)
  ts_stop_point_segment_indices = tf.stack(
      [
          ts_stop_point_segment_index,
          ts_stop_point_segment_index + 1,
      ],
      axis=-1,
  )
  # shape: (num_steps, num_traffic_signals, 2, 4)
  ts_stop_point_segment = tf.gather(
      ts_segments, ts_stop_point_segment_indices, axis=2, batch_dims=2
  )

  # Project the stop point and the object onto the stop point segment, and
  # compare their coordinates along the segment.

  # shape: (num_steps, num_traffic_signals, 2)
  start_to_end = (
      ts_stop_point_segment[..., 1, :2] - ts_stop_point_segment[..., 0, :2]
  )
  # shape: (num_steps, num_traffic_signals)
  stop_point_segment_length2 = geometry_utils.dot_product_2d(
      start_to_end, start_to_end
  )
  # shape: (num_steps, num_traffic_signals, 2)
  start_to_stop_point = ts_stop_point - ts_stop_point_segment[..., 0, :2]
  # shape: (num_objects, num_steps, num_traffic_signals, 2)
  start_to_xy = (
      xy[:, :, tf.newaxis]
      - ts_stop_point_segment[tf.newaxis, :, :, 0, :2]
  )

  # Relative coordinate of the stop point projection along the segment.
  # shape: (num_steps, num_traffic_signals, 2)
  stop_point_rel_t = tf.math.divide_no_nan(
      geometry_utils.dot_product_2d(start_to_stop_point, start_to_end),
      stop_point_segment_length2,
  )
  # Relative coordinate of the object projection along the segment.
  # shape: (num_objects, num_steps, num_traffic_signals, 2)
  object_rel_t = tf.math.divide_no_nan(
      geometry_utils.dot_product_2d(
          start_to_xy, start_to_end[tf.newaxis]
      ),
      stop_point_segment_length2[tf.newaxis],
  )
  # shape: (num_objects, num_steps, num_traffic_signals)
  object_behind_stop_point = tf.less(
      object_rel_t, stop_point_rel_t[tf.newaxis]
  )
  object_ahead_stop_point = tf.greater(
      object_rel_t, stop_point_rel_t[tf.newaxis]
  )
  # shape: (num_objects, num_steps-1, num_traffic_signals)
  object_crossed_stop_point = tf.logical_and(
      object_behind_stop_point[:, :-1], object_ahead_stop_point[:, 1:]
  )
  # shape: (num_objects, num_steps, num_traffic_signals)
  object_crossed_stop_point = tf.concat(
      [
          tf.zeros_like(object_crossed_stop_point[:, 0:1]),
          object_crossed_stop_point,
      ],
      axis=1,
  )

  # Verify all the conditions for a red light violation.
  # shape: (num_objects, num_steps)
  return tf.reduce_any(
      valid[:, :, tf.newaxis]
      & ts_match
      & ts_is_stop[tf.newaxis]
      & ts_lane_valid[tf.newaxis]
      & object_crossed_stop_point,
      axis=-1,
  )


def _get_nearest_lane_segment_index(
    *, xy: tf.Tensor, lane_xyz_valid: tf.Tensor
) -> tf.Tensor:
  """Computes the index of the nearest lane segment in 2D space.

  Args:
    xy: A float Tensor of shape (num_points, 2) containing query points
      as x and y coordinates.
    lane_xyz_valid: A float Tensor of shape (num_points, num_lanes,
      num_segments+1, 4) containing xyz coordinates and a validity flag for all
      points in the polylines.

  Returns:
    A tensor of shape (num_points, 2) containing the index of the nearest lane
    for each 2D query point.
  """

  # shape: (num_points, num_polylines, num_segments+1, 2)
  lane_xy = lane_xyz_valid[:, :, :, :2]
  # shape: (num_points, num_polylines, num_segments+1)
  lane_valid = tf.cast(lane_xyz_valid[:, :, :, 3], dtype=tf.bool)

  # Get distance to each segment.
  # shape: (num_points, num_polylines, num_segments, 2)
  segment_start_xy = lane_xy[:, :, :-1]
  segment_end_xy = lane_xy[:, :, 1:]
  start_to_point = xy[:, tf.newaxis, tf.newaxis] - segment_start_xy
  start_to_end = segment_end_xy - segment_start_xy

  # Relative coordinate of point projection along segment.
  # shape: (num_points, num_polylines, num_segments)
  rel_t = tf.math.divide_no_nan(
      geometry_utils.dot_product_2d(start_to_point, start_to_end),
      geometry_utils.dot_product_2d(start_to_end, start_to_end),
  )
  clipped_rel_t = tf.clip_by_value(rel_t, 0.0, 1.0)
  distance_to_segment = tf.linalg.norm(
      start_to_point + start_to_end * clipped_rel_t[..., tf.newaxis], axis=-1
  )

  # Mask out invalid segments.
  # shape: (num_points, num_polylines, num_segments)
  distance_to_segment = tf.where(
      lane_valid[:, :, :-1],
      distance_to_segment,
      EXTREMELY_LARGE_DISTANCE,
  )

  # shape: (num_points, 2)
  nearest_segment_2d_index = _argmin_2d(distance_to_segment)

  return nearest_segment_2d_index


def _argmin_2d(t: tf.Tensor):
  """Finds the 2D indices of the minimum element in a 3D tensor.

  Args:
    t: A Tensor with shape [B, R, C].

  Returns:
    A tensor with shape [B, 2] containing the 2D indices of the minimum over
    axes R and C.
  """
  flat_indices = tf.argmin(tf.reshape(t, [tf.shape(t)[0], -1]), axis=1)
  num_cols = tf.cast(tf.shape(t)[2], dtype=tf.int64)
  cols = flat_indices % num_cols
  rows = flat_indices // num_cols
  return tf.stack([rows, cols], axis=1)


def _tensorize_traffic_signals(
    traffic_signals: list[list[map_pb2.TrafficSignalLaneState]],
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
  """Converts a per-time, per-lane list of traffic signal states to tensors."""
  num_steps = len(traffic_signals)
  # Get all the lane ids over time.
  all_lane_ids = []
  for signals_at_t in traffic_signals:
    for state in signals_at_t:
      all_lane_ids.append(state.lane)
  all_lane_ids = list(set(all_lane_ids))
  num_tl_lanes = len(all_lane_ids)
  # If there are no traffic signals, introduce a dummy one. This will have
  # `LANE_STATE_UNKNOWN` by definition, so it will not affect the traffic
  # light violation metric.
  if num_tl_lanes == 0:
    all_lane_ids = [-1]
    num_tl_lanes = 1

  # Create tensors with all signals and index per step. Signals that are not
  # present at a given step are set to 0 which is `LANE_STATE_UNKNOWN`.
  lane_ids = tf.constant(all_lane_ids, dtype=tf.int32)
  lane_ids = tf.repeat(lane_ids[None, :], num_steps, axis=0)
  states = np.zeros([num_steps, num_tl_lanes])
  stop_points = np.zeros([num_steps, num_tl_lanes, 2])
  for t, signals_at_t in enumerate(traffic_signals):
    for state in signals_at_t:
      lane_index = all_lane_ids.index(state.lane)
      states[t, lane_index] = state.state
      stop_points[t, lane_index, :] = [state.stop_point.x, state.stop_point.y]
  return (
      tf.convert_to_tensor(lane_ids),
      tf.convert_to_tensor(states),
      tf.convert_to_tensor(stop_points, dtype=tf.float32)
  )
