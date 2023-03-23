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
"""Geometry utils to compute distances between boxes."""

from typing import Tuple
import numpy as np
import tensorflow as tf

# We only consider 2D boxes.
NUM_VERTICES_IN_BOX = 4


def minkowski_sum_of_box_and_box_points(box1_points: tf.Tensor,
                                        box2_points: tf.Tensor) -> tf.Tensor:
  """Batched Minkowski sum of two boxes (counter-clockwise corners in xy).

  The last dimensions of the input and return store the x and y coordinates of
  the points. Both box1_points and box2_points needs to be stored in
  counter-clockwise order. Otherwise the function will return incorrect results
  silently.

  Args:
    box1_points: Tensor of vertices for box 1, with shape:
      (num_boxes, num_points_per_box, 2).
    box2_points: Tensor of vertices for box 2, with shape:
      (num_boxes, num_points_per_box, 2).

  Returns:
    The Minkowski sum of the two boxes, of size (num_boxes,
    num_points_per_box * 2, 2). The points will be stored in counter-clockwise
    order.
  """
  # Hard coded order to pick points from the two boxes. This is a simplification
  # of the generic convex polygons case. For boxes, the adjacent edges are
  # always 90 degrees apart from each other, so the index of vertices can be
  # hard coded.
  point_order_1 = tf.constant([0, 0, 1, 1, 2, 2, 3, 3], dtype=tf.int64)
  point_order_2 = tf.constant([0, 1, 1, 2, 2, 3, 3, 0], dtype=tf.int64)

  box1_start_idx, downmost_box1_edge_direction = _get_downmost_edge_in_box(
      box1_points)
  box2_start_idx, downmost_box2_edge_direction = _get_downmost_edge_in_box(
      box2_points)

  # The cross-product of the unit vectors indicates whether the downmost edge
  # in box2 is pointing to the left side (the inward side of the resulting
  # Minkowski sum) of the downmost edge in box1. If this is the case, pick
  # points from box1 in the order `point_order_2`, and pick points from box2 in
  # the order of `point_order_1`. Otherwise, we switch the order to pick points
  # from the two boxes, pick points from box1 in the order of `point_order_1`,
  # and pick points from box2 in the order of `point_order_2`.
  # Shape: (num_boxes, 1)
  condition = (
      cross_product_2d(
          downmost_box1_edge_direction, downmost_box2_edge_direction
      )
      >= 0.0
  )
  # Tile condition to shape: (num_boxes, num_points_per_box * 2 = 8).
  condition = tf.tile(condition, [1, 8])

  # box1_point_order of size [num_boxes, num_points_per_box * 2 = 8].
  box1_point_order = tf.where(condition, point_order_2, point_order_1)
  # Shift box1_point_order by box1_start_idx, so that the first index in
  # box1_point_order is the downmost vertex in the box.
  box1_point_order = tf.math.mod(box1_point_order + box1_start_idx,
                                 NUM_VERTICES_IN_BOX)
  # Gather points from box1 in order.
  # ordered_box1_points is of size [num_boxes, num_points_per_box * 2, 2].
  ordered_box1_points = tf.gather(
      box1_points, box1_point_order, axis=-2, batch_dims=-1)

  # Gather points from box2 as well.
  box2_point_order = tf.where(condition, point_order_1, point_order_2)
  box2_point_order = tf.math.mod(box2_point_order + box2_start_idx,
                                 NUM_VERTICES_IN_BOX)
  ordered_box2_points = tf.gather(
      box2_points, box2_point_order, axis=-2, batch_dims=-1)
  minkowski_sum = ordered_box1_points + ordered_box2_points
  return minkowski_sum


def signed_distance_from_point_to_convex_polygon(
    query_points: tf.Tensor, polygon_points: tf.Tensor) -> tf.Tensor:
  """Finds the signed distances from query points to convex polygons.

  Each polygon is represented by a 2d tensor storing the coordinates of its
  vertices. The vertices must be ordered in counter-clockwise order. An
  arbitrary number of pairs (point, polygon) can be batched on the 1st
  dimension.

  Note: Each polygon is associated to a single query point.

  Args:
    query_points: (batch_size, 2). The last dimension is the x and y
      coordinates of points.
    polygon_points: (batch_size, num_points_per_polygon, 2). The last
      dimension is the x and y coordinates of vertices.

  Returns:
    A tensor containing the signed distances of the query points to the
    polygons. Shape: (batch_size,).
  """
  tangent_unit_vectors, normal_unit_vectors, edge_lengths = (
      _get_edge_info(polygon_points))

  # Expand the shape of `query_points` to (num_polygons, 1, 2), so that
  # it matches the dimension of `polygons_points` for broadcasting.
  query_points = tf.expand_dims(query_points, axis=1)
  # Compute query points to polygon points distances.
  # Shape (num_polygons, num_points_per_polygon, 2).
  vertices_to_query_vectors = query_points - polygon_points
  # Shape (num_polygons, num_points_per_polygon).
  vertices_distances = tf.linalg.norm(vertices_to_query_vectors, axis=-1)

  # Query point to edge distances are measured as the perpendicular distance
  # of the point from the edge. If the projection of this point on to the edge
  # falls outside the edge itself, this distance is not considered (as there)
  # will be a lower distance with the vertices of this specific edge.

  # Make distances negative if the query point is in the inward side of the
  # edge. Shape: (num_polygons, num_points_per_polygon).
  edge_signed_perp_distances = tf.reduce_sum(
      -normal_unit_vectors * vertices_to_query_vectors, axis=-1)

  # If `edge_signed_perp_distances` are all less than 0 for a
  # polygon-query_point pair, then the query point is inside the convex polygon.
  is_inside = tf.reduce_all(edge_signed_perp_distances <= 0, axis=-1)

  # Project the distances over the tangents of the edge, and verify where the
  # projections fall on the edge.
  # Shape: (num_polygons, num_edges_per_polygon).
  projection_along_tangent = tf.reduce_sum(
      tangent_unit_vectors * vertices_to_query_vectors, axis=-1)
  projection_along_tangent_proportion = tf.divide(projection_along_tangent,
                                                  edge_lengths)
  # Shape: (num_polygons, num_edges_per_polygon).
  is_projection_on_edge = tf.logical_and(
      projection_along_tangent_proportion >= 0.0,
      projection_along_tangent_proportion <= 1.0)

  # If the point projection doesn't lay on the edge, set the distance to inf.
  edge_perp_distances = tf.abs(edge_signed_perp_distances)
  edge_distances = tf.where(is_projection_on_edge,
                            edge_perp_distances, np.inf)

  # Aggregate vertex and edge distances.
  # Shape: (num_polyons, 2 * num_edges_per_polygon).
  edge_and_vertex_distance = tf.concat([edge_distances, vertices_distances],
                                       axis=-1)
  # Aggregate distances per polygon and change the sign if the point lays inside
  # the polygon. Shape: (num_polygons,).
  min_distance = tf.reduce_min(edge_and_vertex_distance, axis=-1)
  signed_distances = tf.where(is_inside, -min_distance, min_distance)
  return signed_distances


def _get_downmost_edge_in_box(box: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
  """Finds the downmost (lowest y-coordinate) edge in the box.

  Note: We assume box edges are given in a counter-clockwise order, so that
  the edge which starts with the downmost vertex (i.e. the downmost edge) is
  uniquely identified.

  Args:
    box: (num_boxes, num_points_per_box, 2). The last dimension contains the x-y
      coordinates of corners in boxes.

  Returns:
    A tuple of two tensors:
      downmost_vertex_idx: The index of the downmost vertex, which is also the
        index of the downmost edge. Shape: (num_boxes, 1).
      downmost_edge_direction: The tangent unit vector of the downmost edge,
        pointing in the counter-clockwise direction of the box.
        Shape: (num_boxes, 1, 2).
  """
  # The downmost vertex is the lowest in the y dimension.
  # Shape: (num_boxes, 1).
  downmost_vertex_idx = tf.argmin(box[..., 1], axis=-1)[:, tf.newaxis]

  # Find the counter-clockwise point edge from the downmost vertex.
  edge_start_vertex = tf.gather(box, downmost_vertex_idx, axis=1, batch_dims=-1)
  edge_end_idx = tf.math.mod(downmost_vertex_idx + 1, NUM_VERTICES_IN_BOX)
  edge_end_vertex = tf.gather(box, edge_end_idx, axis=1, batch_dims=-1)

  # Compute the direction of this downmost edge.
  downmost_edge = edge_end_vertex - edge_start_vertex
  downmost_edge_length = tf.linalg.norm(downmost_edge, axis=-1)
  downmost_edge_direction = downmost_edge / downmost_edge_length[:, :,
                                                                 tf.newaxis]
  return downmost_vertex_idx, downmost_edge_direction


def _get_edge_info(
    polygon_points: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
  """Computes properties about the edges of a polygon.

  Args:
    polygon_points: Tensor containing the vertices of each polygon, with
      shape (num_polygons, num_points_per_polygon, 2). Each polygon is assumed
      to have an equal number of vertices.

  Returns:
    tangent_unit_vectors: A unit vector in (x,y) with the same direction as
      the tangent to the edge. Shape: (num_polygons, num_points_per_polygon, 2).
    normal_unit_vectors: A unit vector in (x,y) with the same direction as
      the normal to the edge.
      Shape: (num_polygons, num_points_per_polygon, 2).
    edge_lengths: Lengths of the edges.
      Shape (num_polygons, num_points_per_polygon).
  """
  # Shift the polygon points by 1 position to get the edges.
  # Shape: (num_polygons, 1, 2).
  first_point_in_polygon = polygon_points[:, 0:1, :]
  # Shape: (num_polygons, num_points_per_polygon, 2).
  shifted_polygon_points = tf.concat(
      [polygon_points[:, 1:, :], first_point_in_polygon], axis=-2)
  # Shape: (num_polygons, num_points_per_polygon, 2).
  edge_vectors = shifted_polygon_points - polygon_points

  # Shape: (num_polygons, num_points_per_polygon).
  edge_lengths = tf.linalg.norm(edge_vectors, axis=-1)
  # Shape: (num_polygons, num_points_per_polygon, 2).
  tangent_unit_vectors = edge_vectors / edge_lengths[:, :, tf.newaxis]
  # Shape: (num_polygons, num_points_per_polygon, 2).
  normal_unit_vectors = tf.stack(
      [-tangent_unit_vectors[..., 1], tangent_unit_vectors[..., 0]], axis=-1)
  return tangent_unit_vectors, normal_unit_vectors, edge_lengths


def cross_product_2d(a: tf.Tensor, b: tf.Tensor) -> tf.Tensor:
  """Computes the signed magnitude of cross product of 2d vectors.

  Args:
    a: A tensor with shape (..., 2).
    b: A tensor with the same shape as `a`.

  Returns:
    An (n-1)-rank tensor that stores the cross products of paired 2d vectors in
    `a` and `b`.
  """
  return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]


def dot_product_2d(a: tf.Tensor, b: tf.Tensor) -> tf.Tensor:
  """Computes the dot product of 2d vectors.

  Args:
    a: A tensor with shape (..., 2).
    b: A tensor with the same shape as `a`.

  Returns:
    An (n-1)-rank tensor that stores the cross products of 2d vectors in a and
    b.
  """
  return a[..., 0] * b[..., 0] + a[..., 1] * b[..., 1]


def rotate_2d_points(xys: tf.Tensor, rotation_yaws: tf.Tensor) -> tf.Tensor:
  """Rotates `xys` counter-clockwise using the `rotation_yaws`.

  Rotates about the origin counter-clockwise in the x-y plane.

  Arguments may have differing shapes as long as they are broadcastable to a
  common shape.

  Args:
    xys: A float Tensor with shape (..., 2) containing xy coordinates.
    rotation_yaws: A float Tensor with shape (..., 1) containing angles in
      radians.

  Returns:
    A float Tensor with shape (..., 2) containing the rotated `xys`.
  """
  rel_cos_yaws = tf.cos(rotation_yaws)
  rel_sin_yaws = tf.sin(rotation_yaws)
  xs_out = rel_cos_yaws * xys[..., 0] - rel_sin_yaws * xys[..., 1]
  ys_out = rel_sin_yaws * xys[..., 0] + rel_cos_yaws * xys[..., 1]
  return tf.stack([xs_out, ys_out], axis=-1)
