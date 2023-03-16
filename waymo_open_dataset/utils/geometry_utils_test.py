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

import tensorflow as tf

from waymo_open_dataset.utils import geometry_utils


class GeometryUtilsTest(tf.test.TestCase):

  def test_signed_distance_from_point_to_convex_polygons_returns_correctly(
      self):
    # `polygon1` is a list of vertices belonging to a square box centered at the
    # origin with side length of 2.0.
    polygon1 = tf.constant([[1.0, -1.0], [1.0, 1.0], [-1.0, 1.0], [-1.0, -1.0]],
                           dtype=tf.float64)
    # `polygon2` is a list of vertices belonging to a square box centered at
    # the origin, whose diagonals are aligned with x and y axes.
    polygon2 = tf.constant([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]],
                           dtype=tf.float64)
    # `polygon2` is queried twice with different query points.
    # polygons_points is of size [num_polygons, num_points_per_polygon, 2].
    polygons_points = tf.stack([polygon1, polygon2, polygon2], axis=0)
    # Query points for each of the polygons. Note: each query point is only
    # associated to a single polygon, so we expect only 3 distances here.
    query_points = tf.constant([[0.0, 0.0], [1.0, 0.0], [5.0, 0.0]],
                               dtype=tf.float64)
    expected_signed_distances = tf.constant([-1, 0, 4], dtype=tf.float64)

    signed_distances = (
        geometry_utils.signed_distance_from_point_to_convex_polygon(
            query_points, polygons_points))
    self.assertAllClose(signed_distances, expected_signed_distances)

  def test_minkowski_sum_of_box_and_box_points_returns_correctly(self):
    # box_1 and box_2 of size [num_boxes, num_points_per_box, 2].
    # Each of the box_1 and box_2 tensor contains the points in the two boxes.
    box_1 = tf.constant([[[1.0, -1.0], [1.0, 1.0], [-1.0, 1.0], [-1.0, -1.0]],
                         [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]]],
                        dtype=tf.float64)
    box_2 = tf.constant([[[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]],
                         [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]]],
                        dtype=tf.float64)

    # Note that we could allow the order of the output polygon points to be
    # different from the expected result below (but still needs to be in
    # counter-clockwise order). However, the implememtation of
    # minkowski_sum_of_box_and_box makes the order of output points
    # deterministic, so we can just hard code the order of points in the
    # expected result to be the same as the output.
    expected_polygon1 = tf.constant([[1, -2], [2, -1], [2, 1], [1, 2], [-1, 2],
                                     [-2, 1], [-2, -1], [-1, -2]],
                                    dtype=tf.float64)
    expected_polygon2 = tf.constant(
        [[0, -2], [1, -1], [2, 0], [1, 1], [0, 2], [-1, 1], [-2, 0], [-1, -1]],
        dtype=tf.float64)
    expected_polygons = tf.stack([expected_polygon1, expected_polygon2], axis=0)

    minkowski_sum = geometry_utils.minkowski_sum_of_box_and_box_points(
        box_1, box_2)
    self.assertAllClose(minkowski_sum, expected_polygons)


if __name__ == '__main__':
  tf.test.main()
