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
"""Tests for object_assset_utils."""

from absl.testing import absltest
import numpy as np
from waymo_open_dataset.v2.perception.utils import object_asset_utils


class ObjectAssetUtilsTest(absltest.TestCase):

  def test_transform_directions_to_box_coord(self):
    directions = np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]])
    # A bounding box centered at (2.0, 2.0, 0.5). The length of the box is
    # 4.2m, width of the box is 1.8m, and height of the box is 1.6m.
    # The heading is set as pi/4 (in radians). The bounding box is either from
    # annotated ground-truth or model prediction.
    # Please refer to label.proto.
    box_3d = np.array([2.0, 2.0, 0.5, 4.2, 1.8, 1.6, np.pi / 4.0])
    # Transform the points normal from SDC coordinate frame to the box-centered
    # frame.
    directions_new = object_asset_utils.transform_directions_to_box_coord(
        directions, box_3d
    )
    directions_ref = np.array(
        [[0.7071, -0.7071, 0.0], [-0.7071, -0.7071, 0.0], [0.0, 0.0, 1.0]]
    )
    np.testing.assert_allclose(directions_new, directions_ref, rtol=1e-3)

  def test_transform_points_to_frame_identity_transform(self):
    points_xyz = np.array([[0.1, 0.3, 0.2], [10.5, -0.9, 4.3], [0.5, -0.7,
                                                                1.5]])
    identity_matrix = np.eye(4)
    points_xyz_new = object_asset_utils.transform_points_to_frame(
        points_xyz, identity_matrix
    )
    np.testing.assert_allclose(points_xyz_new, points_xyz, rtol=1e-6)

  def test_transform_points_to_box_coord_round_trip(self):
    points_xyz = np.array(
        [[0.1, 0.3, 0.2], [10.5, -0.9, 4.3], [0.5, -0.7, 1.5]]
    )
    box_3d = np.asarray(
        [2.0, 2.0, 0.5, 4.2, 1.8, 1.6, np.pi / 4.0], dtype=np.float32
    )
    points_xyz_tfm = object_asset_utils.transform_points_to_box_coord(
        points_xyz, box_3d
    )
    points_xyz_final = object_asset_utils.transform_points_to_box_coord_reverse(
        points_xyz_tfm, box_3d
    )
    np.testing.assert_almost_equal(points_xyz_final, points_xyz, decimal=6)

  def test_transform_directions_to_frame_identity_transform(self):
    directions = np.array(
        [[0.7071, -0.7071, 0.0], [0, -1.0, 0.0], [0.0, -0.7071, -0.7071]]
    )
    identity_matrix = np.eye(4)
    directions_new = object_asset_utils.transform_directions_to_frame(
        directions, identity_matrix
    )
    np.testing.assert_allclose(directions_new, directions, rtol=1e-6)

  def test_get_ray_box_intersects_two(self):
    ray_origin = np.asarray([[-10.0, 0.0, 0.0]], dtype=np.float32)
    ray_dir = np.asarray([[1.0, 0.0, 0.0]], dtype=np.float32)
    box_dim = np.asarray([3, 5, 4], dtype=np.float32)
    num_rays = ray_origin.shape[0]
    isect_flag, tmin, tmax = object_asset_utils.get_ray_box_intersects(
        ray_origin, ray_dir, box_dim
    )
    self.assertEqual(isect_flag.shape[0], num_rays)
    self.assertEqual(tmin.shape[0], num_rays)
    self.assertEqual(tmax.shape[0], num_rays)
    self.assertTrue(isect_flag[0])
    self.assertEqual(tmin[0], 8.5)
    self.assertEqual(tmax[0], 11.5)

  def test_get_ray_box_intersects_one(self):
    ray_origin = np.asarray([[0.0, 0.0, 0.0]], dtype=np.float32)
    ray_dir = np.asarray([[1.0, 0.0, 0.0]], dtype=np.float32)
    box_dim = np.asarray([3, 5, 4], dtype=np.float32)
    num_rays = ray_origin.shape[0]
    isect_flag, tmin, tmax = object_asset_utils.get_ray_box_intersects(
        ray_origin, ray_dir, box_dim
    )
    self.assertEqual(isect_flag.shape[0], num_rays)
    self.assertEqual(tmin.shape[0], num_rays)
    self.assertEqual(tmax.shape[0], num_rays)
    self.assertTrue(isect_flag[0])
    self.assertEqual(tmin[0], -1.5)
    self.assertEqual(tmax[0], 1.5)

  def test_get_ray_box_intersects_none(self):
    ray_origin = np.asarray([[10.0, 0.0, 0.0]], dtype=np.float32)
    ray_dir = np.asarray([[1.0, 0.0, 0.0]], dtype=np.float32)
    box_dim = np.asarray([3, 5, 4], dtype=np.float32)
    num_rays = ray_origin.shape[0]
    isect_flag, tmin, tmax = object_asset_utils.get_ray_box_intersects(
        ray_origin, ray_dir, box_dim
    )
    self.assertEqual(isect_flag.shape[0], num_rays)
    self.assertEqual(tmin.shape[0], num_rays)
    self.assertEqual(tmax.shape[0], num_rays)
    self.assertFalse(isect_flag[0])
    self.assertEqual(tmin[0], -11.5)
    self.assertEqual(tmax[0], -8.5)


if __name__ == "__main__":
  absltest.main()
