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
"""Tests for waymo_open_dataset.v2.column_types."""

from absl.testing import absltest
import numpy as np

from waymo_open_dataset.v2 import column_types


class ColumnTypesTest(absltest.TestCase):

  def test_vec_3d_numpy_basic(self):
    vec_3d = column_types.Vec3d(1.5, -1.6, 1.7)
    self.assertLen(vec_3d.numpy.shape, 1)
    self.assertTupleEqual(vec_3d.numpy.shape, (3,))
    self.assertAlmostEqual(vec_3d.numpy[0], vec_3d.x)
    self.assertAlmostEqual(vec_3d.numpy[1], vec_3d.y)
    self.assertAlmostEqual(vec_3d.numpy[2], vec_3d.z)
    self.assertEqual(vec_3d.numpy.dtype, np.float64)

  def test_box_3d_numpy_basic(self):
    box_3d = column_types.Box3d(
        center=column_types.Vec3d(1.5, -1.6, 1.7),
        size=column_types.Vec3d(0.9384312453, 10.442552123, 1.388204932),
        heading=0.01432353,
    )
    box_3d_numpy = box_3d.numpy(dtype=np.float64)
    self.assertLen(box_3d_numpy.shape, 1)
    self.assertTupleEqual(box_3d_numpy.shape, (7,))
    self.assertAlmostEqual(box_3d_numpy[0], box_3d.center.x)
    self.assertAlmostEqual(box_3d_numpy[1], box_3d.center.y)
    self.assertAlmostEqual(box_3d_numpy[2], box_3d.center.z)
    self.assertAlmostEqual(box_3d_numpy[3], box_3d.size.x)
    self.assertAlmostEqual(box_3d_numpy[4], box_3d.size.y)
    self.assertAlmostEqual(box_3d_numpy[5], box_3d.size.z)
    self.assertAlmostEqual(box_3d_numpy[6], box_3d.heading)


if __name__ == "__main__":
  absltest.main()
