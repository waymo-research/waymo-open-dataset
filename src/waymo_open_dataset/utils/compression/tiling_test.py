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
"""Tests for waymo_open_dataset.utils.compression.tiled_codec."""

import tensorflow as tf

from waymo_open_dataset.utils.compression import tiling as _lib


def _test_data(shape):
  return tf.reshape(tf.range(tf.reduce_prod(shape), dtype=tf.float32), shape)


class UtilsTest(tf.test.TestCase):

  def test_extract_tiles_creates_correct_number_of_tiles(self):
    orig = _test_data((10, 20, 3))
    tiles = _lib.extract_tiles(orig, tile_size=5)
    self.assertEqual(tiles.shape, (2 * 4, 5 * 5 * 3))

  def test_extract_tiles_pads_tiles_using_reflect_mode(self):
    orig = _test_data((4, 4, 1))

    tiles = _lib.extract_tiles(orig, tile_size=3)

    self.assertEqual(tiles.shape, (2 * 2, 3 * 3 * 1))
    tiles_sliced = tf.reshape(tiles, (2, 2, 3, 3, 1))
    self.assertAllClose(
        tiles_sliced[0, 0, :, :, 0], [[0, 1, 2], [4, 5, 6], [8, 9, 10]]
    )
    self.assertAllClose(
        tiles_sliced[0, 1, :, :, 0], [[3, 2, 1], [7, 6, 5], [11, 10, 9]]
    )
    self.assertAllClose(
        tiles_sliced[1, 0, :, :, 0], [[12, 13, 14], [8, 9, 10], [4, 5, 6]]
    )
    self.assertAllClose(
        tiles_sliced[1, 1, :, :, 0], [[15, 14, 13], [11, 10, 9], [7, 6, 5]]
    )

  def test_combine_tiles_reconstructs_original_data(self):
    tiles = tf.reshape(
        tf.constant(
            [
                [
                    [[0, 1, 2], [4, 5, 6], [8, 9, 10]],
                    [[3, 2, 1], [7, 6, 5], [11, 10, 9]],
                ],
                [
                    [[12, 13, 14], [8, 9, 10], [4, 5, 6]],
                    [[15, 14, 13], [11, 10, 9], [7, 6, 5]],
                ],
            ],
            dtype=tf.float32,
        ),
        (2 * 2, 3 * 3 * 1),
    )

    orig = _lib.combine_tiles(tiles, orig_height=4, orig_width=4, tile_size=3)

    self.assertEqual(orig.shape, (4, 4, 1))
    self.assertAllClose(orig, _test_data((4, 4, 1)))


if __name__ == "__main__":
  tf.test.main()
