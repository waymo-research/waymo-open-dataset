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
"""Base class for Codecs to encode/decode vector 2D fields using squate tiles."""
import dataclasses
import tensorflow as tf


def _ceil(x: float) -> int:
  return int(tf.math.ceil(x))


@dataclasses.dataclass(frozen=True)
class _TileShapeInfo:
  """A helper class to get all shape information related to tiling."""

  orig_height: int
  orig_width: int
  tile_size: int
  dims: int

  @classmethod
  def square_tiles(
      cls, orig_shape: tuple[int, int, int], tile_size: int
  ) -> '_TileShapeInfo':
    """Creates the object from the original shape."""
    orig_height, orig_width, dims = orig_shape
    return cls(
        orig_height=orig_height,
        orig_width=orig_width,
        tile_size=tile_size,
        dims=dims,
    )

  @property
  def num_tiles_y(self) -> int:
    return _ceil(self.orig_height / self.tile_size)

  @property
  def num_tiles_x(self) -> int:
    return _ceil(self.orig_width / self.tile_size)

  @property
  def num_tiles(self) -> int:
    return self.num_tiles_y * self.num_tiles_x

  @property
  def padded_height(self) -> int:
    return self.num_tiles_y * self.tile_size

  @property
  def padded_width(self) -> int:
    return self.num_tiles_x * self.tile_size

  @property
  def padded_shape(self) -> tuple[int, int, int]:
    return (
        self.padded_height,
        self.padded_width,
        self.dims,
    )

  @property
  def tile_shape(self) -> tuple[int, int]:
    return (
        self.num_tiles,
        self.tile_size * self.tile_size * self.dims,
    )


def extract_tiles(data: tf.Tensor, tile_size: int) -> tf.Tensor:
  """Extracts square tiles from a 2D vector grid.

  Args:
    data: a 2D array of vectors with shape [height, width, dims].
    tile_size: size of the square tiles to extract.

  Returns:
    an array with the shape
      [num_tiles, tile_size*tile_size*dims].
  """
  info = _TileShapeInfo.square_tiles(data.shape, tile_size)
  padding = tf.constant([
      [0, info.padded_height - info.orig_height],
      [0, info.padded_width - info.orig_width],
      [0, 0],
  ])
  padded = tf.pad(data, padding, mode='REFLECT')
  pre_tiles_5d = tf.reshape(
      padded,
      (
          info.num_tiles_y,
          info.tile_size,
          info.num_tiles_x,
          info.tile_size,
          info.dims,
      ),
  )
  tiles_5d = tf.transpose(pre_tiles_5d, [0, 2, 1, 3, 4])
  return tf.reshape(tiles_5d, info.tile_shape)


def combine_tiles(
    tiles: tf.Tensor, orig_height: int, orig_width: int, tile_size: int
) -> tf.Tensor:
  """Combines tiles into a 2D vector field.

  Args:
    tiles: an array with the shape [num_tiles, tile_size * tile_size * dims],
      where the last dimension represents a tile value tensor with shape
      [tile_size, tile_size, dims] flattened into a vector.
    orig_height: height of the original 2D grid.
    orig_width: width of the original 2D grid.
    tile_size: size of the square tiles to extract.

  Returns:
    a 2D array of vectors (aka 2D vector field) with shape [height, width,
    dims].
  """
  dims = tiles.shape[-1] // (tile_size * tile_size)
  if tiles.shape[-1] != (tile_size * tile_size * dims):
    raise ValueError(f"{tile_size=} doesn't divide {tiles.shape[-1]=} evenly.")
  info = _TileShapeInfo(
      orig_height=orig_height,
      orig_width=orig_width,
      tile_size=tile_size,
      dims=dims,
  )
  tiles_5d = tf.reshape(
      tiles,
      (
          info.num_tiles_y,
          info.num_tiles_x,
          info.tile_size,
          info.tile_size,
          info.dims,
      ),
  )
  pre_tiles_5d = tf.transpose(tiles_5d, [0, 2, 1, 3, 4])
  padded = tf.reshape(pre_tiles_5d, info.padded_shape)
  return padded[: info.orig_height, : info.orig_width, :]
