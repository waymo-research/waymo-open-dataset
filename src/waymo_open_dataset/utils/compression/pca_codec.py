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
"""An implementation of the TileCodes which uses PCA decomposition."""

import dataclasses
import json

import dacite
import numpy as np
from sklearn import decomposition
from sklearn import pipeline
from sklearn import preprocessing
import tensorflow as tf

from waymo_open_dataset.utils.compression import tiling


@dataclasses.dataclass(frozen=True)
class PcaParams:
  """Parameters for the PCA.

  tile_vec_dims = tile_size * tile_size * dims

  Attributes:
    mean: a list with tile_vec_dims elements.
    std: a list with tile_vec_dims elements.
    eigen_vectors: a list with n_components elements. Each element is a list
      with tile_vec_dims values.
    eigen_values: a list with n_components elements.
  """

  mean: list[float]
  std: list[float]
  eigen_vectors: list[list[float]]
  eigen_values: list[float]


def fit_pca(
    data: np.ndarray, num_components: int
) -> tuple[pipeline.Pipeline, PcaParams]:
  """Performs the PCA decomposition.

  It is convenience wrapper for sklearn methods.

  Args:
    data: an array with shape [num_samples, num_features].
    num_components: number of PCA componenents to use.

  Returns:
    2-tuple (pipeline, parameters).
  """
  scaler = preprocessing.StandardScaler()
  pca = decomposition.PCA(n_components=num_components)
  p = pipeline.Pipeline([('scaler', scaler), ('pca', pca)])
  p.fit(data.astype(np.float64))
  return p, PcaParams(
      mean=scaler.mean_.tolist(),
      std=scaler.scale_.tolist(),
      eigen_vectors=pca.components_.tolist(),
      eigen_values=pca.explained_variance_.tolist(),
  )


class Pca:
  """A linear transformation."""

  def __init__(self, params: PcaParams):
    # shape: [1, input_dims]
    # Use float64 to avoid losing precision due to numerical issues.
    to_tensor = lambda x: tf.convert_to_tensor(x, dtype=tf.float64)
    self._mean = to_tensor(params.mean)[tf.newaxis, :]
    # shape: [1, input_dims]
    self._std = to_tensor(params.std)[tf.newaxis, :]
    # shape: [output_dims, input_dims]
    self._eigen_vectors = to_tensor(params.eigen_vectors)

  def transform(self, features: tf.Tensor) -> tf.Tensor:
    scaled = (tf.cast(features, tf.float64) - self._mean) / self._std
    return tf.einsum('ik,nk->ni', self._eigen_vectors, scaled)

  def inverse_transform(self, coeffs: tf.Tensor) -> tf.Tensor:
    scaled = tf.einsum('ik,ni->nk', self._eigen_vectors, coeffs)
    return tf.cast(scaled * self._std + self._mean, tf.float32)


def bucketed_quantization(
    data: np.ndarray, num_buckets: list[int]
) -> list[float]:
  """Returns quantization factors based on number of buckets for each dimension.

  It is a heuristic method to determine quantization factors for the Codec.

  Args:
    data: an array with shape [n_samples, dims].
    num_buckets: a list with dims values.

  Returns:
    a list with dims values.
  """
  return [np.max(np.abs(data[:, i])) / n for i, n in enumerate(num_buckets)]


@dataclasses.dataclass(frozen=True)
class Code:
  """Encoded data.

  Original data is a tensor with shape [height, width, dims].

  Attributes:
    reference: a float32 tensor with shape [dims].
    quantized_values: an int32 tensoro with shape [size].
    shape: a tensor with 2 values [height, width].
  """

  reference: tf.Tensor
  quantized_values: tf.Tensor
  shape: list[int]


@dataclasses.dataclass(frozen=True)
class CodecConfig:
  """Configuration parameters for the `Codec`."""

  pca: PcaParams
  quantization: list[float]
  tile_size: int

  def to_json(self) -> str:
    """Serializes config into json."""
    return json.dumps(dataclasses.asdict(self), indent=2)

  @classmethod
  def from_json(cls, json_str: str) -> 'CodecConfig':
    """Deserializes config from json."""
    return dacite.from_dict(data_class=cls, data=json.loads(json_str))


def codec_extract_tiles(
    data: tf.Tensor, tile_size: int
) -> tuple[tf.Tensor, tf.Tensor]:
  """Extracts tiles for data offsets relative to its mean.

  Args:
    data: a tensor with shape [height, width, dims].
    tile_size: size of the square tiles to extract.

  Returns:
    a 2-tuple (reference, tiles), where reference - is a tensor with shape
    [dims] and tiles is a tensor with shape [n_tiles, tile_size*tile_size*dims].
  """
  tf.assert_rank(data, 3, message=f'Expected rank=3, got shape {data.shape}')
  # shape [1, 1, dims]
  reference = tf.math.reduce_mean(data, axis=[0, 1], keepdims=True)
  # shape [n_tiles, tile_size*tile_size*dims]
  tiles = tiling.extract_tiles(data - reference, tile_size)
  return tf.squeeze(reference, axis=[0, 1]), tiles


def codec_combine_tiles(
    reference: tf.Tensor,
    tiles: tf.Tensor,
    shape: list[int],
    tile_size: int,
) -> tf.Tensor:
  """Combines tiles for the Codec."""
  offsets = tiling.combine_tiles(
      tiles,
      orig_height=shape[0],
      orig_width=shape[1],
      tile_size=tile_size,
  )
  return offsets + reference[tf.newaxis, tf.newaxis, :]


class Codec:
  """A class which implements PCA based encoding of 2D vector fields."""

  def __init__(self, config: CodecConfig):
    self._tile_size = config.tile_size
    self._pca = Pca(config.pca)
    # shape: [n_components]
    self._quantization = tf.convert_to_tensor(
        config.quantization, dtype=tf.float64
    )

  @property
  def quantization_dims(self) -> int:
    return self._quantization.shape[0]

  def _quantize(self, tiles_pca: tf.Tensor) -> tf.Tensor:
    dtype = tf.int32
    clipped = tf.clip_by_value(
        tiles_pca,
        dtype.min * self._quantization,
        dtype.max * self._quantization,
    )
    return tf.cast(tf.round(clipped / self._quantization), dtype)

  def _inv_quantize(self, quantized_values: tf.Tensor) -> tf.Tensor:
    return self._quantization * tf.cast(quantized_values, tf.float64)

  def encode(self, data: tf.Tensor) -> Code:
    """Encodes input data.

    Args:
      data: a tensor with shape [height, width, dims].

    Returns:
      encoded data.
    """
    reference, tiles = codec_extract_tiles(data, self._tile_size)
    # shape [num_tiles, n_components]
    values = self._pca.transform(tiles)
    return Code(
        reference=reference,
        quantized_values=self._quantize(values),
        shape=data.shape.as_list(),
    )

  def decode(self, code: Code) -> tf.Tensor:
    """Decodes the data.

    Args:
      code: encoded data.

    Returns:
      a tensor with shape [height, width, dims].
    """
    # shape [num_tiles, n_components]
    values = self._inv_quantize(code.quantized_values)
    # shape [num_tiles, tile_size*tile_size*dims]
    tiles = self._pca.inverse_transform(values)
    # shape [height, width, dims]
    return codec_combine_tiles(
        code.reference, tiles, code.shape, self._tile_size
    )
