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
"""A binary to generate a configuration file for the PCACodec."""

from typing import Iterable

import numpy as np
import pandas as pd
import tensorflow as tf

from waymo_open_dataset.utils.compression import pca_codec
from waymo_open_dataset.v2.perception import object_asset


def _parse_array(prefix, row):
  return np.array(row[f'{prefix}.values']).reshape(row[f'{prefix}.shape'])


def _parse_rays(row) -> np.ndarray:
  origin = _parse_array('[ObjectAssetRayComponent].ray_origin', row)
  direction = _parse_array('[ObjectAssetRayComponent].ray_direction', row)
  return np.concatenate([origin, direction], axis=-1)


def iter_ray_samples(df: pd.DataFrame) -> Iterable[np.ndarray]:
  """Iterates over object asset samples from a table.

  Args:
    df: a DataFrame with `ObjectAssetRayComponent` columns.

  Yields:
    arrays with shape [height, width, 6] - concatenated origin and directions of
    the rays.
  """
  for _, row in df.iterrows():
    yield _parse_rays(row)


ObjectAssetRayCodecConfig = pca_codec.CodecConfig


class ObjectAssetRayCodec:
  """Wrapper for pca_codec.Codec."""

  def __init__(self, config: ObjectAssetRayCodecConfig):
    self._pca_codec = pca_codec.Codec(config)

  def encode(
      self, src: object_asset.ObjectAssetRayComponent
  ) -> object_asset.ObjectAssetRayCompressedComponent:
    """Encodes an uncompressed ray component into compressed."""
    assert src.ray_origin is not None
    assert src.ray_direction is not None
    vec = tf.concat([src.ray_origin.tensor, src.ray_direction.tensor], axis=-1)
    code = self._pca_codec.encode(vec)
    return object_asset.ObjectAssetRayCompressedComponent(
        key=src.key,
        reference=code.reference.numpy().tolist(),
        quantized_values=code.quantized_values.numpy().ravel().tolist(),
        shape=code.shape,
    )

  def decode(
      self, src: object_asset.ObjectAssetRayCompressedComponent
  ) -> object_asset.ObjectAssetRayComponent:
    """Decodes a compressed ray component into uncompressed."""
    code = pca_codec.Code(
        reference=tf.constant(src.reference),
        quantized_values=tf.reshape(
            tf.constant(src.quantized_values),
            (-1, self._pca_codec.quantization_dims),
        ),
        shape=src.shape,
    )
    vec = self._pca_codec.decode(code)
    origin, direction = tf.split(vec, num_or_size_splits=[3, 3], axis=-1)
    return object_asset.ObjectAssetRayComponent(
        key=src.key,
        ray_origin=object_asset.CameraRayImage(
            values=origin.numpy().ravel().tolist(),
            shape=origin.shape.as_list(),
        ),
        ray_direction=object_asset.CameraRayImage(
            values=direction.numpy().ravel().tolist(),
            shape=direction.shape.as_list(),
        ),
    )
