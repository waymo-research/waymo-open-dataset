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
"""Tests for waymo_open_dataset.utils.compression.pca_codec."""

import numpy as np
import tensorflow as tf

from waymo_open_dataset.utils.compression import pca_codec as _lib


class UtilsTest(tf.test.TestCase):

  def test_fit_pca_params_output_correctly_shaped_parameters(self):
    data = np.random.normal(size=(10000, 20))

    _, params = _lib.fit_pca(data, num_components=10)

    self.assertLen(params.mean, 20)
    self.assertLen(params.std, 20)
    self.assertEqual(np.array(params.eigen_vectors).shape, (10, 20))
    self.assertLen(params.eigen_values, 10)

  def test_pca_class_works_the_same_way_as_a_pca_pipeline_from_sklearn(self):
    data = np.random.normal(size=(10000, 20))
    # Since dimensions of the input vectors uncorrelated we need all components
    # to be able to reconstruct the input.
    num_components = 20

    sklearn_pipeline, pca_params = _lib.fit_pca(data, num_components)
    pca = _lib.Pca(pca_params)

    data_tr = pca.transform(data)
    data_rec = pca.inverse_transform(data_tr)

    data_tr_sklearn = sklearn_pipeline.transform(data)
    data_rec_sklearn = sklearn_pipeline.inverse_transform(data_tr_sklearn)

    self.assertAllClose(data_tr, data_tr_sklearn)
    self.assertAllClose(data_rec, data_rec_sklearn)
    self.assertAllClose(data, data_rec_sklearn)
    self.assertAllClose(data, data_rec)

  def test_codec_config_can_be_saved_and_loaded(self):
    orig = _lib.CodecConfig(
        pca=_lib.PcaParams(
            mean=np.random.uniform(size=(6,)).tolist(),
            std=np.random.uniform(size=(6,)).tolist(),
            eigen_vectors=np.random.uniform(
                size=(
                    10,
                    20,
                )
            ).tolist(),
            eigen_values=np.random.uniform(size=(10)).tolist(),
        ),
        quantization=np.random.uniform(size=(10)).tolist(),
        tile_size=8,
    )

    json = orig.to_json()
    loaded = _lib.CodecConfig.from_json(json)

    self.assertAllClose(orig.pca.mean, loaded.pca.mean)
    self.assertAllClose(orig.pca.std, loaded.pca.std)
    self.assertAllClose(orig.pca.eigen_vectors, loaded.pca.eigen_vectors)
    self.assertAllClose(orig.pca.eigen_values, loaded.pca.eigen_values)
    self.assertAllClose(orig.quantization, loaded.quantization)
    self.assertEqual(orig.tile_size, loaded.tile_size)


def _batch_extract_tiles(data, tile_size):
  return np.concatenate(
      [_lib.codec_extract_tiles(s, tile_size)[1].numpy() for s in data],
      axis=0,
  )


def _random_2d_grid_with_correlated_vectors(
    num_samples: int,
    num_tiles_y: int,
    num_tiles_x: int,
    tile_size: int,
    out_dims: int,
    n_components: int,
) -> np.ndarray:
  """Generates a 2D grid, where vectors within each tile are correlated."""
  # This transformation defines the correlation between output vectors.
  transform = np.random.normal(
      size=(tile_size * tile_size * out_dims, n_components)
  )
  # Create tiles uncorrelated vectors with `n_components` dimensions.
  tiles = np.random.normal(
      size=(num_samples, num_tiles_y, num_tiles_x, n_components)
  )
  # Vectors in these tiles are correlated.
  correlated_tiles = np.einsum('ik,...k->...i', transform, tiles).reshape(
      (num_samples, num_tiles_y, num_tiles_x, tile_size, tile_size, out_dims)
  )
  shape = (
      num_samples,
      num_tiles_y * tile_size,
      num_tiles_x * tile_size,
      out_dims,
  )
  return np.swapaxes(correlated_tiles, 2, 3).reshape(shape).astype(np.float32)


class PcaCodecTest(tf.test.TestCase):

  def test_compressed_data_correctly(self):
    tile_size = 8
    num_components = 30
    data = _random_2d_grid_with_correlated_vectors(
        num_samples=100,
        num_tiles_y=16,
        num_tiles_x=16,
        tile_size=tile_size,
        out_dims=6,
        n_components=20,
    )
    tiles = _batch_extract_tiles(data, tile_size)
    _, pca_params = _lib.fit_pca(tiles, num_components)
    # Small number to ensure there is will be almost no data loss.
    quant = 1e-5
    params = _lib.CodecConfig(
        pca=pca_params,
        quantization=np.array([quant] * num_components),
        tile_size=tile_size,
    )
    sample = tf.constant(data[0])

    codec = _lib.Codec(params)
    code = codec.encode(sample)

    with self.subTest('code_has_correct_shape'):
      self.assertEqual(code.reference.shape, (6,))
      self.assertEqual(code.quantized_values.shape, (16 * 16, num_components))
      # TODO(gorban): refactor to use tuples for Code.shape
      self.assertEqual(code.shape, [128, 128, 6])
    with self.subTest('decodes_original_data'):
      decoded = codec.decode(code)
      self.assertAllClose(sample, decoded, atol=1e-4)


if __name__ == '__main__':
  tf.test.main()
