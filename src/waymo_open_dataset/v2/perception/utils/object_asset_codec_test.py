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
"""Tests for waymo_open_dataset.utils.compression.pca_codec_configurator."""

from absl.testing import absltest
import numpy as np
import pandas as pd

from waymo_open_dataset.utils.compression import pca_codec
from waymo_open_dataset.v2.perception import base
from waymo_open_dataset.v2.perception import object_asset
from waymo_open_dataset.v2.perception.utils import object_asset_codec as _lib


def _generate_data(
    num_samples: int, tile_size: int, n_tiles_y: int, n_tiles_x: int, dims: int
) -> tuple[np.ndarray, np.ndarray]:
  tiles = np.random.normal(
      size=(num_samples * n_tiles_y * n_tiles_x, tile_size * tile_size * dims)
  )
  tiles_before_swap = tiles.reshape(
      (num_samples, n_tiles_y, tile_size, n_tiles_x, tile_size, 6)
  )
  data_before_reshape = np.swapaxes(tiles_before_swap, 2, 3)
  data = data_before_reshape.reshape(
      (num_samples, n_tiles_y * tile_size, n_tiles_x * tile_size, dims)
  )
  return tiles, data


def _create_uncompressed_component(
    sample: np.ndarray,
) -> object_asset.ObjectAssetRayComponent:
  origin, direction = sample[:, :, :3], sample[:, :, 3:]
  return object_asset.ObjectAssetRayComponent(
      key=base.ObjectAssetKey(
          segment_context_name='fake_context_name',
          frame_timestamp_micros=0,
          laser_object_id='fake_object_id',
          camera_name=0,
      ),
      ray_origin=object_asset.CameraRayImage(
          values=origin.ravel().tolist(), shape=list(origin.shape)
      ),
      ray_direction=object_asset.CameraRayImage(
          values=direction.ravel().tolist(), shape=list(direction.shape)
      ),
  )


class PcaCodecConfiguratorTest(absltest.TestCase):

  def test_creates_samples_from_ray_origin_and_directions(self):
    # A table with two rows/samples with shapes:
    s0 = (256, 256, 3)
    n0 = np.prod(s0)
    s1 = (128, 196, 3)
    n1 = np.prod(s1)
    df = pd.DataFrame({
        '[ObjectAssetRayComponent].ray_origin.values': [[1] * n0, [2] * n1],
        '[ObjectAssetRayComponent].ray_origin.shape': [s0, s1],
        '[ObjectAssetRayComponent].ray_direction.values': [[3] * n0, [4] * n1],
        '[ObjectAssetRayComponent].ray_direction.shape': [s0, s1],
    })

    samples = list(_lib.iter_ray_samples(df))

    self.assertLen(samples, 2)
    self.assertEqual(samples[0].shape, (256, 256, 6))
    np.testing.assert_allclose(samples[0][:, :, :3], 1)
    np.testing.assert_allclose(samples[0][:, :, 3:], 3)
    self.assertEqual(samples[1].shape, (128, 196, 6))
    np.testing.assert_allclose(samples[1][:, :, :3], 2)
    np.testing.assert_allclose(samples[1][:, :, 3:], 4)

  def test_codec_encodes_and_decodes_object_asset_rays(self):
    # Fake uncorrelated data to fit codec parameters.
    tile_size = 8
    tiles, data = _generate_data(
        num_samples=100, tile_size=tile_size, n_tiles_y=16, n_tiles_x=16, dims=6
    )
    # Small number to ensure there is will be almost no data loss.
    quant = 1e-5
    # Encoding uncorrelated data requires all components.
    num_components = tile_size * tile_size * 6
    config = pca_codec.CodecConfig(
        pca=pca_codec.fit_pca(tiles, num_components=num_components)[1],
        quantization=np.array([quant] * num_components),
        tile_size=tile_size,
    )
    orig = _create_uncompressed_component(data[0])

    codec = _lib.ObjectAssetRayCodec(config)
    compressed = codec.encode(orig)
    decompressed = codec.decode(compressed)

    self.assertEqual(decompressed.key, orig.key)
    self.assertEqual(decompressed.ray_origin.shape, orig.ray_origin.shape)
    self.assertEqual(decompressed.ray_direction.shape, orig.ray_direction.shape)
    np.testing.assert_allclose(
        decompressed.ray_origin.values, orig.ray_origin.values, atol=1e-4
    )
    np.testing.assert_allclose(
        decompressed.ray_direction.values, orig.ray_direction.values, atol=1e-4
    )


if __name__ == '__main__':
  absltest.main()
