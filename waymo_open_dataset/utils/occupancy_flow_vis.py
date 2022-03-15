# Copyright 2022 The Waymo Open Dataset Authors. All Rights Reserved.
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
"""Visualization functions."""

import math
from typing import Optional, Tuple, Union

import tensorflow as tf

from waymo_open_dataset.utils import occupancy_flow_data


def occupancy_rgb_image(
    agent_grids: occupancy_flow_data.AgentGrids,
    roadgraph_image: tf.Tensor,
    gamma: float = 1.6,
) -> tf.Tensor:
  """Visualize predictions or ground-truth occupancy.

  Args:
    agent_grids: AgentGrids object containing optional
      vehicles/pedestrians/cyclists.
    roadgraph_image: Road graph image [batch_size, height, width, 1] float32.
    gamma: Amplify predicted probabilities so that they are easier to see.

  Returns:
    [batch_size, height, width, 3] float32 RGB image.
  """
  zeros = tf.zeros_like(roadgraph_image)
  ones = tf.ones_like(zeros)

  agents = agent_grids
  veh = zeros if agents.vehicles is None else agents.vehicles
  ped = zeros if agents.pedestrians is None else agents.pedestrians
  cyc = zeros if agents.cyclists is None else agents.cyclists

  veh = tf.math.pow(veh, 1 / gamma)
  ped = tf.math.pow(ped, 1 / gamma)
  cyc = tf.math.pow(cyc, 1 / gamma)

  # Convert layers to RGB.
  rg_rgb = tf.concat([zeros, zeros, zeros], axis=-1)
  veh_rgb = tf.concat([veh, zeros, zeros], axis=-1)  # Red.
  ped_rgb = tf.concat([zeros, ped * 0.67, zeros], axis=-1)  # Green.
  cyc_rgb = tf.concat([cyc * 0.33, zeros, zeros * 0.33], axis=-1)  # Purple.
  bg_rgb = tf.concat([ones, ones, ones], axis=-1)  # White background.
  # Set alpha layers over all RGB channels.
  rg_a = tf.concat([roadgraph_image, roadgraph_image, roadgraph_image], axis=-1)
  veh_a = tf.concat([veh, veh, veh], axis=-1)
  ped_a = tf.concat([ped, ped, ped], axis=-1)
  cyc_a = tf.concat([cyc, cyc, cyc], axis=-1)
  # Stack layers one by one.
  img, img_a = _alpha_blend(fg=rg_rgb, bg=bg_rgb, fg_a=rg_a)
  img, img_a = _alpha_blend(fg=veh_rgb, bg=img, fg_a=veh_a, bg_a=img_a)
  img, img_a = _alpha_blend(fg=ped_rgb, bg=img, fg_a=ped_a, bg_a=img_a)
  img, img_a = _alpha_blend(fg=cyc_rgb, bg=img, fg_a=cyc_a, bg_a=img_a)
  return img


def flow_rgb_image(
    flow: tf.Tensor,
    roadgraph_image: tf.Tensor,
    agent_trails: tf.Tensor,
) -> tf.Tensor:
  """Converts (dx, dy) flow to RGB image.

  Args:
    flow: [batch_size, height, width, 2] float32 tensor holding (dx, dy) values.
    roadgraph_image: Road graph image [batch_size, height, width, 1] float32.
    agent_trails: [batch_size, height, width, 1] float32 tensor containing
      rendered trails for all agents over the past and current time frames.

  Returns:
    [batch_size, height, width, 3] float32 RGB image.
  """
  # Swap x, y for compatibilty with published visualizations.
  flow = tf.roll(flow, shift=1, axis=-1)
  # saturate_magnitude=-1 normalizes highest intensity to largest magnitude.
  flow_image = _optical_flow_to_rgb(flow, saturate_magnitude=-1)
  # Add roadgraph.
  flow_image = _add_grayscale_layer(roadgraph_image, flow_image)  # Black.
  # Overlay agent trails.
  flow_image = _add_grayscale_layer(agent_trails * 0.2, flow_image)  # 0.2 alpha
  return flow_image


def _add_grayscale_layer(
    fg_a: tf.Tensor,
    scene_rgb: tf.Tensor,
) -> tf.Tensor:
  """Adds a black/gray layer using fg_a as alpha over an RGB image."""
  # Create a black layer matching dimensions of fg_a.
  black = tf.zeros_like(fg_a)
  black = tf.concat([black, black, black], axis=-1)
  # Add the black layer with transparency over the scene_rgb image.
  overlay, _ = _alpha_blend(fg=black, bg=scene_rgb, fg_a=fg_a, bg_a=1.0)
  return overlay


def _alpha_blend(
    fg: tf.Tensor,
    bg: tf.Tensor,
    fg_a: Optional[tf.Tensor] = None,
    bg_a: Optional[Union[tf.Tensor, float]] = None,
) -> Tuple[tf.Tensor, tf.Tensor]:
  """Overlays foreground and background image with custom alpha values.

  Implements alpha compositing using Porter/Duff equations.
  https://en.wikipedia.org/wiki/Alpha_compositing

  Works with 1-channel or 3-channel images.

  If alpha values are not specified, they are set to the intensity of RGB
  values.

  Args:
    fg: Foreground: float32 tensor shaped [batch, grid_height, grid_width, d].
    bg: Background: float32 tensor shaped [batch, grid_height, grid_width, d].
    fg_a: Foreground alpha: float32 tensor broadcastable to fg.
    bg_a: Background alpha: float32 tensor broadcastable to bg.

  Returns:
    Output image: tf.float32 tensor shaped [batch, grid_height, grid_width, d].
    Output alpha: tf.float32 tensor shaped [batch, grid_height, grid_width, d].
  """
  if fg_a is None:
    fg_a = fg
  if bg_a is None:
    bg_a = bg
  eps = tf.keras.backend.epsilon()
  out_a = fg_a + bg_a * (1 - fg_a)
  out_rgb = (fg * fg_a + bg * bg_a * (1 - fg_a)) / (out_a + eps)
  return out_rgb, out_a



def _optical_flow_to_hsv(
    flow: tf.Tensor,
    saturate_magnitude: float = -1.0,
    name: Optional[str] = None,
) -> tf.Tensor:
  """Visualize an optical flow field in HSV colorspace.

  This uses the standard color code with hue corresponding to direction of
  motion and saturation corresponding to magnitude.

  The attr `saturate_magnitude` sets the magnitude of motion (in pixels) at
  which the color code saturates. A negative value is replaced with the maximum
  magnitude in the optical flow field.

  Args:
    flow: A `Tensor` of type `float32`. A 3-D or 4-D tensor storing (a batch of)
      optical flow field(s) as flow([batch,] i, j) = (dx, dy). The shape of the
      tensor is [height, width, 2] or [batch, height, width, 2] for the 4-D
      case.
    saturate_magnitude: An optional `float`. Defaults to `-1`.
    name: A name for the operation (optional).

  Returns:
    An tf.float32 HSV image (or image batch) of size [height, width, 3]
    (or [batch, height, width, 3]) compatible with tensorflow color conversion
    ops. The hue at each pixel corresponds to direction of motion. The
    saturation at each pixel corresponds to the magnitude of motion relative to
    the `saturate_magnitude` value. Hue, saturation, and value are in [0, 1].
  """
  with tf.name_scope(name or 'OpticalFlowToHSV'):
    flow_shape = flow.shape
    if len(flow_shape) < 3:
      raise ValueError('flow must be at least 3-dimensional, got'
                       f' `{flow_shape}`')
    if flow_shape[-1] != 2:
      raise ValueError(f'flow must have innermost dimension of 2, got'
                       f' `{flow_shape}`')
    height = flow_shape[-3]
    width = flow_shape[-2]
    flow_flat = tf.reshape(flow, (-1, height, width, 2))

    dx = flow_flat[..., 0]
    dy = flow_flat[..., 1]
    # [batch_size, height, width]
    magnitudes = tf.sqrt(tf.square(dx) + tf.square(dy))
    if saturate_magnitude < 0:
      # [batch_size, 1, 1]
      local_saturate_magnitude = tf.reduce_max(
          magnitudes, axis=(1, 2), keepdims=True)
    else:
      local_saturate_magnitude = tf.convert_to_tensor(saturate_magnitude)

    # Hue is angle scaled to [0.0, 1.0).
    hue = (tf.math.mod(tf.math.atan2(dy, dx), (2 * math.pi))) / (2 * math.pi)
    # Saturation is relative magnitude.
    relative_magnitudes = tf.math.divide_no_nan(magnitudes,
                                                local_saturate_magnitude)
    saturation = tf.minimum(
        relative_magnitudes,
        1.0  # Larger magnitudes saturate.
    )
    # Value is fixed.
    value = tf.ones_like(saturation)
    hsv_flat = tf.stack((hue, saturation, value), axis=-1)
    return tf.reshape(hsv_flat, flow_shape.as_list()[:-1] + [3])


def _optical_flow_to_rgb(
    flow: tf.Tensor,
    saturate_magnitude: float = -1.0,
    name: Optional[str] = None,
) -> tf.Tensor:
  """Visualize an optical flow field in RGB colorspace."""
  name = name or 'OpticalFlowToRGB'
  hsv = _optical_flow_to_hsv(flow, saturate_magnitude, name)
  return tf.image.hsv_to_rgb(hsv)
