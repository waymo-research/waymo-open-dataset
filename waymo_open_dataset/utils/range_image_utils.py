# Copyright 2019 The Waymo Open Dataset Authors. All Rights Reserved.
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
"""Utils to manage range images."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

import tensorflow as tf

__all__ = [
    'compute_range_image_polar', 'compute_range_image_cartesian',
    'build_range_image_from_point_cloud', 'build_camera_depth_image',
    'extract_point_cloud_from_range_image', 'crop_range_image',
    'compute_inclination'
]


def _combined_static_and_dynamic_shape(tensor):
  """Returns a list containing static and dynamic values for the dimensions.

  Returns a list of static and dynamic values for shape dimensions. This is
  useful to preserve static shapes when available in reshape operation.

  Args:
    tensor: A tensor of any type.

  Returns:
    A list of size tensor.shape.ndims containing integers or a scalar tensor.
  """
  static_tensor_shape = tensor.shape.as_list()
  dynamic_tensor_shape = tf.shape(tensor)
  combined_shape = []
  for index, dim in enumerate(static_tensor_shape):
    if dim is not None:
      combined_shape.append(dim)
    else:
      combined_shape.append(dynamic_tensor_shape[index])
  return combined_shape


def _scatter_nd_with_pool(index,
                          value,
                          shape,
                          pool_method=tf.unsorted_segment_max):
  """Similar as tf.scatter_nd but allows custom pool method.

  tf.scatter_nd accumulates (sums) values if there are duplicate indices.

  Args:
    index: [N, 2] tensor. Inner dims are coordinates along height (row) and then
      width (col).
    value: [N] tensor. Values to be scattered.
    shape: (height,width) list that specifies the shape of the output tensor.
    pool_method: pool method when there are multiple points scattered to one
      location.

  Returns:
    image: tensor of shape with value scattered. Missing pixels are set to 0.
  """
  if len(shape) != 2:
    raise ValueError('shape must be of size 2')
  height = shape[0]
  width = shape[1]
  # idx: [N]
  index_encoded, idx = tf.unique(index[:, 0] * width + index[:, 1])
  value_pooled = pool_method(value, idx, tf.size(index_encoded))
  index_unique = tf.stack(
      [index_encoded // width,
       tf.mod(index_encoded, width)], axis=-1)

  image = tf.scatter_nd(index_unique, value_pooled, [height, width])
  return image


def compute_range_image_polar(range_image,
                              extrinsic,
                              inclination,
                              dtype=tf.float64,
                              scope=None):
  """Computes range image polar coordinates.

  Args:
    range_image: [B, H, W] tensor. Lidar range images.
    extrinsic: [B, 4, 4] tensor. Lidar extrinsic.
    inclination: [B, H] tensor. Inclination for each row of the range image.
      0-th entry corresponds to the 0-th row of the range image.
    dtype: float type to use internally. This is needed as extrinsic and
      inclination sometimes have higher resolution than range_image.
    scope: the name scope.

  Returns:
    range_image_polar: [B, H, W, 3] polar coordinates.
  """
  # pylint: disable=unbalanced-tuple-unpacking
  _, height, width = _combined_static_and_dynamic_shape(range_image)
  range_image_dtype = range_image.dtype
  range_image = tf.cast(range_image, dtype)
  extrinsic = tf.cast(extrinsic, dtype)
  inclination = tf.cast(inclination, dtype)

  with tf.name_scope(scope, 'ComputeRangeImagePolar',
                     [range_image, extrinsic, inclination]):
    with tf.name_scope('Azimuth'):
      # [B].
      az_correction = tf.atan2(extrinsic[..., 1, 0], extrinsic[..., 0, 0])
      # [W].
      ratios = (tf.cast(tf.range(width, 0, -1), dtype=dtype) - .5) / tf.cast(
          width, dtype)
      # [B, W].
      azimuth = (ratios * 2. - 1.) * np.pi - tf.expand_dims(az_correction, -1)

    # [B, H, W]
    azimuth_tile = tf.tile(azimuth[:, tf.newaxis, :], [1, height, 1])
    # [B, H, W]
    inclination_tile = tf.tile(inclination[:, :, tf.newaxis], [1, 1, width])
    range_image_polar = tf.stack([azimuth_tile, inclination_tile, range_image],
                                 axis=-1)
    return tf.cast(range_image_polar, dtype=range_image_dtype)


def compute_range_image_cartesian(range_image_polar,
                                  extrinsic,
                                  pixel_pose=None,
                                  frame_pose=None,
                                  dtype=tf.float64,
                                  scope=None):
  """Computes range image cartesian coordinates from polar ones.

  Args:
    range_image_polar: [B, H, W, 3] float tensor. Lidar range image in polar
      coordinate in sensor frame.
    extrinsic: [B, 4, 4] float tensor. Lidar extrinsic.
    pixel_pose: [B, H, W, 4, 4] float tensor. If not None, it sets pose for each
      range image pixel.
    frame_pose: [B, 4, 4] float tensor. This must be set when pixel_pose is set.
      It decides the vehicle frame at which the cartesian points are computed.
    dtype: float type to use internally. This is needed as extrinsic and
      inclination sometimes have higher resolution than range_image.
    scope: the name scope.

  Returns:
    range_image_cartesian: [B, H, W, 3] cartesian coordinates.
  """
  range_image_polar_dtype = range_image_polar.dtype
  range_image_polar = tf.cast(range_image_polar, dtype)
  extrinsic = tf.cast(extrinsic, dtype)
  if pixel_pose is not None:
    pixel_pose = tf.cast(pixel_pose, dtype)
  if frame_pose is not None:
    frame_pose = tf.cast(frame_pose, dtype)

  with tf.name_scope(scope, 'ComputeRangeImageCartesian',
                     [range_image_polar, extrinsic, pixel_pose, frame_pose]):
    azimuth, inclination, range_image_range = tf.unstack(
        range_image_polar, axis=-1)

    cos_azimuth = tf.cos(azimuth)
    sin_azimuth = tf.sin(azimuth)
    cos_incl = tf.cos(inclination)
    sin_incl = tf.sin(inclination)

    # [B, H, W].
    x = cos_azimuth * cos_incl * range_image_range
    y = sin_azimuth * cos_incl * range_image_range
    z = sin_incl * range_image_range

    # [B, H, W, 3]
    range_image_points = tf.stack([x, y, z], -1)
    # [B, 3, 3]
    rotation = extrinsic[..., 0:3, 0:3]
    # translation [B, 1, 3]
    translation = tf.expand_dims(tf.expand_dims(extrinsic[..., 0:3, 3], 1), 1)

    # To vehicle frame.
    # [B, H, W, 3]
    range_image_points = tf.einsum('bkr,bijr->bijk', rotation,
                                   range_image_points) + translation
    if pixel_pose is not None:
      # To global frame.
      # [B, H, W, 3, 3]
      pixel_pose_rotation = pixel_pose[..., 0:3, 0:3]
      # [B, H, W, 3]
      pixel_pose_translation = pixel_pose[..., 0:3, 3]
      # [B, H, W, 3]
      range_image_points = tf.einsum(
          'bhwij,bhwj->bhwi', pixel_pose_rotation,
          range_image_points) + pixel_pose_translation
      if frame_pose is None:
        raise ValueError('frame_pose must be set when pixel_pose is set.')
      # To vehicle frame corresponding to the given frame_pose
      # [B, 4, 4]
      world_to_vehicle = tf.matrix_inverse(frame_pose)
      world_to_vehicle_rotation = world_to_vehicle[:, 0:3, 0:3]
      world_to_vehicle_translation = world_to_vehicle[:, 0:3, 3]
      # [B, H, W, 3]
      range_image_points = tf.einsum(
          'bij,bhwj->bhwi', world_to_vehicle_rotation,
          range_image_points) + world_to_vehicle_translation[:, tf.newaxis,
                                                             tf.newaxis, :]

    range_image_points = tf.cast(
        range_image_points, dtype=range_image_polar_dtype)
    return range_image_points


def build_camera_depth_image(range_image_cartesian,
                             extrinsic,
                             camera_projection,
                             camera_image_size,
                             camera_name,
                             pool_method=tf.unsorted_segment_min,
                             scope=None):
  """Builds camera depth image given camera projections.

  The depth value is the distance between a lidar point and camera frame origin.
  It is decided by cartesian coordinates in vehicle frame and the camera
  extrinsic. Optionally, the cartesian coordinates can be set in the vehicle
  frame corresponding to each pixel pose which makes the depth generated to have
  vehicle motion taken into account.

  Args:
    range_image_cartesian: [B, H, W, 3] tensor. Range image points in vehicle
      frame. Note that if the range image is provided by pixel_pose, then you
      can optionally pass in the cartesian coordinates in each pixel frame.
    extrinsic: [B, 4, 4] tensor. Camera extrinsic.
    camera_projection: [B, H, W, 6] tensor. Each range image pixel is associated
      with at most two camera projections. See dataset.proto for more details.
    camera_image_size: a list of [width, height] integers.
    camera_name: an integer that identifies a camera. See dataset.proto.
    pool_method: pooling method when multiple lidar points are projected to one
      image pixel.
    scope: the name scope.

  Returns:
    image: [B, width, height] depth image generated.
  """
  with tf.name_scope(scope, 'BuildCameraDepthImage',
                     [range_image_cartesian, extrinsic, camera_projection]):
    # [B, 4, 4]
    vehicle_to_camera = tf.matrix_inverse(extrinsic)
    # [B, 3, 3]
    vehicle_to_camera_rotation = vehicle_to_camera[:, 0:3, 0:3]
    # [B, 3]
    vehicle_to_camera_translation = vehicle_to_camera[:, 0:3, 3]
    # [B, H, W, 3]
    range_image_camera = tf.einsum(
        'bij,bhwj->bhwi', vehicle_to_camera_rotation,
        range_image_cartesian) + vehicle_to_camera_translation[:, tf.newaxis,
                                                               tf.newaxis, :]
    # [B, H, W]
    range_image_camera_norm = tf.norm(range_image_camera, axis=-1)
    camera_projection_mask_1 = tf.tile(
        tf.equal(camera_projection[..., 0:1], camera_name), [1, 1, 1, 2])
    camera_projection_mask_2 = tf.tile(
        tf.equal(camera_projection[..., 3:4], camera_name), [1, 1, 1, 2])
    camera_projection_selected = tf.ones_like(
        camera_projection[..., 1:3], dtype=camera_projection.dtype) * -1
    camera_projection_selected = tf.where(camera_projection_mask_2,
                                          camera_projection[..., 4:6],
                                          camera_projection_selected)
    # [B, H, W, 2]
    camera_projection_selected = tf.where(camera_projection_mask_1,
                                          camera_projection[..., 1:3],
                                          camera_projection_selected)
    # [B, H, W]
    camera_projection_mask = tf.logical_or(camera_projection_mask_1,
                                           camera_projection_mask_2)[..., 0]

    def fn(args):
      """Builds depth image for a single frame."""

      # NOTE: Do not use ri_range > 0 as mask as missing range image pixels are
      # not necessarily populated as range = 0.
      mask, ri_range, cp = args
      mask_ids = tf.where(mask)
      index = tf.gather_nd(
          tf.stack([cp[..., 1], cp[..., 0]], axis=-1), mask_ids)
      value = tf.gather_nd(ri_range, mask_ids)
      return _scatter_nd_with_pool(index, value, camera_image_size, pool_method)

    images = tf.map_fn(
        fn,
        elems=[
            camera_projection_mask, range_image_camera_norm,
            camera_projection_selected
        ],
        dtype=range_image_camera_norm.dtype,
        back_prop=False)
    return images


def build_range_image_from_point_cloud(points_vehicle_frame,
                                       num_points,
                                       extrinsic,
                                       inclination,
                                       range_image_size,
                                       dtype=tf.float64,
                                       scope=None):
  """Build virtual range image from point cloud assuming uniform azimuth.

  Args:
    points_vehicle_frame: tf tensor with shape [B, N, 3] in the vehicle frame.
    num_points: [B] int32 tensor indicating the number of points for each frame.
    extrinsic: tf tensor with shape [B, 4, 4].
    inclination: tf tensor of shape [B, H] that is the inclination angle per
      row. sorted from highest value to lowest.
    range_image_size: a size 2 [height, width] list that configures the size of
      the range image.
    dtype: the data type to use.
    scope: tf name scope.

  Returns:
    range_images : [B, H, W, ?] or [B, H, W] tensor. Range images built from the
      given points. Data type is the same as that of points_vehicle_frame. 0.0
      is populated when a pixel is missing.
    ri_indices: tf int32 tensor [B, N, 2]. It represents the range image index
      for each point.
    ri_ranges: [B, N] tensor. It represents the distance between a point and
      sensor frame origin of each point.
  """

  with tf.name_scope(
      scope,
      'BuildRangeImageFromPointCloud',
      values=[points_vehicle_frame, extrinsic, inclination]):
    points_vehicle_frame_dtype = points_vehicle_frame.dtype

    points_vehicle_frame = tf.cast(points_vehicle_frame, dtype)
    extrinsic = tf.cast(extrinsic, dtype)
    inclination = tf.cast(inclination, dtype)

    height, width = range_image_size

    # [B, 4, 4]
    vehicle_to_laser = tf.matrix_inverse(extrinsic)
    # [B, 3, 3]
    rotation = vehicle_to_laser[:, 0:3, 0:3]
    # [B, 1, 3]
    translation = tf.expand_dims(vehicle_to_laser[::, 0:3, 3], 1)
    # Points in sensor frame
    # [B, N, 3]
    points = tf.einsum('bij,bkj->bik', points_vehicle_frame,
                       rotation) + translation
    # [B, N]
    xy_norm = tf.norm(points[..., 0:2], axis=-1)
    # [B, N]
    point_inclination = tf.atan2(points[..., 2], xy_norm)
    # [B, N, H]
    point_inclination_diff = tf.abs(
        tf.expand_dims(point_inclination, axis=-1) -
        tf.expand_dims(inclination, axis=1))
    # [B, N]
    point_ri_row_indices = tf.argmin(
        point_inclination_diff, axis=-1, output_type=tf.int32)

    # [B, 1], within [-pi, pi]
    az_correction = tf.expand_dims(
        tf.atan2(extrinsic[..., 1, 0], extrinsic[..., 0, 0]), -1)
    # [B, N], within [-2pi, 2pi]
    point_azimuth = tf.atan2(points[..., 1], points[..., 0]) + az_correction

    point_azimuth_gt_pi_mask = point_azimuth > np.pi
    point_azimuth_lt_minus_pi_mask = point_azimuth < -np.pi
    point_azimuth = point_azimuth - tf.cast(point_azimuth_gt_pi_mask,
                                            dtype) * 2 * np.pi
    point_azimuth = point_azimuth + tf.cast(point_azimuth_lt_minus_pi_mask,
                                            dtype) * 2 * np.pi

    # [B, N].
    point_ri_col_indices = width - 1.0 + 0.5 - (point_azimuth +
                                                np.pi) / (2.0 * np.pi) * width
    point_ri_col_indices = tf.cast(tf.round(point_ri_col_indices), tf.int32)

    with tf.control_dependencies([
        tf.assert_non_negative(point_ri_col_indices),
        tf.assert_less(point_ri_col_indices, tf.cast(width, tf.int32))
    ]):
      # [B, N, 2]
      ri_indices = tf.stack([point_ri_row_indices, point_ri_col_indices], -1)
      # [B, N]
      ri_ranges = tf.cast(
          tf.norm(points, axis=-1), dtype=points_vehicle_frame_dtype)

      def fn(args):
        """Builds a range image for each frame.

        Args:
          args: a tuple containing:
            - ri_index: [N, 2]
            - ri_value: [N]
            - num_point: scalar tensor

        Returns:
          range_image: [H, W]
        """
        ri_index, ri_value, num_point = args
        # pylint: disable=unbalanced-tuple-unpacking
        ri_index = ri_index[0:num_point, :]
        ri_value = ri_value[0:num_point]
        range_image = _scatter_nd_with_pool(ri_index, ri_value, [height, width],
                                            tf.unsorted_segment_max)
        return range_image

      range_images = tf.map_fn(
          fn,
          elems=[ri_indices, ri_ranges, num_points],
          dtype=points_vehicle_frame_dtype,
          back_prop=False)

      return range_images, ri_indices, ri_ranges


def extract_point_cloud_from_range_image(range_image,
                                         extrinsic,
                                         inclination,
                                         pixel_pose=None,
                                         frame_pose=None,
                                         dtype=tf.float64,
                                         scope=None):
  """Extracts point cloud from range image.

  Args:
    range_image: [B, H, W] tensor. Lidar range images.
    extrinsic: [B, 4, 4] tensor. Lidar extrinsic.
    inclination: [B, H] tensor. Inclination for each row of the range image.
      0-th entry corresponds to the 0-th row of the range image.
    pixel_pose: [B, H, W, 4, 4] tensor. If not None, it sets pose for each range
      image pixel.
    frame_pose: [B, 4, 4] tensor. This must be set when pixel_pose is set. It
      decides the vehicle frame at which the cartesian points are computed.
    dtype: float type to use internally. This is needed as extrinsic and
      inclination sometimes have higher resolution than range_image.
    scope: the name scope.

  Returns:
    range_image_cartesian: [B, H, W, 3] with {x, y, z} as inner dims in vehicle
    frame.
  """
  with tf.name_scope(
      scope, 'ExtractPointCloudFromRangeImage',
      [range_image, extrinsic, inclination, pixel_pose, frame_pose]):
    range_image_polar = compute_range_image_polar(
        range_image, extrinsic, inclination, dtype=dtype)
    range_image_cartesian = compute_range_image_cartesian(
        range_image_polar,
        extrinsic,
        pixel_pose=pixel_pose,
        frame_pose=frame_pose,
        dtype=dtype)
    return range_image_cartesian


def crop_range_image(range_images, new_width, scope=None):
  """Crops range image by shrinking the width.

  Requires: new_width is smaller than the existing width.

  Args:
    range_images: [B, H, W, ...]
    new_width: an integer.
    scope: the name scope.

  Returns:
    range_image_crops: [B, H, new_width, ...]
  """
  # pylint: disable=unbalanced-tuple-unpacking
  shape = _combined_static_and_dynamic_shape(range_images)
  width = shape[2]
  if width == new_width:
    return range_images
  if new_width < 1:
    raise ValueError('new_width must be positive.')
  if width is not None and new_width >= width:
    raise ValueError('new_width {} should be < the old width {}.'.format(
        new_width, width))

  with tf.control_dependencies([tf.assert_less(new_width, width)]):
    with tf.name_scope(scope, 'CropRangeImage', [range_images]):
      diff = width - new_width

      left = diff // 2
      right = diff - left
      range_image_crops = range_images[:, :, left:-right, ...]
      return range_image_crops


def compute_inclination(inclination_range, height, scope=None):
  """Compute uniform inclination range based the given range and height.

  Args:
    inclination_range: [..., 2] tensor. Inner dims are [min inclination, max
      inclination].
    height: an integer indicates height of the range image.
    scope: the name scope.

  Returns:
    inclination: [..., height] tensor. Inclinations computed.
  """
  with tf.name_scope(scope, 'ComputeInclination', [inclination_range]):
    diff = inclination_range[..., 1] - inclination_range[..., 0]
    inclination = (
        (.5 + tf.cast(tf.range(0, height), dtype=inclination_range.dtype)) /
        tf.cast(height, inclination_range.dtype) *
        tf.expand_dims(diff, axis=-1) + inclination_range[..., 0:1])
    return inclination
