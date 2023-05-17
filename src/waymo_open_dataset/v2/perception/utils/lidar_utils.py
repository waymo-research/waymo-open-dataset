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
"""Utils to process LiDAR sensor data components."""

from typing import Optional, TypeVar

import tensorflow as tf

from waymo_open_dataset import dataset_pb2 as _v1_dataset_pb2
from waymo_open_dataset.utils import range_image_utils
from waymo_open_dataset.utils import transform_utils
from waymo_open_dataset.v2.perception import base as _v2_base
from waymo_open_dataset.v2.perception import context as _v2_context
from waymo_open_dataset.v2.perception import lidar as _v2_lidar
from waymo_open_dataset.v2.perception import pose as _v2_pose
from waymo_open_dataset.v2.perception import segmentation as _v2_segmentation
from waymo_open_dataset.v2.perception.compat_v1 import interfaces


def _add_batch_dim(tensor: tf.Tensor):
  return tf.expand_dims(tensor, axis=0)


def convert_lidar_pose_range_image_to_transformation(
    lidar_pose_range_image: _v2_lidar.PoseRangeImage,
) -> tf.Tensor:
  """Converts per-pixel rotation and translation to transformation matrices.

  Args:
    lidar_pose_range_image: A [H, W, 6] tensor of pose of each pixel in range
      image. Inner dimensions are [roll, pitch, yaw, x, y, z], representing a
      transform from vehicle frame to global frame for every range image pixel.
      See lidar.PoseRangeImage for more details.

  Returns:
    A [H, W, 4, 4] tensor for per-pixel transformation matrices.
  """
  ri_pose = lidar_pose_range_image.tensor
  rotation = transform_utils.get_rotation_matrix(
      ri_pose[..., 0],
      ri_pose[..., 1],
      ri_pose[..., 2],
  )
  translation = ri_pose[..., 3:]
  return transform_utils.get_transform(rotation, translation)


def convert_range_image_to_cartesian(
    range_image: _v2_lidar.RangeImage,
    calibration: _v2_context.LiDARCalibrationComponent,
    pixel_pose: Optional[_v2_lidar.PoseRangeImage] = None,
    frame_pose: Optional[_v2_pose.VehiclePoseComponent] = None,
    keep_polar_features=False,
) -> tf.Tensor:
  """Converts one range image from polar coordinates to Cartesian coordinates.

  Args:
    range_image: One range image return captured by a LiDAR sensor.
    calibration: Parameters for calibration of a LiDAR sensor.
    pixel_pose: If not none, it sets pose for each range image pixel.
    frame_pose: This must be set when `pose` is set.
    keep_polar_features: If true, keep the features from the polar range image
      (i.e. range, intensity, and elongation) as the first features in the
      output range image.

  Returns:
    A [H, W, D] image in Cartesian coordinates. D will be 3 if
      keep_polar_features is False (x, y, z) and 6 if keep_polar_features is
      True (range, intensity, elongation, x, y, z).
  """
  if pixel_pose is not None:
    assert frame_pose is not None
  range_image_tensor = range_image.tensor
  extrinsic = tf.reshape(
      tf.convert_to_tensor(calibration.extrinsic.transform),
      (4, 4)
  )

  # Compute inclinations mapping range image rows to circles in the 3D worlds.
  if calibration.beam_inclination.values is not None:
    inclination = tf.convert_to_tensor(calibration.beam_inclination.values)
  else:
    inclination = range_image_utils.compute_inclination(
        inclination_range=tf.convert_to_tensor(
            (
                calibration.beam_inclination.min,
                calibration.beam_inclination.max,
            )
        ),
        height=range_image_tensor.shape[0],
    )
  inclination = tf.reverse(inclination, axis=[-1])

  if pixel_pose is not None and frame_pose is not None:
    pixel_pose_tensor = _add_batch_dim(
        convert_lidar_pose_range_image_to_transformation(pixel_pose),
    )
    frame_pose_tensor = _add_batch_dim(
        tf.reshape(
            tf.convert_to_tensor(frame_pose.world_from_vehicle.transform),
            (4, 4)
        ),
    )
  else:
    pixel_pose_tensor = None
    frame_pose_tensor = None

  range_image_cartesian = (
      range_image_utils.extract_point_cloud_from_range_image(
          range_image=_add_batch_dim(range_image_tensor[..., 0]),
          extrinsic=_add_batch_dim(extrinsic),
          inclination=_add_batch_dim(inclination),
          pixel_pose=pixel_pose_tensor,
          frame_pose=frame_pose_tensor,
      )
  )
  range_image_cartesian = tf.squeeze(range_image_cartesian, axis=0)

  if keep_polar_features:
    range_image_cartesian = tf.concat(
        [
            range_image_tensor[..., 0:3],
            range_image_cartesian,
        ],
        axis=-1,
    )
  return range_image_cartesian


def convert_range_image_to_point_cloud(
    range_image: _v2_lidar.RangeImage,
    calibration: _v2_context.LiDARCalibrationComponent,
    pixel_pose: Optional[_v2_lidar.PoseRangeImage] = None,
    frame_pose: Optional[_v2_pose.VehiclePoseComponent] = None,
    keep_polar_features=False,
) -> tf.Tensor:
  """Converts one range image from polar coordinates to point cloud.

  Args:
    range_image: One range image return captured by a LiDAR sensor.
    calibration: Parameters for calibration of a LiDAR sensor.
    pixel_pose: If not none, it sets pose for each range image pixel.
    frame_pose: This must be set when `pose` is set.
    keep_polar_features: If true, keep the features from the polar range image
      (i.e. range, intensity, and elongation) as the first features in the
      output range image.

  Returns:
    A [N, D] tensor of 3D LiDAR points. D will be 3 if keep_polar_features is
      False (x, y, z) and 6 if keep_polar_features is True (range, intensity,
      elongation, x, y, z).
  """
  range_image_cartesian = convert_range_image_to_cartesian(
      range_image=range_image,
      calibration=calibration,
      pixel_pose=pixel_pose,
      frame_pose=frame_pose,
      keep_polar_features=keep_polar_features,
  )
  range_image_tensor = range_image.tensor
  range_image_mask = range_image_tensor[..., 0] > 0
  points_tensor = tf.gather_nd(
      range_image_cartesian,
      tf.compat.v1.where(range_image_mask)
  )
  return points_tensor


def extract_pointwise_camera_projection(
    range_image: _v2_lidar.RangeImage,
    camera_projection: _v2_lidar.CameraProjectionRangeImage,
) -> tf.Tensor:
  """Extracts information about where in camera images each point is projected.

  Args:
    range_image: One range image return captured by a LiDAR sensor.
    camera_projection: LiDAR point to camera image projections.

  Returns:
    A [N, 6] tensor of camera projection per point. See
      lidar.CameraProjectionRangeImage for definitions of inner dimensions.
  """
  range_image_tensor = range_image.tensor
  range_image_mask = range_image_tensor[..., 0] > 0
  camera_project_tensor = camera_projection.tensor
  pointwise_camera_projection_tensor = tf.gather_nd(
      camera_project_tensor,
      tf.compat.v1.where(range_image_mask)
  )
  return pointwise_camera_projection_tensor


_RangeImage = TypeVar('_RangeImage')


def parse_range_image(
    zlib_compressed_range_image: Optional[bytes],
    range_image_class: type[_RangeImage],
) -> Optional[_RangeImage]:
  """Parses a compressed range image into flattened values and shape."""
  if zlib_compressed_range_image is None or (
      len(zlib_compressed_range_image) == 0  # pylint: disable=g-explicit-length-test
  ):
    return None
  range_image_str_tensor = tf.io.decode_compressed(
      zlib_compressed_range_image, 'ZLIB'
  )
  if range_image_class == _v2_lidar.CameraProjectionRangeImage or (
      range_image_class == _v2_segmentation.LiDARSegmentationRangeImage
  ):
    range_image_matrix = _v1_dataset_pb2.MatrixInt32.FromString(
        bytearray(range_image_str_tensor.numpy())
    )
  else:
    range_image_matrix = _v1_dataset_pb2.MatrixFloat.FromString(
        bytearray(range_image_str_tensor.numpy())
    )
  range_image_value_flattened = (
      tf.convert_to_tensor(value=range_image_matrix.data).numpy().tolist()
  )
  range_image_shape = list(range_image_matrix.shape.dims)
  return range_image_class(
      values=range_image_value_flattened,
      shape=range_image_shape
  )


def get_laser_key(src: interfaces.LiDARComponentSrc) -> _v2_base.LaserKey:
  return _v2_base.LaserKey(
      segment_context_name=src.frame.context.name,
      frame_timestamp_micros=src.frame.timestamp_micros,
      laser_name=src.laser.name,
  )


def get_laser_by_name(
    frame: _v1_dataset_pb2.Frame,
    laser_name: _v2_lidar.LaserName
) -> _v1_dataset_pb2.Laser:
  for laser in frame.lasers:
    if laser.name == laser_name.value:
      return laser
  raise ValueError(f'{laser_name} not found.')
