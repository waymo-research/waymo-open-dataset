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
# =============================================================================
"""Waymo Motion Open Dataset (WOMD) utils to process LiDAR data."""

from typing import Tuple

import numpy as np
import tensorflow as tf

from waymo_open_dataset import dataset_pb2
from waymo_open_dataset.protos import compressed_lidar_pb2
from waymo_open_dataset.protos import scenario_pb2
from waymo_open_dataset.utils import range_image_utils
from waymo_open_dataset.utils import transform_utils
from waymo_open_dataset.utils.compression import delta_encoder


def _get_beam_inclinations(
    beam_inclination_min: float, beam_inclination_max: float, height: int
) -> np.ndarray:
  """Gets the beam inclination information."""
  return range_image_utils.compute_inclination(
      np.array([beam_inclination_min, beam_inclination_max]), height=height
  )


def augment_womd_scenario_with_lidar_points(
    scenario: scenario_pb2.Scenario, lidar_data: scenario_pb2.Scenario
) -> scenario_pb2.Scenario:
  """Aguments the scenario with lidar data for the first 1.1 seconds.

  Args:
    scenario: the WOMD scenario proto containing motion data.
    lidar_data: A WOMD scenario proto which only contains an non-empty
      `compressed_frame_laser_data` field. This field is merged into original
      WOMD scenario's `compressed_frame_laser_data`.

  Returns:
    scenario_augmented: the augmented WOMD scenario proto.
  """

  # Augment the original scenario.
  scenario_augmented = scenario_pb2.Scenario()
  scenario_augmented.CopyFrom(scenario)
  scenario_augmented.compressed_frame_laser_data.extend(
      lidar_data.compressed_frame_laser_data
  )
  return scenario_augmented


def extract_top_lidar_points(
    laser: compressed_lidar_pb2.CompressedLaser,
    frame_pose: tf.Tensor,
    laser_calib: dataset_pb2.LaserCalibration,
) -> Tuple[tf.Tensor, ...]:
  """Extract point clouds from the top laser proto.

  Args:
    laser: the top laser proto.
    frame_pose: a (4, 4) array which decides the vehicle frame at which the
      cartesian points are computed.
    laser_calib: calib proto of the top lidar.

  Returns:
    points_xyz_ri1: a tf.Tensor of shape [#points, 3] for xyz coordinates from
      the 1st range image.
    points_feature_ri1: a tf.Tensor of shape [#points, 3] for point cloud
      features (range, intensity and elongation) from the 1st range image.
    points_xyz_ri2: a tf.Tensor of shape [#points, 3] for xyz coordinates from
      the 2nd range image.
    points_feature_ri2: a tf.Tensor of shape [#points, 3] for point cloud
      features (range, intensity and elongation) from the 2nd range image.
  """
  # Get top pose info and lidar calibrations.
  # -------------------------------------------
  range_image_pose_decompressed = delta_encoder.decompress(
      laser.ri_return1.range_image_pose_delta_compressed
  )
  range_image_top_pose_arr = range_image_pose_decompressed
  range_image_top_pose_rotation = transform_utils.get_rotation_matrix(
      range_image_top_pose_arr[..., 0],
      range_image_top_pose_arr[..., 1],
      range_image_top_pose_arr[..., 2],
  )
  range_image_top_pose_translation = range_image_top_pose_arr[..., 3:]
  range_image_top_pose_arr = transform_utils.get_transform(
      range_image_top_pose_rotation, range_image_top_pose_translation
  )

  pixel_pose_local = range_image_top_pose_arr
  pixel_pose_local = tf.expand_dims(pixel_pose_local, axis=0)
  frame_pose_local = tf.expand_dims(frame_pose, axis=0)

  # Extract point xyz and features.
  # -------------------------------------------
  points_xyz_list = []
  points_feature_list = []
  for ri in [laser.ri_return1, laser.ri_return2]:
    range_image = delta_encoder.decompress(ri.range_image_delta_compressed)
    c = laser_calib
    if not c.beam_inclinations:
      beam_inclinations = _get_beam_inclinations(
          c.beam_inclination_min,
          c.beam_inclination_max,
          range_image.dims[0],
      )
    else:
      beam_inclinations = np.array(c.beam_inclinations)
    beam_inclinations = beam_inclinations[..., ::-1]
    extrinsic = np.reshape(np.array(c.extrinsic.transform), [4, 4])
    range_image_mask = range_image[..., 0] > 0
    range_image_cartesian = (
        range_image_utils.extract_point_cloud_from_range_image(
            tf.expand_dims(range_image[..., 0], axis=0),
            tf.expand_dims(extrinsic, axis=0),
            tf.expand_dims(beam_inclinations, axis=0),
            pixel_pose=pixel_pose_local,
            frame_pose=frame_pose_local,
        )
    )
    range_image_cartesian = tf.squeeze(range_image_cartesian, axis=0)
    # The points XYZ in the vehicle coordinate.
    points_xyz = tf.gather_nd(
        range_image_cartesian, tf.compat.v1.where(range_image_mask)
    )
    points_xyz_list.append(points_xyz)
    # The range, intensity and elongation features.
    points_feature = tf.gather_nd(
        range_image[..., 0:3], tf.compat.v1.where(range_image_mask)
    )
    points_feature_list.append(points_feature)

  return (
      points_xyz_list[0],
      points_feature_list[0],
      points_xyz_list[1],
      points_feature_list[1],
  )


def extract_side_lidar_points(
    laser: compressed_lidar_pb2.CompressedLaser,
    laser_calib: dataset_pb2.LaserCalibration,
) -> Tuple[tf.Tensor, ...]:
  """Extract side lidar points.

  Args:
    laser: the side laser proto.
    laser_calib: calib proto of the side lidar.

  Returns:
    points_xyz_ri1: a tf.Tensor of shape [#points, 3] for xyz coordinates from
      the 1st range image.
    points_feature_ri1: a tf.Tensor of shape [#points, 3] for point cloud
      features (range, intensity and elongation) from the 1st range image.
    points_xyz_ri2: a tf.Tensor of shape [#points, 3] for xyz coordinates from
      the 2nd range image.
    points_feature_ri2: a tf.Tensor of shape [#points, 3] for point cloud
      features (range, intensity and elongation) from the 2nd range image.
  """
  points_xyz_list = []
  points_feature_list = []
  for ri in [laser.ri_return1, laser.ri_return2]:
    range_image = delta_encoder.decompress(ri.range_image_delta_compressed)

    # Lidar calibration info.
    c = laser_calib
    extrinsic = np.reshape(np.array(c.extrinsic.transform), [4, 4])
    if not c.beam_inclinations:
      beam_inclinations = _get_beam_inclinations(
          c.beam_inclination_min, c.beam_inclination_max, range_image.shape[0]
      )
    else:
      beam_inclinations = np.array(c.beam_inclinations)
    beam_inclinations = beam_inclinations[..., ::-1]

    range_image_mask = range_image[..., 0] > 0
    range_image_cartesian = (
        range_image_utils.extract_point_cloud_from_range_image(
            tf.expand_dims(range_image[..., 0], axis=0),
            tf.expand_dims(extrinsic, axis=0),
            tf.expand_dims(beam_inclinations, axis=0),
            pixel_pose=None,
            frame_pose=None,
        )
    )
    range_image_cartesian = tf.squeeze(range_image_cartesian, axis=0)
    # The points XYZ in the vehicle coordinate.
    points_xyz = tf.gather_nd(
        range_image_cartesian, tf.compat.v1.where(range_image_mask)
    )
    points_xyz_list.append(points_xyz)
    # The range, intensity and elongation features.
    points_feature = tf.gather_nd(
        range_image[..., 0:3], tf.compat.v1.where(range_image_mask)
    )
    points_feature_list.append(points_feature)

  return (
      points_xyz_list[0],
      points_feature_list[0],
      points_xyz_list[1],
      points_feature_list[1],
  )
