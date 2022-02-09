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
# ==============================================================================
"""Convenience functions for unit tests for human keypoints."""
from typing import Any, Dict, Tuple, Optional

from waymo_open_dataset import label_pb2
from waymo_open_dataset.protos import keypoint_pb2

KeypointType = keypoint_pb2.KeypointType
ObjectType = label_pb2.Label.Type


def laser_keypoint(point_type: KeypointType = KeypointType.KEYPOINT_TYPE_NOSE,
                   location_m: Tuple[float, float, float] = (0, 0, 0),
                   is_occluded: bool = False) -> keypoint_pb2.LaserKeypoint:
  """Creates a single laser keypoint."""
  x, y, z = location_m
  return keypoint_pb2.LaserKeypoint(
      type=point_type,
      keypoint_3d=keypoint_pb2.Keypoint3d(
          location_m=keypoint_pb2.Vec3d(x=x, y=y, z=z),
          visibility=keypoint_pb2.KeypointVisibility(is_occluded=is_occluded)))


def laser_object(
    obj_id: str,
    box_fields: Optional[Dict[str, Any]] = None,
    has_keypoints: bool = False,
    object_type: ObjectType = ObjectType.TYPE_UNKNOWN) -> label_pb2.Label:
  """Creates a laser object."""
  if has_keypoints:
    # Populate a single keypoint for testing purposes.
    laser_keypoints = keypoint_pb2.LaserKeypoints(keypoint=[laser_keypoint()])
  else:
    laser_keypoints = None
  if box_fields is None:
    box = None
  else:
    box = label_pb2.Label.Box(**box_fields)
  return label_pb2.Label(
      id=obj_id, box=box, laser_keypoints=laser_keypoints, type=object_type)


def camera_keypoint(point_type: KeypointType = KeypointType.KEYPOINT_TYPE_NOSE,
                    location_px: Tuple[float, float] = (0, 0),
                    is_occluded: bool = False) -> keypoint_pb2.CameraKeypoint:
  """Creates a single camera keypoint."""
  x, y = location_px
  return keypoint_pb2.CameraKeypoint(
      type=point_type,
      keypoint_2d=keypoint_pb2.Keypoint2d(
          location_px=keypoint_pb2.Vec2d(x=x, y=y),
          visibility=keypoint_pb2.KeypointVisibility(is_occluded=is_occluded)))


def camera_object(camera_obj_id: str,
                  laser_obj_id: str,
                  box_fields: Optional[Dict[str, Any]] = None,
                  has_keypoints: bool = False) -> label_pb2.Label:
  """Creates a camera object."""
  if has_keypoints:
    # Populate a single keypoint for testing purposes.
    camera_keypoints = keypoint_pb2.CameraKeypoints(
        keypoint=[camera_keypoint()])
  else:
    camera_keypoints = None
  if box_fields is None:
    box = None
  else:
    box = label_pb2.Label.Box(**box_fields)
  return label_pb2.Label(
      id=camera_obj_id,
      box=box,
      association=label_pb2.Label.Association(laser_object_id=laser_obj_id),
      camera_keypoints=camera_keypoints)
