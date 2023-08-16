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
"""Classes which form a public API of the V2 dataset."""

import pathlib as _pathlib

from waymo_open_dataset.v2 import column_types as _column_types
from waymo_open_dataset.v2 import component as _component
from waymo_open_dataset.v2 import dataframe_utils as _dataframe_utils
from waymo_open_dataset.v2.perception import box as _box
from waymo_open_dataset.v2.perception import camera_image as _camera_image
from waymo_open_dataset.v2.perception import context as _context
from waymo_open_dataset.v2.perception import keypoints as _keypoints
from waymo_open_dataset.v2.perception import lidar as _lidar
from waymo_open_dataset.v2.perception import object_asset
from waymo_open_dataset.v2.perception import pose as _pose
from waymo_open_dataset.v2.perception import segmentation as _segmentation
from waymo_open_dataset.v2.perception.utils import lidar_utils as _lidar_utils
from waymo_open_dataset.v2.perception.utils import object_asset_codec
from waymo_open_dataset.v2.perception.utils import object_asset_utils


# Supported components.
CameraBoxComponent = _box.CameraBoxComponent
CameraCalibrationComponent = _context.CameraCalibrationComponent
CameraHumanKeypointsComponent = _keypoints.CameraHumanKeypointsComponent
CameraImageComponent = _camera_image.CameraImageComponent
CameraSegmentationLabelComponent = (
    _segmentation.CameraSegmentationLabelComponent
)
CameraToLiDARBoxAssociationComponent = _box.CameraToLiDARBoxAssociationComponent
LiDARBoxComponent = _box.LiDARBoxComponent
LiDARCalibrationComponent = _context.LiDARCalibrationComponent
LiDARCameraProjectionComponent = _lidar.LiDARCameraProjectionComponent
LiDARCameraSyncedBoxComponent = _box.LiDARCameraSyncedBoxComponent
LiDARComponent = _lidar.LiDARComponent
LiDARHumanKeypointsComponent = _keypoints.LiDARHumanKeypointsComponent
LiDARPoseComponent = _lidar.LiDARPoseComponent
LiDARSegmentationLabelComponent = _segmentation.LiDARSegmentationLabelComponent
ProjectedLiDARBoxComponent = _box.ProjectedLiDARBoxComponent
StatsComponent = _context.StatsComponent
VehiclePoseComponent = _pose.VehiclePoseComponent
ObjectAssetAutoLabelComponent = object_asset.ObjectAssetAutoLabelComponent
ObjectAssetCameraSensorComponent = object_asset.ObjectAssetCameraSensorComponent
ObjectAssetLiDARSensorComponent = object_asset.ObjectAssetLiDARSensorComponent
ObjectAssetRefinedPoseComponent = object_asset.ObjectAssetRefinedPoseComponent
ObjectAssetRayComponent = object_asset.ObjectAssetRayComponent
ObjectAssetRayCompressedComponent = (
    object_asset.ObjectAssetRayCompressedComponent
)
ObjectAssetRayCodec = object_asset_codec.ObjectAssetRayCodec
ObjectAssetRayCodecConfig = object_asset_codec.ObjectAssetRayCodecConfig


# Types and classes
Vec2s = _column_types.Vec2s
Vec2d = _column_types.Vec2d
Vec3s = _column_types.Vec3s
Vec3d = _column_types.Vec3d
Vec2sList = _column_types.Vec2sList
Vec2dList = _column_types.Vec2dList
Vec3sList = _column_types.Vec3sList
Vec3dList = _column_types.Vec3dList

# Tools
merge = _dataframe_utils.merge
convert_range_image_to_point_cloud = (
    _lidar_utils.convert_range_image_to_point_cloud
)

# Canonical string tags for all components.
TAG_BY_COMPONENT = {
    CameraBoxComponent: 'camera_box',
    CameraCalibrationComponent: 'camera_calibration',
    CameraHumanKeypointsComponent: 'camera_hkp',
    CameraImageComponent: 'camera_image',
    CameraSegmentationLabelComponent: 'camera_segmentation',
    CameraToLiDARBoxAssociationComponent: 'camera_to_lidar_box_association',
    LiDARBoxComponent: 'lidar_box',
    LiDARCalibrationComponent: 'lidar_calibration',
    LiDARCameraProjectionComponent: 'lidar_camera_projection',
    LiDARCameraSyncedBoxComponent: 'lidar_camera_synced_box',
    LiDARComponent: 'lidar',
    LiDARHumanKeypointsComponent: 'lidar_hkp',
    LiDARPoseComponent: 'lidar_pose',
    LiDARSegmentationLabelComponent: 'lidar_segmentation',
    ProjectedLiDARBoxComponent: 'projected_lidar_box',
    StatsComponent: 'stats',
    VehiclePoseComponent: 'vehicle_pose',
    ObjectAssetAutoLabelComponent: 'object_asset_auto_label',
    ObjectAssetCameraSensorComponent: 'object_asset_camera_sensor',
    ObjectAssetLiDARSensorComponent: 'object_asset_lidar_sensor',
    ObjectAssetRefinedPoseComponent: 'object_asset_refined_pose',
    ObjectAssetRayComponent: 'object_asset_ray',
    ObjectAssetRayCompressedComponent: 'object_asset_ray_compressed',
}

ALL_COMPONENTS = list(TAG_BY_COMPONENT.keys())
ALL_TAGS = list(TAG_BY_COMPONENT.values())
