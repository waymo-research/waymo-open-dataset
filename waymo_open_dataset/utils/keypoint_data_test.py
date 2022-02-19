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
"""Tests for keypoint_data."""
import numpy as np
import tensorflow as tf

from waymo_open_dataset import dataset_pb2
from waymo_open_dataset import label_pb2
from waymo_open_dataset.utils import keypoint_data as _lib
from waymo_open_dataset.utils import keypoint_test_util as _util

_NOSE = _lib.KeypointType.KEYPOINT_TYPE_NOSE
_LEFT_SHOULDER = _lib.KeypointType.KEYPOINT_TYPE_LEFT_SHOULDER
_RIGHT_SHOULDER = _lib.KeypointType.KEYPOINT_TYPE_RIGHT_SHOULDER


class KeypointUtilsTest(tf.test.TestCase):

  def test_group_object_labels_populates_object_type(self):
    frame = dataset_pb2.Frame(laser_labels=[
        _util.laser_object('fake', object_type=_lib.ObjectType.TYPE_PEDESTRIAN),
    ])

    objects = _lib.group_object_labels(frame)

    self.assertEqual(list(objects.keys()), ['fake'])
    self.assertEqual(objects['fake'].object_type,
                     _lib.ObjectType.TYPE_PEDESTRIAN)

  def test_group_object_labels_populates_laser_box_fields(self):
    frame = dataset_pb2.Frame(laser_labels=[
        _util.laser_object(
            'fake',
            box_fields=dict(
                center_x=1,
                center_y=2,
                center_z=3,
                width=4,
                height=5,
                length=6,
                heading=7)),
    ])

    objects = _lib.group_object_labels(frame)

    self.assertEqual(list(objects.keys()), ['fake'])
    b = objects['fake'].laser.box
    self.assertEqual(b.center_x, 1)
    self.assertEqual(b.center_y, 2)
    self.assertEqual(b.center_z, 3)
    self.assertEqual(b.width, 4)
    self.assertEqual(b.height, 5)
    self.assertEqual(b.length, 6)
    self.assertEqual(b.heading, 7)

  def test_group_object_labels_populates_laser_keypoints(self):
    frame = dataset_pb2.Frame(laser_labels=[
        _util.laser_object('fake', has_keypoints=True),
    ])

    objects = _lib.group_object_labels(frame)

    self.assertEqual(list(objects.keys()), ['fake'])
    self.assertNotEmpty(objects['fake'].laser.keypoints.keypoint)
    self.assertTrue(
        objects['fake'].laser.keypoints.keypoint[0].HasField('keypoint_3d'))

  def test_group_object_labels_associates_camera_with_laser_labels_by_id(self):
    frame = dataset_pb2.Frame(
        laser_labels=[
            _util.laser_object('fake', has_keypoints=True),
        ],
        camera_labels=[
            dataset_pb2.CameraLabels(
                name=_lib.CameraType.FRONT,
                labels=[
                    _util.camera_object(
                        camera_obj_id='vbb_fake',
                        laser_obj_id='fake',
                        box_fields=dict(
                            center_x=1,
                            center_y=2,
                            width=4,
                            length=6,
                            heading=7))
                ])
        ])

    objects = _lib.group_object_labels(frame)

    self.assertEqual(list(objects.keys()), ['fake'])
    self.assertEqual(
        list(objects['fake'].camera.keys()), [_lib.CameraType.FRONT])
    b = objects['fake'].camera[_lib.CameraType.FRONT].box
    self.assertEqual(b.center_x, 1)
    self.assertEqual(b.center_y, 2)
    self.assertEqual(b.width, 4)
    self.assertEqual(b.length, 6)
    self.assertEqual(b.heading, 7)

  def test_group_object_labels_populates_camera_keypoints(self):
    frame = dataset_pb2.Frame(
        laser_labels=[
            _util.laser_object('fake', has_keypoints=True),
        ],
        camera_labels=[
            dataset_pb2.CameraLabels(
                name=_lib.CameraType.FRONT,
                labels=[
                    _util.camera_object(
                        camera_obj_id='vbb_fake',
                        laser_obj_id='fake',
                        box_fields=dict(
                            center_x=1,
                            center_y=2,
                            width=4,
                            length=6,
                            heading=7),
                        has_keypoints=True)
                ])
        ])

    objects = _lib.group_object_labels(frame)

    self.assertEqual(list(objects.keys()), ['fake'])
    self.assertEqual(
        list(objects['fake'].camera.keys()), [_lib.CameraType.FRONT])
    camera_label = objects['fake'].camera[_lib.CameraType.FRONT]
    self.assertTrue(camera_label.keypoints.keypoint[0].HasField('keypoint_2d'))

  def test_select_subset_returns_correct_coordinates(self):
    src_order = (_LEFT_SHOULDER, _NOSE, _RIGHT_SHOULDER)
    dst_order = (_LEFT_SHOULDER, _RIGHT_SHOULDER)
    # batch_size = 2, num_points=3
    coords = tf.constant([[[-1, -1], [0, 0], [1, 1]], [[-2, -2], [0, 0], [2,
                                                                          2]]])

    subset_coords = _lib.select_subset(coords, src_order, dst_order)

    self.assertAllEqual(subset_coords, [[[-1, -1], [1, 1]], [[-2, -2], [2, 2]]])

  def test_create_tensors_from_laser_keypoints_one_missing(self):
    keypoints = [
        _util.laser_keypoint(_NOSE, location_m=(1, 2, 3), is_occluded=False),
        _util.laser_keypoint(
            _LEFT_SHOULDER, location_m=(4, 5, 6), is_occluded=False)
    ]

    tensors = _lib.create_laser_keypoints_tensors(
        keypoints,
        default_location=np.zeros(3, dtype=np.float32),
        order=[_NOSE, _LEFT_SHOULDER, _RIGHT_SHOULDER])

    self.assertAllEqual(tensors.location, [[1, 2, 3], [4, 5, 6], [0, 0, 0]])
    self.assertAllEqual(tensors.visibility, [2, 2, 0])

  def test_create_tensors_from_laser_keypoints_one_occluded(self):
    keypoints = [
        _util.laser_keypoint(
            _RIGHT_SHOULDER, location_m=(1, 2, 3), is_occluded=True),
    ]

    keypoint_tensors = _lib.create_laser_keypoints_tensors(
        keypoints,
        default_location=np.zeros(3, dtype=np.float32),
        order=[_NOSE, _LEFT_SHOULDER, _RIGHT_SHOULDER])

    self.assertAllEqual(keypoint_tensors.location,
                        [[0, 0, 0], [0, 0, 0], [1, 2, 3]])
    self.assertAllEqual(keypoint_tensors.visibility, [0, 0, 1])

  def test_create_tensors_from_laser_keypoints_all_missing(self):
    keypoints = []

    keypoint_tensors = _lib.create_laser_keypoints_tensors(
        keypoints,
        default_location=np.zeros(3, dtype=np.float32),
        order=[_NOSE, _LEFT_SHOULDER, _RIGHT_SHOULDER])

    self.assertAllEqual(keypoint_tensors.location,
                        [[0, 0, 0], [0, 0, 0], [0, 0, 0]])
    self.assertAllEqual(keypoint_tensors.visibility, [0, 0, 0])

  def test_create_tensors_from_camera_keypoints_one_missing(self):
    keypoints = [
        _util.camera_keypoint(_NOSE, location_px=(1, 2), is_occluded=False),
        _util.camera_keypoint(
            _LEFT_SHOULDER, location_px=(4, 5), is_occluded=False)
    ]

    tensors = _lib.create_camera_keypoints_tensors(
        keypoints,
        default_location=np.zeros(2, dtype=np.float32),
        order=[_NOSE, _LEFT_SHOULDER, _RIGHT_SHOULDER])

    self.assertAllEqual(tensors.location, [[1, 2], [4, 5], [0, 0]])
    self.assertAllEqual(tensors.visibility, [2, 2, 0])

  def test_create_tensors_from_camera_keypoints_one_occluded(self):
    keypoints = [
        _util.camera_keypoint(
            _RIGHT_SHOULDER, location_px=(1, 2), is_occluded=True),
    ]

    tensors = _lib.create_camera_keypoints_tensors(
        keypoints,
        default_location=np.zeros(2, dtype=np.float32),
        order=[_NOSE, _LEFT_SHOULDER, _RIGHT_SHOULDER])

    self.assertAllEqual(tensors.location, [[0, 0], [0, 0], [1, 2]])
    self.assertAllEqual(tensors.visibility, [0, 0, 1])

  def test_create_tensors_from_camera_keypoints_all_missing(self):
    keypoints = []

    keypoint_tensors = _lib.create_camera_keypoints_tensors(
        keypoints,
        default_location=np.zeros(2, dtype=np.float32),
        order=[_NOSE, _LEFT_SHOULDER, _RIGHT_SHOULDER])

    self.assertAllEqual(keypoint_tensors.location, [[0, 0], [0, 0], [0, 0]])
    self.assertAllEqual(keypoint_tensors.visibility, [0, 0, 0])

  def test_create_tensors_from_camera_keypoints_default_value_is_tensor(self):
    keypoints = []

    keypoint_tensors = _lib.create_camera_keypoints_tensors(
        keypoints,
        default_location=tf.zeros(2, dtype=np.float32),
        order=[_NOSE, _LEFT_SHOULDER, _RIGHT_SHOULDER])

    self.assertAllEqual(keypoint_tensors.location, [[0, 0], [0, 0], [0, 0]])
    self.assertAllEqual(keypoint_tensors.visibility, [0, 0, 0])

  def test_create_tensors_from_camera_boxes(self):
    box = label_pb2.Label.Box(center_x=1, center_y=2, length=3, width=4)

    tensors = _lib.create_camera_box_tensors(box)

    self.assertAllEqual(tensors.center, [1, 2])
    self.assertAllEqual(tensors.size, [3, 4])

  def test_create_tensors_from_laser_boxes(self):
    box = label_pb2.Label.Box(
        center_x=1,
        center_y=2,
        center_z=3,
        length=4,
        width=5,
        height=6,
        heading=7)

    tensors = _lib.create_laser_box_tensors(box)

    self.assertAllEqual(tensors.center, [1, 2, 3])
    self.assertAllEqual(tensors.size, [4, 5, 6])
    self.assertAllEqual(tensors.heading, 7)


if __name__ == '__main__':
  tf.test.main()
