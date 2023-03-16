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
"""Tests for waymo_open_dataset/v2/perception/compat_v1/keypoints.py."""
from absl.testing import absltest

from waymo_open_dataset import dataset_pb2
from waymo_open_dataset.v2.perception.compat_v1 import interfaces
from waymo_open_dataset.v2.perception.compat_v1 import keypoints


class CameraHumanKeypointsExtractorTest(absltest.TestCase):

  def test_populates_2d_keypoints(self):
    # We populate only fields, which are required/used by the extractor.
    frame = dataset_pb2.Frame(
        context={'name': 'dummy_segment'},
        timestamp_micros=123456,
        camera_labels=[{
            'name': 'FRONT',
            'labels': [{
                'id': 'dummy_id',
                'camera_keypoints': {
                    'keypoint': [
                        {
                            'type': 'KEYPOINT_TYPE_NOSE',
                            'keypoint_2d': {
                                'location_px': {'x': 1, 'y': 2},
                                'visibility': {'is_occluded': False},
                            },
                        },
                        {
                            'type': 'KEYPOINT_TYPE_HEAD_CENTER',
                            'keypoint_2d': {
                                'location_px': {'x': 3, 'y': 4},
                                'visibility': {'is_occluded': True},
                            },
                        },
                    ]
                },
            }],
        }],
    )
    src = interfaces.CameraLabelComponentSrc(
        frame=frame,
        camera_labels=frame.camera_labels[0],
        label=frame.camera_labels[0].labels[0],
    )

    extractor = keypoints.CameraHumanKeypointsFrameExtractor()
    component = next(extractor(src))

    self.assertEqual(component.key.segment_context_name, 'dummy_segment')
    self.assertEqual(component.key.frame_timestamp_micros, 123456)
    self.assertEqual(component.key.camera_name, 1)  # FRONT
    self.assertEqual(component.key.camera_object_id, 'dummy_id')
    self.assertEqual(component.camera_keypoints.type, [1, 20])
    self.assertEqual(
        component.camera_keypoints.keypoint_2d.location_px.x, [1, 3]
    )
    self.assertEqual(
        component.camera_keypoints.keypoint_2d.location_px.y, [2, 4]
    )
    self.assertEqual(
        component.camera_keypoints.keypoint_2d.visibility.is_occluded,
        [False, True],
    )

  def test_returns_none_if_there_is_no_data(self):
    # No camera_keypoints inside.
    frame = dataset_pb2.Frame(
        context={'name': 'dummy_segment'},
        timestamp_micros=123456,
        camera_labels=[{
            'name': 'FRONT',
            'labels': [
                {
                    'id': 'dummy_id',
                }
            ],
        }],
    )
    src = interfaces.CameraLabelComponentSrc(
        frame=frame,
        camera_labels=frame.camera_labels[0],
        label=frame.camera_labels[0].labels[0],
    )

    extractor = keypoints.CameraHumanKeypointsFrameExtractor()
    components = list(extractor(src))

    self.assertEmpty(components)

  def test_raises_an_exception_if_required_fields_are_missing(self):
    # Missing fields field.
    frame = dataset_pb2.Frame(
        context={'name': 'dummy_segment'},
        timestamp_micros=123456,
        camera_labels=[{
            'name': 'FRONT',
            'labels': [{
                'id': 'dummy_id',
                'camera_keypoints': {
                    'keypoint': [
                        {
                            'type': 'KEYPOINT_TYPE_NOSE',
                        },
                        {
                            'type': 'KEYPOINT_TYPE_HEAD_CENTER',
                            'keypoint_2d': {
                                'location_px': {'x': 3, 'y': 4},
                            },
                        },
                    ]
                },
            }],
        }],
    )
    src = interfaces.CameraLabelComponentSrc(
        frame=frame,
        camera_labels=frame.camera_labels[0],
        label=frame.camera_labels[0].labels[0],
    )

    extractor = keypoints.CameraHumanKeypointsFrameExtractor()
    with self.assertRaisesRegex(interfaces.ExtractionError, 'missing'):
      _ = list(extractor(src))


class LiDARHumanKeypointsExtractorTest(absltest.TestCase):

  def test_populates_3d_keypoints(self):
    frame = dataset_pb2.Frame(
        context={'name': 'dummy_segment'},
        timestamp_micros=123456,
        laser_labels=[
            {
                'id': 'dummy_id',
                'laser_keypoints': {
                    'keypoint': [
                        {
                            'type': 'KEYPOINT_TYPE_LEFT_ANKLE',
                            'keypoint_3d': {
                                'location_m': {'x': 1, 'y': 2, 'z': 3},
                                'visibility': {'is_occluded': False},
                            },
                        },
                        {
                            'type': 'KEYPOINT_TYPE_RIGHT_ANKLE',
                            'keypoint_3d': {
                                'location_m': {'x': 4, 'y': 5, 'z': 6},
                                'visibility': {'is_occluded': True},
                            }
                        },
                    ]
                },
            },
        ],
    )
    src = interfaces.LiDARLabelComponentSrc(
        frame=frame,
        lidar_label=frame.laser_labels[0],
    )

    extractor = keypoints.LiDARHumanKeypointsFrameExtractor()
    component = next(extractor(src))

    self.assertEqual(component.key.segment_context_name, 'dummy_segment')
    self.assertEqual(component.key.frame_timestamp_micros, 123456)
    self.assertEqual(component.key.laser_object_id, 'dummy_id')
    self.assertEqual(component.lidar_keypoints.type, [10, 18])
    self.assertEqual(component.lidar_keypoints.keypoint_3d.location_m.x, [1, 4])
    self.assertEqual(component.lidar_keypoints.keypoint_3d.location_m.y, [2, 5])
    self.assertEqual(component.lidar_keypoints.keypoint_3d.location_m.z, [3, 6])
    self.assertEqual(
        component.lidar_keypoints.keypoint_3d.visibility.is_occluded,
        [False, True]
    )

  def test_returns_none_if_there_is_no_data(self):
    frame = dataset_pb2.Frame(
        context={'name': 'dummy_segment'},
        timestamp_micros=123456,
        laser_labels=[
            {
                'id': 'dummy_id',
            },
        ],
    )
    src = interfaces.LiDARLabelComponentSrc(
        frame=frame,
        lidar_label=frame.laser_labels[0],
    )

    extractor = keypoints.LiDARHumanKeypointsFrameExtractor()
    components = list(extractor(src))

    self.assertEmpty(components)

  def test_raises_an_exception_if_required_fields_are_missing(self):
    frame = dataset_pb2.Frame(
        context={'name': 'dummy_segment'},
        timestamp_micros=123456,
        laser_labels=[
            {
                'id': 'dummy_id',
                'laser_keypoints': {
                    'keypoint': [
                        {
                            'type': 'KEYPOINT_TYPE_LEFT_ANKLE',
                        },
                        {
                            'type': 'KEYPOINT_TYPE_RIGHT_ANKLE',
                            'keypoint_3d': {
                                'location_m': {'x': 4, 'y': 5, 'z': 6},
                                'visibility': {'is_occluded': True},
                            }
                        },
                    ]
                },
            },
        ],
    )
    src = interfaces.LiDARLabelComponentSrc(
        frame=frame,
        lidar_label=frame.laser_labels[0],
    )

    extractor = keypoints.LiDARHumanKeypointsFrameExtractor()
    with self.assertRaisesRegex(interfaces.ExtractionError, 'missing'):
      _ = list(extractor(src))


if __name__ == '__main__':
  absltest.main()
