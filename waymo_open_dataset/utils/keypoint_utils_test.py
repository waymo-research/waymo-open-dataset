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
"""Tests for keypoint_utils."""

from matplotlib import collections as mc
from matplotlib import pyplot as plt
import numpy as np
import plotly.graph_objects as go
import tensorflow as tf

from waymo_open_dataset import dataset_pb2
from waymo_open_dataset import label_pb2
from waymo_open_dataset.protos import keypoint_pb2
from waymo_open_dataset.utils import keypoint_utils as _lib

_NOSE = keypoint_pb2.KeypointType.KEYPOINT_TYPE_NOSE
_LEFT_SHOULDER = keypoint_pb2.KeypointType.KEYPOINT_TYPE_LEFT_SHOULDER
_RIGHT_SHOULDER = keypoint_pb2.KeypointType.KEYPOINT_TYPE_RIGHT_SHOULDER


def _laser_keypoint(point_type=_NOSE, location_m=(0, 0, 0), is_occluded=False):
  x, y, z = location_m
  return keypoint_pb2.LaserKeypoint(
      type=point_type,
      keypoint_3d=keypoint_pb2.Keypoint3d(
          location_m=keypoint_pb2.Vec3d(x=x, y=y, z=z),
          visibility=keypoint_pb2.KeypointVisibility(is_occluded=is_occluded)))


def _laser_object(obj_id,
                  box_fields=None,
                  has_keypoints=False,
                  object_type=_lib.ObjectType.TYPE_UNKNOWN):
  if has_keypoints:
    # Populate a single keypoint for testing purposes.
    laser_keypoints = keypoint_pb2.LaserKeypoints(keypoint=[_laser_keypoint()])
  else:
    laser_keypoints = None
  if box_fields is None:
    box = None
  else:
    box = label_pb2.Label.Box(**box_fields)
  return label_pb2.Label(
      id=obj_id, box=box, laser_keypoints=laser_keypoints, type=object_type)


def _camera_keypoint(point_type=_NOSE, location_px=(0, 0), is_occluded=False):
  x, y = location_px
  return keypoint_pb2.CameraKeypoint(
      type=point_type,
      keypoint_2d=keypoint_pb2.Keypoint2d(
          location_px=keypoint_pb2.Vec2d(x=x, y=y),
          visibility=keypoint_pb2.KeypointVisibility(is_occluded=is_occluded)))


def _camera_object(camera_obj_id,
                   laser_obj_id,
                   box_fields=None,
                   has_keypoints=False):
  if has_keypoints:
    # Populate a single keypoint for testing purposes.
    camera_keypoints = keypoint_pb2.CameraKeypoints(
        keypoint=[_camera_keypoint()])
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


def _get_collection(ax, col_type):
  cols = [c for c in ax.collections if isinstance(c, col_type)]
  assert len(cols) == 1, f'Required a single {col_type}, got {cols}'
  return cols[0]


def _vec2d_as_array(location_px: keypoint_pb2.Vec2d) -> np.ndarray:
  return np.array([location_px.x, location_px.y])


class EdgeTest(tf.test.TestCase):

  def test_simple_edge_between_two_points_creates_single_line(self):
    coords = {
        _LEFT_SHOULDER: _lib.Point([0, 0]),
        _RIGHT_SHOULDER: _lib.Point([1, 1])
    }
    colors = {_LEFT_SHOULDER: '#AAAAAA', _RIGHT_SHOULDER: '#BBBBBB'}

    edge = _lib.SolidLineEdge(_LEFT_SHOULDER, _RIGHT_SHOULDER, 1)
    lines = edge.create_lines(coords, colors)

    self.assertLen(lines, 1)
    # Line has the same color as the start point.
    self.assertEqual(lines[0].color, '#AAAAAA')

  def test_bicolored_edge_between_two_points_creates_two_lines(self):
    coords = {
        _LEFT_SHOULDER: _lib.Point([0, 0]),
        _RIGHT_SHOULDER: _lib.Point([1, 1])
    }
    colors = {_LEFT_SHOULDER: '#AAAAAA', _RIGHT_SHOULDER: '#BBBBBB'}

    edge = _lib.BicoloredEdge(_LEFT_SHOULDER, _RIGHT_SHOULDER, 1)
    lines = edge.create_lines(coords, colors)

    self.assertLen(lines, 2)
    self.assertEqual(lines[0].color, '#AAAAAA')
    self.assertEqual(lines[1].color, '#BBBBBB')

  def test_multipoint_edge_averages_locations_to_get_ends(self):
    coords = {
        _LEFT_SHOULDER: _lib.Point([2, 0]),
        _RIGHT_SHOULDER: _lib.Point([0, 2]),
        _NOSE: _lib.Point([3, 3])
    }
    colors = {
        _LEFT_SHOULDER: '#AAAAAA',
        _RIGHT_SHOULDER: '#BBBBBB',
        _NOSE: '#CCCCCC'
    }

    edge = _lib.MultipointEdge([_LEFT_SHOULDER, _RIGHT_SHOULDER], [_NOSE], 1)
    lines = edge.create_lines(coords, colors)

    self.assertLen(lines, 2)
    np.testing.assert_allclose(lines[0].start, [1, 1])
    np.testing.assert_allclose(lines[0].end, [2, 2])
    self.assertEqual(lines[0].color, _lib.DEFAULT_COLOR)
    np.testing.assert_allclose(lines[1].start, [2, 2])
    np.testing.assert_allclose(lines[1].end, [3, 3])
    self.assertEqual(lines[1].color, '#CCCCCC')


class KeypointUtilsTest(tf.test.TestCase):

  def test_group_object_labels_populates_object_type(self):
    frame = dataset_pb2.Frame(laser_labels=[
        _laser_object('fake', object_type=_lib.ObjectType.TYPE_PEDESTRIAN),
    ])

    objects = _lib.group_object_labels(frame)

    self.assertEqual(list(objects.keys()), ['fake'])
    self.assertEqual(objects['fake'].object_type,
                     _lib.ObjectType.TYPE_PEDESTRIAN)

  def test_group_object_labels_populates_laser_box_fields(self):
    frame = dataset_pb2.Frame(laser_labels=[
        _laser_object(
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
        _laser_object('fake', has_keypoints=True),
    ])

    objects = _lib.group_object_labels(frame)

    self.assertEqual(list(objects.keys()), ['fake'])
    self.assertNotEmpty(objects['fake'].laser.keypoints.keypoint)
    self.assertTrue(
        objects['fake'].laser.keypoints.keypoint[0].HasField('keypoint_3d'))

  def test_group_object_labels_associates_camera_with_laser_labels_by_id(self):
    frame = dataset_pb2.Frame(
        laser_labels=[
            _laser_object('fake', has_keypoints=True),
        ],
        camera_labels=[
            dataset_pb2.CameraLabels(
                name=_lib.CameraType.FRONT,
                labels=[
                    _camera_object(
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
            _laser_object('fake', has_keypoints=True),
        ],
        camera_labels=[
            dataset_pb2.CameraLabels(
                name=_lib.CameraType.FRONT,
                labels=[
                    _camera_object(
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

  def test_build_camera_wireframe_dot_and_lines_have_correct_attributes(self):
    camera_keypoints = [
        _camera_keypoint(_NOSE, location_px=(2, 1), is_occluded=True),
        _camera_keypoint(_LEFT_SHOULDER, location_px=(1, 0)),
        _camera_keypoint(_RIGHT_SHOULDER, location_px=(3, 0))
    ]
    config = _lib.WireframeConfig(
        edges=[_lib.BicoloredEdge(_LEFT_SHOULDER, _RIGHT_SHOULDER, width=2)],
        point_colors={
            _NOSE: '#FFFFFF',
            _LEFT_SHOULDER: '#AAAAAA',
            _RIGHT_SHOULDER: '#BBBBBB'
        },
        point_sizes={
            _NOSE: 1,
            _LEFT_SHOULDER: 2,
            _RIGHT_SHOULDER: 3
        })

    wireframe = _lib.build_camera_wireframe(camera_keypoints, config)

    with self.subTest(name='CorrectNumberOfLines'):
      # Two lines per edge, e.g. 2 for the edge between shoulders.
      self.assertLen(wireframe.lines, 2)
    with self.subTest(name='PopulatesLineColors'):
      self.assertEqual([l.color for l in wireframe.lines],
                       ['#AAAAAA', '#BBBBBB'])
    with self.subTest(name='PopulatesLineWidth'):
      self.assertEqual([l.width for l in wireframe.lines], [2, 2])
    with self.subTest(name='CorrectNumberOfDots'):
      # One dot per keypoint.
      self.assertLen(wireframe.dots, 3)
    with self.subTest(name='PopulatesDotColors'):
      self.assertEqual([d.color for d in wireframe.dots],
                       ['#FFFFFF', '#AAAAAA', '#BBBBBB'])
    with self.subTest(name='PopulatesDotSizes'):
      self.assertEqual([d.size for d in wireframe.dots], [1, 2, 3])
    with self.subTest(name='PopulatesBoarderColorForOccludedPoints'):
      self.assertEqual([d.border_color for d in wireframe.dots],
                       [_lib.OCCLUDED_COLOR, None, None])
      self.assertEqual([d.actual_border_color for d in wireframe.dots],
                       [_lib.OCCLUDED_COLOR, '#AAAAAA', '#BBBBBB'])

  def test_build_laser_wireframe_dot_and_lines_have_correct_attributes(self):
    laser_keypoints = [
        _laser_keypoint(_NOSE, location_m=(2, 1, 1), is_occluded=True),
        _laser_keypoint(_LEFT_SHOULDER, location_m=(1, 0, 1)),
        _laser_keypoint(_RIGHT_SHOULDER, location_m=(3, 0, 1))
    ]
    config = _lib.WireframeConfig(
        edges=[_lib.BicoloredEdge(_LEFT_SHOULDER, _RIGHT_SHOULDER, width=2)],
        point_colors={
            _NOSE: '#FFFFFF',
            _LEFT_SHOULDER: '#AAAAAA',
            _RIGHT_SHOULDER: '#BBBBBB'
        },
        point_sizes={
            _NOSE: 1,
            _LEFT_SHOULDER: 2,
            _RIGHT_SHOULDER: 3
        })

    wireframe = _lib.build_laser_wireframe(laser_keypoints, config)

    with self.subTest(name='CorrectNumberOfLines'):
      # Two lines per edge, e.g. 2 for the edge between shoulders.
      self.assertLen(wireframe.lines, 2)
    with self.subTest(name='PopulatesLineColors'):
      self.assertEqual([l.color for l in wireframe.lines],
                       ['#AAAAAA', '#BBBBBB'])
    with self.subTest(name='PopulatesLineWidth'):
      self.assertEqual([l.width for l in wireframe.lines], [2, 2])
    with self.subTest(name='CorrectNumberOfDots'):
      # One dot per keypoint.
      self.assertLen(wireframe.dots, 3)
    with self.subTest(name='PopulatesDotColors'):
      self.assertEqual([d.color for d in wireframe.dots],
                       ['#FFFFFF', '#AAAAAA', '#BBBBBB'])
    with self.subTest(name='PopulatesDotSizes'):
      self.assertEqual([d.size for d in wireframe.dots], [1, 2, 3])
    with self.subTest(name='PopulatesDotNames'):
      self.assertEqual([d.name for d in wireframe.dots],
                       ['NOSE?', 'LEFT_SHOULDER', 'RIGHT_SHOULDER'])
    with self.subTest(name='PopulatesBoarderColorForOccludedPoints'):
      self.assertEqual([d.border_color for d in wireframe.dots],
                       [_lib.OCCLUDED_COLOR, None, None])
      self.assertEqual([d.actual_border_color for d in wireframe.dots],
                       [_lib.OCCLUDED_COLOR, '#AAAAAA', '#BBBBBB'])

  def test_build_camera_wireframe_factors_modify_sizes(self):
    camera_keypoints = [
        _camera_keypoint(_NOSE, location_px=(2, 1)),
        _camera_keypoint(_LEFT_SHOULDER, location_px=(1, 0)),
        _camera_keypoint(_RIGHT_SHOULDER, location_px=(3, 0))
    ]
    config = _lib.WireframeConfig(
        edges=[_lib.BicoloredEdge(_LEFT_SHOULDER, _RIGHT_SHOULDER, width=2)],
        point_colors={
            _NOSE: '#FFFFFF',
            _LEFT_SHOULDER: '#AAAAAA',
            _RIGHT_SHOULDER: '#BBBBBB'
        },
        point_sizes={
            _NOSE: 1,
            _LEFT_SHOULDER: 2,
            _RIGHT_SHOULDER: 3
        },
        dot_size_factor=2.0,
        line_width_factor=3.0)

    wireframe = _lib.build_camera_wireframe(camera_keypoints, config)

    with self.subTest(name='ModfiesLineWidth'):
      self.assertEqual([l.width for l in wireframe.lines], [6, 6])
    with self.subTest(name='ModifiesDotSizes'):
      self.assertEqual([l.size for l in wireframe.dots], [2, 4, 6])

  def test_draw_camera_wireframe_adds_collections_to_the_axis(self):
    wireframe = _lib.Wireframe(
        lines=[
            _lib.Line(start=[0, 0], end=[1, 1], color='#AAAAAA', width=1),
            _lib.Line(start=[1, 1], end=[2, 2], color='#BBBBBB', width=1),
        ],
        dots=[
            _lib.Dot(location=[0, 0], color='#FFFFFF', size=1, name='A'),
            _lib.Dot(location=[1, 1], color='#FFFFFF', size=2, name='B'),
            _lib.Dot(location=[2, 2], color='#FFFFFF', size=3, name='C')
        ])

    _, ax = plt.subplots()
    _lib.draw_camera_wireframe(ax, wireframe)

    with self.subTest(name='CorrectNumberOfLineSegments'):
      line_collection = _get_collection(ax, mc.LineCollection)
      self.assertLen(line_collection.get_segments(), 2)
    with self.subTest(name='CorrectNumberOfDots'):
      el_collection = _get_collection(ax, mc.EllipseCollection)
      self.assertEqual(el_collection.get_offsets().shape, (3, 2))

  def test_draw_laser_wireframe_adds_traces_to_the_figure(self):
    wireframe = _lib.Wireframe(
        lines=[
            _lib.Line(start=[0, 0, 0], end=[1, 1, 1], color='#AAAAAA', width=1),
            _lib.Line(start=[1, 1, 1], end=[2, 2, 2], color='#BBBBBB', width=1),
        ],
        dots=[
            _lib.Dot(location=[0, 0, 0], color='#FFFFFF', size=1, name='A'),
            _lib.Dot(location=[1, 1, 1], color='#FFFFFF', size=2, name='B'),
            _lib.Dot(location=[2, 2, 2], color='#FFFFFF', size=3, name='C')
        ])

    fig = go.Figure()
    _lib.draw_laser_wireframe(fig, wireframe)

    with self.subTest(name='CorrectNumberOfLineSegments'):
      line_traces = list(fig.select_traces({'mode': 'lines'}))
      self.assertLen(line_traces, 2)
    with self.subTest(name='CorrectNumberOfDots'):
      dot_traces = list(fig.select_traces({'mode': 'markers'}))
      self.assertLen(dot_traces, 1)
      self.assertLen(dot_traces[0].x, 3)

  def test_crop_camera_keypoints_returns_updated_keypoints_and_image(self):
    image = np.zeros((450, 250, 3))
    camera_keypoints = [
        _camera_keypoint(location_px=(100, 200)),
        _camera_keypoint(location_px=(150, 200)),
    ]
    # NOTE: Box.length is along OX, Box.width is along OY
    box = label_pb2.Label.Box(center_x=100, center_y=200, length=100, width=200)

    new_image, new_keypoints = _lib.crop_camera_keypoints(
        image, camera_keypoints, box)

    with self.subTest(name='CroppedImageSizeIsCorrect'):
      self.assertEqual(new_image.shape, (200, 100, 3))
    with self.subTest(name='CroppedKeypointsShifted'):
      np.testing.assert_allclose(
          _vec2d_as_array(new_keypoints[0].keypoint_2d.location_px), [50, 100])
      np.testing.assert_allclose(
          _vec2d_as_array(new_keypoints[1].keypoint_2d.location_px), [100, 100])


if __name__ == '__main__':
  tf.test.main()
