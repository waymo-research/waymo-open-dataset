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
"""Utilities for reading WOD motion data and rendering topdown grids."""

import dataclasses
from typing import Dict, Optional

import tensorflow as tf

from waymo_open_dataset.protos import scenario_pb2

_ObjectType = scenario_pb2.Track.ObjectType
ALL_AGENT_TYPES = [
    _ObjectType.TYPE_VEHICLE,
    _ObjectType.TYPE_PEDESTRIAN,
    _ObjectType.TYPE_CYCLIST,
]


# Holds occupancy or flow tensors for different agent classes.  This same data
# structure is used to store topdown tensors rendered from input data as well
# as topdown tensors predicted by a model.
@dataclasses.dataclass
class AgentGrids:
  """Contains any topdown render for vehicles and pedestrians."""
  vehicles: Optional[tf.Tensor] = None
  pedestrians: Optional[tf.Tensor] = None
  cyclists: Optional[tf.Tensor] = None

  def view(self, agent_type: str) -> tf.Tensor:
    """Retrieve topdown tensor for given agent type."""
    if agent_type == _ObjectType.TYPE_VEHICLE:
      return self.vehicles
    elif agent_type == _ObjectType.TYPE_PEDESTRIAN:
      return self.pedestrians
    elif agent_type == _ObjectType.TYPE_CYCLIST:
      return self.cyclists
    else:
      raise ValueError(f'Unknown agent type:{agent_type}.')


def parse_tf_example(tf_example: tf.Tensor) -> Dict[str, tf.Tensor]:
  """Parses a tf.Example into a dict of tensors."""
  features_description = {
      # Scenario ID.
      'scenario/id':
          tf.io.FixedLenFeature([], tf.string, default_value=None),

      # Roadgraph features.
      'roadgraph_samples/dir':
          tf.io.FixedLenFeature([20000, 3], tf.float32, default_value=None),
      'roadgraph_samples/id':
          tf.io.FixedLenFeature([20000, 1], tf.int64, default_value=None),
      'roadgraph_samples/type':
          tf.io.FixedLenFeature([20000, 1], tf.int64, default_value=None),
      'roadgraph_samples/valid':
          tf.io.FixedLenFeature([20000, 1], tf.int64, default_value=None),
      'roadgraph_samples/xyz':
          tf.io.FixedLenFeature([20000, 3], tf.float32, default_value=None),

      # Agent features.
      'state/id':
          tf.io.FixedLenFeature([128], tf.float32, default_value=None),
      'state/type':
          tf.io.FixedLenFeature([128], tf.float32, default_value=None),
      'state/is_sdc':
          tf.io.FixedLenFeature([128], tf.int64, default_value=None),
      'state/tracks_to_predict':
          tf.io.FixedLenFeature([128], tf.int64, default_value=None),
      'state/current/bbox_yaw':
          tf.io.FixedLenFeature([128, 1], tf.float32, default_value=None),
      'state/current/height':
          tf.io.FixedLenFeature([128, 1], tf.float32, default_value=None),
      'state/current/length':
          tf.io.FixedLenFeature([128, 1], tf.float32, default_value=None),
      'state/current/timestamp_micros':
          tf.io.FixedLenFeature([128, 1], tf.int64, default_value=None),
      'state/current/valid':
          tf.io.FixedLenFeature([128, 1], tf.int64, default_value=None),
      'state/current/vel_yaw':
          tf.io.FixedLenFeature([128, 1], tf.float32, default_value=None),
      'state/current/velocity_x':
          tf.io.FixedLenFeature([128, 1], tf.float32, default_value=None),
      'state/current/velocity_y':
          tf.io.FixedLenFeature([128, 1], tf.float32, default_value=None),
      'state/current/width':
          tf.io.FixedLenFeature([128, 1], tf.float32, default_value=None),
      'state/current/x':
          tf.io.FixedLenFeature([128, 1], tf.float32, default_value=None),
      'state/current/y':
          tf.io.FixedLenFeature([128, 1], tf.float32, default_value=None),
      'state/current/z':
          tf.io.FixedLenFeature([128, 1], tf.float32, default_value=None),
      'state/future/bbox_yaw':
          tf.io.FixedLenFeature([128, 80], tf.float32, default_value=None),
      'state/future/height':
          tf.io.FixedLenFeature([128, 80], tf.float32, default_value=None),
      'state/future/length':
          tf.io.FixedLenFeature([128, 80], tf.float32, default_value=None),
      'state/future/timestamp_micros':
          tf.io.FixedLenFeature([128, 80], tf.int64, default_value=None),
      'state/future/valid':
          tf.io.FixedLenFeature([128, 80], tf.int64, default_value=None),
      'state/future/vel_yaw':
          tf.io.FixedLenFeature([128, 80], tf.float32, default_value=None),
      'state/future/velocity_x':
          tf.io.FixedLenFeature([128, 80], tf.float32, default_value=None),
      'state/future/velocity_y':
          tf.io.FixedLenFeature([128, 80], tf.float32, default_value=None),
      'state/future/width':
          tf.io.FixedLenFeature([128, 80], tf.float32, default_value=None),
      'state/future/x':
          tf.io.FixedLenFeature([128, 80], tf.float32, default_value=None),
      'state/future/y':
          tf.io.FixedLenFeature([128, 80], tf.float32, default_value=None),
      'state/future/z':
          tf.io.FixedLenFeature([128, 80], tf.float32, default_value=None),
      'state/past/bbox_yaw':
          tf.io.FixedLenFeature([128, 10], tf.float32, default_value=None),
      'state/past/height':
          tf.io.FixedLenFeature([128, 10], tf.float32, default_value=None),
      'state/past/length':
          tf.io.FixedLenFeature([128, 10], tf.float32, default_value=None),
      'state/past/timestamp_micros':
          tf.io.FixedLenFeature([128, 10], tf.int64, default_value=None),
      'state/past/valid':
          tf.io.FixedLenFeature([128, 10], tf.int64, default_value=None),
      'state/past/vel_yaw':
          tf.io.FixedLenFeature([128, 10], tf.float32, default_value=None),
      'state/past/velocity_x':
          tf.io.FixedLenFeature([128, 10], tf.float32, default_value=None),
      'state/past/velocity_y':
          tf.io.FixedLenFeature([128, 10], tf.float32, default_value=None),
      'state/past/width':
          tf.io.FixedLenFeature([128, 10], tf.float32, default_value=None),
      'state/past/x':
          tf.io.FixedLenFeature([128, 10], tf.float32, default_value=None),
      'state/past/y':
          tf.io.FixedLenFeature([128, 10], tf.float32, default_value=None),
      'state/past/z':
          tf.io.FixedLenFeature([128, 10], tf.float32, default_value=None),

      # Traffic light features.
      'traffic_light_state/current/state':
          tf.io.FixedLenFeature([1, 16], tf.int64, default_value=None),
      'traffic_light_state/current/valid':
          tf.io.FixedLenFeature([1, 16], tf.int64, default_value=None),
      'traffic_light_state/current/x':
          tf.io.FixedLenFeature([1, 16], tf.float32, default_value=None),
      'traffic_light_state/current/y':
          tf.io.FixedLenFeature([1, 16], tf.float32, default_value=None),
      'traffic_light_state/current/z':
          tf.io.FixedLenFeature([1, 16], tf.float32, default_value=None),
      'traffic_light_state/past/state':
          tf.io.FixedLenFeature([10, 16], tf.int64, default_value=None),
      'traffic_light_state/past/valid':
          tf.io.FixedLenFeature([10, 16], tf.int64, default_value=None),
      'traffic_light_state/past/x':
          tf.io.FixedLenFeature([10, 16], tf.float32, default_value=None),
      'traffic_light_state/past/y':
          tf.io.FixedLenFeature([10, 16], tf.float32, default_value=None),
      'traffic_light_state/past/z':
          tf.io.FixedLenFeature([10, 16], tf.float32, default_value=None),
  }

  return tf.io.parse_single_example(tf_example, features_description)


def add_sdc_fields(inputs: Dict[str, tf.Tensor]) -> Dict[str, tf.Tensor]:
  """Extracts current x, y, z of the autonomous vehicle as specific fields."""
  # [batch_size, 2]
  sdc_indices = tf.where(tf.equal(inputs['state/is_sdc'], 1))
  # [batch_size, 1]
  inputs['sdc/current/x'] = tf.gather_nd(inputs['state/current/x'], sdc_indices)
  inputs['sdc/current/y'] = tf.gather_nd(inputs['state/current/y'], sdc_indices)
  inputs['sdc/current/z'] = tf.gather_nd(inputs['state/current/z'], sdc_indices)
  inputs['sdc/current/bbox_yaw'] = tf.gather_nd(
      inputs['state/current/bbox_yaw'], sdc_indices)
  return inputs
