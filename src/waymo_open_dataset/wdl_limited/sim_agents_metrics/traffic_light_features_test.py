# Copyright (c) 2025 Waymo LLC. All rights reserved.

# This is licensed under a BSD+Patent license.
# Please see LICENSE and PATENTS text files.
# ==============================================================================

import numpy as np
import tensorflow as tf

from waymo_open_dataset.protos import map_pb2
from waymo_open_dataset.utils import test_utils
from waymo_open_dataset.utils import trajectory_utils
from waymo_open_dataset.wdl_limited.sim_agents_metrics import traffic_light_features

LaneType = map_pb2.LaneCenter.LaneType


class TrafficLightFeaturesTest(tf.test.TestCase):

  def test_red_light_violation_is_correct_on_synthetic_data(self):
    scenario = test_utils.get_womd_test_scenario()
    trajectories = trajectory_utils.ObjectTrajectories.from_scenario(scenario)

    # We can introduce a traffic light violation by moving the ADV which is
    # stopped at a red light in the test scenario. We move it in the negative
    # y direction, towards the traffic light.
    traj = np.array(trajectories.y)
    traj[scenario.sdc_track_index, 20:] -= np.arange(71)
    trajectories = trajectory_utils.ObjectTrajectories(
        object_id=trajectories.object_id,
        x=trajectories.x,
        y=traj,
        z=trajectories.z,
        heading=trajectories.heading,
        length=trajectories.length,
        width=trajectories.width,
        height=trajectories.height,
        object_type=trajectories.object_type,
        valid=trajectories.valid,
    )

    # Evaluation objects (includes ADV).
    mask = np.array([False] * 83)
    mask[[42, 43, 72, 82]] = True
    mask = tf.convert_to_tensor(mask)

    lane_ids = []
    lane_polylines = []
    for map_feature in scenario.map_features:
      if map_feature.HasField('lane'):
        if map_feature.lane.type == LaneType.TYPE_SURFACE_STREET:
          lane_ids.append(map_feature.id)
          lane_polylines.append(map_feature.lane.polyline)

    traffic_signals = []
    for dynamic_map_state in scenario.dynamic_map_states:
      traffic_signals.append(list(dynamic_map_state.lane_states))

    red_light_violations = traffic_light_features.compute_red_light_violation(
        center_x=trajectories.x,
        center_y=trajectories.y,
        valid=trajectories.valid,
        evaluated_object_mask=mask,
        lane_polylines=lane_polylines,
        lane_ids=lane_ids,
        traffic_signals=traffic_signals,
    )

    self.assertEqual(
        red_light_violations.shape, (4, trajectories.valid.shape[-1])
    )
    # Last object (ADV) should have a red light violation.
    self.assertTrue(tf.reduce_any(red_light_violations, axis=-1)[-1])
    # All other objects should not have a red light violation.
    self.assertFalse(tf.reduce_any(red_light_violations[:-1, :]))

if __name__ == '__main__':
  tf.test.main()
