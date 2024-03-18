# Copyright (c) 2024 Waymo LLC. All rights reserved.

# This is licensed under a BSD+Patent license.
# Please see LICENSE and PATENTS text files.
# ==============================================================================

import math
from typing import List

from absl.testing import parameterized
import numpy as np
import tensorflow as tf

from waymo_open_dataset.utils import test_utils
from waymo_open_dataset.utils.sim_agents import converters
from waymo_open_dataset.utils.sim_agents import test_utils as sim_agents_test_utils
from waymo_open_dataset.wdl_limited.sim_agents_metrics import interaction_features


MAX_HEADING_DIFF = interaction_features.MAX_HEADING_DIFF
SMALL_OVERLAP_THRESHOLD = interaction_features.SMALL_OVERLAP_THRESHOLD
MAX_HEADING_DIFF_FOR_SMALL_OVERLAP = (
    interaction_features.MAX_HEADING_DIFF_FOR_SMALL_OVERLAP
)
MAX_TTC_SEC = interaction_features.MAXIMUM_TIME_TO_COLLISION


def _circle_to_circle_distance(x1: float, y1: float, r1: float, x2: float,
                               y2: float, r2: float) -> float:
  """Computes the distance between two circles.

  Args:
    x1: First coordinate of the first circle.
    y1: Second coordinate of the first circle.
    r1: Radius of the first circle.
    x2: First coordinate of the second circle.
    y2: Second coordinate of the second circle.
    r2: Radius of the second circle.

  Returns:
    The signed distance between the two circles.
  """
  centers_distance = np.sqrt((x1 - x2)**2 + (y2 - y1)**2)
  # Subtract the radii to get the circle to circle signed distance.
  return centers_distance - r1 - r2


class InteractionFeaturesTest(tf.test.TestCase, parameterized.TestCase):

  def test_distance_to_nearest_object_has_correct_shape(self):
    scenario = test_utils.get_womd_test_scenario()
    submission = sim_agents_test_utils.load_test_submission()
    simulated_trajectories = converters.joint_scene_to_trajectories(
        submission.scenario_rollouts[0].joint_scenes[0], scenario)
    mask = tf.convert_to_tensor([True] * 4 + [False] * 42 + [True] * 4)
    distances = interaction_features.compute_distance_to_nearest_object(
        center_x=simulated_trajectories.x,
        center_y=simulated_trajectories.y,
        center_z=simulated_trajectories.z,
        length=simulated_trajectories.length,
        width=simulated_trajectories.width,
        height=simulated_trajectories.height,
        heading=simulated_trajectories.heading,
        valid=simulated_trajectories.valid,
        evaluated_object_mask=mask)
    self.assertEqual(
        distances.shape, (8, simulated_trajectories.valid.shape[-1]))

  def test_distance_to_nearest_returns_correct_distances(self):
    # Collection of N different configurations for box1 and box2, expressed as a
    # tensor (N, 2, 7), where the last dimension represent [x, y, z, length,
    # width, height, heading].

    # Boxes are touching forward. Expected distance is 0m.
    box_config1 = tf.constant([[0.0, 0.0, 0.0, 2.0, 1.0, 1.0, 0.0],
                               [3.0, 0.0, 0.0, 4.0, 1.0, 1.0, 0.0]])
    # Boxes are touching on the right. Expected distance is 0m.
    box_config2 = tf.constant([[0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0],
                               [0.0, 1.5, 0.0, 1.0, 2.0, 1.0, 0.0]])
    # Boxes are touching but box1 is rotated by 90 degrees. Expected distance
    # is 0m.
    box_config3 = tf.constant([[0.0, 0.0, 0.0, 2.0, 1.0, 1.0, np.pi/2],
                               [1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0]])
    # Boxes are 10 meters apart on the x-axis. Box1 is rotated by 180 degrees.
    # Expected distance 10m.
    box_config4 = tf.constant([[6.0, 0.0, 0.0, 2.0, 1.0, 1.0, np.pi],
                               [-6.0, 0.0, 0.0, 2.0, 1.0, 1.0, 0.0]])

    all_boxes = tf.stack([box_config1, box_config2, box_config3, box_config4],
                         axis=0)

    # We can treat the N independent cases as different timesteps, so rearrange
    # the tensor shapes to conform to (n_objects, n_steps).
    all_boxes = tf.transpose(all_boxes, [1, 0, 2])
    # Evaluate only box 1, and consider all the boxes valid at every step.
    evaluated_boxes = tf.constant([True, False])
    validity = tf.fill(all_boxes.shape[:-1], True)

    # List the expected distances. Expand the first index as the number of
    # evaluated objects is 1.
    expected_distances = tf.constant([0.0, 0.0, 0.0, 10.0])[tf.newaxis]

    distances = interaction_features.compute_distance_to_nearest_object(
        center_x=all_boxes[..., 0], center_y=all_boxes[..., 1],
        center_z=all_boxes[..., 2], length=all_boxes[..., 3],
        width=all_boxes[..., 4], height=all_boxes[..., 5],
        heading=all_boxes[..., 6], valid=validity,
        evaluated_object_mask=evaluated_boxes,
        corner_rounding_factor=0.)

    self.assertAllClose(distances, expected_distances)

  @parameterized.named_parameters(
      {'testcase_name': 'valid_box2', 'box2_valid': True},
      {'testcase_name': 'invalid_box2', 'box2_valid': False},
  )
  def test_distance_to_nearest_object_handles_invalid_objects(self, box2_valid):
    all_boxes = tf.constant(
        # Boxes are 10 meters apart on the x-axis. Box1 is rotated by 180
        # degrees. Expected distance 10m. Shape: (2, 1, 7)
        [[[6.0, 0.0, 0.0, 2.0, 1.0, 1.0, np.pi]],
         [[-6.0, 0.0, 0.0, 2.0, 1.0, 1.0, 0.0]]])
    validity = tf.constant([[True], [box2_valid]])
    # Evaluate only box 1, and consider all the boxes valid at every step.
    evaluated_boxes = tf.constant([True, False])

    distances = interaction_features.compute_distance_to_nearest_object(
        center_x=all_boxes[..., 0], center_y=all_boxes[..., 1],
        center_z=all_boxes[..., 2], length=all_boxes[..., 3],
        width=all_boxes[..., 4], height=all_boxes[..., 5],
        heading=all_boxes[..., 6], valid=validity,
        evaluated_object_mask=evaluated_boxes)

    if box2_valid:
      self.assertAllClose(distances, tf.fill(distances.shape, 10.0))
    else:
      self.assertAllClose(distances, tf.fill(
          distances.shape, interaction_features.EXTREMELY_LARGE_DISTANCE))

  def test_distance_to_nearest_object_selects_nearest(self):
    # Create 3 boxes, box1 centered on (0, 0), box2 at 5m from box1 and box3
    # at 10m from box1, in different directions. Shape: (3, 1, 7).
    all_boxes = tf.constant(
        [[[0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0]],
         [[6.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0]],
         [[0.0, 11.0, 0.0, 1.0, 1.0, 1.0, 0.0]]])
    # We expand the time dimension to create 3 different setups:
    # 1. box2 valid, box3 valid, nearest is box2.
    # 2. box2 valid, box3 invalid, nearest is still box2.
    # 3. box2 invalid, box3 valid, nearest is now box3.
    validity = tf.constant([[True, True, True],
                            [True, True, False],
                            [True, False, True]], dtype=tf.bool)
    all_boxes = tf.broadcast_to(all_boxes, [3, 3, 7])
    expected_distances = tf.constant([[5.0, 5.0, 10.0]])

    # Evaluate only box 1, and consider all the boxes valid at every step.
    evaluated_boxes = tf.constant([True, False, False])

    distances = interaction_features.compute_distance_to_nearest_object(
        center_x=all_boxes[..., 0], center_y=all_boxes[..., 1],
        center_z=all_boxes[..., 2], length=all_boxes[..., 3],
        width=all_boxes[..., 4], height=all_boxes[..., 5],
        heading=all_boxes[..., 6], valid=validity,
        evaluated_object_mask=evaluated_boxes,
        corner_rounding_factor=0.)
    self.assertAllClose(distances, expected_distances)

  @parameterized.named_parameters(
      (
          'no_rounding',
          0.,
          # Touching corners
          0.,
      ),
      (
          'max_rounding',
          1.,
          # Circles of radii and centers 1, (0,0) and 2, (3, 3) respectively.
          _circle_to_circle_distance(0., 0., 1., 3., 3., 2.)
      ),
      (
          'medium_rounding',
          0.5,
          # Circles of radii and centers 0.5, (0.5,0.5) and 1, (2, 2)
          # respectively.
          _circle_to_circle_distance(0.5, 0.5, 0.5, 2., 2., 1.)
      ),
  )
  def test_distance_to_nearest_object_with_corner_rounding(
      self, corner_rounding_factor: float, expected_distance: float):
    # Create 2 boxes, box1 shaped 2x2 centered on (0, 0), box2 shaped 6x4
    # centered on (4, 3).
    # Box parameters are ordered as (x,y,z,l,w,h,heading).
    # Shape: (2, 1, 7).
    all_boxes = tf.constant(
        [[[0.0, 0.0, 0.0, 2.0, 2.0, 1.0, 0.0]],
         [[4.0, 3.0, 0.0, 6.0, 4.0, 1.0, 0.0]]])
    # Shape: (objects, steps, box parameters)
    all_boxes = tf.broadcast_to(all_boxes, [2, 3, 7])

    # Evaluate only box 1, and consider all the boxes valid at every step.
    evaluated_boxes = tf.constant([True, False])
    validity = tf.fill(all_boxes.shape[:-1], True)

    expected_distances = tf.constant([[expected_distance]])
    # Shape: (evaluated objects, steps)
    expected_distances = tf.broadcast_to(expected_distances, [1, 3])

    distances = interaction_features.compute_distance_to_nearest_object(
        center_x=all_boxes[..., 0], center_y=all_boxes[..., 1],
        center_z=all_boxes[..., 2], length=all_boxes[..., 3],
        width=all_boxes[..., 4], height=all_boxes[..., 5],
        heading=all_boxes[..., 6], valid=validity,
        evaluated_object_mask=evaluated_boxes,
        corner_rounding_factor=corner_rounding_factor)
    self.assertAllClose(distances, expected_distances)

  def test_time_to_collision_with_object_in_front_has_correct_shape(self):
    scenario = test_utils.get_womd_test_scenario()
    submission = sim_agents_test_utils.load_test_submission()
    simulated_trajectories = converters.joint_scene_to_trajectories(
        submission.scenario_rollouts[0].joint_scenes[0], scenario
    )
    mask = tf.convert_to_tensor([True] * 4 + [False] * 42 + [True] * 4)
    seconds_per_step = 0.1
    ttc = interaction_features.compute_time_to_collision_with_object_in_front(
        center_x=simulated_trajectories.x,
        center_y=simulated_trajectories.y,
        length=simulated_trajectories.length,
        width=simulated_trajectories.width,
        heading=simulated_trajectories.heading,
        valid=simulated_trajectories.valid,
        evaluated_object_mask=mask,
        seconds_per_step=seconds_per_step,
    )
    self.assertEqual(ttc.shape, (8, simulated_trajectories.valid.shape[-1]))

  @parameterized.named_parameters(
      # 9 square boxes in a 3x3 grid.
      (
          '9 squares grid',
          [
              [-3, 3],
              [0, 3],
              [3, 3],
              [-3, 0],
              [0, 0],
              [3, 0],
              [-3, -3],
              [0, -3],
              [3, -3],
          ],
          [0] * 9,
          [[1, 1]] * 9,
          [10, 6, 1] * 3,
          [2 / 4, 2 / 5, MAX_TTC_SEC] * 3,
      ),
      # Rectangles in a line.
      (
          'Lined up rectangles',
          [[0, 0], [5, 0], [10, 0], [15, 0]],
          [0, 0, 0, 0],
          [[4, 2]] * 4,
          [6, 10, 3, 1],
          [MAX_TTC_SEC, 1 / 7, 1 / 2, MAX_TTC_SEC],
      ),
      (
          'test ignore misaligned',
          [[0, 0], [5, 0], [10, 0], [15, 0]],
          [
              0,
              MAX_HEADING_DIFF + 0.01,
              0,
              MAX_HEADING_DIFF - 0.01,
          ],
          [[4, 2]] * 4,
          [10, 6, 3, 1],
          [
              6 / 7,
              MAX_TTC_SEC,
              # Closed-form TTC =
              #   (center_distance - ego_length/2 - rotated_obstacle_length/2)
              #   / relative_speed
              (
                  3
                  - math.cos(MAX_HEADING_DIFF - 0.01) * 2
                  - math.sin(MAX_HEADING_DIFF - 0.01)
              )
              / 2,
              MAX_TTC_SEC,
          ],
      ),
      (
          'test ignore no overlap',
          [[0, 0], [5, 2.1], [10, 1.1], [15, 0]],
          [0, 0, 0, 0],
          [[4, 2]] * 4,
          [10, 6, 3, 1],
          [6 / 7, 1 / 3, 1 / 2, MAX_TTC_SEC],
      ),
      (
          'test ignore small misalignment with low overlap',
          [
              [0, 0],
              [5, 2.5 - SMALL_OVERLAP_THRESHOLD],
              [10, -2.5 + SMALL_OVERLAP_THRESHOLD],
          ],
          [
              0,
              MAX_HEADING_DIFF_FOR_SMALL_OVERLAP + 0.01,
              -MAX_HEADING_DIFF_FOR_SMALL_OVERLAP + 0.01,
          ],
          [[4, 2]] * 3,
          [6, 3, 1],
          [
              # Closed-form TTC =
              #   (center_distance - ego_length/2 - rotated_obstacle_length/2)
              #   / relative_speed
              (
                  8
                  - math.cos(MAX_HEADING_DIFF_FOR_SMALL_OVERLAP - 0.01) * 2
                  - math.sin(MAX_HEADING_DIFF_FOR_SMALL_OVERLAP - 0.01)
              )
              / 5,
              MAX_TTC_SEC,
              MAX_TTC_SEC,
          ],
      ),
  )
  def test_time_to_collision_with_object_in_front(
      self,
      center_xys: List[float],
      headings: List[float],
      boxes_sizes: List[float],
      speeds: List[float],
      expected_ttc_sec: List[float],
  ):
    center_xys = tf.convert_to_tensor(center_xys, dtype=tf.float32)
    headings = tf.convert_to_tensor(headings, dtype=tf.float32)
    boxes_sizes = tf.convert_to_tensor(boxes_sizes, dtype=tf.float32)
    speeds = tf.convert_to_tensor(speeds, dtype=tf.float32)
    expected_ttc_sec = tf.convert_to_tensor(expected_ttc_sec, dtype=tf.float32)

    seconds_per_step = 0.1
    # Simulate 2 steps back and forward to get non-nan speeds with central
    # difference.
    center_x_1 = center_xys[:, 0]
    center_x_2 = center_x_1 + speeds * tf.math.cos(headings) * seconds_per_step
    center_x_0 = center_x_1 - speeds * tf.math.cos(headings) * seconds_per_step
    center_x = tf.stack(
        [center_x_0, center_x_1, center_x_2],
        axis=-1,
    )
    center_y_1 = center_xys[:, 1]
    center_y_2 = center_y_1 - speeds * tf.math.sin(headings) * seconds_per_step
    center_y_0 = center_y_1 + speeds * tf.math.sin(headings) * seconds_per_step
    center_y = tf.stack(
        [center_y_0, center_y_1, center_y_2],
        axis=-1,
    )
    length = tf.broadcast_to(boxes_sizes[:, 0:1], center_x.shape)
    width = tf.broadcast_to(boxes_sizes[:, 1:2], center_x.shape)
    heading = tf.broadcast_to(headings[:, tf.newaxis], center_x.shape)
    valid = tf.ones_like(center_x, dtype=bool)
    mask = tf.ones_like(center_x[:, 0], dtype=bool)
    # `ttc` shape: (num_evaluated_objects, num_steps).
    ttc = interaction_features.compute_time_to_collision_with_object_in_front(
        center_x=center_x,
        center_y=center_y,
        length=length,
        width=width,
        heading=heading,
        valid=valid,
        evaluated_object_mask=mask,
        seconds_per_step=seconds_per_step,
    )
    # Assert `ttc` is correct in the step with non-nan speeds.
    self.assertAllClose(ttc[:, 1], expected_ttc_sec)


if __name__ == '__main__':
  tf.test.main()
