# Copyright 2019 The Waymo Open Dataset Authors. All Rights Reserved.
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
"""Python version of metrics/config_util.h."""

from waymo_open_dataset import label_pb2
from waymo_open_dataset.protos import breakdown_pb2
from waymo_open_dataset.protos import scenario_pb2

__all__ = [
    "get_breakdown_names_from_config", "get_breakdown_names_from_motion_config"
]


def _get_num_breakdown_shards(breakdown_generator_id):
  """Gets the number of breakdown shards for a given breakdown generator ID.

  Args:
    breakdown_generator_id: the breakdown generator ID.

  Returns:
    The number of breakdown shards.
  """
  if breakdown_generator_id == breakdown_pb2.Breakdown.GeneratorId.Value(
      "ONE_SHARD"):
    return 1
  elif breakdown_generator_id == (
      breakdown_pb2.Breakdown.GeneratorId.Value("ALL_BUT_SIGN")):
    return 2
  elif breakdown_generator_id == (
      breakdown_pb2.Breakdown.GeneratorId.Value("SIZE")):
    return 2
  elif breakdown_generator_id == (
      breakdown_pb2.Breakdown.GeneratorId.Value("OBJECT_TYPE")):
    return 4
  elif breakdown_generator_id == breakdown_pb2.Breakdown.GeneratorId.Value(
      "RANGE"):
    return 3 * 4
  elif breakdown_generator_id == breakdown_pb2.Breakdown.GeneratorId.Value(
      "VELOCITY"):
    return 5 * 4
  else:
    raise ValueError("Unsupported breakdown {}.".format(
        breakdown_pb2.Breakdown.GeneratorId.Name(breakdown_generator_id)))


def _get_breakdown_shard_name(breakdown_generator_id, shard):
  """Gets the breakdown shard name.

  Args:
    breakdown_generator_id: the breakdown generator ID.
    shard: the breakdown shard

  Returns:
    The name of this breakdown shard.
  """

  if breakdown_generator_id == breakdown_pb2.Breakdown.GeneratorId.Value(
      "ONE_SHARD"):
    return breakdown_pb2.Breakdown.GeneratorId.Name(breakdown_generator_id)
  elif breakdown_generator_id == (
      breakdown_pb2.Breakdown.GeneratorId.Value("ALL_BUT_SIGN")):
    if shard == 0:
      return "ALL_NS"
    else:
      return "SIGN"
  elif breakdown_generator_id == (
      breakdown_pb2.Breakdown.GeneratorId.Value("OBJECT_TYPE")):
    return "{}_{}".format(
        breakdown_pb2.Breakdown.GeneratorId.Name(breakdown_generator_id),
        label_pb2.Label.Type.Name(shard + 1))
  elif breakdown_generator_id == (
      breakdown_pb2.Breakdown.GeneratorId.Value("VELOCITY")):
    object_type = shard // 5 + 1
    velocity_shard = shard % 5
    velocity_shard_name = ""
    if velocity_shard == 0:
      velocity_shard_name = "STATIONARY"
    elif velocity_shard == 1:
      velocity_shard_name = "SLOW"
    elif velocity_shard == 2:
      velocity_shard_name = "MEDIUM"
    elif velocity_shard == 3:
      velocity_shard_name = "FAST"
    else:
      velocity_shard_name = "VERY_FAST"
    return "{}_{}_{}".format(
        breakdown_pb2.Breakdown.GeneratorId.Name(breakdown_generator_id),
        label_pb2.Label.Type.Name(object_type), velocity_shard_name)
  elif breakdown_generator_id == breakdown_pb2.Breakdown.GeneratorId.Value(
      "RANGE"):
    object_type = shard // 3 + 1
    range_shard = shard % 3
    range_shard_name = ""
    if range_shard == 0:
      range_shard_name = "[0, 30)"
    elif range_shard == 1:
      range_shard_name = "[30, 50)"
    else:
      range_shard_name = "[50, +inf)"
    return "{}_{}_{}".format(
        breakdown_pb2.Breakdown.GeneratorId.Name(breakdown_generator_id),
        label_pb2.Label.Type.Name(object_type), range_shard_name)
  elif breakdown_generator_id == breakdown_pb2.Breakdown.GeneratorId.Value(
      "SIZE"):
    object_type = shard // 2 + 1
    size_shard = shard % 2
    size_shard_name = "small" if size_shard == 0 else "large"
    return "{}_{}_{}".format(
        breakdown_pb2.Breakdown.GeneratorId.Name(breakdown_generator_id),
        label_pb2.Label.Type.Name(object_type), size_shard_name)
  else:
    raise ValueError("Unsupported breakdown {}.".format(
        breakdown_pb2.Breakdown.GeneratorId.Name(breakdown_generator_id)))


def get_breakdown_names_from_config(config):
  """Returns names for each metrics breakdown defined by the config.

  The output vector is ordered as:
  [{generator_i_shard_j_diffculity_level_k}].
  i in [0, num_breakdown_generators).
  j in [0, num_shards for the i-th breakdown generator).
  k in [0, num_difficulty_levels for each shard in the i-th breakdown
  generator).

  The implementation should be kept the same as metrics/config_util.{h,cc}.

  Args:
    config: the metrics config defined in protos/metrics.proto.

  Returns:
    A list of names for each breakdown defined by the config. The order is
      guaranteed to be the same as all public metric lib that produces
      breakdown metrics.
  """

  names = []
  for index, breakdown_generator_id in enumerate(
      config.breakdown_generator_ids):
    for shard_id in range(_get_num_breakdown_shards(breakdown_generator_id)):
      difficulty_levels = config.difficulties[index].levels
      shard_name = _get_breakdown_shard_name(breakdown_generator_id, shard_id)
      if not difficulty_levels:
        names.append("{}_{}".format(shard_name, "LEVEL_2"))
      else:
        for dl in difficulty_levels:
          names.append("{}_{}".format(shard_name,
                                      label_pb2.Label.DifficultyLevel.Name(dl)))
  return names


def get_breakdown_names_from_motion_config(config):
  r"""Returns names for each metrics breakdown defined by the config.

  The output vector is ordered as:
  [{object_type_i_step_j}]
  j \in [0, len(step_configrations) for ith object_type]
  i \in [0, num_object_types (currently at 3: VEHICLE, PEDESTRIAN, CYCLIST)]

  The implementation should be kept the same as metrics/ops/motion_metrics

  Args:
    config: the metrics config defined in protos/metrics.proto.

  Returns:
    A list of names for each breakdown defined by the config. The order is
      guaranteed to be the same as all public metric lib that produces
      breakdown metrics.
  """
  names = []
  for object_type in (scenario_pb2.Track.TYPE_VEHICLE,
                      scenario_pb2.Track.TYPE_PEDESTRIAN,
                      scenario_pb2.Track.TYPE_CYCLIST):
    for step in config.step_configurations:
      names.append("{}_{}".format(
          scenario_pb2.Track.ObjectType.Name(object_type),
          step.measurement_step))
  return names
