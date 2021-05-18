/* Copyright 2019 The Waymo Open Dataset Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "waymo_open_dataset/metrics/ops/utils.h"

#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/types/optional.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "waymo_open_dataset/protos/metrics.pb.h"

namespace waymo {
namespace open_dataset {
namespace {
using LB = Label::Box;
typedef void (LB::*Setter)(double);

// Returns a 3d box from the tensor at a given index. This function does not
// check param validity.
// Requires:: bbox: [N, 7] float tensor. 0 <= box_index < N.
Label::Box GetBox3d(const tensorflow::Tensor& bbox, int box_index) {
  LB box;
  const auto matrix = bbox.matrix<float>();
  static const std::vector<Setter>* setters = new std::vector<Setter>{
      &LB::set_center_x, &LB::set_center_y, &LB::set_center_z, &LB::set_length,
      &LB::set_width,    &LB::set_height,   &LB::set_heading};
  for (int i = 0; i < 7; ++i) {
    (box.*(*setters)[i])(matrix(box_index, i));
  }
  return box;
}

// Returns a 2d box from the tensor at a given index. This function does not
// check param validity.
// Requires:: bbox: [N, 5] float tensor. 0 <= box_index < N.
Label::Box GetBox2d(const tensorflow::Tensor& bbox, int box_index) {
  LB box;
  const auto matrix = bbox.matrix<float>();
  static const std::vector<Setter>* setters = new std::vector<Setter>{
      &LB::set_center_x, &LB::set_center_y, &LB::set_length, &LB::set_width,
      &LB::set_heading};
  for (int i = 0; i < 5; ++i) {
    (box.*(*setters)[i])(matrix(box_index, i));
  }
  return box;
}

// Returns a 2d AA box from the tensor at a given index. This function does not
// check param validity.
// Requires:: bbox: [N, 4] float tensor. 0 <= box_index < N.
Label::Box GetAABox2d(const tensorflow::Tensor& bbox, int box_index) {
  LB box;
  const auto matrix = bbox.matrix<float>();
  static const std::vector<Setter>* setters = new std::vector<Setter>{
      &LB::set_center_x, &LB::set_center_y, &LB::set_length, &LB::set_width};
  for (int i = 0; i < 4; ++i) {
    (box.*(*setters)[i])(matrix(box_index, i));
  }
  return box;
}

// Return a Box from the tensor at a given index. This function does not check
// param validity.
// Requires:: bbox: [N, M] float tensor. 0 <= box_index < N and M in [4, 5, 7].
Label::Box GetBoxByDimension(const tensorflow::Tensor& bbox, int box_index) {
  const int32 box_dof = bbox.dim_size(1);
  switch (box_dof) {
    case 4:
      return GetAABox2d(bbox, box_index);
    case 5:
      return GetBox2d(bbox, box_index);
    case 7:
      return GetBox3d(bbox, box_index);
    default:
      LOG(FATAL) << "Incorrect number of box DOF " << box_dof;
  }
}

}  // namespace

int GetDesiredBoxDOF(Label::Box::Type box_type) {
  switch (box_type) {
    case Label::Box::TYPE_AA_2D:
      return 4;
    case Label::Box::TYPE_2D:
      return 5;
    case Label::Box::TYPE_3D:
      return 7;
    case Label::Box::TYPE_UNKNOWN:
    default:
      LOG(FATAL) << "Unknown box type.";
  }
}

absl::flat_hash_map<int64, std::vector<Object>> ParseObjectFromTensors(
    const tensorflow::Tensor& bbox, const tensorflow::Tensor& type,
    const tensorflow::Tensor& frame_id,
    const absl::optional<const tensorflow::Tensor>& score,
    const absl::optional<const tensorflow::Tensor>& overlap_nlz,
    const absl::optional<const tensorflow::Tensor>& detection_difficulty,
    const absl::optional<const tensorflow::Tensor>& tracking_difficulty,
    const absl::optional<const tensorflow::Tensor>& object_speed) {
  CHECK_EQ(bbox.dim_size(0), type.dim_size(0));
  CHECK_EQ(bbox.dim_size(0), frame_id.dim_size(0));
  if (score.has_value()) {
    CHECK_EQ(bbox.dim_size(0), score.value().dim_size(0));
  }
  if (overlap_nlz.has_value()) {
    CHECK_EQ(bbox.dim_size(0), overlap_nlz->dim_size(0));
  }
  if (detection_difficulty.has_value()) {
    CHECK_EQ(bbox.dim_size(0), detection_difficulty->dim_size(0));
  }
  if (tracking_difficulty.has_value()) {
    CHECK_EQ(bbox.dim_size(0), tracking_difficulty->dim_size(0));
  }
  if (object_speed.has_value()) {
    CHECK_EQ(bbox.dim_size(0), object_speed->dim_size(0));
  }

  absl::flat_hash_map<int64, std::vector<Object>> objects;
  for (int i = 0, n = bbox.dim_size(0); i < n; ++i) {
    Object object;
    object.mutable_object()->set_type(
        static_cast<Label::Type>(type.vec<uint8>()(i)));
    if (score.has_value()) {
      object.set_score(score.value().vec<float>()(i));
    }
    *object.mutable_object()->mutable_box() = GetBoxByDimension(bbox, i);
    if (overlap_nlz.has_value()) {
      object.set_overlap_with_nlz(overlap_nlz.value().vec<bool>()(i));
    }
    if (detection_difficulty.has_value()) {
      object.mutable_object()->set_detection_difficulty_level(
          static_cast<Label::DifficultyLevel>(
              detection_difficulty.value().vec<uint8>()(i)));
    }
    if (tracking_difficulty.has_value()) {
      object.mutable_object()->set_tracking_difficulty_level(
          static_cast<Label::DifficultyLevel>(
              tracking_difficulty.value().vec<uint8>()(i)));
    }
    if (object_speed.has_value()) {
      object.mutable_object()->mutable_metadata()->set_speed_x(
          object_speed.value().matrix<float>()(i, 0));
      object.mutable_object()->mutable_metadata()->set_speed_y(
          object_speed.value().matrix<float>()(i, 1));
    }
    const int64 id = frame_id.vec<int64>()(i);
    objects[id].emplace_back(std::move(object));
  }
  return objects;
}

absl::flat_hash_map<std::string,
                    absl::flat_hash_map<int64, std::vector<Object>>>
ParseObjectGroupedBySequenceFromTensors(
    const tensorflow::Tensor& bbox, const tensorflow::Tensor& type,
    const tensorflow::Tensor& frame_id, const tensorflow::Tensor& sequence_id,
    const tensorflow::Tensor& object_id,
    const absl::optional<const tensorflow::Tensor>& score,
    const absl::optional<const tensorflow::Tensor>& overlap_nlz,
    const absl::optional<const tensorflow::Tensor>& detection_difficulty,
    const absl::optional<const tensorflow::Tensor>& tracking_difficulty,
    const absl::optional<const tensorflow::Tensor>& object_speed) {
  CHECK_EQ(bbox.dim_size(0), type.dim_size(0));
  CHECK_EQ(bbox.dim_size(0), frame_id.dim_size(0));
  CHECK_EQ(bbox.dim_size(0), sequence_id.dim_size(0));
  CHECK_EQ(bbox.dim_size(0), object_id.dim_size(0));
  if (score.has_value()) {
    CHECK_EQ(bbox.dim_size(0), score.value().dim_size(0));
  }
  if (overlap_nlz.has_value()) {
    CHECK_EQ(bbox.dim_size(0), overlap_nlz->dim_size(0));
  }
  if (detection_difficulty.has_value()) {
    CHECK_EQ(bbox.dim_size(0), detection_difficulty->dim_size(0));
  }
  if (tracking_difficulty.has_value()) {
    CHECK_EQ(bbox.dim_size(0), tracking_difficulty->dim_size(0));
  }
  if (object_speed.has_value()) {
    CHECK_EQ(bbox.dim_size(0), object_speed->dim_size(0));
  }

  // Map of sequence ids to (map of frame ids to list of objects in that frame).
  absl::flat_hash_map<std::string,
                      absl::flat_hash_map<int64, std::vector<Object>>>
      objects;
  // Tracking metrics compuation can fail if inputs are repeated, so
  // track_object_frame_sequence tracks the uniqueness of (object id, frame id,
  // sequence id) and logs warning if they are repeated for easier debugging
  // when tracking metrics fails.
  absl::flat_hash_set<std::tuple<std::string, int64, std::string>>
      track_object_frame_sequence;

  for (int i = 0, n = bbox.dim_size(0); i < n; ++i) {
    Object object;
    object.mutable_object()->set_type(
        static_cast<Label::Type>(type.vec<uint8>()(i)));
    if (score.has_value()) {
      object.set_score(score.value().vec<float>()(i));
    }
    *object.mutable_object()->mutable_box() = GetBoxByDimension(bbox, i);
    if (overlap_nlz.has_value()) {
      object.set_overlap_with_nlz(overlap_nlz.value().vec<bool>()(i));
    }
    if (detection_difficulty.has_value()) {
      object.mutable_object()->set_detection_difficulty_level(
          static_cast<Label::DifficultyLevel>(
              detection_difficulty.value().vec<uint8>()(i)));
    }
    if (tracking_difficulty.has_value()) {
      object.mutable_object()->set_tracking_difficulty_level(
          static_cast<Label::DifficultyLevel>(
              tracking_difficulty.value().vec<uint8>()(i)));
    }
    if (object_speed.has_value()) {
      object.mutable_object()->mutable_metadata()->set_speed_x(
          object_speed.value().matrix<float>()(i, 0));
      object.mutable_object()->mutable_metadata()->set_speed_y(
          object_speed.value().matrix<float>()(i, 1));
    }
    const std::string object_id_i = absl::StrCat(object_id.vec<int64>()(i));
    object.mutable_object()->set_id(object_id_i);
    const int64 frame_id_i = frame_id.vec<int64>()(i);
    const std::string sequence_id_i =
        static_cast<std::string>(sequence_id.vec<tensorflow::tstring>()(i));
    objects[sequence_id_i][frame_id_i].emplace_back(std::move(object));

    // Tracking metrics computation can fail if the inputs are repeated, e.g.
    // there are boxes with the same object ids in one frame.
    const std::tuple<std::string, int64, std::string> object_frame_sequence_i =
        std::make_tuple(object_id_i, frame_id_i, sequence_id_i);
    if (track_object_frame_sequence.contains(object_frame_sequence_i)) {
      LOG(WARNING) << "Saw repeated input of object_id " << object_id_i
                   << " frame_id " << frame_id_i << " sequence_id "
                   << sequence_id_i;
    } else {
      track_object_frame_sequence.insert(object_frame_sequence_i);
    }
  }
  return objects;
}

absl::flat_hash_map<std::string, std::pair<Scenario, ScenarioPredictions>>
ParseScenarioAndPredictonsFromTensors(
    const tensorflow::Tensor& pred_trajectory,
    const tensorflow::Tensor& pred_score,
    const tensorflow::Tensor& gt_trajectory,
    const tensorflow::Tensor& gt_is_valid,
    const tensorflow::Tensor& pred_gt_indices,
    const tensorflow::Tensor& pred_gt_indices_mask,
    const tensorflow::Tensor& object_type, const tensorflow::Tensor& object_id,
    const tensorflow::Tensor& scenario_id) {
  CHECK_EQ(pred_trajectory.dims(), 6);
  CHECK_EQ(pred_score.dims(), 3);
  CHECK_EQ(gt_trajectory.dims(), 4);
  CHECK_EQ(gt_is_valid.dims(), 3);
  CHECK_EQ(pred_gt_indices.dims(), 3);
  CHECK_EQ(pred_gt_indices_mask.dims(), 3);
  CHECK_EQ(object_type.dims(), 2);
  CHECK_EQ(object_id.dims(), 2);
  CHECK_EQ(scenario_id.dims(), 1);

  const int batch_size = pred_trajectory.dim_size(0);
  const int num_pred_groups = pred_trajectory.dim_size(1);
  const int top_k = pred_trajectory.dim_size(2);
  const int num_agents_per_group = pred_trajectory.dim_size(3);
  const int num_pred_steps = pred_trajectory.dim_size(4);
  const int num_total_agents = gt_trajectory.dim_size(1);
  const int num_gt_steps = gt_trajectory.dim_size(2);

  CHECK_EQ(pred_score.dim_size(0), batch_size);
  CHECK_EQ(pred_score.dim_size(1), num_pred_groups);
  CHECK_EQ(pred_score.dim_size(2), top_k);
  CHECK_EQ(gt_trajectory.dim_size(0), batch_size);
  CHECK_EQ(gt_trajectory.dim_size(3), 7);
  CHECK_EQ(gt_is_valid.dim_size(0), batch_size);
  CHECK_EQ(gt_is_valid.dim_size(1), num_total_agents);
  CHECK_EQ(gt_is_valid.dim_size(2), num_gt_steps);
  CHECK_EQ(pred_gt_indices.dim_size(0), batch_size);
  CHECK_EQ(pred_gt_indices.dim_size(1), num_pred_groups);
  CHECK_EQ(pred_gt_indices.dim_size(2), num_agents_per_group);
  CHECK_EQ(pred_gt_indices_mask.dim_size(0), batch_size);
  CHECK_EQ(pred_gt_indices_mask.dim_size(1), num_pred_groups);
  CHECK_EQ(pred_gt_indices_mask.dim_size(2), num_agents_per_group);
  CHECK_EQ(object_type.dim_size(0), batch_size);
  CHECK_EQ(object_type.dim_size(1), num_total_agents);
  CHECK_EQ(scenario_id.dim_size(0), batch_size);
  CHECK_EQ(object_id.dim_size(0), batch_size);
  CHECK_EQ(object_id.dim_size(1), num_total_agents);

  absl::flat_hash_map<std::string, std::pair<Scenario, ScenarioPredictions>>
      results;

  for (int i = 0; i < batch_size; ++i) {
    const std::string cur_scenario_id =
        scenario_id.vec<tensorflow::tstring>()(i);
    Scenario* scenario = &(results[cur_scenario_id].first);
    ScenarioPredictions* predictions = &(results[cur_scenario_id].second);
    scenario->set_scenario_id(cur_scenario_id);
    predictions->set_scenario_id(cur_scenario_id);

    for (int j = 0; j < num_total_agents; ++j) {
      const int64 cur_object_id = object_id.matrix<int64>()(i, j);
      bool trajectory_is_valid = false;
      for (int t = 0; t < num_gt_steps; ++t) {
        if (gt_is_valid.tensor<bool, 3>()(i, j, t)) {
          trajectory_is_valid = true;
          break;
        }
      }
      if (trajectory_is_valid) {
        Track* track = scenario->add_tracks();
        track->set_id(cur_object_id);
        track->set_object_type(
            static_cast<Track_ObjectType>(object_type.matrix<int64>()(i, j)));
        for (int t = 0; t < num_gt_steps; ++t) {
          auto state = track->add_states();
          state->set_center_x(gt_trajectory.tensor<float, 4>()(i, j, t, 0));
          state->set_center_y(gt_trajectory.tensor<float, 4>()(i, j, t, 1));
          state->set_length(gt_trajectory.tensor<float, 4>()(i, j, t, 2));
          state->set_width(gt_trajectory.tensor<float, 4>()(i, j, t, 3));
          state->set_heading(gt_trajectory.tensor<float, 4>()(i, j, t, 4));
          state->set_velocity_x(gt_trajectory.tensor<float, 4>()(i, j, t, 5));
          state->set_velocity_y(gt_trajectory.tensor<float, 4>()(i, j, t, 6));
          state->set_valid(gt_is_valid.tensor<bool, 3>()(i, j, t));
        }
      }
    }

    for (int m = 0; m < num_pred_groups; ++m) {
      for (int n = 0; n < num_agents_per_group; ++n) {
        if (pred_gt_indices_mask.tensor<bool, 3>()(i, m, n)) {
          int64 index = pred_gt_indices.tensor<int64, 3>()(i, m, n);
          CHECK_GE(index, 0);
          CHECK_LT(index, num_total_agents);
          RequiredPrediction& required_track =
              *scenario->add_tracks_to_predict();
          required_track.set_track_index(index);
          required_track.set_difficulty(RequiredPrediction::LEVEL_1);

          scenario->add_objects_of_interest(
              object_id.matrix<int64>()(i, scenario->tracks(index).id()));
        }
      }
    }

    for (int m = 0; m < num_pred_groups; ++m) {
      auto* multi_modal_predictions =
          predictions->add_multi_modal_predictions();

      for (int k = 0; k < top_k; ++k) {
        const float cur_score = pred_score.tensor<float, 3>()(i, m, k);
        JointTrajectories* joint_prediction =
            multi_modal_predictions->add_joint_predictions();
        joint_prediction->set_confidence(cur_score);
        for (int j = 0; j < num_agents_per_group; ++j) {
          if (pred_gt_indices_mask.tensor<bool, 3>()(i, m, j)) {
            auto new_trajectory = joint_prediction->add_trajectories();
            int64 index = pred_gt_indices.tensor<int64, 3>()(i, m, j);
            CHECK_GE(index, 0);
            CHECK_LT(index, num_total_agents);
            new_trajectory->set_object_id(scenario->tracks(index).id());
            for (int t = 0; t < num_pred_steps; ++t) {
              new_trajectory->add_center_x(
                  pred_trajectory.tensor<float, 6>()(i, m, k, j, t, 0));
              new_trajectory->add_center_y(
                  pred_trajectory.tensor<float, 6>()(i, m, k, j, t, 1));
            }
          }
        }
      }
    }
  }
  return results;
}

}  // namespace open_dataset
}  // namespace waymo
