/* Copyright 2021 The Waymo Open Dataset Authors.

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

#include "waymo_open_dataset/metrics/motion_metrics_utils.h"

#include <set>

#include "waymo_open_dataset/math/math_util.h"
#include "waymo_open_dataset/math/vec2d.h"
#include "waymo_open_dataset/protos/motion_metrics.pb.h"
#include "waymo_open_dataset/protos/motion_submission.pb.h"
#include "waymo_open_dataset/protos/scenario.pb.h"

namespace waymo {
namespace open_dataset {

absl::optional<TrajectoryType> ClassifyTrack(int prediction_step,
                                             const Track& track) {
  // Parameters for classification.
  static constexpr float kMaxSpeedForStationary = 2.0;                 // (m/s)
  static constexpr float kMaxDisplacementForStationary = 3.0;          // (m)
  static constexpr float kMaxLateralDisplacementForStraight = 2.5;     // (m)
  static constexpr float kMinLongitudinalDisplacementForUTurn = 0.0;  // (m)
  static constexpr float kMaxAbsHeadingDiffForStraight = M_PI / 6.0;   // (rad)

  // Find the last valid point in the track.
  const int num_states = track.states_size();
  int last_valid_index = -1;
  for (int i = num_states - 1; i > prediction_step; --i) {
    if (track.states(i).valid()) {
      last_valid_index = i;
      break;
    }
  }
  if (last_valid_index == -1) {
    return absl::nullopt;
  }

  // Compute the distance from first position to the last valid position,
  // heading difference, and x and y differences.
  const ObjectState& start_state = track.states(prediction_step);
  if (!start_state.valid()) {
    return absl::nullopt;
  }

  const ObjectState end_state = track.states(last_valid_index);
  const double x_delta = end_state.center_x() - start_state.center_x();
  const double y_delta = end_state.center_y() - start_state.center_y();
  const double final_displacement = std::hypot(x_delta, y_delta);
  const float heading_diff =
      NormalizeAngle(end_state.heading() - start_state.heading());
  const Vec2d normalized_delta =
      Vec2d(x_delta, y_delta).Rotated(-start_state.heading());
  const float start_speed =
      std::hypot(start_state.velocity_x(), start_state.velocity_y());
  const float end_speed =
      std::hypot(end_state.velocity_x(), end_state.velocity_y());
  const float max_speed = std::max(start_speed, end_speed);
  const float dx = normalized_delta.x();
  const float dy = normalized_delta.y();

  // Check for different trajectory types based on the computed parameters.
  if (max_speed < kMaxSpeedForStationary &&
      final_displacement < kMaxDisplacementForStationary) {
    return TrajectoryType::STATIONARY;
  }
  if (std::abs(heading_diff) < kMaxAbsHeadingDiffForStraight) {
    if (std::abs(normalized_delta.y()) < kMaxLateralDisplacementForStraight) {
      return TrajectoryType::STRAIGHT;
    }
    return dy < 0 ? TrajectoryType::STRAIGHT_RIGHT
                  : TrajectoryType::STRAIGHT_LEFT;
  }
  if (dy < 0) {
    return dx < kMinLongitudinalDisplacementForUTurn
               ? TrajectoryType::RIGHT_U_TURN
               : TrajectoryType::RIGHT_TURN;
  }
  if (dx < kMinLongitudinalDisplacementForUTurn) {
    return TrajectoryType::LEFT_U_TURN;
  }
  return TrajectoryType::LEFT_TURN;
}

Status ValidateChallengePredictions(
    const ScenarioPredictions& scenario_predictions,
    MotionChallengeSubmission::SubmissionType submission_type) {
  if (submission_type != MotionChallengeSubmission::MOTION_PREDICTION &&
      submission_type != MotionChallengeSubmission::INTERACTION_PREDICTION) {
    return InvalidArgumentError("Invalid submission type");
  }
  const bool is_interactive_submission =
      submission_type == MotionChallengeSubmission::INTERACTION_PREDICTION;

  // Validate that all interactive submissions have a single
  // MultiModalPrediction.
  const int expected_num_trajectories = is_interactive_submission ? 2 : 1;
  if (is_interactive_submission &&
      scenario_predictions.multi_modal_predictions_size() > 1) {
    return InvalidArgumentError(
        "Interactive submissions must have a single MultiModalPrediction");
  }

  // Verify that all joint predictions have the correct number of objects.
  for (const auto& multi_modal_prediction :
       scenario_predictions.multi_modal_predictions()) {
    for (const auto& prediction : multi_modal_prediction.joint_predictions()) {
      if (prediction.trajectories_size() != expected_num_trajectories) {
        const std::string message =
            "Invalid number of predictions. For the behavior prediction "
            "challenge each MulitModalPrediction should predict only a "
            "single object. For the interactive prediction challenge each "
            "MulitModalPrediction should predict two objects. : \n" +
            prediction.DebugString();
        return InvalidArgumentError(message);
      }
    }
  }
  return OkStatus();
}

namespace {

Status ConvertSinglePredictions(const PredictionSet& prediction_set,
                                const std::string& scenario_id,
                                ScenarioPredictions* result) {
  result->set_scenario_id(scenario_id);
  for (const auto& prediction : prediction_set.predictions()) {
    MultimodalPrediction& mm_prediction =
        *result->add_multi_modal_predictions();
    for (const auto& scored_trajectory : prediction.trajectories()) {
      // Add a JointTrajectories object with a single prediction for each object
      // in the prediction set.
      const Trajectory& trajectory = scored_trajectory.trajectory();
      JointTrajectories& joint_trajectories =
          *mm_prediction.add_joint_predictions();
      joint_trajectories.set_confidence(scored_trajectory.confidence());
      SingleTrajectory& single_trajectory =
          *joint_trajectories.add_trajectories();
      single_trajectory.set_object_id(prediction.object_id());
      if (trajectory.center_x_size() != trajectory.center_y_size()) {
        return InvalidArgumentError(
            "Trajectory center_x size does not match center_y size");
      }
      for (int i = 0; i < trajectory.center_x_size(); ++i) {
        single_trajectory.add_center_x(trajectory.center_x(i));
        single_trajectory.add_center_y(trajectory.center_y(i));
      }
    }
  }
  return OkStatus();
}

Status ConvertJointPrediction(const JointPrediction& joint_prediction,
                              const std::string& scenario_id,
                              ScenarioPredictions* result) {
  result->set_scenario_id(scenario_id);
  MultimodalPrediction& mm_prediction = *result->add_multi_modal_predictions();
  for (const auto& joint_trajectory : joint_prediction.joint_trajectories()) {
    if (joint_trajectory.trajectories_size() != 2) {
      return InvalidArgumentError(
          "Joint predictions must predict 2 objects for the challenge.");
    }
    JointTrajectories& joint_trajectories =
        *mm_prediction.add_joint_predictions();
    joint_trajectories.set_confidence(joint_trajectory.confidence());
    for (const auto& trajectory : joint_trajectory.trajectories()) {
      SingleTrajectory& dst_trajectory = *joint_trajectories.add_trajectories();
      dst_trajectory.set_object_id(trajectory.object_id());
      if (trajectory.trajectory().center_x_size() !=
          trajectory.trajectory().center_y_size()) {
        return InvalidArgumentError(
            "Trajectory center_x size does not match center_y size");
      }
      for (int i = 0; i < trajectory.trajectory().center_x_size(); ++i) {
        dst_trajectory.add_center_x(trajectory.trajectory().center_x(i));
        dst_trajectory.add_center_y(trajectory.trajectory().center_y(i));
      }
    }
  }
  return OkStatus();
}

// Returns a vector position for the given step in a trajectory.
Vec2d Position(const SingleTrajectory& trajectory, int step) {
  return Vec2d(trajectory.center_x(step), trajectory.center_y(step));
}

}  // namespace

Status ConvertChallengePredictions(
    const ChallengeScenarioPredictions& predictions,
    ScenarioPredictions* result) {
  if (predictions.has_single_predictions()) {
    return ConvertSinglePredictions(predictions.single_predictions(),
                                    predictions.scenario_id(), result);
  } else if (predictions.has_joint_prediction()) {
    return ConvertJointPrediction(predictions.joint_prediction(),
                                  predictions.scenario_id(), result);
  }
  return InvalidArgumentError("Submission is missing trajectories.");
}

Status GetTrack(int object_id,
                const absl::flat_hash_map<int, const Track*>& ids_to_tracks,
                const Track** track) {
  const auto iter = ids_to_tracks.find(object_id);
  if (iter == ids_to_tracks.end()) {
    return InvalidArgumentError(absl::StrCat(
        "Invalid prediction object ID. Track not found for : ", object_id));
  }
  *track = iter->second;
  return OkStatus();
}

int PredictionToTrackStep(const MotionMetricsConfig& config,
                          int prediction_step) {
  const int ratio =
      config.track_steps_per_second() / config.prediction_steps_per_second();
  return (prediction_step + 1) * ratio + CurrentTrackStep(config);
}

// Track step corresponding to the current (not future) state.
int CurrentTrackStep(const MotionMetricsConfig& config) {
  return config.track_history_samples();
}

Polygon2d PredictionToPolygon(const MotionMetricsConfig& config,
                              const SingleTrajectory& trajectory,
                              int trajectory_step, const Track& track,
                              bool use_current_box_dimensions) {
  const Vec2d center(trajectory.center_x(trajectory_step),
                     trajectory.center_y(trajectory_step));

  // Compute the heading from the positions.
  double heading;
  if (trajectory.center_x_size() < 2) {
    return Polygon2d();
  }
  if (trajectory_step == 0) {
    heading = (Position(trajectory, 1) - Position(trajectory, 0)).Angle();
  } else if (trajectory_step == trajectory.center_x_size() - 1) {
    heading = (Position(trajectory, trajectory_step) -
               Position(trajectory, trajectory_step - 1))
                  .Angle();
  } else {
    // Compute the mean heading using the vectors from the previous and next
    // steps.
    const Vec2d previous = Position(trajectory, trajectory_step - 1);
    const Vec2d current = Position(trajectory, trajectory_step);
    const Vec2d next = Position(trajectory, trajectory_step + 1);
    heading = ((next - current).Angle() + (current - previous).Angle()) / 2.0;
  }

  const int track_step = use_current_box_dimensions
                             ? CurrentTrackStep(config)
                             : PredictionToTrackStep(config, trajectory_step);
  const ObjectState state = track.states(track_step);
  return Polygon2d(Box2d(center, heading, state.length(), state.width()));
}

Polygon2d StateToPolygon(const ObjectState& state) {
  const Vec2d center(state.center_x(), state.center_y());
  return Polygon2d(
      Box2d(center, state.heading(), state.length(), state.width()));
}

std::vector<float> GetNormalizedConfidences(
    const MultimodalPrediction& prediction) {
  const int num_predictions = prediction.joint_predictions_size();
  if (num_predictions == 0) {
    return {};
  }
  std::vector<float> result(num_predictions);
  float sum = 0.0;
  for (int i = 0; i < num_predictions; ++i) {
    const float confidence = prediction.joint_predictions(i).confidence();
    result[i] = confidence;
    sum += confidence;
  }

  // Normalize the confidences.
  float nominal = 1.0f / num_predictions;
  for (int i = 0; i < num_predictions; ++i) {
    result[i] = sum == 0.0 ? nominal : result[i] / sum;
  }
  return result;
}

}  // namespace open_dataset
}  // namespace waymo
