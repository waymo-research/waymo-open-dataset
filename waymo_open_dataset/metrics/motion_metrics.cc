/* Copyright 2021 The Waymo Open Dataset Authors. All Rights Reserved.

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

// Code to compute motion prediction metrics for the waymo open dataset.

#include "waymo_open_dataset/metrics/motion_metrics.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <unordered_map>

#include "google/protobuf/text_format.h"
#include "absl/container/flat_hash_map.h"
#include "absl/strings/str_cat.h"
#include "absl/types/optional.h"
#include "waymo_open_dataset/math/box2d.h"
#include "waymo_open_dataset/math/polygon2d.h"
#include "waymo_open_dataset/math/vec2d.h"
#include "waymo_open_dataset/metrics/motion_metrics_utils.h"
#include "waymo_open_dataset/protos/metrics.pb.h"
#include "waymo_open_dataset/protos/motion_submission.pb.h"
#include "waymo_open_dataset/protos/scenario.pb.h"

namespace waymo {
namespace open_dataset {
namespace {

const std::unordered_map<Track::ObjectType, int>& GetObjectTypePriority() {
  static const auto* priorities =
      new std::unordered_map<Track::ObjectType, int>({
          {Track::TYPE_UNSET, 0},
          {Track::TYPE_OTHER, 1},
          {Track::TYPE_VEHICLE, 2},
          {Track::TYPE_PEDESTRIAN, 3},
          {Track::TYPE_CYCLIST, 4},
      });
  return *priorities;
}

const std::unordered_map<TrajectoryType, int>& GetTrajectoryTypePriority() {
  static const auto* priorities = new std::unordered_map<TrajectoryType, int>(
      {{TrajectoryType::STATIONARY, 0},
       {TrajectoryType::STRAIGHT, 1},
       {TrajectoryType::STRAIGHT_RIGHT, 2},
       {TrajectoryType::STRAIGHT_LEFT, 3},
       {TrajectoryType::RIGHT_TURN, 4},
       {TrajectoryType::LEFT_TURN, 5},
       {TrajectoryType::LEFT_U_TURN, 6},
       {TrajectoryType::RIGHT_U_TURN, 7}});
  return *priorities;
}

// Validate the metrics configuration.
Status ValidateConfig(const MotionMetricsConfig& config) {
  // Validate the configuration.
  if (config.prediction_steps_per_second() > config.track_steps_per_second()) {
    return InvalidArgumentError(
        "prediction_steps_per_second cannot be greater than "
        "track_steps_per_second");
  }
  if (config.track_steps_per_second() % config.prediction_steps_per_second() !=
      0) {
    return InvalidArgumentError(
        "track_steps_per_second must be an integer multiple of "
        "prediction_steps_per_second");
  }

  const int prediction_steps = config.track_future_samples() *
                               config.prediction_steps_per_second() /
                               config.track_steps_per_second();
  for (const auto& step_config : config.step_configurations()) {
    if (step_config.measurement_step() > prediction_steps ||
        step_config.measurement_step() < 0) {
      return InvalidArgumentError("Requested miss rate time is invalid");
    }
  }
  return OkStatus();
}

// Validate a given MultimodalPrediction. Returns the set of object IDs that are
// predicted in each joint prediction.
Status ValidateMultiModalPrediction(
    const MultimodalPrediction& multi_modal_prediction,
    const int expected_trajectory_size, std::set<int>* prediction_ids) {
  // Use this set to validate that all predicted trajectories in the
  // multi_modal_prediction proto are for the same object.
  std::set<int> trajectory_ids;
  for (const auto& prediction : multi_modal_prediction.joint_predictions()) {
    // Validate the trajectory lengths.
    std::set<int> ids;
    for (const auto& trajectory : prediction.trajectories()) {
      if (trajectory.center_x_size() != expected_trajectory_size ||
          trajectory.center_y_size() != expected_trajectory_size) {
        return InvalidArgumentError(absl::StrCat(
            "Invalid predicted trajectory length. Expected : ",
            expected_trajectory_size, " found : ", trajectory.center_x_size(),
            " trajectory : ", trajectory.DebugString()));
      }
      prediction_ids->insert(trajectory.object_id());
      ids.insert(trajectory.object_id());
    }

    // Verify that every joint prediction contains the same objects.
    if (trajectory_ids.empty()) {
      trajectory_ids = ids;
    }
    if (ids != trajectory_ids) {
      return InvalidArgumentError(
          "Joint predictions do not contain the same predicted objects.");
    }
  }
  return OkStatus();
}

// Validates that predicted_object_ids contains all of the objects in the
// scenario tracks_to_predict field.
Status ValidateRequiredIds(const Scenario& scenario,
                           const std::set<int>& predicted_object_ids) {
  // Validate that the predictions contains trajectory predictions for the
  // required objects.
  std::set<int> required_object_ids;
  for (const auto& required_track : scenario.tracks_to_predict()) {
    const int track_index = required_track.track_index();
    if (track_index >= scenario.tracks_size() || track_index < 0) {
      return InvalidArgumentError("Internal error : Invalid track index : " +
                                  scenario.scenario_id());
    }
    const int required_id = scenario.tracks(track_index).id();
    required_object_ids.insert(required_id);
  }
  if (predicted_object_ids != required_object_ids) {
    std::string required;
    std::string predicted;
    for (const auto id : required_object_ids) {
      required += absl::StrCat(id, ", ");
    }
    for (const auto id : predicted_object_ids) {
      predicted += absl::StrCat(id, ", ");
    }
    return InvalidArgumentError(
        absl::StrCat("Missing required object prediction for scenario : ",
                     scenario.scenario_id(), " required : ", required,
                     " predicted : ", predicted));
  }
  return OkStatus();
}

// Validates that the scenario_predictions proto contains the required data for
// a behavior prediction submission.
Status ValidatePredictions(const MotionMetricsConfig& config,
                           const ScenarioPredictions& scenario_predictions,
                           const Scenario& scenario) {
  // Validate that the scenario IDs match.
  if (scenario_predictions.scenario_id() != scenario.scenario_id()) {
    return InvalidArgumentError(
        "Scenario IDs do not match : " + scenario_predictions.scenario_id() +
        " vs. " + scenario.scenario_id());
  }

  // Validate the predictions trajectory lengths and construct a set of the
  // predicted objects.
  const int trajectory_size = config.track_future_samples() *
                              config.prediction_steps_per_second() /
                              config.track_steps_per_second();

  // Create a set to validate that all required objects from the scenario have
  // a prediction.
  std::set<int> predicted_object_ids;
  std::set<int> prediction_ids;
  for (const auto& multi_modal_prediction :
       scenario_predictions.multi_modal_predictions()) {
    // Validate the prediction and store all IDs in each prediction.
    Status status = ValidateMultiModalPrediction(
        multi_modal_prediction, trajectory_size, &prediction_ids);
    if (!status.ok()) {
      return status;
    }
    predicted_object_ids.insert(prediction_ids.begin(), prediction_ids.end());
  }

  // Validate that there is a prediction for all required objects.
  return ValidateRequiredIds(scenario, predicted_object_ids);
}

// Validate that all of the given tracks have the required length.
Status ValidateTracks(const MotionMetricsConfig& config,
                      absl::flat_hash_map<int, const Track*>& ids_to_tracks) {
  const int track_size =
      config.track_history_samples() + config.track_future_samples() + 1;
  for (const auto& [id, track] : ids_to_tracks) {
    if (track->states_size() != track_size) {
      return InvalidArgumentError("Invalid Scenario track size.");
    }
  }
  return OkStatus();
}

// Validate that all predicted objects correspond to an object listed in the
// scenario tracks_to_predict or objects_of_interest fields.
Status ValidatePredictionIds(
    const ScenarioPredictions& predictions,
    absl::flat_hash_map<int, const Track*>& ids_to_tracks) {
  for (const auto& multi_modal_prediction :
       predictions.multi_modal_predictions()) {
    for (const auto& joint_prediction :
         multi_modal_prediction.joint_predictions()) {
      for (const auto& trajectory : joint_prediction.trajectories()) {
        if (ids_to_tracks.find(trajectory.object_id()) == ids_to_tracks.end()) {
          return InvalidArgumentError("Invalid predicted object ID found : " +
                                      absl::StrCat(trajectory.object_id()));
        }
      }
    }
  }
  return OkStatus();
}

using ConstTrackPtr = const Track*;

// Get the track with the given ID from ids_to_tracks. Returns an error if
// the ID is not found in the map.
Status GetTrack(int object_id,
                const absl::flat_hash_map<int, ConstTrackPtr>& ids_to_tracks,
                ConstTrackPtr* track) {
  const auto iter = ids_to_tracks.find(object_id);
  if (iter == ids_to_tracks.end()) {
    return InvalidArgumentError(absl::StrCat(
        "Invalid prediction object ID. Track not found for : ", object_id));
  }
  *track = iter->second;
  return OkStatus();
}

// Returns the track step corresponding to a given prediction step.
int PredictionToTrack(const MotionMetricsConfig& config, int prediction_step) {
  const int ratio =
      config.track_steps_per_second() / config.prediction_steps_per_second();
  return (prediction_step + 1) * ratio + config.track_history_samples();
}

// Computes L2 deviation of the trajectory from the track at the given step.
// Returns nullopt if the track state is not valid at the given step.
absl::optional<double> DisplacementAtStep(const MotionMetricsConfig& config,
                                          const SingleTrajectory& trajectory,
                                          const Track& track,
                                          int trajectory_step) {
  const int track_step = PredictionToTrack(config, trajectory_step);
  const ObjectState& track_state = track.states(track_step);
  if (!track_state.valid()) {
    return absl::nullopt;
  }
  // Compute the deviation at the given time step.
  const double dx =
      trajectory.center_x(trajectory_step) - track_state.center_x();
  const double dy =
      trajectory.center_y(trajectory_step) - track_state.center_y();
  return std::hypot(dx, dy);
}

// Displacement error in the vehicle frame.
struct Displacement {
  Displacement() {}
  Displacement(double lat, double lon) : lateral(lat), longitudinal(lon) {}

  // The lateral and longitudinal displacement in the agent frame.
  double lateral = std::numeric_limits<double>::max();
  double longitudinal = std::numeric_limits<double>::max();
};

// Computes lateral and longitudinal displacement of the trajectory from the
// track at the given step. Returns nullopt if the track state is not valid at
// the given step.
absl::optional<Displacement> GetDisplacementAtStep(
    const MotionMetricsConfig& config, const SingleTrajectory& trajectory,
    const Track& track, int trajectory_step) {
  const int track_step = PredictionToTrack(config, trajectory_step);
  const ObjectState& track_state = track.states(track_step);
  if (!track_state.valid()) {
    return absl::nullopt;
  }

  // Compute the displacement vector at the given time step.
  const double dx =
      trajectory.center_x(trajectory_step) - track_state.center_x();
  const double dy =
      trajectory.center_y(trajectory_step) - track_state.center_y();

  // Transform the displacement into the agent frame.
  Vec2d displacement = Vec2d(dx, dy).Rotate(-track_state.heading());

  // In the agent frame lateral displacement is along the Y axis, and
  // longitudinal displacement is along the X axis.
  return Displacement(displacement.y(), displacement.x());
}

// Computes the average displacement of a trajectory from the given ground truth
// track from index 0 to max_prediction_step. The trajectory and track lengths
// must be previously verified.
absl::optional<double> AverageDisplacement(const MotionMetricsConfig& config,
                                           const SingleTrajectory& trajectory,
                                           const Track& track,
                                           const int max_prediction_step) {
  double error = 0.0;
  int state_count = 0;
  for (int i = 0; i <= max_prediction_step; ++i) {
    const ObjectState& track_state = track.states(PredictionToTrack(config, i));
    if (track_state.valid()) {
      const double dx = trajectory.center_x(i) - track_state.center_x();
      const double dy = trajectory.center_y(i) - track_state.center_y();
      error += std::hypot(dx, dy);
      ++state_count;
    }
  }
  if (state_count == 0) {
    return absl::nullopt;
  }
  return error / state_count;
}

// Computes the ADE or FDE for a joint prediction (the mean of the ADE or FDE
// values for all joint trajectories). If compute_final_displacement is true,
// the displacement value is only computed at the time step given by
// prediction_step. Otherwise the average displacement is computed up to
// prediction_step. Returns nullopt if there are not enough valid states to
// compute the displacement for any of the trajectories.
Status AverageDisplacement(
    const MotionMetricsConfig& config,
    const JointTrajectories& joint_prediction,
    const absl::flat_hash_map<int, const Track*>& ids_to_tracks,
    const int prediction_step, bool compute_final_displacement,
    absl::optional<double>* result) {
  double total_joint_ade = 0.0;
  int ade_count = 0;
  for (const auto& trajectory : joint_prediction.trajectories()) {
    // Get the track associated with the trajectory prediction.
    ConstTrackPtr track;
    Status status = GetTrack(trajectory.object_id(), ids_to_tracks, &track);
    if (!status.ok()) {
      return InternalError(
          "Internal error : object ID not found in ids_to_tracks.");
    }
    absl::optional<double> ade =
        compute_final_displacement
            ? DisplacementAtStep(config, trajectory, *track, prediction_step)
            : AverageDisplacement(config, trajectory, *track, prediction_step);

    // If one trajectory has no valid states to compute ADE, bypass the rest
    // of the trajectories in the joint prediction.
    if (!ade) {
      break;
    }

    total_joint_ade += *ade;
    ++ade_count;
  }

  // If all trajectories in the joint prediction were successfully evaluated,
  // return the mean ADE value.
  if (ade_count != joint_prediction.trajectories_size()) {
    *result = absl::nullopt;
  } else {
    *result = total_joint_ade / ade_count;
  }
  return OkStatus();
}

// Computes the minimum average displacement error (minADE) metric or final
// average displacement error(minFDE). If compute_final_displacement is true,
// the displacements will only be computed at the given time step
// (the min FDE), otherwise the average displacement up to prediction_step will
// be used (the min ADE).
Status ComputeMinAverageDisplacement(
    const MotionMetricsConfig& config,
    const MultimodalPrediction& multi_modal_prediction,
    const absl::flat_hash_map<int, const Track*>& ids_to_tracks,
    const int prediction_step, const bool compute_final_displacement,
    Accumulator* result) {
  // Find the joint prediction with the minimum ADE for this track.
  double min_joint_ade = std::numeric_limits<double>::max();
  int count = 0;
  for (const auto& joint_prediction :
       multi_modal_prediction.joint_predictions()) {
    // Compute the ADE for this joint prediction.
    absl::optional<double> joint_ade;
    Status status = AverageDisplacement(config, joint_prediction, ids_to_tracks,
                                        prediction_step,
                                        compute_final_displacement, &joint_ade);
    if (!status.ok()) {
      return status;
    }

    // Keep the minimum valid ADE value.
    if (joint_ade && *joint_ade < min_joint_ade) {
      min_joint_ade = *joint_ade;
    }

    // Only evaluate the desired K predictions.
    if (++count == config.max_predictions()) {
      break;
    }
  }

  // If the minADE was successfully computed, add a single measurement for this
  // joint prediction. If not, it means that valid ground truth data was not
  // available to compute the metric.
  if (min_joint_ade != std::numeric_limits<double>::max()) {
    result->Add(min_joint_ade);
  }
  return OkStatus();
}

// Computes a scale factor for the miss rate thresholds that varies with the
// track's initial speed.
Status SpeedScaleFactor(const MotionMetricsConfig& config, const Track& track,
                        double* result) {
  if (track.states_size() < config.track_history_samples()) {
    return InvalidArgumentError("Internal Error : Track length is invalid.");
  }

  if (config.speed_lower_bound() >= config.speed_upper_bound()) {
    return InvalidArgumentError(
        "Speed upper bound must be greater than the speed lower bound.");
  }

  const ObjectState& state = track.states(config.track_history_samples());
  const double speed = std::hypot(state.velocity_x(), state.velocity_y());
  if (speed < config.speed_lower_bound()) {
    *result = config.speed_scale_lower();
    return OkStatus();
  }
  if (speed > config.speed_upper_bound()) {
    *result = config.speed_scale_upper();
    return OkStatus();
  }

  // Linearly interpolate in between the bounds.
  const double fraction =
      (speed - config.speed_lower_bound()) /
      (config.speed_upper_bound() - config.speed_lower_bound());
  *result =
      config.speed_scale_lower() +
      (config.speed_scale_upper() - config.speed_scale_lower()) * fraction;
  return OkStatus();
}

Status IsTruePositive(
    const MotionMetricsConfig& config,
    const JointTrajectories& joint_prediction,
    const absl::flat_hash_map<int, const Track*>& ids_to_tracks,
    const MotionMetricsConfig::MeasurementStepConfig& step_config,
    absl::optional<bool>* is_tp) {
  // Compute the displacements at prediction_step for all trajectories.
  const int num_trajectories = joint_prediction.trajectories_size();
  std::vector<Displacement> displacements(num_trajectories);
  int measurement_count = 0;
  for (int index = 0; index < num_trajectories; ++index) {
    const SingleTrajectory& trajectory = joint_prediction.trajectories(index);
    ConstTrackPtr track_ptr;
    auto status = GetTrack(trajectory.object_id(), ids_to_tracks, &track_ptr);
    if (!status.ok()) {
      return status;
    }
    absl::optional<Displacement> displacement = GetDisplacementAtStep(
        config, trajectory, *track_ptr, step_config.measurement_step());

    // If one trajectory has no valid states to compute the displacement,
    // bypass the rest of the trajectories in the joint prediction.
    if (!displacement) {
      break;
    }

    // Compute the velocity scale factor for the thresholds.
    double scale;
    status = SpeedScaleFactor(config, *track_ptr, &scale);
    if (!status.ok()) {
      return status;
    }

    // Scale the displacement by the inverse of the speed scale to allow
    // thresholding with the original threshold.
    displacement->lateral /= scale;
    displacement->longitudinal /= scale;

    // Store the scaled displacements for each trajectory.
    displacements[index] = *displacement;
    ++measurement_count;
  }

  // Return absl::nullopt if there are not enough valid track states to compute
  // the displacements.
  if (measurement_count != num_trajectories) {
    *is_tp = absl::nullopt;
    return OkStatus();
  }

  // Only return true if all trajectories in the joint prediction are within the
  // miss rate thresholds;
  bool miss = false;
  for (const Displacement displacement : displacements) {
    if (std::abs(displacement.lateral) > step_config.lateral_miss_threshold() ||
        std::abs(displacement.longitudinal) >
            step_config.longitudinal_miss_threshold()) {
      miss = true;
    }
  }
  *is_tp = !miss;
  return OkStatus();
}

// Computes the miss rate metric.
Status ComputeMissRate(
    const MotionMetricsConfig& config,
    const MotionMetricsConfig::MeasurementStepConfig& step_config,
    const MultimodalPrediction& multi_modal_prediction,
    const absl::flat_hash_map<int, const Track*>& ids_to_tracks,
    Accumulator* result) {
  // Determine if all trajectories within each joint prediction are within the
  // miss rate radius at the prediction step.
  bool has_valid_measurements = false;
  int count = 0;
  for (const auto& joint_prediction :
       multi_modal_prediction.joint_predictions()) {
    absl::optional<bool> true_positive;
    Status status = IsTruePositive(config, joint_prediction, ids_to_tracks,
                                   step_config, &true_positive);
    if (!status.ok()) {
      return status;
    }
    if (true_positive.has_value()) {
      // If a true positive is found add a single negative miss rate
      // measurement.
      if (*true_positive) {
        result->Add(0.0);
        return OkStatus();
      }
      has_valid_measurements = true;
    }

    // Only evaluate the desired K predictions.
    if (++count == config.max_predictions()) {
      break;
    }
  }

  // If no valid measurements were made, return an empty accumulator.
  if (!has_valid_measurements) {
    return OkStatus();
  }

  // If all joint trajectories are misses, add a single 1.0 miss rate
  // measurement.
  result->Add(1.0);
  return OkStatus();
}

// Returns a normalized vector of the prediction trajectory confidences.
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

// Returns a vector position for the given step in a trajectory.
Vec2d Position(const SingleTrajectory& trajectory, int step) {
  return Vec2d(trajectory.center_x(step), trajectory.center_y(step));
}

// Returns a bounding box polygon for the given trajectory at the given step
// using the dimensions of the original track box.
Polygon2d PredictionToPolygon(const MotionMetricsConfig& config,
                              const SingleTrajectory& trajectory,
                              int trajectory_step, const Track& track) {
  const int track_step = PredictionToTrack(config, trajectory_step);
  const ObjectState state = track.states(track_step);
  const Vec2d center(trajectory.center_x(trajectory_step),
                     trajectory.center_y(trajectory_step));

  // Compute the heading from the positions.
  double heading;
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
  return Polygon2d(Box2d(center, heading, state.length(), state.width()));
}

// Returns a bounding box polygon for the given object state.
Polygon2d StateToPolygon(const ObjectState& state) {
  const Vec2d center(state.center_x(), state.center_y());
  return Polygon2d(
      Box2d(center, state.heading(), state.length(), state.width()));
}

// Computes the overlap rate for a single MultimodalPrediction.
Status ComputeOverlapRate(
    const MotionMetricsConfig& config, int last_trajectory_step,
    const MultimodalPrediction& prediction,
    const absl::flat_hash_map<int, const Track*>& ids_to_tracks,
    const Scenario& scenario, Accumulator* result) {
  // Get a list of objects that are valid at the prediction time.
  const int current_time_track_step = config.track_history_samples();
  std::vector<int> tracks_to_test_indices;
  for (int i = 0; i < scenario.tracks_size(); ++i) {
    if (scenario.tracks(i).states(current_time_track_step).valid()) {
      tracks_to_test_indices.push_back(i);
    }
  }

  // Find the highest probability trajectory prediction.
  std::vector<float> confidences = GetNormalizedConfidences(prediction);

  // Only include the desired K predictions.
  confidences.resize(config.max_predictions());
  const int max_index =
      std::distance(confidences.begin(),
                    std::max_element(confidences.begin(), confidences.end()));

  // Check for overlaps up to the given step.
  const JointTrajectories& joint_prediction =
      prediction.joint_predictions(max_index);
  for (const auto& trajectory : joint_prediction.trajectories()) {
    for (int step = 0; step <= last_trajectory_step; ++step) {
      ConstTrackPtr track_ptr;
      Status status =
          GetTrack(trajectory.object_id(), ids_to_tracks, &track_ptr);
      if (!status.ok()) {
        return status;
      }
      const Track& prediction_track = *track_ptr;
      const Polygon2d prediction_box =
          PredictionToPolygon(config, trajectory, step, prediction_track);
      for (const int track_index : tracks_to_test_indices) {
        const Track& track = scenario.tracks(track_index);
        if (track.id() == prediction_track.id()) {
          continue;
        }
        const int track_step = PredictionToTrack(config, step);
        const ObjectState state = track.states(track_step);
        if (!state.valid()) {
          continue;
        }
        const Polygon2d track_box = StateToPolygon(track.states(track_step));

        // If a overlap is found return a single 1.0 accumulated measurement.
        if (prediction_box.MaybeHasIntersectionWith(track_box) &&
            prediction_box.ComputeIntersectionArea(track_box) > 0) {
          result->Add(1.0);
          return OkStatus();
        }
      }
    }
  }

  // If no overlaps were found accumulate a 0.0 measurement.
  result->Add(0.0);
  return OkStatus();
}

// Return the indices of the predicted trajectories in descending order of
// confidence. This only includes the first K trajectories as defined in config.
std::vector<int> GetSortedTrajectoryIndices(
    const MotionMetricsConfig& config, const MultimodalPrediction& prediction) {
  const int num_predictions =
      std::min(prediction.joint_predictions_size(), config.max_predictions());
  std::vector<int> result(num_predictions);
  for (int i = 0; i < num_predictions; ++i) {
    result[i] = i;
  }
  std::sort(result.begin(), result.end(), [prediction](int i1, int i2) {
    return prediction.joint_predictions(i1).confidence() >
           prediction.joint_predictions(i2).confidence();
  });
  return result;
}

// Computes the mean average precision stats for a single trajectory at the
// given prediction step.
Status ComputeMeanAveragePrecision(
    const MotionMetricsConfig& config,
    const MotionMetricsConfig::MeasurementStepConfig& step_config,
    const MultimodalPrediction& prediction,
    const absl::flat_hash_map<int, const Track*>& ids_to_tracks,
    MeanAveragePrecisionStats* result) {
  // Initialize all bucket vector sizes and initialize all threshold stats.
  result->pr_buckets.resize(TrajectoryType::NUM_TYPES);

  // Determine the trajectory type bucket for the first track in the first
  // joint prediction.
  if (prediction.joint_predictions().empty() ||
      prediction.joint_predictions(0).trajectories().empty()) {
    return OkStatus();
  }

  // Find the trajectory_type that best describes the prediction. If it is a
  // joint prediction, the trajectory_type is selected according pre-defined
  // priorities among all agents.
  TrajectoryType type = TrajectoryType::STATIONARY;
  bool valid_type_found = false;

  const std::unordered_map<TrajectoryType, int>& priorities =
      GetTrajectoryTypePriority();

  for (const auto& trajectory :
       prediction.joint_predictions(0).trajectories()) {
    ConstTrackPtr track_ptr;

    Status status = GetTrack(trajectory.object_id(), ids_to_tracks, &track_ptr);
    if (!status.ok()) {
      return status;
    }

    absl::optional<TrajectoryType> trajectory_type =
        ClassifyTrack(config.track_history_samples(), *track_ptr);

    if (trajectory_type) {
      if (priorities.find(*trajectory_type) == priorities.end()) {
        return InvalidArgumentError(
            absl::StrCat("trajectory_type", *trajectory_type,
                         " not defined in GetObjectTypePriority"));
      }

      if (priorities.at(*trajectory_type) > priorities.at(type)) {
        type = *trajectory_type;
      }
      valid_type_found = true;
    }
  }

  // If the tracks do not have enough valid states, return an empty result.
  if (!valid_type_found) {
    return OkStatus();
  }

  // Bin right u-turns with right turns.
  if (type == TrajectoryType::RIGHT_U_TURN) {
    type = TrajectoryType::RIGHT_TURN;
  }

  // Compute the P/R stats for the predictions.
  PredictionStats& bucket = result->pr_buckets[type];
  std::vector<int> sorted_indices =
      GetSortedTrajectoryIndices(config, prediction);
  bool already_found_positive = false;
  bool measurement_taken = false;
  for (int joint_prediction_index : sorted_indices) {
    const JointTrajectories& joint_prediction =
        prediction.joint_predictions(joint_prediction_index);

    // Determine if the prediction is a true positive.
    absl::optional<bool> is_true_positive_opt;
    Status status = IsTruePositive(config, joint_prediction, ids_to_tracks,
                                   step_config, &is_true_positive_opt);
    if (!status.ok()) {
      return status;
    }
    if (!is_true_positive_opt.has_value()) {
      continue;
    }
    const bool is_true_positive = is_true_positive_opt.value();

    // Store a true positive only if a true positive has not yet been stored
    // for these joint object tracks.
    const bool sample_result =
        already_found_positive ? false : is_true_positive;
    bucket.samples.push_back(
        PredictionSample(joint_prediction.confidence(), sample_result));
    if (is_true_positive) {
      already_found_positive = true;
    }
    measurement_taken = true;
  }

  // If a measurement was taken, increment the total number of possible
  // true positives in the set.
  if (measurement_taken) {
    bucket.num_trajectories += 1;
  }
  return OkStatus();
}

// Computes accumulated metrics stats for all predictions in the given
// ScenarioPredictions proto.
Status ComputeAllMetrics(
    const MotionMetricsConfig& config, const ScenarioPredictions& predictions,
    const absl::flat_hash_map<int, const Track*>& ids_to_tracks,
    const Scenario& scenario, BucketedMetricsStats* result) {
  // Accumulate the metric values across all object predictions.
  for (const auto& multi_modal_prediction :
       predictions.multi_modal_predictions()) {
    // Determine the object type bucket for the first track in the first
    // joint prediction.
    if (multi_modal_prediction.joint_predictions().empty() ||
        multi_modal_prediction.joint_predictions(0).trajectories().empty()) {
      return OkStatus();
    }

    // Find the object_type that best describes the prediction. If it is a joint
    // prediction, the type is selected according predefined priorities among
    // all agents.
    Track::ObjectType object_type = Track::TYPE_UNSET;

    const std::unordered_map<Track::ObjectType, int>& priorities =
        GetObjectTypePriority();

    for (const auto& trajectory :
         multi_modal_prediction.joint_predictions(0).trajectories()) {
      ConstTrackPtr track_ptr;
      Status status =
          GetTrack(trajectory.object_id(), ids_to_tracks, &track_ptr);
      if (!status.ok()) {
        return status;
      }

      if (priorities.find(track_ptr->object_type()) == priorities.end()) {
        return InvalidArgumentError(
            absl::StrCat("object_type ", track_ptr->object_type(),
                         " not defined in GetObjectTypePriority"));
      }

      if (priorities.at(track_ptr->object_type()) >
          priorities.at(object_type)) {
        object_type = track_ptr->object_type();
      }
    }

    // Compute the per time step metrics.
    for (const auto& step_config : config.step_configurations()) {
      const int evaluation_step = step_config.measurement_step();

      // Accumulate the results into the stats for this object's type.
      std::map<int, MetricsStats>& step_to_metric_stats =
          result->stats[object_type];
      MetricsStats& metrics_stats = step_to_metric_stats[evaluation_step];

      Status status;

      // Compute the minADE metric.
      Accumulator min_ade_stats;
      status = ComputeMinAverageDisplacement(
          config, multi_modal_prediction, ids_to_tracks, evaluation_step,
          /*compute_final_displacement=*/false, &min_ade_stats);
      if (!status.ok()) {
        return status;
      }
      metrics_stats.min_average_displacement.Accumulate(min_ade_stats);

      // Compute the minFDE metrics.
      Accumulator fde_stats;
      status = ComputeMinAverageDisplacement(
          config, multi_modal_prediction, ids_to_tracks, evaluation_step,
          /*compute_final_displacement=*/true, &fde_stats);
      if (!status.ok()) {
        return status;
      }
      metrics_stats.min_final_displacement.Accumulate(fde_stats);

      // Compute the miss rate metrics.
      Accumulator miss_rate_stats;
      status = ComputeMissRate(config, step_config, multi_modal_prediction,
                               ids_to_tracks, &miss_rate_stats);
      if (!status.ok()) {
        return status;
      }
      metrics_stats.miss_rate.Accumulate(miss_rate_stats);

      // Compute the overlap rate metrics.
      Accumulator overlap_stats;
      status =
          ComputeOverlapRate(config, evaluation_step, multi_modal_prediction,
                             ids_to_tracks, scenario, &overlap_stats);
      if (!status.ok()) {
        return status;
      }
      metrics_stats.overlap_rate.Accumulate(overlap_stats);

      // Compute the mean average precision metrics.
      MeanAveragePrecisionStats mean_average_precision_stats;
      status = ComputeMeanAveragePrecision(
          config, step_config, multi_modal_prediction, ids_to_tracks,
          &mean_average_precision_stats);
      if (!status.ok()) {
        return status;
      }
      metrics_stats.mean_average_precision.Accumulate(
          mean_average_precision_stats);
    }
  }
  return OkStatus();
}

}  // namespace

void MetricsStats::Accumulate(const MetricsStats& metrics_stats) {
  min_average_displacement.Accumulate(metrics_stats.min_average_displacement);
  min_final_displacement.Accumulate(metrics_stats.min_final_displacement);
  miss_rate.Accumulate(metrics_stats.miss_rate);
  overlap_rate.Accumulate(metrics_stats.overlap_rate);
  mean_average_precision.Accumulate(metrics_stats.mean_average_precision);
}

void PredictionStats::Accumulate(const PredictionStats& prediction_stats) {
  samples.insert(samples.end(), prediction_stats.samples.begin(),
                 prediction_stats.samples.end());
  num_trajectories += prediction_stats.num_trajectories;
}

void MeanAveragePrecisionStats::Accumulate(
    const MeanAveragePrecisionStats& mean_ap_stats) {
  if (mean_ap_stats.pr_buckets.empty()) {
    return;
  }
  if (pr_buckets.empty()) {
    pr_buckets.resize(mean_ap_stats.pr_buckets.size());
  }
  CHECK_EQ(pr_buckets.size(), mean_ap_stats.pr_buckets.size());
  for (int i = 0; i < pr_buckets.size(); ++i) {
    pr_buckets[i].Accumulate(mean_ap_stats.pr_buckets[i]);
  }
}

void BucketedMetricsStats::Accumulate(
    const BucketedMetricsStats& metrics_stats) {
  for (const auto& [type, step_to_stats] : metrics_stats.stats) {
    for (const auto& [step, other_stats] : step_to_stats) {
      stats[type][step].Accumulate(other_stats);
    }
  }
}

std::map<int, MetricsStats> BucketedMetricsStats::AccumulateAcrossTypes() {
  std::map<int, MetricsStats> result;
  for (const auto& [type, step_to_stats] : stats) {
    for (const auto& [step, metrics_stats] : step_to_stats) {
      result[step].Accumulate(metrics_stats);
    }
  }
  return result;
}

// Sorts samples first by confidence and second by true positive status such
// that for samples with identical confidences the false positives are listed
// first.
void SortSamples(std::vector<PredictionSample>* samples_ptr) {
  CHECK(samples_ptr != nullptr);
  std::vector<PredictionSample>& samples = *samples_ptr;
  std::sort(
      samples.begin(), samples.end(),
      [](const PredictionSample& sample1, const PredictionSample& sample2) {
        if (sample1.confidence != sample2.confidence) {
          return sample1.confidence > sample2.confidence;
        }
        // Secondary sort to move false positives in front of true positives.
        return !sample1.true_positive && sample2.true_positive;
      });
}

std::vector<PrSample> ComputePrCurve(
    std::vector<PredictionSample>* prediction_samples, int total_gt_count) {
  // Sort samples by confidence and secondarily to move false positives ahead
  // of true positives for samples with equal confidence.
  SortSamples(prediction_samples);
  std::vector<PredictionSample>& samples = *prediction_samples;

  const int num_samples = samples.size();
  std::vector<PrSample> result(num_samples);
  int true_positives = 0;
  for (int i = 0; i < num_samples; ++i) {
    // Compute cumulative true positives.
    if (samples[i].true_positive) {
      true_positives += 1;
    }
    float positives = static_cast<float>(true_positives);
    result[i].precision = positives / (i + 1);
    result[i].recall = positives / total_gt_count;
  }
  return result;
}

// Compute the mean average precision metric for a set of samples. All samples
// in prediction_samples will be sorted by confidence.
double ComputeMeanAveragePrecision(PredictionStats* prediction_samples) {
  if (prediction_samples->samples.empty()) {
    return 0.0;
  }
  // Compute the precision recall curve.
  std::vector<PrSample> pr_curve = ComputePrCurve(
      &prediction_samples->samples, prediction_samples->num_trajectories);

  // Compute the mean average precision.
  const int num_samples = pr_curve.size();
  PrSample highest = pr_curve[num_samples - 1];
  double total_area = 0;
  for (int i = num_samples - 1; i >= 0; --i) {
    if (pr_curve[i].precision > highest.precision) {
      total_area += highest.precision * (highest.recall - pr_curve[i].recall);
      highest = pr_curve[i];
    }
  }

  // Add the remaining area at the far left of the curve.
  total_area += highest.recall * highest.precision;
  return total_area;
}

double ComputeMapMetric(MeanAveragePrecisionStats* stats) {
  double total = 0;
  int count = 0;
  for (auto& bucket : stats->pr_buckets) {
    if (!bucket.samples.empty()) {
      total += ComputeMeanAveragePrecision(&bucket);
      ++count;
    }
  }
  if (count == 0) {
    return 0.0;
  }
  return total / count;
}

Status ComputeMetricsStats(const MotionMetricsConfig& config,
                           const ScenarioPredictions& predictions,
                           const Scenario& scenario,
                           BucketedMetricsStats* result) {
  Status status = ValidateConfig(config);
  if (!status.ok()) {
    return status;
  }

  // Validate the contents of the predictions proto.
  status = ValidatePredictions(config, predictions, scenario);
  if (!status.ok()) {
    return status;
  }

  // Create a map from object ID to The object's Track for the objects to be
  // predicted.
  absl::flat_hash_map<int, const Track*> ids_to_tracks;
  // Create a map of tracks for the tracks_to_predict field.
  for (const auto& required_track : scenario.tracks_to_predict()) {
    const int track_index = required_track.track_index();
    if (track_index >= scenario.tracks_size() || track_index < 0) {
      return InvalidArgumentError("Invalid track index in scenario.");
    }
    const Track& track = scenario.tracks(track_index);
    ids_to_tracks[track.id()] = &scenario.tracks(track_index);
  }

  status = ValidateTracks(config, ids_to_tracks);
  if (!status.ok()) {
    return status;
  }

  // Validate that all predicted IDs exist in the ids_to_tracks map.
  status = ValidatePredictionIds(predictions, ids_to_tracks);
  if (!status.ok()) {
    return status;
  }

  // Compute the metrics for the predictions.
  status =
      ComputeAllMetrics(config, predictions, ids_to_tracks, scenario, result);
  if (!status.ok()) {
    return status;
  }

  return OkStatus();
}

namespace {

// Compute a MotionMetricsBundle from a MetricsStats object for the given
// measurement step and object type. All precision measurements in stats will
// be sorted by confidence.
void ComputeBundle(int step, Track::ObjectType object_type, MetricsStats* stats,
                   MotionMetricsBundle* bundle) {
  bundle->set_measurement_step(step);
  if (object_type != Track::TYPE_UNSET) {
    bundle->set_object_filter(object_type);
  }
  bundle->set_min_ade(stats->min_average_displacement.Mean());
  bundle->set_min_fde(stats->min_final_displacement.Mean());
  bundle->set_miss_rate(stats->miss_rate.Mean());
  bundle->set_overlap_rate(stats->overlap_rate.Mean());
  bundle->set_mean_average_precision(
      ComputeMapMetric(&stats->mean_average_precision));
}

}  // namespace

MotionMetrics ComputeMotionMetrics(BucketedMetricsStats* total_stats) {
  // Compute the metrics values from the accumulated statistics for each type
  // and each measurement step.
  MotionMetrics metrics;
  for (auto& [object_type, step_to_stats] : total_stats->stats) {
    for (auto& [step, stats] : step_to_stats) {
      MotionMetricsBundle* bundle = metrics.add_metrics_bundles();
      ComputeBundle(step, object_type, &stats, bundle);
    }
  }
  return metrics;
}

MotionMetricsConfig GetChallengeConfig() {
  const std::string scenario_str = R"(
    track_steps_per_second: 10
    prediction_steps_per_second: 2
    track_history_samples: 10
    track_future_samples: 80
    speed_lower_bound: 1.4
    speed_upper_bound: 11.0
    speed_scale_lower: 0.5
    speed_scale_upper: 1.0
    step_configurations {
      measurement_step: 5
      lateral_miss_threshold: 1.0
      longitudinal_miss_threshold: 2.0
    }
    step_configurations {
      measurement_step: 9
      lateral_miss_threshold: 1.8
      longitudinal_miss_threshold: 3.6
    }
    step_configurations {
      measurement_step: 15
      lateral_miss_threshold: 3.0
      longitudinal_miss_threshold: 6.0
    }
    max_predictions: 6
  )";

  MotionMetricsConfig result;
  CHECK(google::protobuf::TextFormat::ParseFromString(scenario_str, &result));
  return result;
}

Status ComputeMotionMetrics(
    const MotionMetricsConfig& config,
    const absl::flat_hash_map<std::string, ScenarioPredictions>& predictions,
    const absl::flat_hash_map<std::string, Scenario>& scenarios,
    MotionMetrics* metrics) {
  BucketedMetricsStats total_stats;

  // Compute statistics for all prediction, scenario pairs.
  for (const auto& [key, predictions] : predictions) {
    auto iter = scenarios.find(key);
    if (iter == scenarios.end()) {
      return InvalidArgumentError("Scenario not found.  Id : " + key);
    }
    const Scenario& scenario = iter->second;

    BucketedMetricsStats metrics_stats;
    Status status =
        ComputeMetricsStats(config, predictions, scenario, &metrics_stats);
    if (!status.ok()) {
      return status;
    }
    total_stats.Accumulate(metrics_stats);
  }

  // Compute the metrics values from the accumulated statistics.
  *metrics = ComputeMotionMetrics(&total_stats);
  return OkStatus();
}

}  // namespace open_dataset
}  // namespace waymo
