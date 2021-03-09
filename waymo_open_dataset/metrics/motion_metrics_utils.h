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

#ifndef WAYMO_OPEN_DATASET_METRICS_MOTION_METRICS_UTILS_H_
#define WAYMO_OPEN_DATASET_METRICS_MOTION_METRICS_UTILS_H_

#include "absl/types/optional.h"
#include "waymo_open_dataset/common/status.h"
#include "waymo_open_dataset/protos/motion_metrics.pb.h"
#include "waymo_open_dataset/protos/motion_submission.pb.h"
#include "waymo_open_dataset/protos/scenario.pb.h"

namespace waymo {
namespace open_dataset {

// Trajectory classes used to compute the mean average precision metric.
enum TrajectoryType {
  STATIONARY = 0,
  STRAIGHT = 1,
  STRAIGHT_LEFT = 2,
  STRAIGHT_RIGHT = 3,
  LEFT_U_TURN = 4,
  LEFT_TURN = 5,
  RIGHT_U_TURN = 6,
  RIGHT_TURN = 7,
  NUM_TYPES  // The max value of a TrajectoryType enum. Must be last.
};

// Returns the classification bucket for the trajectory. Returns nullopt if the
// track does not have enough valid states to classify.
absl::optional<TrajectoryType> ClassifyTrack(const int prediction_step,
                                             const Track& track);

// Validates that the given scenario predictions proto has the proper number
// of predictions and trajectories per prediction.
Status ValidateChallengePredictions(
    const ScenarioPredictions& scenario_predictions,
    MotionChallengeSubmission::SubmissionType submission_type);

// Converts a ChallengeScenarioPredictions to a ScenarioPredictions for metrics
// computations.
Status ConvertChallengePredictions(
    const ChallengeScenarioPredictions& predictions,
    ScenarioPredictions* result);

}  // namespace open_dataset
}  // namespace waymo

#endif  // WAYMO_OPEN_DATASET_METRICS_MOTION_METRICS_UTILS_H_
