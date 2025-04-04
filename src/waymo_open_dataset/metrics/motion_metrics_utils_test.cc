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

#include <vector>

#include <glog/logging.h>
#include "google/protobuf/text_format.h"
#include "google/protobuf/util/message_differencer.h"
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "waymo_open_dataset/math/vec2d.h"
#include "waymo_open_dataset/protos/motion_metrics.pb.h"
#include "waymo_open_dataset/protos/motion_submission.pb.h"

namespace waymo {
namespace open_dataset {
namespace {

// Creates a track from a given set of points. Headings and velocities are
// computed from the points.
Track CreateTrackFromPoints(const std::vector<Vec2d>& points) {
  Track track;
  track.set_id(0);
  track.set_object_type(Track::TYPE_VEHICLE);
  for (const auto& point : points) {
    ObjectState& state = *track.add_states();
    state.set_center_x(point.x());
    state.set_center_y(point.y());
    state.set_valid(true);
  }

  // Compute velocities and headings.
  for (int i = 0; i < track.states_size(); ++i) {
    if (i < track.states_size() - 1) {
      const ObjectState& cur_state = track.states(i);
      const ObjectState& next_state = track.states(i + 1);
      Vec2d delta = Vec2d(next_state.center_x(), next_state.center_y()) -
                    Vec2d(cur_state.center_x(), cur_state.center_y());
      track.mutable_states(i)->set_heading(delta.Angle());
      track.mutable_states(i)->set_velocity_x(delta.x() * 10);
      track.mutable_states(i)->set_velocity_y(delta.y() * 10);
    } else {
      track.mutable_states(i)->set_heading(track.states(i - 1).heading());
      track.mutable_states(i)->set_velocity_x(track.states(i - 1).velocity_x());
      track.mutable_states(i)->set_velocity_y(track.states(i - 1).velocity_y());
    }
  }
  return track;
}

// Rotates a track by a given angle.
Track RotateTrack(const Track& track, const double angle) {
  Track result;
  result.set_id(track.id());
  result.set_object_type(track.object_type());
  const Vec2d s0(track.states(0).center_x(), track.states(0).center_y());
  for (const auto& track_state : track.states()) {
    ObjectState& state = *result.add_states();
    Vec2d new_point =
        (Vec2d(track_state.center_x(), track_state.center_y()) - s0)
            .Rotated(angle) +
        s0;
    state.set_center_x(new_point.x());
    state.set_center_y(new_point.y());
    Vec2d v = Vec2d(track_state.velocity_x(), track_state.velocity_y())
                  .Rotated(angle);
    state.set_velocity_x(v.x());
    state.set_velocity_y(v.y());
    state.set_heading(track_state.heading() + angle);
    state.set_valid(true);
  }
  return result;
}

// Returns the given set of points rotated by a given angle in radians.
std::vector<Vec2d> RotatePoints(const std::vector<Vec2d>& points,
                                double angle_radians) {
  std::vector<Vec2d> result;
  result.reserve(points.size());
  for (const auto& point : points) {
    result.push_back(point.Rotated(angle_radians));
  }
  return result;
}

void TestTrackClassification(const std::vector<Vec2d>& points,
                             const TrajectoryType expected_type) {
  const Track track = CreateTrackFromPoints(points);
  absl::optional<TrajectoryType> type = ClassifyTrack(1, track);
  EXPECT_EQ(*type, expected_type);

  // Rotate the track by 2 different angles and verify the same classification.
  Track track_rotated = RotateTrack(track, .52);
  type = ClassifyTrack(1, track_rotated);
  EXPECT_EQ(*type, expected_type);
  track_rotated = RotateTrack(track, -1.0);
  type = ClassifyTrack(1, track_rotated);
  EXPECT_EQ(*type, expected_type);

  // Rotate the trajectory points through a range of angles.
  for (double angle = -M_PI; angle <= M_PI; angle += M_PI / 32) {
    const std::vector<Vec2d> rotated_points = RotatePoints(points, angle);
    const Track track = CreateTrackFromPoints(rotated_points);
    auto type = ClassifyTrack(1, track);
    EXPECT_EQ(*type, expected_type);
  }
}

TEST(MotionMetricsUtils, ClassifyTrackStationary) {
  std::vector<Vec2d> points = {{0, 0}, {0, .1}, {0, .2}, {0, .3}, {0, .4}};
  TestTrackClassification(points, TrajectoryType::STATIONARY);
}

TEST(MotionMetricsUtils, ClassifyTrackStraight) {
  std::vector<Vec2d> points = {{0, 0}, {0, 1}, {0, 2}, {0, 3}, {0, 4}};
  TestTrackClassification(points, TrajectoryType::STRAIGHT);
}

TEST(MotionMetricsUtils, ClassifyTrackSemiStraightAcrossPiBoundary) {
  std::vector<Vec2d> points = {
      {0, 0}, {-1, 0.05}, {-2, 0}, {-3, -0.5}, {-4, -0.5}};
  TestTrackClassification(points, TrajectoryType::STRAIGHT);
}

TEST(MotionMetricsUtils, ClassifyTrackStraightLeft) {
  std::vector<Vec2d> points = {{0, 0},    {0, 2},   {0, 4},     {-.5, 6},
                               {-1.5, 8}, {-2, 10}, {-2.5, 12}, {-3, 14},
                               {-4, 16},  {-5, 18}, {-5.5, 20}};
  TestTrackClassification(points, TrajectoryType::STRAIGHT_LEFT);
}

TEST(MotionMetricsUtils, ClassifyTrackStraightRight) {
  std::vector<Vec2d> points = {{0, 0},   {0, 2},  {0, 4},    {.5, 6},
                               {1.5, 8}, {2, 10}, {2.5, 12}, {3, 14},
                               {4, 16},  {5, 18}, {5.5, 20}};
  TestTrackClassification(points, TrajectoryType::STRAIGHT_RIGHT);
}

TEST(MotionMetricsUtils, ClassifyTrackRight) {
  std::vector<Vec2d> points = {{0, 0},   {0, 2},   {0, 4},  {.5, 6},
                               {2, 8},   {4, 10},  {6, 12}, {8, 14},
                               {10, 16}, {12, 18}, {14, 20}};
  TestTrackClassification(points, TrajectoryType::RIGHT_TURN);
}

TEST(MotionMetricsUtils, ClassifyTrackLeft) {
  std::vector<Vec2d> points = {{0, 0},    {0, 2},    {0, 4},   {-.5, 6},
                               {-2, 8},   {-4, 10},  {-6, 12}, {-8, 14},
                               {-10, 16}, {-12, 18}, {-14, 20}};
  TestTrackClassification(points, TrajectoryType::LEFT_TURN);
}

TEST(MotionMetricsUtils, ClassifyTrackLeftUTurn) {
  std::vector<Vec2d> points = {{0, 0},    {-2, 2},   {-2, 4},   {-4, 4},
                               {-6, 2},   {-7, 0},   {-8, -2},  {-9, -4},
                               {-10, -6}, {-12, -8}, {-14, -10}};
  TestTrackClassification(points, TrajectoryType::LEFT_U_TURN);
}

TEST(MotionMetricsUtils, ClassifyTrackLeftUTurn180Deg) {
  std::vector<Vec2d> points = {{0, 0},   {-2, 0},  {-4, 0},  {-6, -2},
                               {-6, -4}, {-4, -6}, {-2, -8}, {0, -10},
                               {2, -10}, {4, -10}, {6, -10}};
  TestTrackClassification(points, TrajectoryType::LEFT_U_TURN);
}

TEST(MotionMetricsUtils, ClassifyTrackLeftUTurnGT180Deg) {
  std::vector<Vec2d> points = {{0, 0},   {-2, 0},  {-4, 0},  {-6, -2},
                               {-6, -4}, {-4, -6}, {-2, -8}, {0, -10},
                               {2, -10}, {4, -9},  {6, -8}};
  TestTrackClassification(points, TrajectoryType::LEFT_U_TURN);
}

TEST(MotionMetricsUtils, ClassifyTrackLeftUTurnLT180Deg) {
  std::vector<Vec2d> points = {{0, 0},   {-2, 0},  {-4, 0},  {-6, -2},
                               {-6, -4}, {-4, -6}, {-2, -8}, {0, -10},
                               {2, -10}, {4, -10}, {6, -11}};
  TestTrackClassification(points, TrajectoryType::LEFT_U_TURN);
}

TEST(MotionMetricsUtils, ClassifyTrackRightUTurn180Deg) {
  std::vector<Vec2d> points = {{0, 0},  {-2, 0}, {-4, 0}, {-6, 2},
                               {-6, 4}, {-4, 6}, {-2, 8}, {0, 10},
                               {2, 10}, {4, 10}, {6, 10}};
  TestTrackClassification(points, TrajectoryType::RIGHT_U_TURN);
}

TEST(MotionMetricsUtils, ClassifyTrackRightUTurnGT180Deg) {
  std::vector<Vec2d> points = {{0, 0},  {-2, 0}, {-4, 0}, {-6, 2},
                               {-6, 4}, {-4, 6}, {-2, 8}, {0, 10},
                               {2, 10}, {4, 9},  {6, 8}};
  TestTrackClassification(points, TrajectoryType::RIGHT_U_TURN);
}

TEST(MotionMetricsUtils, ClassifyTrackRightUTurnLT180Deg) {
  std::vector<Vec2d> points = {{0, 0},  {-2, 0}, {-4, 0}, {-6, 2},
                               {-6, 4}, {-4, 6}, {-2, 8}, {0, 10},
                               {2, 10}, {4, 11}, {6, 12}};
  TestTrackClassification(points, TrajectoryType::RIGHT_U_TURN);
}

TEST(MotionMetricsUtils, ClassifyTrackRightUTurn) {
  std::vector<Vec2d> points = {{0, 0},   {2, 2},   {2, 4},   {4, 4},
                               {6, 2},   {7, 0},   {8, -2},  {9, -4},
                               {10, -6}, {12, -8}, {14, -10}};
  TestTrackClassification(points, TrajectoryType::RIGHT_U_TURN);
}

TEST(MotionMetricsUtils, ValidateChallengePredictionsJoint) {
  ScenarioPredictions predictions;
  MultimodalPrediction& prediction = *predictions.add_multi_modal_predictions();
  JointTrajectories& joint_prediction = *prediction.add_joint_predictions();
  joint_prediction.add_trajectories();
  joint_prediction.add_trajectories();
  Status status = ValidateChallengePredictions(
      predictions, MotionChallengeSubmission::INTERACTION_PREDICTION);
  EXPECT_TRUE(status.ok());
  joint_prediction.add_trajectories();
  status = ValidateChallengePredictions(
      predictions, MotionChallengeSubmission::INTERACTION_PREDICTION);
  EXPECT_FALSE(status.ok());
}

TEST(MotionMetricsUtils, ValidateChallengePredictionsSingle) {
  ScenarioPredictions predictions;
  MultimodalPrediction& prediction = *predictions.add_multi_modal_predictions();
  JointTrajectories& joint_prediction = *prediction.add_joint_predictions();
  joint_prediction.add_trajectories();
  MultimodalPrediction& prediction2 =
      *predictions.add_multi_modal_predictions();
  JointTrajectories& joint_prediction2 = *prediction2.add_joint_predictions();
  joint_prediction2.add_trajectories();
  Status status = ValidateChallengePredictions(
      predictions, MotionChallengeSubmission::MOTION_PREDICTION);
  EXPECT_TRUE(status.ok());
  joint_prediction.add_trajectories();
  status = ValidateChallengePredictions(
      predictions, MotionChallengeSubmission::MOTION_PREDICTION);
  EXPECT_FALSE(status.ok());
}

TEST(MotionMetricsUtils, SubmissionToPredictionsSingle) {
  const std::string predictions_str = R"(
    scenario_id: "test"
    single_predictions {
      predictions {
        object_id: 1;
        trajectories {
          confidence: .1
          trajectory {
            center_x: [4, 6, 8, 10]
            center_y: [4, 6, 8, 10]
          }
        }
        trajectories {
          confidence: .2
          trajectory {
            center_x: [-2, -3, -4, -5]
            center_y: [0, 0, 0, 0]
          }
        }
      }
      predictions {
        object_id: 2;
        trajectories {
          confidence: .3
          trajectory {
            center_x: [4, 6, 8, 10]
            center_y: [4, 6, 8, 10]
          }
        }
      }
    }
  )";
  ChallengeScenarioPredictions predictions;
  CHECK(google::protobuf::TextFormat::ParseFromString(predictions_str, &predictions));
  ScenarioPredictions result;
  Status status = ConvertChallengePredictions(predictions, &result);
  ASSERT_TRUE(status.ok());

  const std::string results_str = R"(
    scenario_id : "test"
    multi_modal_predictions {
      joint_predictions {
        trajectories {
          object_id: 1
          center_x: [4, 6, 8, 10]
          center_y: [4, 6, 8, 10]
        }
        confidence: 0.1
      }
      joint_predictions {
        trajectories {
          object_id: 1
          center_x: [-2, -3, -4, -5]
          center_y: [0, 0, 0, 0]
        }
        confidence: 0.2
      }
    }
    multi_modal_predictions {
      joint_predictions {
        trajectories {
          object_id: 2
          center_x: [4, 6, 8, 10]
          center_y: [4, 6, 8, 10]
        }
        confidence: 0.3
      }
    }
  )";
  ScenarioPredictions expected_result;
  CHECK(google::protobuf::TextFormat::ParseFromString(results_str, &expected_result));
  EXPECT_TRUE(
      google::protobuf::util::MessageDifferencer::Equals(result, expected_result));
}

TEST(MotionMetricsUtils, SubmissionToPredictionsJoint) {
  const std::string predictions_str = R"(
    scenario_id: "test"
    joint_prediction {
      joint_trajectories {
        confidence : 0.3;
        trajectories {
          object_id: 1
          trajectory {
            center_x: [4, 6, 8, 10]
            center_y: [4, 6, 8, 10]
          }
        }
        trajectories {
          object_id: 2
          trajectory {
            center_x: [-2, -3, -4, -5]
            center_y: [0, 0, 0, 0]
          }
        }
      }
      joint_trajectories {
        confidence : 0.2;
        trajectories {
          object_id: 1
          trajectory {
            center_x: [5, 6, 8, 10]
            center_y: [5, 6, 8, 10]
          }
        }
        trajectories {
          object_id: 2
          trajectory {
            center_x: [-3, -3, -4, -5]
            center_y: [0, 0, 0, 0]
          }
        }
      }
    }
  )";
  ChallengeScenarioPredictions predictions;
  CHECK(google::protobuf::TextFormat::ParseFromString(predictions_str, &predictions));
  ScenarioPredictions result;
  Status status = ConvertChallengePredictions(predictions, &result);
  ASSERT_TRUE(status.ok());

  const std::string results_str = R"(
    scenario_id: "test"
    multi_modal_predictions {
      joint_predictions {
        trajectories {
          object_id: 1
          center_x: [4, 6, 8, 10]
          center_y: [4, 6, 8, 10]
        }
        trajectories {
          object_id: 2
          center_x: [-2, -3, -4, -5]
          center_y: [0, 0, 0, 0]
        }
        confidence: 0.3
      }
      joint_predictions {
        trajectories {
          object_id: 1
          center_x: [5, 6, 8, 10]
          center_y: [5, 6, 8, 10]
        }
        trajectories {
          object_id: 2
          center_x: [-3, -3, -4, -5]
          center_y: [0, 0, 0, 0]
        }
        confidence: 0.2
      }
    }
  )";
  ScenarioPredictions expected_result;
  CHECK(google::protobuf::TextFormat::ParseFromString(results_str, &expected_result));
  EXPECT_TRUE(
      google::protobuf::util::MessageDifferencer::Equals(result, expected_result));
}

TEST(MotionMetricsUtils, PredictionToTrackStep) {
  MotionMetricsConfig metrics_config;
  CHECK(google::protobuf::TextFormat::ParseFromString(R"(
    track_steps_per_second: 10
    prediction_steps_per_second: 2
    track_history_samples: 5
    track_future_samples: 80
  )",
                                            &metrics_config));
  EXPECT_EQ(PredictionToTrackStep(metrics_config, 1), 15);
}

TEST(MotionMetricsUtils, CurrentTrackStep) {
  MotionMetricsConfig metrics_config;
  CHECK(google::protobuf::TextFormat::ParseFromString(R"(
    track_steps_per_second: 10
    prediction_steps_per_second: 2
    track_history_samples: 5
    track_future_samples: 80
  )",
                                            &metrics_config));
  EXPECT_EQ(CurrentTrackStep(metrics_config), 5);
}

TEST(MotionMetricsUtils, PredictionToPolygon) {
  MotionMetricsConfig metrics_config;
  CHECK(google::protobuf::TextFormat::ParseFromString(R"(
    track_steps_per_second: 1
    prediction_steps_per_second: 1
    track_history_samples: 1
    track_future_samples: 1
  )",
                                            &metrics_config));

  Track track;
  CHECK(google::protobuf::TextFormat::ParseFromString(R"(
    object_type: TYPE_VEHICLE
    states {
      length: 2
      width: 2
      valid: true
    }
    states {
      length: 4
      width: 2
      valid: true
    }
    states {
      length: 6
      width: 2
      valid: true
    }
  )",
                                            &track));
  SingleTrajectory trajectory;
  CHECK(google::protobuf::TextFormat::ParseFromString(R"(
    center_x: [1, 2, 3]
    center_y: [0, 0, 0]
  )",
                                            &trajectory));

  const int trajectory_step = 0;
  {
    const bool use_current_box_dimensions = false;
    const Polygon2d got =
        PredictionToPolygon(metrics_config, trajectory, trajectory_step, track,
                            use_current_box_dimensions);
    const Polygon2d want(Box2d(/*center= */ {3, 0},
                               /*heading=*/0,
                               /*length=*/6,
                               /*width=*/2));
    EXPECT_EQ(got.Area(), want.Area());
  }
  {
    const bool use_current_box_dimensions = true;
    const Polygon2d got =
        PredictionToPolygon(metrics_config, trajectory, trajectory_step, track,
                            use_current_box_dimensions);
    const Polygon2d want(Box2d(/*center= */ {3, 0},
                               /*heading=*/0,
                               /*length=*/4,
                               /*width=*/2));
    EXPECT_EQ(got.Area(), want.Area());
  }
}

TEST(MotionMetricsUtils, AverageAngle) {
  EXPECT_NEAR(AverageAngle(0.0, 0.0), 0.0, 1e-5);
  EXPECT_NEAR(AverageAngle(0.0, M_PI), M_PI / 2.0, 1e-5);
  EXPECT_NEAR(AverageAngle(M_PI / 2.0, M_PI), 3 * M_PI / 4.0, 1e-5);
  EXPECT_NEAR(AverageAngle(-M_PI + .05, M_PI - .05), M_PI, 1e-5);
}

}  // namespace
}  // namespace open_dataset
}  // namespace waymo
