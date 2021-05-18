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

#include "waymo_open_dataset/metrics/motion_metrics.h"

#include <glog/logging.h>
#include "google/protobuf/text_format.h"
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "waymo_open_dataset/math/vec2d.h"
#include "waymo_open_dataset/protos/motion_metrics.pb.h"
#include "waymo_open_dataset/protos/motion_submission.pb.h"

namespace waymo {
namespace open_dataset {
namespace {

// Creates a test submission proto with predictions containing the ground truth
// trajectories offset by constant values. This submission corresponds to a
// MotionMetricsConfig with future_seconds=8, history_seconds=1,
// prediction_steps_per_second=2, and track_steps_per_second=10. These are
// hard coded here with expected values to avoid duplicating the indexing logic
// in the metrics code.
ScenarioPredictions CreateTestSubmissionProto(
    const Scenario& scenario, std::vector<int> offsets = {1, 2, 3, 4},
    std::vector<float> confidences = {0.8f, 0.2f, 0.2f, 0.2f}) {
  CHECK_EQ(offsets.size(), confidences.size());
  ScenarioPredictions submission;
  submission.set_scenario_id(scenario.scenario_id());
  for (const auto& required_track : scenario.tracks_to_predict()) {
    const int track_index = required_track.track_index();
    CHECK_LT(track_index, scenario.tracks_size());
    const Track& track = scenario.tracks(track_index);
    auto* prediction = submission.add_multi_modal_predictions();

    // Add predicted trajectories each offset from the ground truth by a
    // constant value.
    for (int offset_index = 0; offset_index < offsets.size(); ++offset_index) {
      auto* joint_prediction = prediction->add_joint_predictions();

      // Add only a single trajectory to the joint prediction for a BP challenge
      // submission (interactive challenge submissions will have multiple).
      auto* trajectory = joint_prediction->add_trajectories();
      trajectory->set_object_id(track.id());
      for (int i = 15; i < 91; i += 5) {
        const ObjectState& track_state = track.states(i);
        if (track_state.valid()) {
          trajectory->add_center_x(track_state.center_x() +
                                   offsets[offset_index]);
          trajectory->add_center_y(track_state.center_y());
        } else {
          trajectory->add_center_x(-1.0);
          trajectory->add_center_y(-1.0);
        }
      }
      joint_prediction->set_confidence(confidences[offset_index]);
    }
  }
  return submission;
}
}  // namespace


// Creates a sythetic test ScenarioPredictions proto.
ScenarioPredictions CreateTestPredictions() {
  const std::string predictions_str = R"(
    scenario_id: "test"
    multi_modal_predictions {
      joint_predictions {
         confidence : 0.5
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
      }
    }
  )";
  ScenarioPredictions predictions;
  CHECK(google::protobuf::TextFormat::ParseFromString(predictions_str, &predictions));
  return predictions;
}

Scenario CreateTestScenario() {
  const std::string scenario_str = R"(
    scenario_id: "test"
    timestamps_seconds: [0.1, 0.2, 0.3, 0.4, 0.5]
    current_time_index: 0
    tracks {
      id: 1
      object_type: TYPE_VEHICLE
      states {
        center_x: 2
        center_y: 2
        heading: 0.78539816
        valid: true
        velocity_x: 20.0
        velocity_y: 20.0
      }
      states {
        center_x: 4
        center_y: 4
        heading: 0.78539816
        valid: true
      }
      states {
        center_x: 6
        center_y: 6
        heading: 0.78539816
        valid: true
      }
      states {
        center_x: 8
        center_y: 8
        heading: 0.78539816
        valid: true
      }
      states {
        center_x: 10
        center_y: 10
        heading: 0.78539816
        valid: true
      }
    }
    tracks {
      id: 2
      object_type: TYPE_VEHICLE
      states {
        center_x: -1
        center_y: 0
        heading: 3.14159
        valid: true
        velocity_x: -10
        velocity_y: 0
      }
      states {
        center_x: -2
        center_y: 0
        heading: 3.14159
        valid: true
      }
      states {
        center_x: -3
        center_y: 0
        heading: 3.14159
        valid: true
      }
      states {
        center_x: -4
        center_y: 0
        heading: 3.14159
        valid: true
      }
      states {
        center_x: -5
        center_y: 0
        heading: 3.14159
        valid: true
      }
    }
    tracks_to_predict: {
      track_index: 0
      difficulty: LEVEL_1
    }
    tracks_to_predict: {
      track_index: 1
      difficulty: LEVEL_1
    }
  )";
  Scenario scenario;
  CHECK(google::protobuf::TextFormat::ParseFromString(scenario_str, &scenario));
  return scenario;
}

// Return a MotionMetricsConfig to be used with the test objects created with
// CreateTestPredictions and CreateTestScenario.
MotionMetricsConfig GetTestConfig() {
  const std::string scenario_str = R"(
    track_steps_per_second: 10
    prediction_steps_per_second: 10
    track_history_samples: 0
    track_future_samples: 4
    speed_lower_bound: 1.4
    speed_upper_bound: 11.0
    speed_scale_lower: 1.0
    speed_scale_upper: 1.0
    step_configurations {
      measurement_step: 3
      lateral_miss_threshold: 1.0
      longitudinal_miss_threshold: 2.0
    }
    max_predictions: 6
  )";
  MotionMetricsConfig config;
  CHECK(google::protobuf::TextFormat::ParseFromString(scenario_str, &config));
  return config;
}

class TestJointMetricsSynthetic : public ::testing::Test {
 public:
  void SetUp() override {
    predictions_ = CreateTestPredictions();
    scenario_ = CreateTestScenario();
    config_ = GetTestConfig();
  }

 protected:
  Scenario scenario_;
  MotionMetricsConfig config_;
  ScenarioPredictions predictions_;
  MotionChallengeSubmission::SubmissionType submission_type_ =
      MotionChallengeSubmission::INTERACTION_PREDICTION;
};

TEST_F(TestJointMetricsSynthetic, ComputeMissRateNoMisses) {
  BucketedMetricsStats metrics_stats;
  Status status =
      ComputeMetricsStats(config_, predictions_, scenario_, &metrics_stats);
  EXPECT_TRUE(status.ok());

  // Expect miss rate to be zero and mAP metric to be 1.0
  MetricsStats& stats = metrics_stats.stats[Track::TYPE_VEHICLE][3];
  EXPECT_EQ(stats.miss_rate.Mean(), 0.0);
  EXPECT_EQ(ComputeMapMetric(&stats.mean_average_precision), 1.0);
}

TEST_F(TestJointMetricsSynthetic, ComputeMissRateLateral_2) {
  const std::string predictions_str = R"(
    scenario_id: "test"
    multi_modal_predictions {
      joint_predictions {
         confidence : 0.5
         trajectories {
           object_id: 1
           center_x: [4, 6, 8, 10]
           center_y: [4, 6, 8, 10]
         }
         trajectories {
           object_id: 2
           center_x: [-2, -3, -4, -5]
           center_y: [1.01, 1.01, 1.01, 1.01]
         }
      }
    }
  )";
  ScenarioPredictions predictions;
  CHECK(google::protobuf::TextFormat::ParseFromString(predictions_str, &predictions));
  BucketedMetricsStats metrics_stats;
  Status status =
      ComputeMetricsStats(config_, predictions, scenario_, &metrics_stats);
  EXPECT_TRUE(status.ok());

  // Expect miss rate to be 1 and mAP metric to be 0.
  MetricsStats& stats = metrics_stats.stats[Track::TYPE_VEHICLE][3];
  EXPECT_EQ(stats.miss_rate.Mean(), 1.0);
  EXPECT_EQ(ComputeMapMetric(&stats.mean_average_precision), 0.0);
}

TEST_F(TestJointMetricsSynthetic, ComputeMissRateLateral_1) {
  const std::string predictions_str = R"(
    scenario_id: "test"
    multi_modal_predictions {
      joint_predictions {
         confidence : 0.5
         trajectories {
           object_id: 1
           center_x: [4, 6, 8, 9.292]
           center_y: [4, 6, 8, 10.708]
         }
         trajectories {
           object_id: 2
           center_x: [-2, -3, -4, -5]
           center_y: [0, 0, 0, 0]
         }
      }
    }
  )";
  ScenarioPredictions predictions;
  CHECK(google::protobuf::TextFormat::ParseFromString(predictions_str, &predictions));
  BucketedMetricsStats metrics_stats;
  Status status =
      ComputeMetricsStats(config_, predictions, scenario_, &metrics_stats);
  EXPECT_TRUE(status.ok());

  // Expect miss rate to be 1 and mAP metric to be 0.
  MetricsStats& stats = metrics_stats.stats[Track::TYPE_VEHICLE][3];
  EXPECT_EQ(stats.miss_rate.Mean(), 1.0);
  EXPECT_EQ(ComputeMapMetric(&stats.mean_average_precision), 0.0);
}

TEST_F(TestJointMetricsSynthetic, ComputeMissRateLongitudinal_2) {
  const std::string predictions_str = R"(
    scenario_id: "test"
    multi_modal_predictions {
      joint_predictions {
         confidence : 0.5
         trajectories {
           object_id: 1
           center_x: [4, 6, 8, 10]
           center_y: [4, 6, 8, 10]
         }
         trajectories {
           object_id: 2
           center_x: [-2, -3, -4, -7.01]
           center_y: [0, 0, 0, 0]
         }
      }
    }
  )";
  ScenarioPredictions predictions;
  CHECK(google::protobuf::TextFormat::ParseFromString(predictions_str, &predictions));
  BucketedMetricsStats metrics_stats;
  Status status =
      ComputeMetricsStats(config_, predictions, scenario_, &metrics_stats);
  EXPECT_TRUE(status.ok());

  // Expect miss rate to be 1 and mAP metric to be 0.
  MetricsStats& stats = metrics_stats.stats[Track::TYPE_VEHICLE][3];
  EXPECT_EQ(stats.miss_rate.Mean(), 1.0);
  EXPECT_EQ(ComputeMapMetric(&stats.mean_average_precision), 0.0);
}

TEST_F(TestJointMetricsSynthetic, ComputeMissRateLongitudinal_1) {
  const std::string predictions_str = R"(
    scenario_id: "test"
    multi_modal_predictions {
      joint_predictions {
         confidence : 0.5
         trajectories {
           object_id: 1
           center_x: [4, 6, 8, 11.415]
           center_y: [4, 6, 8, 11.415]
         }
         trajectories {
           object_id: 2
           center_x: [-2, -3, -4, -5]
           center_y: [0, 0, 0, 0]
         }
      }
    }
  )";
  ScenarioPredictions predictions;
  CHECK(google::protobuf::TextFormat::ParseFromString(predictions_str, &predictions));
  BucketedMetricsStats metrics_stats;
  Status status =
      ComputeMetricsStats(config_, predictions, scenario_, &metrics_stats);
  EXPECT_TRUE(status.ok());

  // Expect miss rate to be 1 and mAP metric to be 0.
  MetricsStats& stats = metrics_stats.stats[Track::TYPE_VEHICLE][3];
  EXPECT_EQ(stats.miss_rate.Mean(), 1.0);
  EXPECT_EQ(ComputeMapMetric(&stats.mean_average_precision), 0.0);
}

TEST_F(TestJointMetricsSynthetic, ComputeNoMissLongitudinal_1) {
  const std::string predictions_str = R"(
    scenario_id: "test"
    multi_modal_predictions {
      joint_predictions {
         confidence : 0.5
         trajectories {
           object_id: 1
           center_x: [4, 6, 8, 11.414]
           center_y: [4, 6, 8, 11.414]
         }
         trajectories {
           object_id: 2
           center_x: [-2, -3, -4, -5]
           center_y: [0, 0, 0, 0]
         }
      }
    }
  )";
  ScenarioPredictions predictions;
  CHECK(google::protobuf::TextFormat::ParseFromString(predictions_str, &predictions));
  BucketedMetricsStats metrics_stats;
  Status status =
      ComputeMetricsStats(config_, predictions, scenario_, &metrics_stats);
  EXPECT_TRUE(status.ok());

  // Expect miss rate to be 0 and mAP metric to be 1.
  MetricsStats& stats = metrics_stats.stats[Track::TYPE_VEHICLE][3];
  EXPECT_EQ(stats.miss_rate.Mean(), 0.0);
  EXPECT_EQ(ComputeMapMetric(&stats.mean_average_precision), 1.0);
}

TEST_F(TestJointMetricsSynthetic, ComputeVelocityScalingLatitudinal) {
  const std::string predictions_str = R"(
    scenario_id: "test"
    multi_modal_predictions {
      joint_predictions {
         confidence : 0.5
         trajectories {
           object_id: 1
           center_x: [4, 6, 8, 10]
           center_y: [4, 6, 8, 10]
         }
         trajectories {
           object_id: 2
           center_x: [-2, -3, -4, -5]
           center_y: [0, 0, 0, 0.75]
         }
      }
    }
  )";
  ScenarioPredictions predictions;
  MotionMetricsConfig config = GetTestConfig();
  config.set_speed_scale_lower(0.5);
  config.set_speed_scale_upper(1.0);
  config.set_speed_lower_bound(1.0);
  config.set_speed_upper_bound(3.0);
  CHECK(google::protobuf::TextFormat::ParseFromString(predictions_str, &predictions));
  BucketedMetricsStats metrics_stats;
  Status status =
      ComputeMetricsStats(config, predictions, scenario_, &metrics_stats);
  EXPECT_TRUE(status.ok());

  // Expect miss rate to be 0 and mAP metric to be 1.
  MetricsStats stats = metrics_stats.stats[Track::TYPE_VEHICLE][3];
  EXPECT_EQ(stats.miss_rate.Mean(), 0.0);
  EXPECT_EQ(ComputeMapMetric(&stats.mean_average_precision), 1.0);

  // Decrease the velocity below the speed lower bound.
  Scenario scenario;
  scenario.CopyFrom(scenario_);
  scenario.mutable_tracks(1)->mutable_states(0)->set_velocity_x(0.0);
  scenario.mutable_tracks(1)->mutable_states(0)->set_velocity_y(0.0);
  BucketedMetricsStats metrics_stats_2;
  status = ComputeMetricsStats(config, predictions, scenario, &metrics_stats_2);
  EXPECT_TRUE(status.ok());
  stats = metrics_stats_2.stats[Track::TYPE_VEHICLE][3];
  EXPECT_EQ(stats.miss_rate.Mean(), 1.0);

  // Set the velocity to just below the speed required for object2 to fit.
  scenario.mutable_tracks(1)->mutable_states(0)->set_velocity_x(1.9999);
  BucketedMetricsStats metrics_stats_3;
  status = ComputeMetricsStats(config, predictions, scenario, &metrics_stats_3);
  EXPECT_TRUE(status.ok());
  stats = metrics_stats_3.stats[Track::TYPE_VEHICLE][3];
  EXPECT_EQ(stats.miss_rate.Mean(), 1.0);

  // Set the velocity to just above the speed required for object2 to fit.
  scenario.mutable_tracks(1)->mutable_states(0)->set_velocity_x(2.001);
  BucketedMetricsStats metrics_stats_4;
  status = ComputeMetricsStats(config, predictions, scenario, &metrics_stats_4);
  stats = metrics_stats_4.stats[Track::TYPE_VEHICLE][3];
  EXPECT_EQ(stats.miss_rate.Mean(), 0.0);
}

TEST_F(TestJointMetricsSynthetic, ComputeVelocityScalingLongitudinal) {
  const std::string predictions_str = R"(
    scenario_id: "test"
    multi_modal_predictions {
      joint_predictions {
         confidence : 0.5
         trajectories {
           object_id: 1
           center_x: [4, 6, 8, 10]
           center_y: [4, 6, 8, 10]
         }
         trajectories {
           object_id: 2
           center_x: [-2, -3, -4, -6.5]
           center_y: [0, 0, 0, 0]
         }
      }
    }
  )";
  ScenarioPredictions predictions;
  MotionMetricsConfig config = GetTestConfig();
  config.set_speed_scale_lower(0.5);
  config.set_speed_scale_upper(1.0);
  config.set_speed_lower_bound(1.0);
  config.set_speed_upper_bound(3.0);
  CHECK(google::protobuf::TextFormat::ParseFromString(predictions_str, &predictions));

  BucketedMetricsStats metrics_stats;
  Status status =
      ComputeMetricsStats(config, predictions, scenario_, &metrics_stats);
  EXPECT_TRUE(status.ok());

  // Expect miss rate to be 0 and mAP metric to be 1.
  MetricsStats stats = metrics_stats.stats[Track::TYPE_VEHICLE][3];
  EXPECT_EQ(stats.miss_rate.Mean(), 0.0);
  EXPECT_EQ(ComputeMapMetric(&stats.mean_average_precision), 1.0);

  // Decrease the velocity below the speed lower bound.
  Scenario scenario;
  scenario.CopyFrom(scenario_);
  scenario.mutable_tracks(1)->mutable_states(0)->set_velocity_x(0.0);
  scenario.mutable_tracks(1)->mutable_states(0)->set_velocity_y(0.0);
  BucketedMetricsStats metrics_stats_2;
  status = ComputeMetricsStats(config, predictions, scenario, &metrics_stats_2);
  EXPECT_TRUE(status.ok());
  stats = metrics_stats_2.stats[Track::TYPE_VEHICLE][3];
  EXPECT_EQ(stats.miss_rate.Mean(), 1.0);

  // Set the velocity to just below the speed required for object2 to fit.
  scenario.mutable_tracks(1)->mutable_states(0)->set_velocity_x(1.9999);
  BucketedMetricsStats metrics_stats_3;
  ComputeMetricsStats(config, predictions, scenario, &metrics_stats_3);
  EXPECT_TRUE(status.ok());
  stats = metrics_stats_3.stats[Track::TYPE_VEHICLE][3];
  EXPECT_EQ(stats.miss_rate.Mean(), 1.0);

  // Set the velocity to just above the speed required for object2 to fit.
  scenario.mutable_tracks(1)->mutable_states(0)->set_velocity_x(2.001);
  BucketedMetricsStats metrics_stats_4;
  status = ComputeMetricsStats(config, predictions, scenario, &metrics_stats_4);
  EXPECT_TRUE(status.ok());
  stats = metrics_stats_4.stats[Track::TYPE_VEHICLE][3];
  EXPECT_EQ(stats.miss_rate.Mean(), 0.0);
}

TEST_F(TestJointMetricsSynthetic, ComputeNoMissLateral_2) {
  const std::string predictions_str = R"(
    scenario_id: "test"
    multi_modal_predictions {
      joint_predictions {
         confidence : 0.5
         trajectories {
           object_id: 1
           center_x: [4, 6, 8, 9.294]
           center_y: [4, 6, 8, 10.706]
         }
         trajectories {
           object_id: 2
           center_x: [-2, -3, -4, -5]
           center_y: [0, 0, 0, 0]
         }
      }
    }
  )";
  ScenarioPredictions predictions;
  CHECK(google::protobuf::TextFormat::ParseFromString(predictions_str, &predictions));
  BucketedMetricsStats metrics_stats;
  Status status =
      ComputeMetricsStats(config_, predictions, scenario_, &metrics_stats);
  EXPECT_TRUE(status.ok());

  // Expect miss rate to be 0 and mAP metric to be 1.
  MetricsStats& stats = metrics_stats.stats[Track::TYPE_VEHICLE][3];
  EXPECT_EQ(stats.miss_rate.Mean(), 0);
  EXPECT_EQ(ComputeMapMetric(&stats.mean_average_precision), 1.0);
}

TEST_F(TestJointMetricsSynthetic, TwoJointPredictionsNoMiss) {
  const std::string predictions_str = R"(
    scenario_id: "test"
    multi_modal_predictions {
      joint_predictions {
        confidence : 0.8
        trajectories {
         object_id: 1
         center_x: [4, 6, 8, 10]
         center_y: [4, 6, 8, 10]
        }
        trajectories {
         object_id: 2
         center_x: [-2, -3, -4, -7.01]
         center_y: [0, 0, 0, 0]
        }
      }
      joint_predictions {
        confidence : 0.5
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
      }
    }
  )";
  ScenarioPredictions predictions;
  CHECK(google::protobuf::TextFormat::ParseFromString(predictions_str, &predictions));
  BucketedMetricsStats metrics_stats;
  Status status =
      ComputeMetricsStats(config_, predictions, scenario_, &metrics_stats);
  EXPECT_TRUE(status.ok());

  // Expect miss rate to be 0 and mAP metric to be 0.5.
  MetricsStats& stats = metrics_stats.stats[Track::TYPE_VEHICLE][3];
  EXPECT_EQ(stats.miss_rate.Mean(), 0.0);
  EXPECT_EQ(ComputeMapMetric(&stats.mean_average_precision), 0.5);
}

TEST_F(TestJointMetricsSynthetic, TwoJointPredictionsObjectAndTrajectoryTypes) {
  const std::string predictions_str = R"(
    scenario_id: "test"
    multi_modal_predictions {
      joint_predictions {
        confidence : 0.8
        trajectories {
         object_id: 1
         center_x: [0, 0, 0, 0]
         center_y: [0, 0, 0, 0]
        }
        trajectories {
         object_id: 2
         center_x: [-2, -3, -4, -7.01]
         center_y: [0, 0, 0, 0]
        }
      }
      joint_predictions {
        confidence : 0.5
        trajectories {
         object_id: 1
         center_x: [0, 0, 0, 0]
         center_y: [0, 0, 0, 0]
        }
        trajectories {
         object_id: 2
         center_x: [-2, -3, -4, -5]
         center_y: [0, 0, 0, 0]
        }
      }
    }
  )";
  ScenarioPredictions predictions;
  CHECK(google::protobuf::TextFormat::ParseFromString(predictions_str, &predictions));

  Scenario scenario = scenario_;
  // Change the first track to something boring (STATIC) so that the track_type
  // is inherited from the second track.
  auto* track = scenario.mutable_tracks(0);
  for (auto& states : *(track->mutable_states())) {
    states.set_center_x(0);
    states.set_center_y(0);
    states.set_velocity_x(0);
    states.set_velocity_y(0);
  }
  // Change the second track's object type to PEDESTRIAN so that the object_type
  // is inherited from the second track.
  track = scenario.mutable_tracks(1);
  track->set_object_type(Track::TYPE_PEDESTRIAN);

  BucketedMetricsStats metrics_stats;
  Status status =
      ComputeMetricsStats(config_, predictions, scenario, &metrics_stats);
  EXPECT_TRUE(status.ok());

  EXPECT_TRUE(metrics_stats.stats.find(Track::TYPE_VEHICLE) ==
              metrics_stats.stats.end());
  // Expect miss rate to be 0 and mAP metric to be 0.5.
  MetricsStats& stats = metrics_stats.stats[Track::TYPE_PEDESTRIAN][3];
  EXPECT_EQ(stats.miss_rate.Mean(), 0.0);
  // The STRAIGHT bucket should have 1 trajectory, and STATIC bucket should have
  // 0 trajectory.
  EXPECT_EQ(stats.mean_average_precision.pr_buckets[0].num_trajectories, 0);
  EXPECT_EQ(stats.mean_average_precision.pr_buckets[1].num_trajectories, 1);
  EXPECT_EQ(ComputeMapMetric(&stats.mean_average_precision), 0.5);
}

TEST_F(TestJointMetricsSynthetic, TwoJointPredictionsMiss) {
  const std::string predictions_str = R"(
    scenario_id: "test"
    multi_modal_predictions {
      joint_predictions {
        confidence : 0.8
        trajectories {
         object_id: 1
         center_x: [4, 6, 8, 10]
         center_y: [4, 6, 8, 10]
        }
        trajectories {
         object_id: 2
         center_x: [-2, -3, -4, -7.01]
         center_y: [0, 0, 0, 0]
        }
      }
      joint_predictions {
        confidence : 0.5
        trajectories {
         object_id: 1
         center_x: [4, 6, 8, 14]
         center_y: [4, 6, 8, 14]
        }
        trajectories {
         object_id: 2
         center_x: [-2, -3, -4, -5]
         center_y: [0, 0, 0, 0]
        }
      }
    }
  )";
  ScenarioPredictions predictions;
  CHECK(google::protobuf::TextFormat::ParseFromString(predictions_str, &predictions));
  BucketedMetricsStats metrics_stats;
  Status status =
      ComputeMetricsStats(config_, predictions, scenario_, &metrics_stats);
  EXPECT_TRUE(status.ok());

  // Expect miss rate to be 1 and mAP metric to be 0.0.
  MetricsStats& stats = metrics_stats.stats[Track::TYPE_VEHICLE][3];
  EXPECT_EQ(stats.miss_rate.Mean(), 1.0);
  EXPECT_EQ(ComputeMapMetric(&stats.mean_average_precision), 0.0);
}

TEST_F(TestJointMetricsSynthetic, InvalidJointPredictions) {
  const std::string predictions_str = R"(
    scenario_id: "test"
    multi_modal_predictions {
      joint_predictions {
        confidence : 0.8
        trajectories {
         object_id: 1
         center_x: [4, 6, 8, 10]
         center_y: [4, 6, 8, 10]
        }
        trajectories {
         object_id: 2
         center_x: [-2, -3, -4, -7.01]
         center_y: [0, 0, 0, 0]
        }
      }
      joint_predictions {
        confidence : 0.5
        trajectories {
         object_id: 1
         center_x: [4, 6, 8, 14]
         center_y: [4, 6, 8, 14]
        }
        trajectories {
         object_id: 1
         center_x: [-2, -3, -4, -5]
         center_y: [0, 0, 0, 0]
        }
      }
    }
  )";
  ScenarioPredictions predictions;
  CHECK(google::protobuf::TextFormat::ParseFromString(predictions_str, &predictions));

  BucketedMetricsStats metrics_stats;
  Status status =
      ComputeMetricsStats(config_, predictions, scenario_, &metrics_stats);
  EXPECT_FALSE(status.ok());
}

TEST_F(TestJointMetricsSynthetic, MissingObjectPredictions) {
  const std::string predictions_str = R"(
    scenario_id: "test"
    multi_modal_predictions {
      joint_predictions {
        confidence : 0.8
        trajectories {
         object_id: 3
         center_x: [4, 6, 8, 10]
         center_y: [4, 6, 8, 10]
        }
        trajectories {
         object_id: 4
         center_x: [-2, -3, -4, -7.01]
         center_y: [0, 0, 0, 0]
        }
      }
     }
  )";
  ScenarioPredictions predictions;
  CHECK(google::protobuf::TextFormat::ParseFromString(predictions_str, &predictions));
  BucketedMetricsStats metrics_stats;
  Status status =
      ComputeMetricsStats(config_, predictions, scenario_, &metrics_stats);
  EXPECT_FALSE(status.ok());
}

TEST_F(TestJointMetricsSynthetic, ComputeMinADE) {
  const std::string predictions_str = R"(
    scenario_id: "test"
    multi_modal_predictions {
      joint_predictions {
         confidence : 0.5
         trajectories {
           object_id: 1
           center_x: [4, 6, 8, 10]
           center_y: [0, 0, 0, 0]
         }
         trajectories {
           object_id: 2
           center_x: [0, 0, 0, 0]
           center_y: [2, 3, 4, 5]
         }
      }
      joint_predictions {
         confidence : 0.5
         trajectories {
           object_id: 1
           center_x: [14, 16, 18, 20]
           center_y: [0, 0, 0, 0]
         }
         trajectories {
           object_id: 2
           center_x: [0, 0, 0, 0]
           center_y: [22, 23, 24, 25]
         }
      }
    }
  )";
  ScenarioPredictions predictions;
  CHECK(google::protobuf::TextFormat::ParseFromString(predictions_str, &predictions));
  BucketedMetricsStats metrics_stats;
  Status status =
      ComputeMetricsStats(config_, predictions, scenario_, &metrics_stats);
  EXPECT_TRUE(status.ok());

  // Validate the min ADE and FDE.
  MetricsStats& stats = metrics_stats.stats[Track::TYPE_VEHICLE][3];
  EXPECT_NEAR(stats.min_average_displacement.Mean(), 5.97487, 1e-4);
  EXPECT_NEAR(stats.min_final_displacement.Mean(), 8.53553, 1e-4);
}

TEST(MotionMetrics, ComputePrCurve) {
  std::vector<PredictionSample> samples = {{.3, true}, {.7, false}, {.4, true},
                                           {.8, true}, {.5, false}, {.2, true}};

  std::vector<PrSample> pr_curve = ComputePrCurve(&samples, samples.size());

  std::vector<PrSample> expected_result = {
      {1.0 / 6, 1.0}, {1.0 / 6, 0.5}, {1.0 / 6, 1.0 / 3},
      {1.0 / 3, 0.5}, {0.5, 3.0 / 5}, {2.0 / 3, 2.0 / 3}};
  for (int i = 0; i < pr_curve.size(); ++i) {
    EXPECT_NEAR(pr_curve[i].precision, expected_result[i].precision, 1e-5);
    EXPECT_NEAR(pr_curve[i].recall, expected_result[i].recall, 1e-5);
  }
}

TEST(MotionMetrics, ComputePrCurveEqualConfidences) {
  std::vector<PredictionSample> samples = {{.8, true},  {.5, false},
                                           {.5, true},  {.5, true},
                                           {.5, false}, {.5, false}};

  std::vector<PrSample> pr_curve = ComputePrCurve(&samples, samples.size());

  std::vector<PrSample> expected_result = {
      {1.0 / 6, 1},       {1.0 / 6, 0.5},     {1.0 / 6, 1.0 / 3},
      {1.0 / 6, 1.0 / 4}, {1.0 / 3, 2.0 / 5}, {0.5, 0.5}};
  for (int i = 0; i < pr_curve.size(); ++i) {
    EXPECT_NEAR(pr_curve[i].precision, expected_result[i].precision, 1e-5);
    EXPECT_NEAR(pr_curve[i].recall, expected_result[i].recall, 1e-5);
  }
}

TEST(MotionMetrics, ComputeMeanAveragePrecision) {
  PredictionStats stats;
  stats.samples = {{.3, true}, {.7, true},  {.4, false},
                   {.8, true}, {.5, false}, {.2, false}};
  stats.num_trajectories = 4;

  EXPECT_NEAR(ComputeMeanAveragePrecision(&stats), 0.65, 1e-4);
}

TEST(MotionMetrics, ComputeMeanAveragePrecisionEqualConfidences) {
  PredictionStats stats;
  stats.samples = {{.8, true}, {.5, false}, {.5, true},
                   {.5, true}, {.5, false}, {.5, false}};
  stats.num_trajectories = 6;

  EXPECT_NEAR(ComputeMeanAveragePrecision(&stats), 1.0 / 3, 1e-4);
}

}  // namespace open_dataset
}  // namespace waymo
