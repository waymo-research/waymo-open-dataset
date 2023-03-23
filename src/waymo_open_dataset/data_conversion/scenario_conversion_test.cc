/* Copyright 2022 The Waymo Open Dataset Authors.

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

#include "waymo_open_dataset/data_conversion/scenario_conversion.h"

#include "google/protobuf/text_format.h"
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/strings/str_cat.h"
#include "tensorflow/core/example/example.pb.h"
#include "waymo_open_dataset/protos/metrics.pb.h"
#include "waymo_open_dataset/protos/scenario.pb.h"

namespace waymo {
namespace open_dataset {
namespace internal {
namespace {

using ::google::protobuf::TextFormat;
using ::testing::ElementsAre;

using ::testing::FloatEq;
using ::testing::FloatNear;
using ::testing::UnorderedElementsAre;

const char kTestConfigStr[] = R"(
  num_future_steps: 1
  num_past_steps: 2
  max_num_agents: 2
  max_roadgraph_samples: 6
  max_traffic_light_control_points_per_step: 2
)";

const char kScenarioStr[] = R"(
  scenario_id: "123"
  timestamps_seconds: 1.0
  timestamps_seconds: 2.0
  timestamps_seconds: 3.0
  timestamps_seconds: 4.0
  tracks {
    id: 101
    object_type: TYPE_VEHICLE
    states {
      center_x: 0.5
      center_y: 0.2
      center_z: 0.1
      length: 1
      width: 1
      height: 1
      heading: 0.3
      velocity_x: 0.1
      velocity_y: 0.1
      valid: true
    }
    states {
      center_x: 0.6
      center_y: 0.3
      center_z: 0.1
      length: 1
      width: 1
      height: 1
      heading: 0.3
      velocity_x: 0
      velocity_y: 0
      valid: true
    }
    states {
      center_x: 0.6
      center_y: 0.3
      center_z: 0.1
      length: 1
      width: 1
      height: 1
      heading: 0.3
      velocity_x: 0.1
      velocity_y: 0.1
      valid: true
    }
    states {
      center_x: 0.7
      center_y: 0.4
      center_z: 0.2
      length: 1
      width: 1
      height: 1
      heading: 0.3
      velocity_x: 0.1
      velocity_y: 0.1
      valid: true
    }
  }
  tracks {
    id: 102
    object_type: TYPE_VEHICLE
    states {
      center_x: 1.5
      center_y: 1.2
      center_z: 1.1
      length: 1
      width: 1
      height: 1
      heading: 0.3
      velocity_x: 0.1
      velocity_y: 0.1
      valid: true
    }
    states {
      center_x: 1.6
      center_y: 1.3
      center_z: 1.1
      length: 1
      width: 1
      height: 1
      heading: 0.3
      velocity_x: 0
      velocity_y: 0
      valid: true
    }
    states {
      center_x: 1.6
      center_y: 1.3
      center_z: 1.1
      length: 1
      width: 1
      height: 1
      heading: 0.3
      velocity_x: 0.1
      velocity_y: 0.1
      valid: true
    }
    states {
      center_x: 1.7
      center_y: 1.4
      center_z: 1.2
      length: 1
      width: 1
      height: 1
      heading: 0.3
      velocity_x: 0.1
      velocity_y: 0.1
      valid: true
    }
  }
  tracks {
    id: 103
    object_type: TYPE_PEDESTRIAN
    states {
      center_x: 2.5
      center_y: 2.2
      center_z: 2.1
      length: 1
      width: 1
      height: 1
      heading: 0.3
      velocity_x: 0.1
      velocity_y: 0.1
      valid: true
    }
    states {
      center_x: 2.6
      center_y: 2.3
      center_z: 2.1
      length: 1
      width: 1
      height: 1
      heading: 0.3
      velocity_x: 0
      velocity_y: 0
      valid: false
    }
    states {
      center_x: 2.6
      center_y: 2.3
      center_z: 2.1
      length: 1
      width: 1
      height: 1
      heading: 0.3
      velocity_x: 0.1
      velocity_y: 0.1
      valid: false
    }
    states {
      center_x: 2.7
      center_y: 2.4
      center_z: 2.2
      length: 1
      width: 1
      height: 1
      heading: 0.3
      velocity_x: 0.1
      velocity_y: 0.1
      valid: false
    }
  }
  tracks {
    id: 104
    object_type: TYPE_VEHICLE
    states {
      center_x: 2.5
      center_y: 2.2
      center_z: 2.1
      length: 1
      width: 1
      height: 1
      heading: 0.3
      velocity_x: 0.1
      velocity_y: 0.1
      valid: true
    }
    states {
      center_x: 2.6
      center_y: 2.3
      center_z: 2.1
      length: 1
      width: 1
      height: 1
      heading: 0.3
      velocity_x: 0
      velocity_y: 0
      valid: true
    }
    states {
      center_x: 2.6
      center_y: 2.3
      center_z: 2.1
      length: 1
      width: 1
      height: 1
      heading: 0.3
      velocity_x: 0.1
      velocity_y: 0.1
      valid: true
    }
    states {
      center_x: 2.7
      center_y: 2.4
      center_z: 2.2
      length: 1
      width: 1
      height: 1
      heading: 0.3
      velocity_x: 0.1
      velocity_y: 0.1
      valid: true
    }
  }
  sdc_track_index: 0
  map_features {
    id: 1
    stop_sign {
      lane: 3
      position {
        x: 5
        y: 5
        z: 3
      }
    }
  }
  map_features {
    id: 2
    lane {
      speed_limit_mph: 35
      type: TYPE_SURFACE_STREET
      interpolating: false
      polyline {
        x: 1
        y: 1
        z: 1
      }
      polyline {
        x: 1
        y: 2
        z: 1
      }
      polyline {
        x: 1
        y: 3
        z: 1
      }
    }
  }
  dynamic_map_states {
    lane_states {
      lane: 2
      state: LANE_STATE_ARROW_STOP
      stop_point {
        x: 1.0
        y: 0.0
        z: 3.0
      }
    }
  }
  dynamic_map_states {
    lane_states {
      lane: 2
      state: LANE_STATE_ARROW_STOP
      stop_point {
        x: 2.0
        y: 0.0
        z: 3.0
      }
    }
  }
  dynamic_map_states {
    lane_states {
      lane: 2
      state: LANE_STATE_ARROW_STOP
      stop_point {
        x: 3.0
        y: 0.0
        z: 3.0
      }
    }
  }
  dynamic_map_states {
    lane_states {
      lane: 2
      state: LANE_STATE_ARROW_STOP
      stop_point {
        x: 2.0
        y: 0.0
        z: 3.0
      }
    }
  }
  objects_of_interest: 102
  objects_of_interest: 103
  tracks_to_predict {
    track_index: 0
    difficulty: LEVEL_1
  }
  tracks_to_predict: {
    track_index: 1
    difficulty: LEVEL_1
  }
)";

const char kMapFeatureStr[] = R"(
  id: 2
  lane {
    speed_limit_mph: 35
    type: TYPE_SURFACE_STREET
    interpolating: false
    polyline {
      x: 1
      y: 1
      z: 1
    }
    polyline {
      x: 1
      y: 1.5
      z: 1
    }
    polyline {
      x: 1.5
      y: 1.5
      z: 1
    }
    polyline {
      x: 2.20710678119
      y: 2.20710678119
      z: 1
    }
    polyline {
      x: 2.20710678119
      y: 2.70710678119
      z: 1
    }
  }
)";

TEST(IncrementCounter, KeyExists) {
  std::map<std::string, int> counters;
  counters["key"] = 3;
  IncrementCounter("key", &counters);
  EXPECT_EQ(counters["key"], 4);
  IncrementCounter("key", &counters, 2);
  EXPECT_EQ(counters["key"], 6);
}

TEST(IncrementCounter, KeyNotExists) {
  std::map<std::string, int> counters;
  IncrementCounter("key", &counters);
  EXPECT_EQ(counters["key"], 1);
  counters.clear();
  IncrementCounter("key", &counters, 5);
  EXPECT_EQ(counters["key"], 5);
}

TEST(GetModeledAgents, IncludeSdc) {
 MotionExampleConversionConfig config;
TextFormat::ParseFromString(kTestConfigStr, &config);  // 
  Scenario scenario;
TextFormat::ParseFromString(kScenarioStr, &scenario);  // const
  std::map<std::string, int> counters;
  auto modeled_agents =
      GetModeledAgents(config, scenario, &scenario.tracks()[0],
                       /*last_timestamps_step=*/2, &counters);
  EXPECT_THAT(modeled_agents, UnorderedElementsAre(&scenario.tracks()[0],
                                                   &scenario.tracks()[1]));
}

void ValidateStateFields(
    const absl::flat_hash_map<std::string, tensorflow::Feature>& features,
    const std::string prefix, const std::string time_frame,
    const int64_t valid = 0, const float x = -1, const float y = -1,
    const float z = -1, const float width = -1, const float length = -1,
    const float height = -1, const float velocity_x = -1,
    const float velocity_y = -1, const float speed = -1,
    const float bbox_yaw = -1, const float vel_yaw = -1) {
  std::string timeframe_prefix = absl::StrCat(prefix, time_frame, "/");
  EXPECT_THAT(
      features.at(absl::StrCat(timeframe_prefix, "valid")).int64_list().value(),
      ElementsAre(valid));
  EXPECT_THAT(
      features.at(absl::StrCat(timeframe_prefix, "x")).float_list().value(),
      ElementsAre(x));
  EXPECT_THAT(
      features.at(absl::StrCat(timeframe_prefix, "y")).float_list().value(),
      ElementsAre(y));
  EXPECT_THAT(
      features.at(absl::StrCat(timeframe_prefix, "z")).float_list().value(),
      ElementsAre(z));
  EXPECT_THAT(
      features.at(absl::StrCat(timeframe_prefix, "width")).float_list().value(),
      ElementsAre(width));
  EXPECT_THAT(features.at(absl::StrCat(timeframe_prefix, "length"))
                  .float_list()
                  .value(),
              ElementsAre(length));
  EXPECT_THAT(features.at(absl::StrCat(timeframe_prefix, "height"))
                  .float_list()
                  .value(),
              ElementsAre(height));
  EXPECT_THAT(features.at(absl::StrCat(timeframe_prefix, "velocity_x"))
                  .float_list()
                  .value(),
              ElementsAre(velocity_x));
  EXPECT_THAT(features.at(absl::StrCat(timeframe_prefix, "velocity_y"))
                  .float_list()
                  .value(),
              ElementsAre(velocity_y));
  EXPECT_THAT(
      features.at(absl::StrCat(timeframe_prefix, "speed")).float_list().value(),
      ElementsAre(speed));
  EXPECT_THAT(features.at(absl::StrCat(timeframe_prefix, "bbox_yaw"))
                  .float_list()
                  .value(),
              ElementsAre(bbox_yaw));
  EXPECT_THAT(features.at(absl::StrCat(timeframe_prefix, "vel_yaw"))
                  .float_list()
                  .value(),
              ElementsAre(FloatEq(vel_yaw)));
}

TEST(AddStateFeatures, ActualStateFeatures) {
 MotionExampleConversionConfig config;
TextFormat::ParseFromString(kTestConfigStr, &config);  // 
  Scenario scenario;
TextFormat::ParseFromString(kScenarioStr, &scenario);  // const
  std::map<std::string, int> counters;
  absl::flat_hash_map<std::string, tensorflow::Feature> features;
  AddStateFeatures(config, scenario.timestamps_seconds(), 2, "state/",
                   &scenario.tracks()[0], /*is_sdc=*/true, /*is_context=*/true,
                   /*objects_of_interest=*/false, /*difficulty_level=*/2,
                   &features, &counters);
  EXPECT_EQ(counters["Agents_State_Feature"], 1);
  EXPECT_EQ(counters["Object_Type_1"], 1);
  EXPECT_EQ(counters["Valid_Object_State"], 3);
  EXPECT_THAT(features["state/id"].float_list().value(), ElementsAre(101));
  EXPECT_THAT(features["state/type"].float_list().value(), ElementsAre(1));
  EXPECT_THAT(features["state/is_context"].int64_list().value(),
              ElementsAre(1));
  EXPECT_THAT(features["state/is_sdc"].int64_list().value(), ElementsAre(1));
  EXPECT_THAT(features["state/objects_of_interest"].int64_list().value(),
              ElementsAre(0));
  EXPECT_THAT(features["state/tracks_to_predict"].int64_list().value(),
              ElementsAre(0));
  EXPECT_THAT(features["state/difficulty_level"].int64_list().value(),
              ElementsAre(2));
  EXPECT_THAT(features["state/past/timestamp_micros"].int64_list().value(),
              ElementsAre(1e6));
  ValidateStateFields(features, "state/", "past", 1, 0.5, 0.2, 0.1, 1, 1, 1,
                      0.1, 0.1, std::sqrt(0.02), 0.3, 0.785398);
  EXPECT_THAT(features["state/current/timestamp_micros"].int64_list().value(),
              ElementsAre(2e6));
  ValidateStateFields(features, "state/", "current", 1, 0.6, 0.3, 0.1, 1, 1, 1,
                      0, 0, 0, 0.3, 0.0);
  EXPECT_THAT(features["state/future/timestamp_micros"].int64_list().value(),
              ElementsAre(3e6));
  ValidateStateFields(features, "state/", "future", 1, 0.6, 0.3, 0.1, 1, 1, 1,
                      0.1, 0.1, std::sqrt(0.02), 0.3, 0.785398);
}

TEST(AddStateFeatures, SdcFeatures) {
 MotionExampleConversionConfig config;
TextFormat::ParseFromString(kTestConfigStr, &config);  // 
  Scenario scenario;
TextFormat::ParseFromString(kScenarioStr, &scenario);  // const
  std::map<std::string, int> counters;
  absl::flat_hash_map<std::string, tensorflow::Feature> features;
  AddStateFeatures(config, scenario.timestamps_seconds(), 2, "sdc/",
                   &scenario.tracks()[0], /*is_sdc=*/true, /*is_context=*/false,
                   /*objects_of_interest=*/false, /*difficulty_level=*/0,
                   &features, &counters);
  EXPECT_EQ(counters["Sdc_State_Feature"], 1);
  EXPECT_THAT(features["sdc/id"].float_list().value(), ElementsAre(101));
  EXPECT_THAT(features["sdc/type"].float_list().value(), ElementsAre(1));
  EXPECT_THAT(features["sdc/past/timestamp_micros"].int64_list().value(),
              ElementsAre(1e6));
  ValidateStateFields(features, "sdc/", "past", 1, 0.5, 0.2, 0.1, 1, 1, 1, 0.1,
                      0.1, std::sqrt(0.02), 0.3, 0.785398);
  EXPECT_THAT(features["sdc/current/timestamp_micros"].int64_list().value(),
              ElementsAre(2e6));
  ValidateStateFields(features, "sdc/", "current", 1, 0.6, 0.3, 0.1, 1, 1, 1, 0,
                      0, 0, 0.3, 0.0);
  EXPECT_THAT(features["sdc/future/timestamp_micros"].int64_list().value(),
              ElementsAre(3e6));
  ValidateStateFields(features, "sdc/", "future", 1, 0.6, 0.3, 0.1, 1, 1, 1,
                      0.1, 0.1, std::sqrt(0.02), 0.3, 0.785398);
}

TEST(AddStateFeatures, PaddedFeatures) {
 MotionExampleConversionConfig config;
TextFormat::ParseFromString(kTestConfigStr, &config);  // 
  Scenario scenario;
TextFormat::ParseFromString(kScenarioStr, &scenario);  // const
  std::map<std::string, int> counters;
  absl::flat_hash_map<std::string, tensorflow::Feature> features;
  AddStateFeatures(config, scenario.timestamps_seconds(), 2, "state/", nullptr,
                   /*is_sdc=*/false, /*is_context=*/true,
                   /*objects_of_interest=*/false, /*difficulty_level=*/-1,
                   &features, &counters);
  EXPECT_EQ(counters["Padded_State_Feature"], 1);
  EXPECT_THAT(features["state/id"].float_list().value(), ElementsAre(-1));
  EXPECT_THAT(features["state/type"].float_list().value(), ElementsAre(-1));
  EXPECT_THAT(features["state/is_context"].int64_list().value(),
              ElementsAre(-1));
  EXPECT_THAT(features["state/is_sdc"].int64_list().value(), ElementsAre(-1));
  EXPECT_THAT(features["state/tracks_to_predict"].int64_list().value(),
              ElementsAre(-1));
  EXPECT_THAT(features["state/difficulty_level"].int64_list().value(),
              ElementsAre(-1));
  EXPECT_THAT(features["state/objects_of_interest"].int64_list().value(),
              ElementsAre(-1));
  EXPECT_THAT(features["state/past/timestamp_micros"].int64_list().value(),
              ElementsAre(-1));
  ValidateStateFields(features, "state/", "past");
  EXPECT_THAT(features["state/current/timestamp_micros"].int64_list().value(),
              ElementsAre(-1));
  ValidateStateFields(features, "state/", "current");
  EXPECT_THAT(features["state/future/timestamp_micros"].int64_list().value(),
              ElementsAre(-1));
  ValidateStateFields(features, "state/", "future");
}

TEST(AddTrafficLightStateFeatures, Normal) {
 MotionExampleConversionConfig config;
TextFormat::ParseFromString(kTestConfigStr, &config);  // 
  Scenario scenario;
TextFormat::ParseFromString(kScenarioStr, &scenario);  // const
  std::map<std::string, int> counters;
  absl::flat_hash_map<std::string, tensorflow::Feature> features;
  auto status = AddTrafficLightStateFeatures(
      config, scenario.dynamic_map_states(), scenario.timestamps_seconds(), 2,
      &features, &counters);
  EXPECT_TRUE(status.ok());
  EXPECT_EQ(counters["Actual_Traffic_Light_State_Features"], 3);
  EXPECT_EQ(counters["Padded_Traffic_Light_State_Features"], 3);
  EXPECT_THAT(features["traffic_light_state/past/timestamp_micros"]
                  .int64_list()
                  .value(),
              ElementsAre(1e6));
  EXPECT_THAT(features["traffic_light_state/current/timestamp_micros"]
                  .int64_list()
                  .value(),
              ElementsAre(2e6));
  EXPECT_THAT(features["traffic_light_state/future/timestamp_micros"]
                  .int64_list()
                  .value(),
              ElementsAre(3e6));
  EXPECT_THAT(features["traffic_light_state/past/x"].float_list().value(),
              ElementsAre(1, -1));
  EXPECT_THAT(features["traffic_light_state/current/x"].float_list().value(),
              ElementsAre(2, -1));
  EXPECT_THAT(features["traffic_light_state/future/x"].float_list().value(),
              ElementsAre(3, -1));
  EXPECT_THAT(features["traffic_light_state/past/y"].float_list().value(),
              ElementsAre(0, -1));
  EXPECT_THAT(features["traffic_light_state/current/y"].float_list().value(),
              ElementsAre(0, -1));
  EXPECT_THAT(features["traffic_light_state/future/y"].float_list().value(),
              ElementsAre(0, -1));
  EXPECT_THAT(features["traffic_light_state/past/z"].float_list().value(),
              ElementsAre(3, -1));
  EXPECT_THAT(features["traffic_light_state/current/z"].float_list().value(),
              ElementsAre(3, -1));
  EXPECT_THAT(features["traffic_light_state/future/z"].float_list().value(),
              ElementsAre(3, -1));
  EXPECT_THAT(features["traffic_light_state/past/state"].int64_list().value(),
              ElementsAre(1, -1));
  EXPECT_THAT(
      features["traffic_light_state/current/state"].int64_list().value(),
      ElementsAre(1, -1));
  EXPECT_THAT(features["traffic_light_state/future/state"].int64_list().value(),
              ElementsAre(1, -1));
  EXPECT_THAT(features["traffic_light_state/past/id"].int64_list().value(),
              ElementsAre(2, -1));
  EXPECT_THAT(features["traffic_light_state/current/id"].int64_list().value(),
              ElementsAre(2, -1));
  EXPECT_THAT(features["traffic_light_state/future/id"].int64_list().value(),
              ElementsAre(2, -1));
  EXPECT_THAT(features["traffic_light_state/past/valid"].int64_list().value(),
              ElementsAre(1, 0));
  EXPECT_THAT(
      features["traffic_light_state/current/valid"].int64_list().value(),
      ElementsAre(1, 0));
  EXPECT_THAT(features["traffic_light_state/future/valid"].int64_list().value(),
              ElementsAre(1, 0));
}

TEST(AddSinglePointFeature, MiddlePoint) {
  Scenario scenario;
TextFormat::ParseFromString(kScenarioStr, &scenario);  // const
  const auto& map_feature = scenario.map_features()[1];

  absl::flat_hash_map<std::string, tensorflow::Feature> features;
  AddSinglePointFeature(map_feature.id(), 2, map_feature.lane().polyline()[0],
                        map_feature.lane().polyline()[1], /*valid=*/1,
                        /*prefix=*/"roadgraph_samples/", &features);
  EXPECT_THAT(features["roadgraph_samples/id"].int64_list().value(),
              ElementsAre(2));
  EXPECT_THAT(features["roadgraph_samples/type"].int64_list().value(),
              ElementsAre(2));
  EXPECT_THAT(features["roadgraph_samples/xyz"].float_list().value(),
              ElementsAre(1.0, 1.0, 1.0));
  EXPECT_THAT(features["roadgraph_samples/valid"].int64_list().value(),
              ElementsAre(1));
  EXPECT_THAT(features["roadgraph_samples/dir"].float_list().value(),
              ElementsAre(0.0, 1.0, 0.0));
}

TEST(AddSinglePointFeature, LastPoint) {
  Scenario scenario;
TextFormat::ParseFromString(kScenarioStr, &scenario);  // const
  const auto& map_feature = scenario.map_features()[1];

  absl::flat_hash_map<std::string, tensorflow::Feature> features;
  AddSinglePointFeature(map_feature.id(), 6, map_feature.lane().polyline()[2],
                        map_feature.lane().polyline()[2], /*valid=*/1,
                        /*prefix=*/"roadgraph_samples/", &features);
  EXPECT_THAT(features["roadgraph_samples/id"].int64_list().value(),
              ElementsAre(2));
  EXPECT_THAT(features["roadgraph_samples/type"].int64_list().value(),
              ElementsAre(6));
  EXPECT_THAT(features["roadgraph_samples/xyz"].float_list().value(),
              ElementsAre(1.0, 3.0, 1.0));
  EXPECT_THAT(features["roadgraph_samples/valid"].int64_list().value(),
              ElementsAre(1));
  EXPECT_THAT(features["roadgraph_samples/dir"].float_list().value(),
              ElementsAre(0.0, 0.0, 0.0));
}

TEST(AddSinglePointFeature, LastPointStopSign) {
  Scenario scenario;
TextFormat::ParseFromString(kScenarioStr, &scenario);  // const
  const auto& map_feature = scenario.map_features()[0];

  absl::flat_hash_map<std::string, tensorflow::Feature> features;
  AddSinglePointFeature(map_feature.id(), 17,
                        map_feature.stop_sign().position(),
                        map_feature.stop_sign().position(), /*valid=*/1,
                        /*prefix=*/"roadgraph_segments/", &features);
  EXPECT_THAT(features["roadgraph_segments/id"].int64_list().value(),
              ElementsAre(1));
  EXPECT_THAT(features["roadgraph_segments/type"].int64_list().value(),
              ElementsAre(17));
  EXPECT_THAT(features["roadgraph_segments/xyz"].float_list().value(),
              ElementsAre(5.0, 5.0, 3.0));
  EXPECT_THAT(features["roadgraph_segments/valid"].int64_list().value(),
              ElementsAre(1));
  EXPECT_THAT(features["roadgraph_segments/dir"].float_list().value(),
              ElementsAre(0.0, 0.0, 0.0));
}

TEST(AddInterpolatedRoadgraphSamples, InterpolateEqualToSource) {
  MapFeature map_feature;
TextFormat::ParseFromString(kMapFeatureStr, &map_feature);  // const

  absl::flat_hash_map<std::string, tensorflow::Feature> features;
  MotionExampleConversionConfig config;
  config.set_polyline_sample_spacing(0.5);
  int num_points = 0;
  AddInterpolatedRoadGraphSamples(config, map_feature.id(), /*type=*/5,
                                  map_feature.lane().polyline(),
                                  /*valid=*/1, &num_points, &features);
  EXPECT_THAT(features["roadgraph_samples/id"].int64_list().value(),
              ElementsAre(2, 2, 2, 2, 2));
  EXPECT_THAT(features["roadgraph_samples/type"].int64_list().value(),
              ElementsAre(5, 5, 5, 5, 5));
  EXPECT_THAT(
      features["roadgraph_samples/xyz"].float_list().value(),
      ElementsAre(1, 1, 1, 1, 1.5, 1, 1.5, 1.5, 1, FloatNear(2.2071068, 1e-5),
                  FloatNear(2.2071068, 1e-5), 1, FloatNear(2.2071068, 1e-5),
                  FloatNear(2.7071068, 1e-5), 1));
  EXPECT_THAT(features["roadgraph_samples/valid"].int64_list().value(),
              ElementsAre(1, 1, 1, 1, 1));
  EXPECT_THAT(features["roadgraph_samples/dir"].float_list().value(),
              ElementsAre(0, 1, 0, 1, 0, 0, FloatNear(0.70710677, 1e-5),
                          FloatNear(0.70710677, 1e-5), 0, 0, 1, 0, 0, 0, 0));
}

TEST(AddInterpolatedRoadgraphSamples, InterpolateDoubleSource) {
  MapFeature map_feature;
TextFormat::ParseFromString(kMapFeatureStr, &map_feature);  // const

  absl::flat_hash_map<std::string, tensorflow::Feature> features;
  MotionExampleConversionConfig config;
  config.set_polyline_sample_spacing(1.0);
  int num_points = 0;
  AddInterpolatedRoadGraphSamples(config, map_feature.id(), /*type=*/5,
                                  map_feature.lane().polyline(),
                                  /*valid=*/1, &num_points, &features);
  EXPECT_THAT(features["roadgraph_samples/id"].int64_list().value(),
              ElementsAre(2, 2, 2));
  EXPECT_THAT(features["roadgraph_samples/type"].int64_list().value(),
              ElementsAre(5, 5, 5));
  EXPECT_THAT(features["roadgraph_samples/xyz"].float_list().value(),
              ElementsAre(1, 1, 1, 1.5, 1.5, 1, FloatNear(2.2071068, 1e-5),
                          FloatNear(2.7071068, 1e-5), 1));
  EXPECT_THAT(features["roadgraph_samples/valid"].int64_list().value(),
              ElementsAre(1, 1, 1));
  EXPECT_THAT(features["roadgraph_samples/dir"].float_list().value(),
              ElementsAre(FloatNear(0.707107, 1e-5), FloatNear(0.707107, 1e-5),
                          0, FloatNear(0.505449, 1e-5),
                          FloatNear(0.862856, 1e-5), 0, 0, 0, 0));
}

TEST(AddInterpolatedRoadgraphSamples, InterpolateFractional) {
  MapFeature map_feature;
TextFormat::ParseFromString(kMapFeatureStr, &map_feature);  // const

  absl::flat_hash_map<std::string, tensorflow::Feature> features;
  MotionExampleConversionConfig config;
  config.set_polyline_sample_spacing(0.75);
  int num_points = 0;
  AddInterpolatedRoadGraphSamples(config, map_feature.id(), /*type=*/5,
                                  map_feature.lane().polyline(),
                                  /*valid=*/1, &num_points, &features);
  EXPECT_THAT(features["roadgraph_samples/id"].int64_list().value(),
              ElementsAre(2, 2, 2, 2));
  EXPECT_THAT(features["roadgraph_samples/type"].int64_list().value(),
              ElementsAre(5, 5, 5, 5));
  EXPECT_THAT(
      features["roadgraph_samples/xyz"].float_list().value(),
      ElementsAre(1, 1, 1, FloatNear(1.20581, 1e-4), FloatNear(1.48706, 1e-4),
                  1, FloatNear(2.20711, 1e-4), FloatNear(2.20711, 1e-4), 1,
                  FloatNear(2.20711, 1e-4), FloatNear(2.70711, 1e-4), 1));
  EXPECT_THAT(features["roadgraph_samples/valid"].int64_list().value(),
              ElementsAre(1, 1, 1, 1));
  EXPECT_THAT(features["roadgraph_samples/dir"].float_list().value(),
              ElementsAre(FloatNear(0.389229, 1e-4), FloatNear(0.921141, 1e-4),
                          0, FloatNear(0.811875, 1e-4),
                          FloatNear(0.583832, 1e-4), 0, 0, 1, 0, 0, 0, 0));
}

TEST(ConvertExampleUtils, AddCrosswalk) {
  const char kCrosswalkStr[] = R"(
    id: 2
    crosswalk {
      polygon{
        x: 0
        y: 0
        z: 0
      }
      polygon{
        x: 9
        y: 0
        z: 0
      }
      polygon{
        x: 9
        y: 6
        z: 0
      }
      polygon{
        x: 0
        y: 6
        z: 0
      }
    }
  )";
  MapFeature map_feature;
TextFormat::ParseFromString(kCrosswalkStr, &map_feature);  // const
  absl::flat_hash_map<std::string, tensorflow::Feature> features;

  MotionExampleConversionConfig config;
  config.set_polygon_sample_spacing(3.0);
  config.set_max_roadgraph_samples(10000);
  int num_points = 0;
  AddPolygonSamples(config, map_feature.id(), /*type=*/5,
                    map_feature.crosswalk().polygon(),
                    /*valid=*/1, &num_points, &features);
  EXPECT_THAT(features["roadgraph_samples/id"].int64_list().value(),
              ElementsAre(2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2));
  EXPECT_THAT(features["roadgraph_samples/type"].int64_list().value(),
              ElementsAre(5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5));
  EXPECT_THAT(features["roadgraph_samples/xyz"].float_list().value(),
              ElementsAre(0, 0, 0, 3, 0, 0, 6, 0, 0, 9, 0, 0, 9, 3, 0, 9, 6, 0,
                          6, 6, 0, 3, 6, 0, 0, 6, 0, 0, 3, 0, 0, 0, 0));
  EXPECT_THAT(features["roadgraph_samples/valid"].int64_list().value(),
              ElementsAre(1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1));
  EXPECT_THAT(features["roadgraph_samples/dir"].float_list().value(),
              ElementsAre(1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, -1, 0, 0,
                          -1, 0, 0, -1, 0, 0, 0, -1, 0, 0, -1, 0, 0, 0, 0));
}

TEST(ValidateScenarioProto, ValidScenario) {
  Scenario scenario;
TextFormat::ParseFromString(kScenarioStr, &scenario);  // const
  std::map<std::string, int> counters;

  EXPECT_TRUE(ValidateScenarioProto(scenario, &counters));
  EXPECT_EQ(counters["Valid_Scenario"], 1);
}

TEST(ValidateScenarioProto, EmptyScenario) {
  const Scenario scenario;
  std::map<std::string, int> counters;
  EXPECT_FALSE(ValidateScenarioProto(scenario, &counters));
  EXPECT_EQ(counters["Empty_Scenario"], 1);
}

TEST(ValidateScenarioProto, ScenarioInvalidSdcIndex) {
 Scenario scenario;
TextFormat::ParseFromString(kScenarioStr, &scenario);  // 
  scenario.set_sdc_track_index(200);
  std::map<std::string, int> counters;

  EXPECT_FALSE(ValidateScenarioProto(scenario, &counters));
  EXPECT_EQ(counters["Scenario_Sdc_Index_No_Match"], 1);
}

TEST(ValidateScenarioProto, TrackStatesSizeNotMatched) {
  Scenario scenario;
TextFormat::ParseFromString(R"pb(
    timestamps_seconds: 1.0
    timestamps_seconds: 2.0
    tracks {
      id: 101
      object_type: TYPE_VEHICLE
      states {
        center_x: 0.5
        center_y: 0.2
        center_z: 0.1
        length: 1
        width: 1
        height: 1
        heading: 0.3
        velocity_x: 0.1
        velocity_y: 0.1
        valid: true
      }
    }
    objects_of_interest: 101
  )pb", &scenario);  // const
  std::map<std::string, int> counters;

  EXPECT_FALSE(ValidateScenarioProto(scenario, &counters));
  EXPECT_EQ(counters["Track_States_Size_Not_Match_Timestamps_Step_Size"], 1);
}

TEST(ScenarioToExample, Convert) {
 MotionExampleConversionConfig config;
TextFormat::ParseFromString(kTestConfigStr, &config);  // 
  config.set_max_num_agents(4);
  config.set_num_future_steps(2);
  config.set_polygon_sample_spacing(0);
  config.set_polyline_sample_spacing(0);

 Scenario scenario;
TextFormat::ParseFromString(kScenarioStr, &scenario);  // 
  scenario.clear_tracks_to_predict();
  RequiredPrediction& required_track = *scenario.add_tracks_to_predict();
  required_track.set_track_index(1);
  required_track.set_difficulty(RequiredPrediction::LEVEL_1);
  RequiredPrediction& required_track2 = *scenario.add_tracks_to_predict();
  required_track2.set_track_index(3);
  required_track2.set_difficulty(RequiredPrediction::LEVEL_2);

  absl::StatusOr<tensorflow::Example> example =
      ScenarioToExample(scenario, config);
  ASSERT_TRUE(example.ok());

  const auto& example_features = example.value().features().feature();
  absl::flat_hash_map<std::string, tensorflow::Feature> features(
      example_features.begin(), example_features.end());

  std::vector<std::string> expected_features = {
      "roadgraph_samples/xyz",
      "roadgraph_samples/dir",
      "roadgraph_samples/type",
      "roadgraph_samples/valid",
      "roadgraph_samples/id",
      "scenario/id",
      "state/current/x",
      "state/current/y",
      "state/current/z",
      "state/current/bbox_yaw",
      "state/current/length",
      "state/current/width",
      "state/current/height",
      "state/current/speed",
      "state/current/timestamp_micros",
      "state/current/vel_yaw",
      "state/current/velocity_x",
      "state/current/velocity_y",
      "state/current/valid",
      "state/future/x",
      "state/future/y",
      "state/future/z",
      "state/future/bbox_yaw",
      "state/future/length",
      "state/future/width",
      "state/future/height",
      "state/future/speed",
      "state/future/timestamp_micros",
      "state/future/vel_yaw",
      "state/future/velocity_x",
      "state/future/velocity_y",
      "state/future/valid",
      "state/past/x",
      "state/past/y",
      "state/past/z",
      "state/past/bbox_yaw",
      "state/past/length",
      "state/past/width",
      "state/past/height",
      "state/past/speed",
      "state/past/timestamp_micros",
      "state/past/vel_yaw",
      "state/past/velocity_x",
      "state/past/velocity_y",
      "state/past/valid",
      "state/tracks_to_predict",
      "state/objects_of_interest",
      "state/id",
      "state/is_sdc",
      "state/type",
      "traffic_light_state/current/state",
      "traffic_light_state/current/x",
      "traffic_light_state/current/y",
      "traffic_light_state/current/z",
      "traffic_light_state/current/id",
      "traffic_light_state/current/valid",
      "traffic_light_state/current/timestamp_micros",
      "traffic_light_state/future/state",
      "traffic_light_state/future/x",
      "traffic_light_state/future/y",
      "traffic_light_state/future/z",
      "traffic_light_state/future/timestamp_micros",
      "traffic_light_state/future/id",
      "traffic_light_state/future/valid",
      "traffic_light_state/past/state",
      "traffic_light_state/past/x",
      "traffic_light_state/past/y",
      "traffic_light_state/past/z",
      "traffic_light_state/past/id",
      "traffic_light_state/past/valid",
      "traffic_light_state/past/timestamp_micros"};
  for (const auto& feature_key : expected_features) {
    ASSERT_TRUE(features.contains(feature_key.c_str()));
  }

  EXPECT_THAT(features["scenario/id"].bytes_list().value(), ElementsAre("123"));
  EXPECT_THAT(features["state/id"].float_list().value(),
              ElementsAre(102, 104, 101, 103));
  EXPECT_THAT(features["state/type"].float_list().value(),
              ElementsAre(1, 1, 1, 2));
  // Only objects in tracks_to_predict will be modeled.
  EXPECT_THAT(features["state/is_context"].int64_list().value(),
              ElementsAre(0, 0, 1, 1));
  EXPECT_THAT(features["state/is_sdc"].int64_list().value(),
              ElementsAre(0, 0, 1, 0));
  EXPECT_THAT(features["state/tracks_to_predict"].int64_list().value(),
              ElementsAre(1, 1, 0, 0));
  EXPECT_THAT(features["state/difficulty_level"].int64_list().value(),
              ElementsAre(1, 2, 0, 0));
  EXPECT_THAT(features["state/objects_of_interest"].int64_list().value(),
              ElementsAre(1, 0, 0, 1));
  EXPECT_THAT(features["state/past/timestamp_micros"].int64_list().value(),
              ElementsAre(1e6, 1e6, 1e6, 1e6));
  EXPECT_THAT(features["state/current/timestamp_micros"].int64_list().value(),
              ElementsAre(2e6, 2e6, 2e6, -1));
  EXPECT_THAT(features["state/future/timestamp_micros"].int64_list().value(),
              ElementsAre(3e6, 4e6, 3e6, 4e6, 3e6, 4e6, -1, -1));
  EXPECT_THAT(features["roadgraph_samples/id"].int64_list().value(),
              ElementsAre(2, 2, 2, 1, -1, -1));
  EXPECT_THAT(features["roadgraph_samples/type"].int64_list().value(),
              ElementsAre(2, 2, 2, 17, -1, -1));
  EXPECT_THAT(features["roadgraph_samples/xyz"].float_list().value(),
              ElementsAre(1.0, 1.0, 1.0, 1.0, 2.0, 1.0, 1.0, 3.0, 1.0, 5.0, 5.0,
                          3.0, -1, -1, -1, -1, -1, -1));
  EXPECT_THAT(features["roadgraph_samples/valid"].int64_list().value(),
              ElementsAre(1, 1, 1, 1, 0, 0));
  EXPECT_THAT(features["roadgraph_samples/dir"].float_list().value(),
              ElementsAre(0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                          0.0, -1, -1, -1, -1, -1, -1));
  EXPECT_THAT(features["state/current/valid"].int64_list().value(),
              ElementsAre(1, 1, 1, 0));
  EXPECT_THAT(features["state/current/x"].float_list().value(),
              ElementsAre(1.6, 2.6, 0.6, -1));
  EXPECT_THAT(features["state/past/valid"].int64_list().value(),
              ElementsAre(1, 1, 1, 1));
  EXPECT_THAT(features["state/past/x"].float_list().value(),
              ElementsAre(1.5, 2.5, 0.5, 2.5));
  EXPECT_THAT(features["state/future/valid"].int64_list().value(),
              ElementsAre(1, 1, 1, 1, 1, 1, 0, 0));
  EXPECT_THAT(features["state/future/x"].float_list().value(),
              ElementsAre(1.6, 1.7, 2.6, 2.7, 0.6, 0.7, -1, -1));
  EXPECT_THAT(features["state/id"].float_list().value(),
              ElementsAre(102, 104, 101, 103));
  EXPECT_THAT(features["traffic_light_state/past/timestamp_micros"]
                  .int64_list()
                  .value(),
              ElementsAre(1e6));
  EXPECT_THAT(features["traffic_light_state/current/timestamp_micros"]
                  .int64_list()
                  .value(),
              ElementsAre(2e6));
  EXPECT_THAT(features["traffic_light_state/future/timestamp_micros"]
                  .int64_list()
                  .value(),
              ElementsAre(3e6, 4e6));
}

}  // namespace
}  // namespace internal
}  // namespace open_dataset
}  // namespace waymo
