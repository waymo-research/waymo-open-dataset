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

#ifndef WAYMO_OPEN_DATASET_DATA_CONVERSION_SCENARIO_CONVERSION_H_
#define WAYMO_OPEN_DATASET_DATA_CONVERSION_SCENARIO_CONVERSION_H_

#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "tensorflow/core/example/example.pb.h"
#include "tensorflow/core/example/feature.pb.h"
#include "waymo_open_dataset/protos/conversion_config.pb.h"
#include "waymo_open_dataset/protos/map.pb.h"
#include "waymo_open_dataset/protos/scenario.pb.h"

namespace waymo {
namespace open_dataset {

// All invalid fields will be set to this value.
constexpr float kInvalidFieldValue = -1.0;

// Converts a Scenario proto to a tensorflow Example proto.
// The features are described at https://waymo.com/open/data/motion/tfexample.
// scenario: The scenario proto to be converted.
// conversion_config: Contains configuration settings to convert from Scenario
//    proto to tf.Examples used in bp model training.
// counters: A pointer to the counters map to be updated.
// results: A pointer to the result Example proto.
absl::StatusOr<tensorflow::Example> ScenarioToExample(
    const Scenario& scenario, const MotionExampleConversionConfig& config,
    std::map<std::string, int>* counters = nullptr);

// Functions in the internal namespace are exposed for testing only.
namespace internal {

// Increments counters for the given key.
void IncrementCounter(const std::string& key,
                      std::map<std::string, int>* counters,
                      int increment_by = 1);

// Gets a list of modeled agents from scenario given the last timestamp step.
// The last step index determines the last scenario step that will be converted.
// conversion_config: Contains configuration settings to convert from Scenario
//    proto to tf.Examples used in bp model training.
// scenario: A Scenario object storing all track related informations.
// sdc_track: SDC track state for the current timestamp step.
// last_timestamps_step: The timestamps index in the Scenario proto of the last
//   step in the tf.Example time window.
// counters: A map object storing all counters' values.
absl::flat_hash_set<const Track*> GetModeledAgents(
    const waymo::open_dataset::MotionExampleConversionConfig& conversion_config,
    const Scenario& scenario, const Track* sdc_track, int last_timestamps_step,
    std::map<std::string, int>* counters);

// Adds state/* fields in tensorflow features for each state of each agent.
// Details in go/multipath-tf-example.
// conversion_config: Contains configuration settings to convert from Scenario
//   proto to tf.Examples.
// timestamps_seconds: Timestamps corresponding to the track states for each
//   step in the scenario.
// last_timestamps_step: The timestamps index in the Scenario proto of the last
//   step in the tf.Example time window.
// prefix: Prefix for the to-be-added feature fields. Can either be `sdc/` or
// `state/`.
// track: A Track object pointer. When track is nullptr, all the state/* fields
//   will be filled with default value: -1.
// is_sdc: Whether the state is sdc's state.
// is_context: Whether the state is a context state, only used when adding state
// feature which is not specific for sdc.
// objects_of_interest: Whether the state is a state of an object of
// interest, only used when adding state feature which is not specific for sdc.
// features: A hash map pointer where state features should be stored.
// counters: A map object storing all counters' values.
void AddStateFeatures(
    const waymo::open_dataset::MotionExampleConversionConfig& conversion_config,
    const google::protobuf::RepeatedField<double>& timestamps_seconds,
    int last_timestamps_step, const std::string& prefix, const Track* track,
    bool is_sdc, bool is_context, bool objects_of_interest,
    int difficulty_level,
    absl::flat_hash_map<std::string, tensorflow::Feature>* features,
    std::map<std::string, int>* counters);

// Adds a feature for a single map point.
// id: An unique ID to identify the map feature of the map point.
// type: An unique number to identify road graph map feature type.
// map_point: The current map point, position in meters.
// next_map_point: The next map point of this map feature in the map.
// valid: Non-zero if the point is not a padded map point. Valid values are
//   {0, 1}.
// prefix: Prefix string for keys of tensorflow features. Can either be
// `roadgraph_samples/` or `roadgraph_segments/`.
// features: A hash map pointer where state features should be stored.
void AddSinglePointFeature(
    int64_t id, int64_t type, const MapPoint& map_point,
    const MapPoint& next_map_point, int valid, const std::string& prefix,
    absl::flat_hash_map<std::string, tensorflow::Feature>* features);

// Adds roadgraph_samples/* and roadgraph_segments/* fields in tensorflow
// features for static map features. Details in go/multipath-tf-example.
// example_config: ExampleConfig contains configuration settings to convert
//   from Scenario proto to tf.Examples used in bp model training.
// map_features: A list of static map features.
// features: A hash map pointer where state features should be stored.
// counters: A map object storing all counters' values.
absl::Status AddRoadGraphFeatures(
    const waymo::open_dataset::MotionExampleConversionConfig& conversion_config,
    const google::protobuf::RepeatedPtrField<MapFeature>& map_features,
    absl::flat_hash_map<std::string, tensorflow::Feature>* features,
    std::map<std::string, int>* counters);

// Adds interpolated road graph sample points with spacing defined by the
// given config.
// config: The conversion configuration proto.
// id: The feature ID.
// type: The feature type.
// map_points: A list of polyline points in the feature.
// valid: True if the sample is valid.
// num_points: A pointer to the current number of points that have been added.
// features: The output features map.
void AddInterpolatedRoadGraphSamples(
    const MotionExampleConversionConfig& config, const int64_t id,
    const int type, const google::protobuf::RepeatedPtrField<MapPoint>& map_points,
    const int valid, int* num_points,
    absl::flat_hash_map<std::string, tensorflow::Feature>* features);

// Adds interpolated polygon sample points with spacing defined by the
// given config.
// config: The conversion configuration proto.
// id: The feature ID.
// type: The feature type.
// map_points: A list of polyline points in the feature.
// valid: True if the sample is valid.
// num_points: A pointer to the current number of points that have been added.
// features: The output features map.
void AddPolygonSamples(
    const MotionExampleConversionConfig& config, const int64_t id,
    const int type, const google::protobuf::RepeatedPtrField<MapPoint>& map_points,
    const int valid, int* num_points,
    absl::flat_hash_map<std::string, tensorflow::Feature>* features);

// Adds traffic_light_state/* fields in tensorflow features for dynamic map
// features. Details in go/multipath-tf-example.
// conversion_config: Contains configuration settings for the conversion
//   from Scenario proto to tf.Examples.
// dynamic_map_states: A list of dynamic map states.
// timestamps_seconds: Timestamps corresponding to the track states for each
//   step in the scenario.
// last_timestamps_step: The timestamps index in the Scenario proto of the last
//   step in the tf.Example time window.
// features: A hash map pointer where state features should be stored.
// counters: A map object storing all counters' values.
absl::Status AddTrafficLightStateFeatures(
    const waymo::open_dataset::MotionExampleConversionConfig& conversion_config,
    const google::protobuf::RepeatedPtrField<DynamicMapState>& dynamic_map_states,
    const google::protobuf::RepeatedField<double>& timestamps_seconds,
    int last_timestamps_step,
    absl::flat_hash_map<std::string, tensorflow::Feature>* features,
    std::map<std::string, int>* counters);

// Converts from a Map::FeatureDataCase value to an Example proto feature type.
int64_t GetMapFeatureType(int64_t map_feature_base_type,
                          int64_t map_feature_sub_type = kInvalidFieldValue);

// Gets a list of feature types in order of priority to be included.
std::vector<MapFeature::FeatureDataCase> GetFeaturePriorityList();

// Returns the difficulty level of the given track.
int DifficultyLevel(const Scenario& scenario, const Track* t);

// Validate Scenario proto - not empty and the number of states in each track
// matches the number of timestamps seconds.
bool ValidateScenarioProto(const Scenario& scenario,
                           std::map<std::string, int>* counters);

// Adds tensorflow features for a given timestamp step.
// scenario: A Scenario object storing all track related informations.
// conversion_config: Contains configuration settings to convert
//   from Scenario proto to tf.Examples used in bp model training.
// last_timestamps_step: The timestamps index in the Scenario proto of the
// last
//   step in the tf.Example time window.
// features: A hash map pointer where state features should be stored.
// counters: A map object storing all counters' values.
absl::Status AddTfFeatures(
    const Scenario& scenario,
    const MotionExampleConversionConfig& conversion_config,
    int last_timestamps_step,
    absl::flat_hash_map<std::string, tensorflow::Feature>* features,
    std::map<std::string, int>* counters);

}  // namespace internal
}  // namespace open_dataset
}  // namespace waymo

#endif  // WAYMO_OPEN_DATASET_DATA_CONVERSION_SCENARIO_CONVERSION_H_
