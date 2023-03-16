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

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <map>
#include <set>
#include <string>
#include <utility>
#include <vector>


#include <glog/logging.h>
#include "google/protobuf/repeated_field.h"
#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "tensorflow/core/example/example.pb.h"
#include "tensorflow/core/example/feature.pb.h"
#include "waymo_open_dataset/math/vec3d.h"
#include "waymo_open_dataset/protos/conversion_config.pb.h"
#include "waymo_open_dataset/protos/map.pb.h"
#include "waymo_open_dataset/protos/scenario.pb.h"

namespace waymo {
namespace open_dataset {
namespace {

int64_t GetLaneType(int64_t lane_type) {
  switch (lane_type) {
    case LaneCenter::TYPE_UNDEFINED:
      return 0;
    case LaneCenter::TYPE_FREEWAY:
      return 1;
    case LaneCenter::TYPE_SURFACE_STREET:
      return 2;
    case LaneCenter::TYPE_BIKE_LANE:
      return 3;
    default:
      break;
  }
  return kInvalidFieldValue;
}

int64_t GetRoadLineType(int64_t road_line_type) {
  switch (road_line_type) {
    case RoadLine::TYPE_UNKNOWN:
      return 5;
    case RoadLine::TYPE_BROKEN_SINGLE_WHITE:
      return 6;
    case RoadLine::TYPE_SOLID_SINGLE_WHITE:
      return 7;
    case RoadLine::TYPE_SOLID_DOUBLE_WHITE:
      return 8;
    case RoadLine::TYPE_BROKEN_SINGLE_YELLOW:
      return 9;
    case RoadLine::TYPE_BROKEN_DOUBLE_YELLOW:
      return 10;
    case RoadLine::TYPE_SOLID_SINGLE_YELLOW:
      return 11;
    case RoadLine::TYPE_SOLID_DOUBLE_YELLOW:
      return 12;
    case RoadLine::TYPE_PASSING_DOUBLE_YELLOW:
      return 13;
    default:
      break;
  }
  return kInvalidFieldValue;
}

int64_t GetRoadEdgeType(int64_t road_edge_type) {
  switch (road_edge_type) {
    case RoadEdge::TYPE_UNKNOWN:
      return 14;
    case RoadEdge::TYPE_ROAD_EDGE_BOUNDARY:
      return 15;
    case RoadEdge::TYPE_ROAD_EDGE_MEDIAN:
      return 16;
    default:
      break;
  }
  return kInvalidFieldValue;
}

const int64_t kStopSignType = 17;
const int64_t kCrosswalkType = 18;
const int64_t kSpeedBumpType = 19;
const int64_t kDrivewayType = 20;

void AddBytesFeature(
    const std::string& key, const std::string& value,
    absl::flat_hash_map<std::string, tensorflow::Feature>* features) {
  (*features)[key].mutable_bytes_list()->add_value(value);
}

void AddFloatFeature(
    const std::string& key, const float value,
    absl::flat_hash_map<std::string, tensorflow::Feature>* features) {
  (*features)[key].mutable_float_list()->add_value(value);
}

void AddFloatListFeature(
    const std::string& key, const std::vector<float>& values,
    absl::flat_hash_map<std::string, tensorflow::Feature>* features) {
  for (const float value : values) {
    (*features)[key].mutable_float_list()->add_value(value);
  }
}

void AddInt64Feature(
    const std::string& key, const int64_t value,
    absl::flat_hash_map<std::string, tensorflow::Feature>* features) {
  (*features)[key].mutable_int64_list()->add_value(value);
}

// Returns a set of objects of interest.
// scenario: A Scenario object storing all track related information.
// current_timestamp_steps: The timestamps index in the Scenario proto of the
// current timestamp step.
// keep_invalid_tracks: Remove invalid tracks if any in objects of interest.
absl::flat_hash_set<const Track*> GetObjectsOfInterest(
    const Scenario& scenario, const int current_timestamp_steps,
    bool keep_invalid_tracks) {
  absl::flat_hash_set<const Track*> objects_of_interest;
  for (auto object_id : scenario.objects_of_interest()) {
    for (int i = 0; i < scenario.tracks().size(); ++i) {
      const Track* track = &(scenario.tracks()[i]);
      if (track->id() == object_id &&
          (keep_invalid_tracks ||
           track->states()[current_timestamp_steps].valid())) {
        objects_of_interest.insert(track);
        break;
      }
    }
  }
  return objects_of_interest;
}

// Convert a MapPoint to a Vec3d.
inline Vec3d ToVec3d(const MapPoint& point) {
  return Vec3d(point.x(), point.y(), point.z());
}

// Interpolate a single point given two points and alpha, the fraction between
// them.
Vec3d InterpolatePoint(const MapPoint& left_point, const MapPoint& right_point,
                       const double alpha) {
  const Vec3d left = ToVec3d(left_point);
  return left + alpha * (ToVec3d(right_point) - left);
}

// Compute normalized direction vectors for the given set of points.
std::vector<Vec3d> ComputeDirectionVectors(const std::vector<Vec3d>& points) {
  // Compute normalized direction vectors for each point.
  std::vector<Vec3d> result(points.size());
  for (int i = 0; i < points.size() - 1; ++i) {
    result[i] = (points[i + 1] - points[i]).Normalized();
  }

  // Set the last direction vector to [0, 0, 0].
  result[points.size() - 1] = Vec3d(0, 0, 0);
  return result;
}

// Computes the Catmull-Rom cubic spline interpolation in 3D. The 4 input points
// must be equally spaced. The alpha parameter defines the interpolation
// fraction between p1 and p2 as a value from 0 to 1. Returns the interpolated
// 3D value.
Vec3d InterpolateCubicSpline(const Vec3d& p0, const Vec3d& p1, const Vec3d& p2,
                             const Vec3d& p3, double alpha) {
  const Vec3d a = 0.5 * (-p0 + 3.0 * p1 - 3.0 * p2 + p3);
  const Vec3d b = 0.5 * (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3);
  const Vec3d c = 0.5 * (-p0 + p2);
  const Vec3d d = p1;
  return d + alpha * (c + alpha * (b + alpha * a));
}

}  // namespace

namespace internal {

int64_t GetMapFeatureType(int64_t map_feature_base_type,
                          int64_t map_feature_sub_type) {
  switch (map_feature_base_type) {
    case MapFeature::kLane: {
      return GetLaneType(map_feature_sub_type);
    }
    case MapFeature::kRoadLine: {
      return GetRoadLineType(map_feature_sub_type);
    }
    case MapFeature::kRoadEdge: {
      return GetRoadEdgeType(map_feature_sub_type);
    }
    case MapFeature::kStopSign: {
      return kStopSignType;
    }
    case MapFeature::kCrosswalk: {
      return kCrosswalkType;
    }
    case MapFeature::kSpeedBump: {
      return kSpeedBumpType;
    }
    case MapFeature::kDriveway: {
      return kDrivewayType;
    }
    default: {
      break;
    }
  }
  return kInvalidFieldValue;
}

// Gets a list of feature types in order of priority to be included. All map
// features of the first type in this list will be included first in the
// roadgraph samples.
std::vector<MapFeature::FeatureDataCase> GetFeaturePriorityList() {
  return {MapFeature::kLane,      MapFeature::kRoadLine,
          MapFeature::kStopSign,  MapFeature::kCrosswalk,
          MapFeature::kSpeedBump, MapFeature::kRoadEdge};
}

absl::Status AddSdcAndAgentsStateFeatures(
    const waymo::open_dataset::MotionExampleConversionConfig& conversion_config,
    const Scenario& scenario, const int last_timestamps_step,
    absl::flat_hash_map<std::string, tensorflow::Feature>* features,
    std::map<std::string, int>* counters) {
  // Gather objects of interest in the Scenario proto.
  const int current_timestamp_steps =
      last_timestamps_step - conversion_config.num_future_steps();
  absl::flat_hash_set<const Track*> objects_of_interest = GetObjectsOfInterest(
      scenario, current_timestamp_steps, /*keep_invalid_tracks=*/true);

  // A lambda function to determine if a track is for an object of interest.
  auto is_objects_of_interest = [&objects_of_interest](const Track* t) {
    return objects_of_interest.find(t) != objects_of_interest.end();
  };

  // Add sdc state.
  const Track* sdc_track = nullptr;
  if (scenario.sdc_track_index() >= 0 &&
      scenario.sdc_track_index() < scenario.tracks().size()) {
    sdc_track = &(scenario.tracks()[scenario.sdc_track_index()]);
  } else {
    return absl::FailedPreconditionError("SDC current state not found.");
  }

  // Add modeled agent states.
  absl::flat_hash_set<const Track*> modeled_agents = GetModeledAgents(
      conversion_config, scenario, sdc_track, last_timestamps_step, counters);
  if (modeled_agents.empty()) {
    IncrementCounter("Zero_Modeled_Agents", counters, 1);
    return absl::FailedPreconditionError(
        "No valid modeled agents in the current track.");
  }

  IncrementCounter("Num_Modeled_Agents", counters, modeled_agents.size());
  std::vector<const Track*> sorted_modeled_agents(modeled_agents.begin(),
                                                  modeled_agents.end());
  // Sort the modeled agents based on their ID to make the order deterministic.
  auto compare_modeled_agents = [](const Track* a, const Track* b) {
    return a->id() < b->id();
  };
  std::sort(sorted_modeled_agents.begin(), sorted_modeled_agents.end(),
            compare_modeled_agents);
  for (const auto agent : sorted_modeled_agents) {
    AddStateFeatures(conversion_config, scenario.timestamps_seconds(),
                     last_timestamps_step, "state/", agent,
                     /*is_sdc=*/agent == sdc_track,
                     /*is_context=*/false,
                     /*objects_of_interest*/ is_objects_of_interest(agent),
                     /*difficulty_level=*/DifficultyLevel(scenario, agent),
                     features, counters);
  }

  // Add context agent states.
  const int max_num_agents = conversion_config.max_num_agents();
  std::vector<const Track*> context_agents;
  for (int i = 0; i < scenario.tracks().size(); ++i) {
    const Track* track = &(scenario.tracks()[i]);
    if (modeled_agents.find(track) != modeled_agents.end()) {
      continue;
    }
    context_agents.push_back(track);
  }

  // Sort the context agents based on their distances to SDC and the total
  // number of valid states.
  auto compare_context_agents = [current_timestamp_steps, sdc_track](
                                    const Track* a, const Track* b) {
    const ObjectState& obj_a = (a->states())[current_timestamp_steps];
    const ObjectState& obj_b = (b->states())[current_timestamp_steps];
    if (obj_a.valid() || obj_b.valid()) {
      if (obj_a.valid() && obj_b.valid()) {
        // If the object state of a and the object state of b are all valid at
        // the current timestamp step, the closer object to the SDC should be
        // prioritized to be used as the context agent.
        const ObjectState& obj_sdc =
            (sdc_track->states())[current_timestamp_steps];
        const double dist_a = std::hypot(obj_a.center_x() - obj_sdc.center_x(),
                                         obj_a.center_y() - obj_sdc.center_y());
        double dist_b = std::hypot(obj_b.center_x() - obj_sdc.center_x(),
                                   obj_b.center_y() - obj_sdc.center_y());
        return dist_a < dist_b;
      } else {
        // If only one of the two objects is valid, the valid object will be
        // prioritized to be used as the context agent.
        return obj_a.valid() ? true : false;
      }
    } else {
      // Otherwise, we count the total number of valid states for the track a
      // and the track b.
      int valid_states_a = 0;
      int valid_states_b = 0;

      for (const auto& state : a->states()) {
        if (state.valid()) {
          valid_states_a += 1;
        }
      }

      for (const auto& state : b->states()) {
        if (state.valid()) {
          valid_states_b += 1;
        }
      }
      return valid_states_a > valid_states_b;
    }
  };

  std::sort(context_agents.begin(), context_agents.end(),
            compare_context_agents);
  const int max_context_agents_size = max_num_agents - modeled_agents.size();
  IncrementCounter("Num_Context_Agents", counters,
                   context_agents.size() > max_context_agents_size
                       ? max_context_agents_size
                       : context_agents.size());
  context_agents.resize(max_num_agents - modeled_agents.size(), nullptr);
  for (const auto agent : context_agents) {
    AddStateFeatures(conversion_config, scenario.timestamps_seconds(),
                     last_timestamps_step, "state/", agent,
                     /*is_sdc=*/agent == sdc_track,
                     /*is_context=*/true,
                     /*objects_of_interest=*/is_objects_of_interest(agent),
                     /*difficulty_level=*/DifficultyLevel(scenario, agent),
                     features, counters);
  }
  return absl::OkStatus();
}

// Add point and direction vector data to the given features map.
void AddRoadgraphData(
    const int64_t id, const int64_t type, const std::vector<Vec3d>& points,
    const std::vector<Vec3d>& direction_vectors, const int valid,
    const std::string& prefix,
    absl::flat_hash_map<std::string, tensorflow::Feature>* features) {
  CHECK(prefix == "roadgraph_samples/" || prefix == "roadgraph_segments/");

  const int num_points = points.size();
  CHECK_EQ(direction_vectors.size(), num_points);

  // Add each point and direction vector to the features.
  for (int i = 0; i < num_points; ++i) {
    AddInt64Feature(absl::StrCat(prefix, "id"), id, features);
    AddInt64Feature(absl::StrCat(prefix, "type"), type, features);
    AddInt64Feature(absl::StrCat(prefix, "valid"), valid, features);
    const Vec3d& point = points[i];
    AddFloatListFeature(absl::StrCat(prefix, "xyz"),
                        std::vector<float>{static_cast<float>(point.x()),
                                           static_cast<float>(point.y()),
                                           static_cast<float>(point.z())},
                        features);
    const Vec3d direction = direction_vectors[i];
    AddFloatListFeature(absl::StrCat(prefix, "dir"),
                        std::vector<float>{static_cast<float>(direction.x()),
                                           static_cast<float>(direction.y()),
                                           static_cast<float>(direction.z())},
                        features);
  }
}

void AddRoadgraphSample(
    const waymo::open_dataset::MotionExampleConversionConfig& conversion_config,
    const MapFeature& map_feature, int* num_points,
    absl::flat_hash_map<std::string, tensorflow::Feature>* features) {
  const int max_points = conversion_config.max_roadgraph_samples();
  switch (map_feature.feature_data_case()) {
    case MapFeature::kLane: {
      if (*num_points >= max_points) break;
      AddInterpolatedRoadGraphSamples(
          conversion_config, map_feature.id(),
          GetMapFeatureType(MapFeature::kLane, map_feature.lane().type()),
          map_feature.lane().polyline(), /*valid=*/1, num_points, features);
      break;
    }
    case MapFeature::kRoadLine: {
      if (*num_points >= max_points) break;
      AddInterpolatedRoadGraphSamples(
          conversion_config, map_feature.id(),
          GetMapFeatureType(MapFeature::kRoadLine,
                            map_feature.road_line().type()),
          map_feature.road_line().polyline(), /*valid=*/1, num_points,
          features);
      break;
    }
    case MapFeature::kRoadEdge: {
      if (*num_points >= max_points) break;
      AddInterpolatedRoadGraphSamples(
          conversion_config, map_feature.id(),
          GetMapFeatureType(MapFeature::kRoadEdge,
                            map_feature.road_edge().type()),
          map_feature.road_edge().polyline(), /*valid=*/1, num_points,
          features);
      break;
    }
    case MapFeature::kStopSign: {
      if (*num_points >= max_points) break;
      std::vector<MapPoint> stop_sign({map_feature.stop_sign().position()});
      AddInterpolatedRoadGraphSamples(conversion_config, map_feature.id(),
                                      GetMapFeatureType(MapFeature::kStopSign),
                                      google::protobuf::RepeatedPtrField<MapPoint>(
                                          stop_sign.begin(), stop_sign.end()),
                                      /*valid=*/1, num_points, features);
      break;
    }
    case MapFeature::kCrosswalk: {
      if (*num_points >= max_points) break;
      AddPolygonSamples(conversion_config, map_feature.id(),
                        GetMapFeatureType(MapFeature::kCrosswalk),
                        map_feature.crosswalk().polygon(),
                        /*valid=*/1, num_points, features);
      break;
    }
    case MapFeature::kSpeedBump: {
      // TODO(settinger): Add polygon sampling for speed bumps.
      if (*num_points >= max_points) break;
      AddPolygonSamples(conversion_config, map_feature.id(),
                        GetMapFeatureType(MapFeature::kSpeedBump),
                        map_feature.speed_bump().polygon(),
                        /*valid=*/1, num_points, features);
      break;
    }
    case MapFeature::kDriveway: {
      if (*num_points >= max_points) break;
      AddPolygonSamples(conversion_config, map_feature.id(),
                        GetMapFeatureType(MapFeature::kDriveway),
                        map_feature.driveway().polygon(),
                        /*valid=*/1, num_points, features);
      break;
    }
    case MapFeature::FEATURE_DATA_NOT_SET: {
      break;
    }
  }
}

// Add roadgraph_samples/* fields in tensorflow features for static map
// features.
// conversion_config: ExampleConfig contains configuration settings to convert
//   from Scenario proto to tf.Examples.
// map_features: A list of static map features.
// features: A hash map pointer where state features should be stored.
// counters: A map object storing all counters' values.
void AddRoadGraphSamples(
    const waymo::open_dataset::MotionExampleConversionConfig& conversion_config,
    const google::protobuf::RepeatedPtrField<MapFeature>& map_features,
    absl::flat_hash_map<std::string, tensorflow::Feature>* features,
    std::map<std::string, int>* counters) {
  const int max_points = conversion_config.max_roadgraph_samples();

  int num_points = 0;

  // Get the priority ordering for including different road feature types.
  std::vector<MapFeature::FeatureDataCase> priority_types =
      GetFeaturePriorityList();

  // Add map features prioritized by type.
  for (const auto& priority_type : priority_types) {
    for (const auto& map_feature : map_features) {
      if (map_feature.feature_data_case() != priority_type) {
        continue;
      }
      AddRoadgraphSample(conversion_config, map_feature, &num_points, features);
    }
  }

  // Add any remaining map features not in the prioritized list.
  absl::flat_hash_set<MapFeature::FeatureDataCase> priority_type_set(
      priority_types.begin(), priority_types.end());
  for (const auto& map_feature : map_features) {
    if (priority_type_set.contains(map_feature.feature_data_case())) {
      continue;
    }
    AddRoadgraphSample(conversion_config, map_feature, &num_points, features);
  }

  // Add padding to fill the fixed size tensor to the max_roadgraph_samples
  // size defined in the conversion config.
  const int num_padding = max_points - num_points;
  IncrementCounter("Road_Graph_Samples_Valid_Points", counters, num_points);
  IncrementCounter("Road_Graph_Samples_Padded_Points", counters, num_padding);

  for (int i = 0; i < num_padding; i++) {
    AddInt64Feature("roadgraph_samples/id", kInvalidFieldValue, features);
    AddInt64Feature("roadgraph_samples/type", kInvalidFieldValue, features);
    std::vector<float> invalid_field_values{
        kInvalidFieldValue, kInvalidFieldValue, kInvalidFieldValue};
    AddFloatListFeature("roadgraph_samples/xyz", invalid_field_values,
                        features);
    AddFloatListFeature("roadgraph_samples/dir", invalid_field_values,
                        features);
    AddInt64Feature("roadgraph_samples/valid", 0, features);
  }
}

// Add samples for a single polygon segment. This will increment num_points for
// every sample added and will exit early if num_points >=
// config.max_roadgraph_samples.
void AddSampledPolygonSegment(const Vec3d& start_point, const Vec3d& end_point,
                              const MotionExampleConversionConfig& config,
                              int* num_points, std::vector<Vec3d>* points) {
  const int max_samples = config.max_roadgraph_samples();
  if (*num_points >= max_samples) return;

  // If the sample spacing is not set, only include the first point.
  if (config.polygon_sample_spacing() <= 0) {
    points->push_back(start_point);
    ++(*num_points);
    return;
  }

  // Add equally spaced samples, excluding the last point.
  const Vec3d ray = end_point - start_point;
  const int num_samples = std::max(
      static_cast<int>(1 + ray.Length() / config.polygon_sample_spacing()), 2);
  for (int j = 0; j < num_samples - 1 && *num_points < max_samples; ++j) {
    points->push_back(start_point +
                      ray * static_cast<double>(j) / (num_samples - 1));
    ++(*num_points);
  }
}

void IncrementCounter(const std::string& key,
                      std::map<std::string, int>* counters, int increment_by) {
  if (counters == nullptr) {
    return;
  }
  auto iter = counters->find(key);
  if (iter == counters->end()) {
    (*counters)[key] = increment_by;
  } else {
    iter->second += increment_by;
  }
}

// Returns the difficulty level of the given track.
int DifficultyLevel(const Scenario& scenario, const Track* t) {
  if (t == nullptr) return -1;
  int difficulty_level = 0;
  for (const auto& track_to_predict : scenario.tracks_to_predict()) {
    const Track* tmp = &(scenario.tracks()[track_to_predict.track_index()]);
    if (tmp == t) {
      difficulty_level = static_cast<int>(track_to_predict.difficulty());
      break;
    }
  }
  return difficulty_level;
}

absl::flat_hash_set<const Track*> GetModeledAgents(
    const waymo::open_dataset::MotionExampleConversionConfig& conversion_config,
    const Scenario& scenario, const Track* sdc_track,
    const int last_timestamps_step, std::map<std::string, int>* counters) {
  // The open dataset defines a maximum of 8 tracks to predict per example.
  const int max_num_modeled_agents = conversion_config.max_num_modeled_agents();
  const int current_timestamp_steps =
      last_timestamps_step - conversion_config.num_future_steps();
  absl::flat_hash_set<const Track*> modeled_agents;

  // Find objects of interest in the Scenario proto.
  absl::flat_hash_set<const Track*> objects_of_interest = GetObjectsOfInterest(
      scenario, current_timestamp_steps, /*keep_invalid_tracks=*/false);

  if (!scenario.tracks_to_predict().empty()) {
    // If tracks_to_predict is non-empty, all tracks in tracks_to_predict should
    // be included in the modeled agents.
    for (const auto& required_tracks : scenario.tracks_to_predict()) {
      const int track_to_predict = required_tracks.track_index();
      CHECK_LT(track_to_predict, scenario.tracks_size());
      const Track* track = &(scenario.tracks()[track_to_predict]);

      if (track->states()[current_timestamp_steps].valid() &&
          modeled_agents.size() < max_num_modeled_agents) {
        modeled_agents.insert(track);
      }
    }
  }
  return modeled_agents;
}

void AddStateFeatures(
    const waymo::open_dataset::MotionExampleConversionConfig& conversion_config,
    const google::protobuf::RepeatedField<double>& timestamps_seconds,
    const int last_timestamps_step, const std::string& prefix,
    const Track* track, bool is_sdc, bool is_context, bool objects_of_interest,
    int difficulty_level,
    absl::flat_hash_map<std::string, tensorflow::Feature>* features,
    std::map<std::string, int>* counters) {
  CHECK(prefix == "sdc/" || prefix == "state/");

  const bool padded_feature = (track == nullptr);
  const bool is_sdc_state = prefix == "sdc/";
  // Set id and type state fields of current track.
  if (padded_feature) {
    AddFloatFeature(absl::StrCat(prefix, "id"), kInvalidFieldValue, features);
    AddFloatFeature(absl::StrCat(prefix, "type"), kInvalidFieldValue, features);
    AddInt64Feature(absl::StrCat(prefix, "objects_of_interest"),
                    kInvalidFieldValue, features);
    AddInt64Feature(absl::StrCat(prefix, "is_sdc"), kInvalidFieldValue,
                    features);
    AddInt64Feature(absl::StrCat(prefix, "is_context"), kInvalidFieldValue,
                    features);
    AddInt64Feature(absl::StrCat(prefix, "tracks_to_predict"),
                    kInvalidFieldValue, features);
    AddInt64Feature(absl::StrCat(prefix, "difficulty_level"),
                    kInvalidFieldValue, features);
    IncrementCounter("Padded_State_Feature", counters);
  } else {
    AddFloatFeature(absl::StrCat(prefix, "id"), (*track).id(), features);
    AddFloatFeature(absl::StrCat(prefix, "type"),
                    static_cast<float>((*track).object_type()), features);
    if (is_sdc_state) {
      IncrementCounter("Sdc_State_Feature", counters);
    } else {
      AddInt64Feature(absl::StrCat(prefix, "is_sdc"), is_sdc, features);
      AddInt64Feature(absl::StrCat(prefix, "is_context"), is_context, features);
      AddInt64Feature(absl::StrCat(prefix, "tracks_to_predict"), !is_context,
                      features);
      AddInt64Feature(absl::StrCat(prefix, "difficulty_level"),
                      difficulty_level, features);
      AddInt64Feature(absl::StrCat(prefix, "objects_of_interest"),
                      static_cast<int>(objects_of_interest), features);
      IncrementCounter("Agents_State_Feature", counters);
      IncrementCounter(absl::StrCat("Object_Type_",
                                    static_cast<float>((*track).object_type())),
                       counters);
    }
  }

  const int num_past_steps = conversion_config.num_past_steps();
  const int num_steps = (conversion_config.num_future_steps() + num_past_steps);

  // Add state fields per-time step.
  for (int t = 0; t < num_steps; ++t) {
    std::string time_frame;
    if (t < num_past_steps - 1) {
      time_frame = "past";
    } else if (t == num_past_steps - 1) {
      time_frame = "current";
    } else {
      time_frame = "future";
    }

    std::string state_and_time_frame_prefix =
        absl::StrCat(prefix, time_frame, "/");
    // Compute the timestamps step for t given the last timestamps index -
    // timestamps_step.
    const int t_timestamps_step = last_timestamps_step - (num_steps - 1 - t);
    // Set state fields for the object state at timestamps step of t.
    if (track != nullptr && t_timestamps_step >= track->states().size()) {
      IncrementCounter("Track_State_Not_Enough_Steps", counters);
    }
    const auto* obj_state =
        (track == nullptr || track->object_type() == Track::TYPE_UNSET ||
         t_timestamps_step >= track->states().size() ||
         !track->states()[t_timestamps_step].valid())
            ? nullptr
            : &((*track).states()[t_timestamps_step]);

    const float t_timestamps_seconds =
        t_timestamps_step < timestamps_seconds.size()
            ? static_cast<float>(timestamps_seconds[t_timestamps_step])
            : -1;
    AddInt64Feature(
        absl::StrCat(state_and_time_frame_prefix, "timestamp_micros"),
        obj_state == nullptr ? kInvalidFieldValue : t_timestamps_seconds * 1e6,
        features);
    AddInt64Feature(absl::StrCat(state_and_time_frame_prefix, "valid"),
                    obj_state == nullptr ? 0 : obj_state->valid(), features);

    if (obj_state != nullptr) {
      IncrementCounter(
          absl::StrCat(obj_state->valid() ? "Valid" : "Invalid",
                       is_sdc_state ? "_Sdc" : "", "_Object_State"),
          counters);
    }

    // Add all state features.
    AddFloatFeature(
        absl::StrCat(state_and_time_frame_prefix, "x"),
        obj_state == nullptr ? kInvalidFieldValue : obj_state->center_x(),
        features);
    AddFloatFeature(
        absl::StrCat(state_and_time_frame_prefix, "y"),
        obj_state == nullptr ? kInvalidFieldValue : obj_state->center_y(),
        features);
    AddFloatFeature(
        absl::StrCat(state_and_time_frame_prefix, "z"),
        obj_state == nullptr ? kInvalidFieldValue : obj_state->center_z(),
        features);
    AddFloatFeature(
        absl::StrCat(state_and_time_frame_prefix, "width"),
        obj_state == nullptr ? kInvalidFieldValue : obj_state->width(),
        features);
    AddFloatFeature(
        absl::StrCat(state_and_time_frame_prefix, "length"),
        obj_state == nullptr ? kInvalidFieldValue : obj_state->length(),
        features);
    AddFloatFeature(
        absl::StrCat(state_and_time_frame_prefix, "height"),
        obj_state == nullptr ? kInvalidFieldValue : obj_state->height(),
        features);
    AddFloatFeature(
        absl::StrCat(state_and_time_frame_prefix, "velocity_x"),
        obj_state == nullptr ? kInvalidFieldValue : obj_state->velocity_x(),
        features);
    AddFloatFeature(
        absl::StrCat(state_and_time_frame_prefix, "velocity_y"),
        obj_state == nullptr ? kInvalidFieldValue : obj_state->velocity_y(),
        features);
    AddFloatFeature(
        absl::StrCat(state_and_time_frame_prefix, "speed"),
        obj_state == nullptr
            ? kInvalidFieldValue
            : std::sqrt(obj_state->velocity_x() * obj_state->velocity_x() +
                        obj_state->velocity_y() * obj_state->velocity_y()),
        features);
    AddFloatFeature(
        absl::StrCat(state_and_time_frame_prefix, "bbox_yaw"),
        obj_state == nullptr ? kInvalidFieldValue : obj_state->heading(),
        features);
    AddFloatFeature(absl::StrCat(state_and_time_frame_prefix, "vel_yaw"),
                    obj_state == nullptr ? kInvalidFieldValue
                                         : atan2(obj_state->velocity_y(),
                                                 obj_state->velocity_x()),
                    features);
  }
}

void AddSinglePointFeature(
    const int64_t id, const int64_t type, const MapPoint& map_point,
    const MapPoint& next_map_point, const int valid, const std::string& prefix,
    absl::flat_hash_map<std::string, tensorflow::Feature>* features) {
  CHECK(prefix == "roadgraph_samples/" || prefix == "roadgraph_segments/");

  AddInt64Feature(absl::StrCat(prefix, "id"), id, features);
  AddInt64Feature(absl::StrCat(prefix, "type"), type, features);
  AddFloatListFeature(absl::StrCat(prefix, "xyz"),
                      std::vector<float>{static_cast<float>(map_point.x()),
                                         static_cast<float>(map_point.y()),
                                         static_cast<float>(map_point.z())},
                      features);
  AddInt64Feature(absl::StrCat(prefix, "valid"), valid, features);

  // Compute direction from the current point to the next point.
  const Vec3d xyz(map_point.x(), map_point.y(), map_point.z());
  const Vec3d next_xyz(next_map_point.x(), next_map_point.y(),
                       next_map_point.z());
  const Vec3d zeroed_direction(0, 0, 0);
  const Vec3d direction = (next_xyz - xyz).Length() < 0.1
                              ? zeroed_direction
                              : (next_xyz - xyz).Normalized();
  AddFloatListFeature(absl::StrCat(prefix, "dir"),
                      std::vector<float>{static_cast<float>(direction.x()),
                                         static_cast<float>(direction.y()),
                                         static_cast<float>(direction.z())},
                      features);
}

void AddPolygonSamples(
    const MotionExampleConversionConfig& config, const int64_t id,
    const int type, const google::protobuf::RepeatedPtrField<MapPoint>& map_points,
    const int valid, int* num_points,
    absl::flat_hash_map<std::string, tensorflow::Feature>* features) {
  // Add the first N-1 segments of the polygon.
  const int feature_size = map_points.size();
  std::vector<Vec3d> points;
  for (int i = 0; i < feature_size - 1; ++i) {
    const Vec3d left_point = ToVec3d(map_points[i]);
    const Vec3d right_point = ToVec3d(map_points[i + 1]);
    AddSampledPolygonSegment(left_point, right_point, config, num_points,
                             &points);
    if (*num_points >= config.max_roadgraph_samples()) {
      break;
    }
  }

  // Add the last polygon segment from the last point to the first point.
  AddSampledPolygonSegment(ToVec3d(map_points[feature_size - 1]),
                           ToVec3d(map_points[0]), config, num_points, &points);
  // Add the first point to complete the polygon.
  if (*num_points < config.max_roadgraph_samples()) {
    points.push_back(ToVec3d(map_points[0]));
    (*num_points)++;
  }

  // Compute direction vectors for the interpolated polyline.
  std::vector<Vec3d> direction_vectors = ComputeDirectionVectors(points);

  // Add points and direction vectors to the features list.
  AddRoadgraphData(id, type, points, direction_vectors, /*valid=*/1,
                   "roadgraph_samples/", features);
}

void AddInterpolatedRoadGraphSamples(
    const MotionExampleConversionConfig& config, const int64_t id,
    const int type, const google::protobuf::RepeatedPtrField<MapPoint>& map_points,
    const int valid, int* num_points,
    absl::flat_hash_map<std::string, tensorflow::Feature>* features) {
  std::vector<Vec3d> points;
  const double sample_spacing = config.polyline_sample_spacing() <= 0.0
                                    ? config.source_polyline_spacing()
                                    : config.polyline_sample_spacing();
  const double source_spacing = config.source_polyline_spacing();

  if (map_points.empty()) {
    return;
  }

  // Interpolate points along the polyline at the desired spacing.
  const int approx_num_points =
      map_points.size() * source_spacing / sample_spacing;
  points.reserve(approx_num_points);
  int point_index = 0;
  while (*num_points < config.max_roadgraph_samples()) {
    // Interpolate the next point along the polyline.
    const double distance = point_index * sample_spacing;
    const int left_index = static_cast<int>(distance / source_spacing);
    if (left_index >= map_points.size() - 1) {
      break;
    }
    const double alpha =
        (distance - left_index * source_spacing) / source_spacing;

    // Store the list of points in the interpolated polyline.
    if (left_index < 1 || left_index > map_points.size() - 3) {
      points.push_back(InterpolatePoint(map_points[left_index],
                                        map_points[left_index + 1], alpha));
    } else {
      points.push_back(InterpolateCubicSpline(
          ToVec3d(map_points[left_index - 1]), ToVec3d(map_points[left_index]),
          ToVec3d(map_points[left_index + 1]),
          ToVec3d(map_points[left_index + 2]), alpha));
    }
    ++(*num_points);
    ++point_index;
  }

  // Add the last point if it is not too close to the most recently stored
  // point.
  if (*num_points < config.max_roadgraph_samples()) {
    const Vec3d last_point = ToVec3d(map_points[map_points.size() - 1]);
    const double kThreshold = 0.01;
    if (points.empty() ||
        (last_point - points.back()).Sqr() > kThreshold * kThreshold) {
      points.push_back(last_point);
      ++(*num_points);
    }
  }

  // Compute direction vectors for the interpolated polyline.
  std::vector<Vec3d> direction_vectors = ComputeDirectionVectors(points);

  // Add points and direction vectors to the features list.
  AddRoadgraphData(id, type, points, direction_vectors, /*valid=*/1,
                   "roadgraph_samples/", features);
}

absl::Status AddTrafficLightStateFeatures(
    const waymo::open_dataset::MotionExampleConversionConfig& conversion_config,
    const google::protobuf::RepeatedPtrField<DynamicMapState>& dynamic_map_states,
    const google::protobuf::RepeatedField<double>& timestamps_seconds,
    int last_timestamps_step,
    absl::flat_hash_map<std::string, tensorflow::Feature>* features,
    std::map<std::string, int>* counters) {
  const auto max_points =
      conversion_config.max_traffic_light_control_points_per_step();
  const int num_past_steps = conversion_config.num_past_steps();
  const int num_steps = (conversion_config.num_future_steps() + num_past_steps);

  // Add traffic state fields per-time step.
  for (int t = 0; t < num_steps; ++t) {
    std::string time_frame;
    if (t < num_past_steps - 1) {
      time_frame = "past";
    } else if (t == num_past_steps - 1) {
      time_frame = "current";
    } else {
      time_frame = "future";
    }

    std::string state_and_time_frame_prefix =
        absl::StrCat("traffic_light_state/", time_frame, "/");
    // Compute the timestamps step for t given the last timestamps index -
    // timestamps_step.
    const int t_timestamps_step = last_timestamps_step - (num_steps - 1 - t);
    // Set state fields for the object state at timestamps step of t.
    if (t_timestamps_step >= dynamic_map_states.size()) {
      IncrementCounter("Traffic_Light_State_Not_Enough_Steps", counters);
    }
    const auto* dynamic_map_state =
        (t_timestamps_step >= dynamic_map_states.size())
            ? nullptr
            : &(dynamic_map_states[t_timestamps_step]);

    const float t_timestamps_seconds =
        t_timestamps_step < timestamps_seconds.size()
            ? static_cast<float>(timestamps_seconds[t_timestamps_step])
            : -1;
    AddInt64Feature(
        absl::StrCat(state_and_time_frame_prefix, "timestamp_micros"),
        dynamic_map_state == nullptr ? kInvalidFieldValue
                                     : t_timestamps_seconds * 1e6,
        features);

    for (int i = 0; i < max_points; ++i) {
      const TrafficSignalLaneState* lane_state = nullptr;
      if (dynamic_map_state != nullptr &&
          i < dynamic_map_state->lane_states().size())
        lane_state = &(dynamic_map_state->lane_states()[i]);

      AddFloatFeature(absl::StrCat(state_and_time_frame_prefix, "x"),
                      lane_state == nullptr ? kInvalidFieldValue
                                            : lane_state->stop_point().x(),
                      features);
      AddFloatFeature(absl::StrCat(state_and_time_frame_prefix, "y"),
                      lane_state == nullptr ? kInvalidFieldValue
                                            : lane_state->stop_point().y(),
                      features);
      AddFloatFeature(absl::StrCat(state_and_time_frame_prefix, "z"),
                      lane_state == nullptr ? kInvalidFieldValue
                                            : lane_state->stop_point().z(),
                      features);
      AddInt64Feature(absl::StrCat(state_and_time_frame_prefix, "state"),
                      lane_state == nullptr
                          ? kInvalidFieldValue
                          : static_cast<int>(lane_state->state()),
                      features);
      AddInt64Feature(
          absl::StrCat(state_and_time_frame_prefix, "id"),
          lane_state == nullptr ? kInvalidFieldValue : lane_state->lane(),
          features);
      AddInt64Feature(absl::StrCat(state_and_time_frame_prefix, "valid"),
                      lane_state == nullptr ? 0 : 1, features);
    }

    if (dynamic_map_state == nullptr) {
      IncrementCounter("Padded_Traffic_Light_State_Features", counters,
                       max_points);
    } else {
      IncrementCounter("Actual_Traffic_Light_State_Features", counters,
                       dynamic_map_state->lane_states().size());
      IncrementCounter("Padded_Traffic_Light_State_Features", counters,
                       max_points - dynamic_map_state->lane_states().size());
    }
  }

  return absl::OkStatus();
}

bool ValidateScenarioProto(const Scenario& scenario,
                           std::map<std::string, int>* counters) {
  // Ensure the scenario proto is non-empty
  bool valid = true;
  if (scenario.timestamps_seconds().empty()) {
    IncrementCounter("Empty_Scenario", counters);
    valid = false;
  }

  if (scenario.sdc_track_index() >= scenario.tracks().size() ||
      scenario.sdc_track_index() < 0) {
    IncrementCounter("Scenario_Sdc_Index_No_Match", counters);
    valid = false;
  }

  for (const auto& track : scenario.tracks()) {
    if (scenario.timestamps_seconds().size() != track.states().size()) {
      IncrementCounter("Track_States_Size_Not_Match_Timestamps_Step_Size",
                       counters);
      valid = false;
    }
  }

  if (valid) {
    IncrementCounter("Valid_Scenario", counters);
  }
  return valid;
}

absl::Status AddTfFeatures(
    const Scenario& scenario,
    const MotionExampleConversionConfig& conversion_config,
    int last_timestamps_step,
    absl::flat_hash_map<std::string, tensorflow::Feature>* features,
    std::map<std::string, int>* counters) {
  // Add the Scenario ID feature to the tf example.
  AddBytesFeature("scenario/id", scenario.scenario_id(), features);

  // Add road graph samples to the tensorflow Features.
  AddRoadGraphSamples(conversion_config, scenario.map_features(), features,
                      counters);

  // Add traffic light states in tensorflow Features.
  absl::Status status = AddTrafficLightStateFeatures(
      conversion_config, scenario.dynamic_map_states(),
      scenario.timestamps_seconds(), last_timestamps_step, features, counters);
  if (!status.ok()) return status;

  // Add state features for the sdc and the agents.
  status = AddSdcAndAgentsStateFeatures(
      conversion_config, scenario, last_timestamps_step, features, counters);
  if (!status.ok()) return status;

  return absl::OkStatus();
}

}  // namespace internal

absl::StatusOr<tensorflow::Example> ScenarioToExample(
    const Scenario& scenario, const MotionExampleConversionConfig& config,
    std::map<std::string, int>* counters) {
  internal::IncrementCounter("Num_Attempted_TF_Examples", counters);

  if (!internal::ValidateScenarioProto(scenario, counters)) {
    return absl::InternalError(
        absl::StrCat("Invalid Scenario Proto: ", scenario.ShortDebugString()));
  }

  // Create the example feature tensors from the Scenario proto data.
  absl::flat_hash_map<std::string, tensorflow::Feature> features;
  const int num_steps = config.num_future_steps() + config.num_past_steps();
  absl::Status status = internal::AddTfFeatures(
      scenario, config, /*last_timestamps_step=*/num_steps - 1, &features,
      counters);
  if (!status.ok()) {
    return status;
  }

  // Create a tensorflow Example proto and populate it with the feature tensors.
  tensorflow::Example result;
  for (const auto& kv : features) {
    (*result.mutable_features()->mutable_feature())[kv.first] = kv.second;
  }

  internal::IncrementCounter("Num_Success_Generate_TF_Examples", counters);
  return result;
}

}  // namespace open_dataset
}  // namespace waymo
