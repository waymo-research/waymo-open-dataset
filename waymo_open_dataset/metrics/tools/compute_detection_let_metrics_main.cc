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

// A tool to compute Longtitudal Error Tolerant (LET) detection metrics using
// the command line.
// Usage:
// /path/to/compute_detection_let_metrics_main pd_filename gt_filename
//
// Modify this GetConfig() function below to adapt the configuration to your
// usage if you need a different behavior.
//
// pd_filename is the name of a file that has prediction boxes in format of
// waymo::open_dataset::Objects proto.
// gt_filename is the name of a file that has groud truth boxes in format of
// waymo::open_dataset::Objects proto.
//
// Note:
// 1. The metrics will be computed using `camera_synced_box` from ground truths
//    and `box` from the predictions. Users do not need to populate the
//    `most_visible_camera_name` and `camera_synced_box`.
// 2. The default config is used to evaluate the submissions to the 3D
//    Camera-Only Detection Challenge. Users can modify the GetConfig() function
//    below to customize the metrics behavior.
//
// Results when running on ground_truths.bin and fake_predictions.bin in the
// directory gives the following result.
//
// NOLINTBEGIN(whitespace/line_length)
// OBJECT_TYPE_TYPE_VEHICLE_LEVEL_2: [LET-mAPL 0.211287] [LET-mAP 0.23023] [LET-mAPH 0.181574]
// OBJECT_TYPE_TYPE_PEDESTRIAN_LEVEL_2: [LET-mAPL 0.213379] [LET-mAP 0.224676] [LET-mAPH 0.170878]
// OBJECT_TYPE_TYPE_SIGN_LEVEL_2: [LET-mAPL 0.0989853] [LET-mAP 0.100288] [LET-mAPH 0.0797784]
// OBJECT_TYPE_TYPE_CYCLIST_LEVEL_2: [LET-mAPL 0] [LET-mAP 0] [LET-mAPH 0]
// RANGE_TYPE_VEHICLE_[0, 30)_LEVEL_2: [LET-mAPL 0.158502] [LET-mAP 0.200029] [LET-mAPH 0.163429]
// RANGE_TYPE_VEHICLE_[30, 50)_LEVEL_2: [LET-mAPL 0.231184] [LET-mAP 0.244745] [LET-mAPH 0.205553]
// RANGE_TYPE_VEHICLE_[50, +inf)_LEVEL_2: [LET-mAPL 0.232808] [LET-mAP 0.241251] [LET-mAPH 0.175992]
// RANGE_TYPE_PEDESTRIAN_[0, 30)_LEVEL_2: [LET-mAPL 0.214398] [LET-mAP 0.228197] [LET-mAPH 0.17427]
// RANGE_TYPE_PEDESTRIAN_[30, 50)_LEVEL_2: [LET-mAPL 0.202346] [LET-mAP 0.21114] [LET-mAPH 0.158337]
// RANGE_TYPE_PEDESTRIAN_[50, +inf)_LEVEL_2: [LET-mAPL 0.193137] [LET-mAP 0.201656] [LET-mAPH 0.158395]
// RANGE_TYPE_SIGN_[0, 30)_LEVEL_2: [LET-mAPL 0.0196413] [LET-mAP 0.0196586] [LET-mAPH 0.018337]
// RANGE_TYPE_SIGN_[30, 50)_LEVEL_2: [LET-mAPL 0.191365] [LET-mAP 0.191489] [LET-mAPH 0.150585]
// RANGE_TYPE_SIGN_[50, +inf)_LEVEL_2: [LET-mAPL 0.149141] [LET-mAP 0.151854] [LET-mAPH 0.125403]
// RANGE_TYPE_CYCLIST_[0, 30)_LEVEL_2: [LET-mAPL 0] [LET-mAP 0] [LET-mAPH 0]
// RANGE_TYPE_CYCLIST_[30, 50)_LEVEL_2: [LET-mAPL 0] [LET-mAP 0] [LET-mAPH 0]
// RANGE_TYPE_CYCLIST_[50, +inf)_LEVEL_2: [LET-mAPL 0] [LET-mAP 0] [LET-mAPH 0]
// CAMERA_TYPE_VEHICLE_FRONT_LEVEL_2: [LET-mAPL 0.333815] [LET-mAP 0.36573] [LET-mAPH 0.30147]
// CAMERA_TYPE_VEHICLE_FRONT-LEFT_LEVEL_2: [LET-mAPL 0.567318] [LET-mAP 0.736161] [LET-mAPH 0.645462]
// CAMERA_TYPE_VEHICLE_FRONT-RIGHT_LEVEL_2: [LET-mAPL 0.33577] [LET-mAP 0.352934] [LET-mAPH 0.267835]
// CAMERA_TYPE_VEHICLE_SIDE-LEFT_LEVEL_2: [LET-mAPL 0] [LET-mAP 0] [LET-mAPH 0]
// CAMERA_TYPE_VEHICLE_SIDE-RIGHT_LEVEL_2: [LET-mAPL 0.125353] [LET-mAP 0.222222] [LET-mAPH 0.113782]
// CAMERA_TYPE_PEDESTRIAN_FRONT_LEVEL_2: [LET-mAPL 0.367701] [LET-mAP 0.389814] [LET-mAPH 0.292392]
// CAMERA_TYPE_PEDESTRIAN_FRONT-LEFT_LEVEL_2: [LET-mAPL 0.251196] [LET-mAP 0.260676] [LET-mAPH 0.198859]
// CAMERA_TYPE_PEDESTRIAN_FRONT-RIGHT_LEVEL_2: [LET-mAPL 0.140006] [LET-mAP 0.154568] [LET-mAPH 0.119544]
// CAMERA_TYPE_PEDESTRIAN_SIDE-LEFT_LEVEL_2: [LET-mAPL 0.172725] [LET-mAP 0.173382] [LET-mAPH 0.161921]
// CAMERA_TYPE_PEDESTRIAN_SIDE-RIGHT_LEVEL_2: [LET-mAPL 0.258021] [LET-mAP 0.261462] [LET-mAPH 0.185914]
// CAMERA_TYPE_SIGN_FRONT_LEVEL_2: [LET-mAPL 0.240335] [LET-mAP 0.245807] [LET-mAPH 0.195759]
// CAMERA_TYPE_SIGN_FRONT-LEFT_LEVEL_2: [LET-mAPL 0.123899] [LET-mAP 0.125459] [LET-mAPH 0.119571]
// CAMERA_TYPE_SIGN_FRONT-RIGHT_LEVEL_2: [LET-mAPL 0.0708887] [LET-mAP 0.070911] [LET-mAPH 0.0535817]
// CAMERA_TYPE_SIGN_SIDE-LEFT_LEVEL_2: [LET-mAPL 0] [LET-mAP 0] [LET-mAPH 0]
// CAMERA_TYPE_SIGN_SIDE-RIGHT_LEVEL_2: [LET-mAPL 0.0312435] [LET-mAP 0.03125] [LET-mAPH 0.0142438]
// CAMERA_TYPE_CYCLIST_FRONT_LEVEL_2: [LET-mAPL 0] [LET-mAP 0] [LET-mAPH 0]
// CAMERA_TYPE_CYCLIST_FRONT-LEFT_LEVEL_2: [LET-mAPL 0] [LET-mAP 0] [LET-mAPH 0]
// CAMERA_TYPE_CYCLIST_FRONT-RIGHT_LEVEL_2: [LET-mAPL 0] [LET-mAP 0] [LET-mAPH 0]
// CAMERA_TYPE_CYCLIST_SIDE-LEFT_LEVEL_2: [LET-mAPL 0] [LET-mAP 0] [LET-mAPH 0]
// CAMERA_TYPE_CYCLIST_SIDE-RIGHT_LEVEL_2: [LET-mAPL 0] [LET-mAP 0] [LET-mAPH 0]
// NOLINTEND(whitespace/line_length)

#include <array>
#include <fstream>
#include <iostream>
#include <streambuf>
#include <string>
#include <unordered_map>
#include <utility>

#include "absl/strings/str_cat.h"
#include "waymo_open_dataset/common/integral_types.h"
#include "waymo_open_dataset/dataset.pb.h"
#include "waymo_open_dataset/label.pb.h"
#include "waymo_open_dataset/metrics/config_util.h"
#include "waymo_open_dataset/metrics/detection_metrics.h"
#include "waymo_open_dataset/protos/breakdown.pb.h"
#include "waymo_open_dataset/protos/metrics.pb.h"

namespace waymo {
namespace open_dataset {
namespace {
// Generates a simple metrics config with one difficulty level (LEVEL_2 assumed)
// for each breakdown.

// The longitudinal tolerance for LET-metrics.
constexpr double kLongitudinalTolerancePercentage = 0.1;
// The minimal longitudinal tolerance in meters.
constexpr double kMinLongitudinalToleranceMeter = 0.5;
// The cameras' mean location (x, y, z) in the vehicle frame.
constexpr std::array<double, 3> kCameraLocation = {1.43, 0, 2.18};

Config GetConfig() {
  Config config;
  {
    config.add_breakdown_generator_ids(Breakdown::OBJECT_TYPE);
    Difficulty* d = config.add_difficulties();
    d->add_levels(Label::LEVEL_2);
  }
  {
    config.add_breakdown_generator_ids(Breakdown::RANGE);
    Difficulty* d = config.add_difficulties();
    d->add_levels(Label::LEVEL_2);
  }
  {
    config.add_breakdown_generator_ids(Breakdown::CAMERA);
    Difficulty*d = config.add_difficulties();
    d->add_levels(Label::LEVEL_2);
  }

  config.set_matcher_type(MatcherProto::TYPE_HUNGARIAN);
  config.add_iou_thresholds(0.0);
  // TYPE_VEHICLE.
  config.add_iou_thresholds(0.5);
  // TYPE_PEDESTRIAN.
  config.add_iou_thresholds(0.3);
  // TYPE_SIGN.
  config.add_iou_thresholds(0.3);
  // TYPE_CYCLIST.
  config.add_iou_thresholds(0.3);
  config.set_box_type(Label::Box::TYPE_3D);

  for (int i = 0; i < 100; ++i) {
    config.add_score_cutoffs(i * 0.01);
  }
  config.add_score_cutoffs(1.0);

  // Enable LET metrics.
  auto* let_metrics_config = config.mutable_let_metric_config();
  let_metrics_config->set_enabled(true);
  let_metrics_config->set_align_type(
      Config::LongitudinalErrorTolerantConfig::TYPE_RANGE_ALIGNED);
  let_metrics_config->set_longitudinal_tolerance_percentage(
      kLongitudinalTolerancePercentage);
  let_metrics_config->set_min_longitudinal_tolerance_meter(
      kMinLongitudinalToleranceMeter);
  auto* sensor_location = let_metrics_config->mutable_sensor_location();
  sensor_location->set_x(kCameraLocation[0]);
  sensor_location->set_y(kCameraLocation[1]);
  sensor_location->set_z(kCameraLocation[2]);

  return config;
}

// Computes the detection metrics.
void Compute(const std::string& pd_str, const std::string& gt_str) {
  Objects pd_objects;
  if (!pd_objects.ParseFromString(pd_str)) {
    std::cerr << "Failed to parse predictions.";
    return;
  }
  Objects gt_objects_ori;
  if (!gt_objects_ori.ParseFromString(gt_str)) {
    std::cerr << "Failed to parse ground truths.";
    return;
  }

  // Set detection difficulty.
  Objects gt_objects;
  constexpr int kDetectionLevel2NumPointsThreshold = 5;
  for (auto& o : *gt_objects_ori.mutable_objects()) {
    // Note that we use `num_top_lidar_points_in_box` instead of
    // `num_lidar_points_in_box` because cameras have roughly the same FOV as
    // the top lidar.
    if (o.object().num_top_lidar_points_in_box() <= 0) continue;
    // Decide detection difficulty by the number of points inside the box if the
    // boxes don't come with human annotated difficulty.
    if (!o.object().has_detection_difficulty_level() ||
        o.object().detection_difficulty_level() == Label::UNKNOWN) {
      o.mutable_object()->set_detection_difficulty_level(
          o.object().num_top_lidar_points_in_box() <=
                  kDetectionLevel2NumPointsThreshold
              ? Label::LEVEL_2
              : Label::LEVEL_1);
    }
    // Ignore boxes that are not visible to any of the cameras.
    if (!o.object().has_most_visible_camera_name() ||
        o.object().most_visible_camera_name().empty()){
      continue;
    }
    CameraName::Name camera_name;
    CHECK(CameraName::Name_Parse(o.object().most_visible_camera_name(),
                                 &camera_name))
        << "Object has invalid `most_visible_camera_name` "
        << o.object().most_visible_camera_name();
    // Replace the label box with the camera-synced box. In the 3D Camera-Only
    // Detection Challenge, we use camera-synced boxes as the ground truth boxes
    // as well as the evaluation target boxes.
    *o.mutable_object()->mutable_box() = o.object().camera_synced_box();
    *gt_objects.add_objects() = o;
  }

  std::map<std::pair<std::string, int64>, std::vector<Object>> pd_map;
  std::map<std::pair<std::string, int64>, std::vector<Object>> gt_map;
  std::set<std::pair<std::string, int64>> all_example_keys;
  auto get_key = [](const Object& object) {
    return std::make_pair(
        absl::StrCat(object.context_name(), "_", object.camera_name()),
        object.frame_timestamp_micros());
  };
  for (auto& o : *pd_objects.mutable_objects()) {
    const auto key = get_key(o);
    pd_map[key].push_back(std::move(o));
    all_example_keys.insert(key);
  }
  for (auto& o : *gt_objects.mutable_objects()) {
    const auto key = get_key(o);
    gt_map[key].push_back(std::move(o));
    all_example_keys.insert(key);
  }

  std::cout << all_example_keys.size() << " examples found.\n";

  std::vector<std::vector<Object>> pds;
  std::vector<std::vector<Object>> gts;
  for (auto& example_key : all_example_keys) {
    auto gt_it = gt_map.find(example_key);
    if (gt_it == gt_map.end()) {
      gts.push_back({});
    } else {
      gts.push_back(std::move(gt_it->second));
    }
    auto pd_it = pd_map.find(example_key);
    if (pd_it == pd_map.end()) {
      pds.push_back({});
    } else {
      pds.push_back(std::move(pd_it->second));
    }
  }

  const Config config = GetConfig();

  const std::vector<DetectionMetrics> detection_metrics =
      ComputeDetectionMetrics(config, pds, gts);
  const std::vector<std::string> breakdown_names =
      GetBreakdownNamesFromConfig(config);
  if (breakdown_names.size() != detection_metrics.size()) {
    std::cerr << "Metrics (size of " << detection_metrics.size()
              << ") and breadown (size of " << breakdown_names.size()
              << ") does not match.\n";
    return;
  }
  std::cout << "\n";
  for (int i = 0; i < detection_metrics.size(); ++i) {
    const DetectionMetrics& metric = detection_metrics[i];
    std::cout << breakdown_names[i] << ": [LET-mAPL "
              << metric.mean_average_precision_longitudinal_affinity_weighted()
              << "]"
              << " [LET-mAP " << metric.mean_average_precision() << "]"
              << " [LET-mAPH " << metric.mean_average_precision_ha_weighted()
              << "]\n";
  }
}

}  // namespace
}  // namespace open_dataset
}  // namespace waymo

namespace {
// Read all content in a file to a string.
std::string ReadFileToString(const char* filename) {
  std::ifstream s(filename);
  const std::string content((std::istreambuf_iterator<char>(s)),
                            std::istreambuf_iterator<char>());
  return content;
}
}  // namespace

int main(int argc, char** argv) {
  if (argc < 3) {
    std::cerr << "Usage: " << argv[0]
              << " prediction_filename ground_truth_filename.\n";
    return 1;
  }

  const char* pd_filename = argv[1];
  const char* gt_filename = argv[2];
  const std::string pd_str = ReadFileToString(pd_filename);
  const std::string gt_str = ReadFileToString(gt_filename);
  if (gt_str.empty()) {
    std::cerr << "Must specify ground truth.\n";
    return 1;
  }

  waymo::open_dataset::Compute(pd_str, gt_str);

  return 0;
}
