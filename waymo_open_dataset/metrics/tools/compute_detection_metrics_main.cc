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

// A tool to compute detection metrics from the command line.
// Usage:
// /path/to/compute_detection_metrics_main pd_filename gt_filename
//
// By default, this tool assumes 3d upright box and no difficulty based
// breakdown.
// Modify this GetConfig() function below to adapt the configuration to your
// usage if you need a different behavior.
//
// pd_filename is the name of a file that has prediction boxes in format of
// waymo::open_dataset::Objects proto.
// gt_filename is the name of a file that has groud truth boxes in format of
// waymo::open_dataset::Objects proto.
//
// Note:
// 1. gt_filename does not need to populate no_label_zone_objects as this tool
//    is not yet able to check whether a box overlaps with an no-label-zone
//    polygon. This is to be added.
// 2. As a result of 1, pd_filename must have overlap_with_nlz populated if you
//    think your predictions overlaps with no label zones.
//    NOTE: overlap_with_nlz does not need to be populated when you submit to
//    our leaderboard (not public yet). It will be ignored even if it is
//    populated.
//
// Results when running on ground_truths.bin and fake_predictions.bin in the
// directory gives the following result.
//
// OBJECT_TYPE_TYPE_VEHICLE_LEVEL_2: [mAP 0.0650757] [mAPH 0.055601]
// OBJECT_TYPE_TYPE_PEDESTRIAN_LEVEL_2: [mAP 0.101786] [mAPH 0.0473937]
// OBJECT_TYPE_TYPE_SIGN_LEVEL_2: [mAP 0.0500082] [mAPH 0.0366725]
// OBJECT_TYPE_TYPE_CYCLIST_LEVEL_2: [mAP 0] [mAPH 0]
// RANGE_TYPE_VEHICLE_[0, 30)_LEVEL_2: [mAP 0.0784186] [mAPH 0.0709105]
// RANGE_TYPE_VEHICLE_[30, 50)_LEVEL_2: [mAP 0.133955] [mAPH 0.103888]
// RANGE_TYPE_VEHICLE_[50, +inf)_LEVEL_2: [mAP 0.0522752] [mAPH 0.0460845]
// RANGE_TYPE_PEDESTRIAN_[0, 30)_LEVEL_2: [mAP 0.0458333] [mAPH 2.20291e-09]
// RANGE_TYPE_PEDESTRIAN_[30, 50)_LEVEL_2: [mAP 0.110526] [mAPH 0.0512746]
// RANGE_TYPE_PEDESTRIAN_[50, +inf)_LEVEL_2: [mAP 0.15] [mAPH 0.074034]
// RANGE_TYPE_SIGN_[0, 30)_LEVEL_2: [mAP 0.128993] [mAPH 0.100171]
// RANGE_TYPE_SIGN_[30, 50)_LEVEL_2: [mAP 0.0354839] [mAPH 0.0171309]
// RANGE_TYPE_SIGN_[50, +inf)_LEVEL_2: [mAP 0.0164904] [mAPH 0.01548]
// RANGE_TYPE_CYCLIST_[0, 30)_LEVEL_2: [mAP 0] [mAPH 0]
// RANGE_TYPE_CYCLIST_[30, 50)_LEVEL_2: [mAP 0] [mAPH 0]
// RANGE_TYPE_CYCLIST_[50, +inf)_LEVEL_2: [mAP 0] [mAPH 0]

#include <fstream>
#include <iostream>
#include <streambuf>
#include <string>
#include <unordered_map>
#include <utility>

#include "absl/strings/str_cat.h"
#include "waymo_open_dataset/common/integral_types.h"
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
Config GetConfig() {
  Config config;
  config.add_breakdown_generator_ids(Breakdown::OBJECT_TYPE);
  auto* d = config.add_difficulties();
  d->add_levels(Label::LEVEL_1);
  d->add_levels(Label::LEVEL_2);
  config.add_breakdown_generator_ids(Breakdown::RANGE);
  d = config.add_difficulties();
  d->add_levels(Label::LEVEL_1);
  d->add_levels(Label::LEVEL_2);

  config.set_matcher_type(MatcherProto::TYPE_HUNGARIAN);
  config.add_iou_thresholds(0.0);
  config.add_iou_thresholds(0.7);
  config.add_iou_thresholds(0.5);
  config.add_iou_thresholds(0.5);
  config.add_iou_thresholds(0.5);
  config.set_box_type(Label::Box::TYPE_3D);

  for (int i = 0; i < 100; ++i) {
    config.add_score_cutoffs(i * 0.01);
  }
  config.add_score_cutoffs(1.0);

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
    if (o.object().num_lidar_points_in_box() <= 0) continue;
    // Decide detection difficulty by the number of points inside the box if the
    // boxes don't come with human annotated difficulty.
    if (!o.object().has_detection_difficulty_level() ||
        o.object().detection_difficulty_level() == Label::UNKNOWN) {
      o.mutable_object()->set_detection_difficulty_level(
          o.object().num_lidar_points_in_box() <=
                  kDetectionLevel2NumPointsThreshold
              ? Label::LEVEL_2
              : Label::LEVEL_1);
    }
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
    std::cout << breakdown_names[i] << ": [mAP "
              << metric.mean_average_precision() << "]"
              << " [mAPH " << metric.mean_average_precision_ha_weighted()
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
