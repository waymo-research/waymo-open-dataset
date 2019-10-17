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

// A tool to compute tracking metrics from the command line.
// Usage:
// /path/to/compute_tracking_metrics_main pd_filename gt_filename
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
/*
OBJECT_TYPE_TYPE_VEHICLE_LEVEL_1: [MOTA 0] [MOTP 0] [Miss 0] [Mismatch 0] [FP 0]
OBJECT_TYPE_TYPE_VEHICLE_LEVEL_2: [MOTA 0] [MOTP 0] [Miss 0] [Mismatch 0] [FP 0]
OBJECT_TYPE_TYPE_PEDESTRIAN_LEVEL_1: [MOTA 0] [MOTP 0] [Miss 0] [Mismatch 0] [FP 0]
OBJECT_TYPE_TYPE_PEDESTRIAN_LEVEL_2: [MOTA 0] [MOTP 0] [Miss 0] [Mismatch 0] [FP 0]
OBJECT_TYPE_TYPE_SIGN_LEVEL_1: [MOTA 0] [MOTP 0] [Miss 0] [Mismatch 0] [FP 0]
OBJECT_TYPE_TYPE_SIGN_LEVEL_2: [MOTA 0] [MOTP 0] [Miss 0] [Mismatch 0] [FP 0]
OBJECT_TYPE_TYPE_CYCLIST_LEVEL_1: [MOTA 0] [MOTP 0] [Miss 0] [Mismatch 0] [FP 0]
OBJECT_TYPE_TYPE_CYCLIST_LEVEL_2: [MOTA 0] [MOTP 0] [Miss 0] [Mismatch 0] [FP 0]
RANGE_TYPE_VEHICLE_[0, 30)_LEVEL_1: [MOTA 0.145455] [MOTP 0.13926] [Miss 0.636364] [Mismatch 0] [FP 0.218182]
RANGE_TYPE_VEHICLE_[0, 30)_LEVEL_2: [MOTA 0.145455] [MOTP 0.13926] [Miss 0.636364] [Mismatch 0] [FP 0.218182]
RANGE_TYPE_VEHICLE_[30, 50)_LEVEL_1: [MOTA 0.21875] [MOTP 0.15987] [Miss 0.59375] [Mismatch 0] [FP 0.1875]
RANGE_TYPE_VEHICLE_[30, 50)_LEVEL_2: [MOTA 0.21875] [MOTP 0.15987] [Miss 0.59375] [Mismatch 0] [FP 0.1875]
RANGE_TYPE_VEHICLE_[50, +inf)_LEVEL_1: [MOTA 0] [MOTP 0] [Miss 0] [Mismatch 0] [FP 0]
RANGE_TYPE_VEHICLE_[50, +inf)_LEVEL_2: [MOTA 0] [MOTP 0] [Miss 0] [Mismatch 0] [FP 0]
RANGE_TYPE_PEDESTRIAN_[0, 30)_LEVEL_1: [MOTA 0] [MOTP 0] [Miss 0] [Mismatch 0] [FP 0]
RANGE_TYPE_PEDESTRIAN_[0, 30)_LEVEL_2: [MOTA 0] [MOTP 0] [Miss 0] [Mismatch 0] [FP 0]
RANGE_TYPE_PEDESTRIAN_[30, 50)_LEVEL_1: [MOTA 0.25] [MOTP 0.0845365] [Miss 0.5] [Mismatch 0] [FP 0.25]
RANGE_TYPE_PEDESTRIAN_[30, 50)_LEVEL_2: [MOTA 0.25] [MOTP 0.0845365] [Miss 0.5] [Mismatch 0] [FP 0.25]
RANGE_TYPE_PEDESTRIAN_[50, +inf)_LEVEL_1: [MOTA 0] [MOTP 0] [Miss 0] [Mismatch 0] [FP 0]
RANGE_TYPE_PEDESTRIAN_[50, +inf)_LEVEL_2: [MOTA 0] [MOTP 0] [Miss 0] [Mismatch 0] [FP 0]
RANGE_TYPE_SIGN_[0, 30)_LEVEL_1: [MOTA 0.227273] [MOTP 0.151606] [Miss 0.545455] [Mismatch 0] [FP 0.227273]
RANGE_TYPE_SIGN_[0, 30)_LEVEL_2: [MOTA 0.227273] [MOTP 0.151606] [Miss 0.545455] [Mismatch 0] [FP 0.227273]
RANGE_TYPE_SIGN_[30, 50)_LEVEL_1: [MOTA 0] [MOTP 0] [Miss 0] [Mismatch 0] [FP 0]
RANGE_TYPE_SIGN_[30, 50)_LEVEL_2: [MOTA 0] [MOTP 0] [Miss 0] [Mismatch 0] [FP 0]
RANGE_TYPE_SIGN_[50, +inf)_LEVEL_1: [MOTA 0] [MOTP 0] [Miss 0] [Mismatch 0] [FP 0]
RANGE_TYPE_SIGN_[50, +inf)_LEVEL_2: [MOTA 0] [MOTP 0] [Miss 0] [Mismatch 0] [FP 0]
RANGE_TYPE_CYCLIST_[0, 30)_LEVEL_1: [MOTA 0] [MOTP 0] [Miss 0] [Mismatch 0] [FP 0]
RANGE_TYPE_CYCLIST_[0, 30)_LEVEL_2: [MOTA 0] [MOTP 0] [Miss 0] [Mismatch 0] [FP 0]
RANGE_TYPE_CYCLIST_[30, 50)_LEVEL_1: [MOTA 0] [MOTP 0] [Miss 0] [Mismatch 0] [FP 0]
RANGE_TYPE_CYCLIST_[30, 50)_LEVEL_2: [MOTA 0] [MOTP 0] [Miss 0] [Mismatch 0] [FP 0]
RANGE_TYPE_CYCLIST_[50, +inf)_LEVEL_1: [MOTA 0] [MOTP 0] [Miss 0] [Mismatch 0] [FP 0]
RANGE_TYPE_CYCLIST_[50, +inf)_LEVEL_2: [MOTA 0] [MOTP 0] [Miss 0] [Mismatch 0] [FP 0]
*/

#include <fstream>
#include <iostream>
#include <iterator>
#include <map>
#include <set>
#include <streambuf>
#include <string>
#include <utility>
#include <vector>

#include "waymo_open_dataset/common/integral_types.h"
#include "waymo_open_dataset/label.pb.h"
#include "waymo_open_dataset/metrics/config_util.h"
#include "waymo_open_dataset/metrics/tracking_metrics.h"
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

// Computes the tracking metrics by taking serialized prediction objects and
// serialized groundtruth objects.
void Compute(const std::string& pd_str, const std::string& gt_str) {
  Objects pd_objects;
  if (!pd_objects.ParseFromString(pd_str)) {
    std::cerr << "Failed to parse predictions.";
    return;
  }
  Objects gt_objects;
  if (!gt_objects.ParseFromString(gt_str)) {
    std::cerr << "Failed to parse ground truths.";
    return;
  }

  std::map<std::string, std::map<int64, std::vector<Object>>> pd_map;
  std::map<std::string, std::map<int64, std::vector<Object>>> gt_map;
  std::map<std::string, std::set<int64>> all_example_keys;
  for (auto& o : *pd_objects.mutable_objects()) {
    all_example_keys[o.context_name()].insert(o.frame_timestamp_micros());
    pd_map[o.context_name()][o.frame_timestamp_micros()].push_back(
        std::move(o));
  }
  for (auto& o : *gt_objects.mutable_objects()) {
    all_example_keys[o.context_name()].insert(o.frame_timestamp_micros());
    gt_map[o.context_name()][o.frame_timestamp_micros()].push_back(
        std::move(o));
  }

  std::vector<std::vector<std::vector<Object>>> pds(all_example_keys.size());
  std::vector<std::vector<std::vector<Object>>> gts(all_example_keys.size());
  int i = 0;
  for (const auto& example_key : all_example_keys) {
    for (int64 ts : example_key.second) {
      pds[i].push_back(pd_map[example_key.first][ts]);
      gts[i].push_back(gt_map[example_key.first][ts]);
    }
    i++;
  }

  const Config config = GetConfig();

  const std::vector<TrackingMetrics> tracking_metrics =
      ComputeTrackingMetrics(config, pds, gts);
  const std::vector<std::string> breakdown_names =
      GetBreakdownNamesFromConfig(config);
  if (breakdown_names.size() != tracking_metrics.size()) {
    std::cerr << "Metrics (size of " << tracking_metrics.size()
              << ") and breadown (size of " << breakdown_names.size()
              << ") does not match.\n";
    return;
  }
  std::cout << "\n";
  for (int i = 0; i < tracking_metrics.size(); ++i) {
    const TrackingMetrics& metric = tracking_metrics[i];
    std::cout << breakdown_names[i] << ": [MOTA " << metric.mota() << "]"
              << " [MOTP " << metric.motp() << "]"
              << " [Miss " << metric.miss() << "]"
              << " [Mismatch " << metric.mismatch() << "]"
              << " [FP " << metric.fp() << "]\n";
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
