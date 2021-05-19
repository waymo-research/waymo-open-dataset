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

// A binary to compare an objects file from the latency evaluator (containing
// results from a subset of the test set's frames) against a submission proto
// (containing results from the full test set) for equivalence.
#include <fstream>
#include <iostream>
#include <memory>
#include <numeric>
#include <tuple>
#include <vector>

#include <glog/logging.h>
#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "waymo_open_dataset/common/integral_types.h"
#include "waymo_open_dataset/dataset.pb.h"
#include "waymo_open_dataset/metrics/matcher.h"
#include "waymo_open_dataset/protos/metrics.pb.h"
#include "waymo_open_dataset/protos/submission.pb.h"

ABSL_FLAG(std::string, latency_result_filename, {},
          "File that contains the car.open_dataset.Objects proto from the "
          "latency evaluation scripts.");
ABSL_FLAG(std::vector<std::string>, full_result_filenames, {},
          "Comma separated list of sharded files that contains "
          "car.open_dataset.Objects proto from user provided submissions.");
ABSL_FLAG(double, iou_threshold, 0.9,
          "IOU threshold to match detections between the latency evaluator "
          "results and the user submission.");
ABSL_FLAG(double, minimum_score, 0.0,
          "Minimum score of detections to consider. Detections with scores "
          "lower than this will not be checked for equivalence between the "
          "submission proto and the latency evaluation script.");

namespace waymo {
namespace open_dataset {
namespace {

// Generates a simple metrics config to match the results from the submissions
// proto with ones from the objects file. Uses very high IOU thresholds since
// the boxes should be nearly identical. If is_3d is true, this binary will do 3
// IOU matching; otherwise, it will do 2D axis-aligned IOU matching.
Config GetConfig(bool is_3d, double iou_threshold) {
  Config config;

  config.set_matcher_type(MatcherProto::TYPE_HUNGARIAN);
  config.add_iou_thresholds(iou_threshold);
  config.add_iou_thresholds(iou_threshold);
  config.add_iou_thresholds(iou_threshold);
  config.add_iou_thresholds(iou_threshold);
  config.add_iou_thresholds(iou_threshold);
  if (is_3d) {
    config.set_box_type(Label::Box::TYPE_3D);
  } else {
    config.set_box_type(Label::Box::TYPE_AA_2D);
  }

  return config;
}

// Reads a file into an Objects proto. The file can either be a serialized
// Objects proto or a serialized Submission proto (from which the Objects proto
// will be extracted).
Objects ReadObjectsFromFile(const std::vector<std::string>& paths) {
  Objects objects_merged;
  for (const auto& path : paths) {
    Objects objs;
    std::ifstream s(path);
    const std::string content((std::istreambuf_iterator<char>(s)),
                              std::istreambuf_iterator<char>());
    if (!objs.ParseFromString(content)) {
      Submission submission;
      if (!submission.ParseFromString(content)) {
        LOG(FATAL) << "Could not parse " << path << " as submission either.";
      }
      objs = std::move(submission.inference_results());
    }
    for (const auto& o : objs.objects()) {
      *objects_merged.add_objects() = o;
    }
  }

  return objects_merged;
}

// Compare the results from the latency evaluator to the results from the full
// submission. For each frame in the latency results, this function uses a
// Hungarian matcher to match corresponding Object protos in the two sets of
// results and then ensures that the relevant fields of the Object (the box
// dimensions, confidence score, and class name) are nearly identical.
// Returns 0 if the two sets of results match and 1 otherwise.
int Compute(const std::string& latency_result_filename,
            const std::vector<std::string>& full_result_filename,
            double iou_threshold, double minimum_score) {
  using KeyTuple = std::tuple<std::string, int64, CameraName::Name>;
  Objects latency_result_objs = ReadObjectsFromFile({latency_result_filename});
  Objects full_result_objs = ReadObjectsFromFile(full_result_filename);

  // Maps from frames (identified by their context name, timestamps, and camera
  // names) to a vector of Object protos, for both the latency result subset and
  // the full results.
  std::map<KeyTuple, std::vector<Object>> latency_result_map;
  std::map<KeyTuple, std::vector<Object>> full_result_map;

  auto print_key = [](const KeyTuple& key) {
    std::ostringstream oss;
    oss << std::get<0>(key) << " ts " << std::get<1>(key) << " cam name "
        << CameraName::Name_Name(std::get<2>(key));
    return oss.str();
  };

  bool is_2d;
  for (auto& o : *latency_result_objs.mutable_objects()) {
    is_2d = o.object().box().has_heading();
    if (o.score() >= minimum_score && o.object().type() > 0) {
      const KeyTuple key(o.context_name(), o.frame_timestamp_micros(),
                         o.camera_name());
      latency_result_map[key].push_back(std::move(o));
    }
  }
  for (auto& o : *full_result_objs.mutable_objects()) {
    if (o.score() >= minimum_score && o.object().type() > 0) {
      const KeyTuple key(o.context_name(), o.frame_timestamp_micros(),
                         o.camera_name());
      full_result_map[key].push_back(std::move(o));
    }
  }

  const Config config = GetConfig(is_2d, iou_threshold);
  std::unique_ptr<Matcher> matcher = Matcher::Create(config);

  // This loop iterates over the key-value pairs in the latency result map
  // rather than the full result map because it assumes that the latency results
  // are generated on a subset of the frames in the submissions proto.
  for (const auto& kv : latency_result_map) {
    const auto& example_key = kv.first;
    const auto& latency_results = kv.second;
    auto full_result_it = full_result_map.find(example_key);
    if (full_result_it == full_result_map.end()) {
      LOG(FATAL) << print_key(example_key)
                 << " in latency evaluator results but not in submission.";
      return 1;
    }
    const auto& full_results = full_result_it->second;

    const size_t num_detections = latency_results.size();

    // Keep track of the number of detections that do not match, starting by
    // subtracting the number of detections in the latency results from the
    // number of detections in the full results since that difference
    // constitutes detections in the full results that cannot have a match in
    // the latency results.
    size_t unmatched_detections = 0;
    if (full_results.size() > num_detections) {
      unmatched_detections = full_results.size() - num_detections;
    }

    // Run the Hungarian matcher on the two sets of results from this frame.
    matcher->SetPredictions(latency_results);
    matcher->SetGroundTruths(full_results);

    std::vector<int> pred_subset(num_detections);
    std::iota(pred_subset.begin(), pred_subset.end(), 0);
    matcher->SetPredictionSubset(pred_subset);
    std::vector<int> gt_subset(full_results.size());
    std::iota(gt_subset.begin(), gt_subset.end(), 0);
    matcher->SetGroundTruthSubset(gt_subset);

    std::vector<int> matches;
    matcher->Match(&matches, nullptr);

    for (int latency_ind = 0; latency_ind < num_detections; ++latency_ind) {
      const Object& latency_obj = latency_results[latency_ind];
      const int full_ind = matches[latency_ind];
      if (full_ind < 0) {
        LOG(INFO) << "No match found for object " << latency_ind
                  << " for frame " << print_key(example_key) << std::endl
                  << latency_obj.DebugString();
        ++unmatched_detections;
        continue;
      }
      const Object& full_obj = full_results[full_ind];

      if (std::abs(latency_obj.score() - full_obj.score()) > 0.05) {
        LOG(INFO) << "Matched objects for frame " << print_key(example_key)
                  << " are not identical: " << std::endl
                  << latency_obj.DebugString() << std::endl
                  << "vs" << std::endl
                  << full_obj.DebugString();
        ++unmatched_detections;
      }
    }

    if (unmatched_detections > 0.05 * num_detections) {
      LOG(FATAL) << "Latency evaluator results did not match submission "
                 << "proto for " << print_key(example_key) << std::endl
                 << unmatched_detections << " detections out of "
                 << num_detections << " did not match. This exceeds our "
                 << "cut-off of 5% of detections being unmatched.";
      return 1;
    }

    std::cout << "Results matched for " << print_key(example_key) << std::endl;
  }

  std::cout << "Latency results are identical to full results!" << std::endl;
  return 0;
}

}  // namespace
}  // namespace open_dataset
}  // namespace waymo

int main(int argc, char* argv[]) {
  absl::ParseCommandLine(argc, argv);

  const std::string latency_result_filename =
      absl::GetFlag(FLAGS_latency_result_filename);
  const std::vector<std::string> full_result_filennames =
      absl::GetFlag(FLAGS_full_result_filenames);
  const double iou_threshold = absl::GetFlag(FLAGS_iou_threshold);
  const double minimum_score = absl::GetFlag(FLAGS_minimum_score);

  return waymo::open_dataset::Compute(latency_result_filename,
                                    full_result_filennames, iou_threshold,
                                    minimum_score);
}
