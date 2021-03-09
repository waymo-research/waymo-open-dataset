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

#include <glog/logging.h>
#include "waymo_open_dataset/common/integral_types.h"
#include "waymo_open_dataset/dataset.pb.h"
#include "waymo_open_dataset/metrics/matcher.h"
#include "waymo_open_dataset/protos/metrics.pb.h"
#include "waymo_open_dataset/protos/submission.pb.h"

namespace waymo {
namespace open_dataset {
namespace {

// Generates a simple metrics config to match the results from the submissions
// proto with ones from the objects file. Uses very high IOU thresholds since
// the boxes should be nearly identical. If is_3d is true, this binary will do 3
// IOU matching; otherwise, it will do 2D axis-aligned IOU matching.
Config GetConfig(bool is_3d) {
  Config config;

  config.set_matcher_type(MatcherProto::TYPE_HUNGARIAN);
  config.add_iou_thresholds(0.9);
  config.add_iou_thresholds(0.9);
  config.add_iou_thresholds(0.9);
  config.add_iou_thresholds(0.9);
  config.add_iou_thresholds(0.9);
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
Objects ReadObjectsFromFile(const std::string& path) {
  Objects objs;
  std::ifstream s(path);
  const std::string content((std::istreambuf_iterator<char>(s)),
                            std::istreambuf_iterator<char>());
  if (!objs.ParseFromString(content)) {
    LOG(ERROR) << "Could not parse " << path
               << " as Objects file. Trying as a Submission file.";
    Submission submission;
    if (!submission.ParseFromString(content)) {
      LOG(FATAL) << "Could not parse " << path << " as submission either.";
    }
    objs = std::move(submission.inference_results());
  }

  return objs;
}

// Compare the results from the latency evaluator to the results from the full
// submission. For each frame in the latency results, this function uses a
// Hungarian matcher to match corresponding Object protos in the two sets of
// results and then ensures that the relevant fields of the Object (the box
// dimensions, confidence score, and class name) are nearly identical.
// Returns 0 if the two sets of results match and 1 otherwise.
int Compute(const std::string& latency_result_filename,
            const std::string& full_result_filename) {
  using KeyTuple = std::tuple<std::string, int64, CameraName::Name>;
  Objects latency_result_objs = ReadObjectsFromFile(latency_result_filename);
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
    const KeyTuple key(o.context_name(), o.frame_timestamp_micros(),
                       o.camera_name());
    is_2d = o.object().box().has_heading();
    latency_result_map[key].push_back(std::move(o));
  }
  for (auto& o : *full_result_objs.mutable_objects()) {
    const KeyTuple key(o.context_name(), o.frame_timestamp_micros(),
                       o.camera_name());
    full_result_map[key].push_back(std::move(o));
  }

  std::cout << latency_result_map.size() << " frames found.\n";

  const Config config = GetConfig(is_2d);
  std::unique_ptr<Matcher> matcher = Matcher::Create(config);

  // This loop iterates over the key-value pairs in the latency result map
  // rather than the full result map because it assumes that the latency results
  // are generated on a subset of the frames in the submissions proto.
  for (const auto& kv : latency_result_map) {
    const auto& example_key = kv.first;
    const auto& latency_results = kv.second;
    auto full_result_it = full_result_map.find(example_key);
    if (full_result_it == full_result_map.end()) {
      std::cerr << print_key(example_key) << " not found in full results"
                << std::endl;
      return 1;
    }
    const auto& full_results = full_result_it->second;

    const size_t num_detections = latency_results.size();
    if (full_results.size() != num_detections) {
      std::cerr << "Different number of detections found: " << num_detections
                << " in latency results, " << full_results.size()
                << " in full results for frame " << print_key(example_key)
                << std::endl;
      return 1;
    }

    // Run the Hungarian matcher on the two sets of results from this frame.
    matcher->SetPredictions(latency_results);
    matcher->SetGroundTruths(full_results);

    std::vector<int> subset(num_detections);
    std::iota(subset.begin(), subset.end(), 0);
    matcher->SetPredictionSubset(subset);
    matcher->SetGroundTruthSubset(subset);

    std::vector<int> matches;
    matcher->Match(&matches, nullptr);

    for (int latency_ind = 0; latency_ind < num_detections; ++latency_ind) {
      const Object& latency_obj = latency_results[latency_ind];
      const int full_ind = matches[latency_ind];
      if (full_ind < 0) {
        std::cerr << "No match found for object " << latency_ind
                  << " for frame " << print_key(example_key) << std::endl;
        return 1;
      }
      const Object& full_obj = full_results[full_ind];

      if (std::abs(latency_obj.score() - full_obj.score()) > 1e-3 ||
          std::abs(latency_obj.object().box().center_x() -
                   full_obj.object().box().center_x()) > 1e-3 ||
          std::abs(latency_obj.object().box().center_y() -
                   full_obj.object().box().center_y()) > 1e-3 ||
          std::abs(latency_obj.object().box().center_z() -
                   full_obj.object().box().center_z()) > 1e-3 ||
          std::abs(latency_obj.object().box().length() -
                   full_obj.object().box().length()) > 1e-3 ||
          std::abs(latency_obj.object().box().width() -
                   full_obj.object().box().width()) > 1e-3 ||
          std::abs(latency_obj.object().box().height() -
                   full_obj.object().box().height()) > 1e-3 ||
          std::abs(latency_obj.object().box().heading() -
                   full_obj.object().box().heading()) > 1e-3 ||
          latency_obj.object().type() != full_obj.object().type()) {
        std::cerr << "Matched objects for frame " << print_key(example_key)
                  << " are not identical: " << std::endl
                  << latency_obj.DebugString() << std::endl
                  << "vs" << std::endl
                  << full_obj.DebugString();
        return 1;
      }
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
  if (argc < 3) {
    std::cerr << "Usage: " << argv[0]
              << " prediction_filename ground_truth_filename.\n";
    return 1;
  }

  const std::string latency_result_filename(argv[1]);
  const std::string full_result_filename(argv[2]);

  return waymo::open_dataset::Compute(latency_result_filename,
                                    full_result_filename);
}
