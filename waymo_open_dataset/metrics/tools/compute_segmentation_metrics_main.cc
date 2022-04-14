/* Copyright 2022 The Waymo Open Dataset Authors. All Rights Reserved.

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

// A tool to compute 3D semantic segmentation metrics from the command line.
// Usage:
// /path/to/compute_semantic_segmentation_metrics_main pd_filename
// gt_filename
//
// pd_filename is the name of a file that has prediction frames, represented
// as waymo::open_dataset::SegmentationFrameList protos
// gt_filename is the name of a file that has groud truth frames, represented
// as waymo::open_dataset::SegmentationFrameList protos
//
//
// Results when running on ground_truths.bin and fake_predictions.bin in the
// directory gives the following result: (Output orders might vary).
// 6 frames found in prediction.
// 6 frames found in groundtruth.
// TYPE_CAR:0.0434832
// TYPE_BICYCLIST:0
// TYPE_CONSTRUCTION_CONE:0
// TYPE_LANE_MARKER:0
// TYPE_TRUCK:0
// TYPE_PEDESTRIAN:0.0434832
// TYPE_BICYCLE:0
// TYPE_OTHER_GROUND:0
// TYPE_BUS:0
// TYPE_SIGN:0
// TYPE_MOTORCYCLE:0
// TYPE_TREE_TRUNK:0
// TYPE_WALKABLE:0
// TYPE_OTHER_VEHICLE:0
// TYPE_TRAFFIC_LIGHT:0
// TYPE_BUILDING:0
// TYPE_SIDEWALK:0
// TYPE_CURB:0
// TYPE_MOTORCYCLIST:0
// TYPE_POLE:0
// TYPE_VEGETATION:0
// TYPE_ROAD:0
// miou=0.00395302

#include <fstream>
#include <iostream>
#include <map>
#include <ostream>
#include <set>
#include <streambuf>
#include <string>
#include <utility>
#include <vector>

#include "absl/strings/str_cat.h"
#include "waymo_open_dataset/common/integral_types.h"
#include "waymo_open_dataset/dataset.pb.h"
#include "waymo_open_dataset/metrics/segmentation_metrics.h"
#include "waymo_open_dataset/protos/segmentation.pb.h"
#include "waymo_open_dataset/protos/segmentation_metrics.pb.h"
#include "zlib.h"

namespace waymo {
namespace open_dataset {
namespace {

// Util function for Zlib uncompression.
std::string Uncompress(std::string const& s) {
  unsigned int source_size = s.size();
  const char* source = s.c_str();
  constexpr int kMaxUncompressedLen = 1 << 30;  // 1GB
  uLongf max_len = kMaxUncompressedLen;
  char* destination = new char[kMaxUncompressedLen];
  int result = uncompress((unsigned char*)destination, &max_len,
                          (const unsigned char*)source, source_size);
  if (result != Z_OK) {
    delete[] destination;
    std::cerr << "Uncompress error occured! Error code: " << result << "\n";
  }
  // Since we don't know the output size, we use the max len.
  std::string ret = std::string(destination, max_len);
  delete[] destination;
  return ret;
}

// Helper function to convert a frame into a vector.
std::vector<Segmentation::Type> Flatten(MatrixInt32 frame) {
  std::vector<Segmentation::Type> flattened;
  flattened.reserve(frame.data().size());
  for (int i = 0; i < frame.data().size(); ++i) {
    flattened.push_back(static_cast<Segmentation::Type>(frame.data().at(i)));
  }
  return flattened;
}

// Helper function to create a frame with all points to be UNDEFINED.
std::vector<Segmentation::Type> CreateEmptyPrediction(int num_points) {
  std::vector<Segmentation::Type> flattened(num_points,
                                            Segmentation::TYPE_UNDEFINED);
  return flattened;
}

// Computes the 3D semantic segmentation metrics.
void Compute(const std::string& pd_str, const std::string& gt_str) {
  SegmentationMetricsConfig segmentation_metrics_config;
  // Create a segmentation_metrics_config, where we evaluate:
  // 1. All classes except the TYPE_UNDEFINED
  // 2. The TOP lidar and both return range images.
  const auto segmention_type_descriptor = Segmentation::Type_descriptor();
  for (int i = 0; i < segmention_type_descriptor->value_count(); i++) {
    // Loop over all Segmentation::Type except the TYPE_UNDEFINED
    Segmentation::Type segmentation_type = static_cast<Segmentation::Type>(
        segmention_type_descriptor->value(i)->number());
    if (segmentation_type != Segmentation::TYPE_UNDEFINED) {
      segmentation_metrics_config.mutable_segmentation_types()->Add(
          segmentation_type);
    }
  }
  SegmentationMetricsIOU miou =
      SegmentationMetricsIOU(segmentation_metrics_config);
  SegmentationFrameList pd_frames;
  if (!pd_frames.ParseFromString(pd_str)) {
    std::cerr << "Failed to parse predictions.\n";
    return;
  }
  std::cout << pd_frames.frames_size() << " frames found in prediction.\n";
  SegmentationFrameList gt_frames;
  if (!gt_frames.ParseFromString(gt_str)) {
    std::cerr << "Failed to parse ground truths.\n";
    return;
  }
  std::cout << gt_frames.frames_size() << " frames found in groundtruth.\n";
  std::map<std::pair<std::string, int64>, SegmentationFrame> pd_map;
  std::map<std::pair<std::string, int64>, SegmentationFrame> gt_map;
  std::set<std::pair<std::string, int64>> all_example_keys;
  auto get_key = [](const SegmentationFrame& segmentation_frame) {
    return std::make_pair(segmentation_frame.context_name(),
                          segmentation_frame.frame_timestamp_micros());
  };
  for (auto& f : *pd_frames.mutable_frames()) {
    const auto key = get_key(f);
    pd_map[key] = std::move(f);
    all_example_keys.insert(key);
  }
  for (auto& f : *gt_frames.mutable_frames()) {
    const auto key = get_key(f);
    gt_map[key] = std::move(f);
    all_example_keys.insert(key);
  }
  int counter = 0;
  for (auto& example_key : all_example_keys) {
    if (counter % 100 == 0) {
      std::cout << "Processing example " << counter << " out of "
                << all_example_keys.size() << std::endl;
    }
    ++counter;
    auto gt_it = gt_map.find(example_key);
    if (gt_it == gt_map.end() || gt_it->second.segmentation_labels().empty()) {
      // We skip frames which do not have ground truth.
      continue;
    }
    if (gt_it->second.segmentation_labels().size() > 1 ||
        gt_it->second.segmentation_labels()[0].name() != LaserName::TOP) {
      std::cerr << "Only TOP laser is supported right now.\n";
      return;
    }
    auto pd_it = pd_map.find(example_key);
    bool has_valid_prediction =
        pd_it != pd_map.end() &&
        pd_it->second.segmentation_labels().size() == 1 &&
        pd_it->second.segmentation_labels()[0].name() == LaserName::TOP;
    MatrixInt32 gt_matrix;
    // Measure the first return range image.
    gt_matrix.ParseFromString(Uncompress(gt_it->second.segmentation_labels()[0]
                                             .ri_return1()
                                             .segmentation_label_compressed()));
    const int num_points = gt_matrix.data().size();

    std::vector<Segmentation::Type> pd_vector;
    if (has_valid_prediction) {
      MatrixInt32 pd_matrix;
      pd_matrix.ParseFromString(
          Uncompress(pd_it->second.segmentation_labels()[0]
                         .ri_return1()
                         .segmentation_label_compressed()));
      pd_vector = Flatten(pd_matrix);
    } else {
      pd_vector = CreateEmptyPrediction(num_points);
    }
    miou.Update(pd_vector, Flatten(gt_matrix));
    // Measure the second return range image.
    gt_matrix.ParseFromString(Uncompress(gt_it->second.segmentation_labels()[0]
                                             .ri_return2()
                                             .segmentation_label_compressed()));
    if (has_valid_prediction) {
      MatrixInt32 pd_matrix;
      pd_matrix.ParseFromString(
          Uncompress(pd_it->second.segmentation_labels()[0]
                         .ri_return2()
                         .segmentation_label_compressed()));
      pd_vector = Flatten(pd_matrix);
    } else {
      pd_vector = CreateEmptyPrediction(num_points);
    }
    miou.Update(pd_vector, Flatten(gt_matrix));
  }
  const SegmentationMetrics segmentation_metrics = miou.ComputeIOU();
  for (auto it = segmentation_metrics.per_class_iou().begin();
       it != segmentation_metrics.per_class_iou().end(); it++) {
    std::cout << Segmentation::Type_Name(static_cast<Segmentation::Type>(
                     it->first))    // string (key)
              << ':' << it->second  // string's value
              << std::endl;
  }
  std::cout << "miou=" << segmentation_metrics.miou() << std::endl;
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
