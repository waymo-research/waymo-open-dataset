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
// pd_filename is the name of a file that has prediction frames in format of
// waymo::open_dataset::SegmentationFrameList proto.
// gt_filename is the name of a file that has groud truth frames in format of
// waymo::open_dataset::SegmentationFrameList proto.
//
//
// Results when running on ground_truths.bin and fake_predictions.bin in the
// directory gives the following result:
// miou=0.25

#include <ctime>
#include <fstream>
#include <iostream>
#include <ostream>
#include <streambuf>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "absl/strings/str_cat.h"
#include "waymo_open_dataset/dataset.pb.h"
#include "waymo_open_dataset/label.pb.h"
#include "waymo_open_dataset/metrics/segmentation_metrics.h"
#include "waymo_open_dataset/protos/segmentation_metrics.pb.h"

namespace waymo {
namespace open_dataset {
namespace {

// Helper function to convert a frame into a vector.
std::vector<Segmentation::Type> Flatten(MatrixInt32 frame) {
  std::vector<Segmentation::Type> flattened;
  flattened.reserve(frame.data().size());
  for (int i = 0; i < frame.data().size(); ++i) {
    flattened.push_back(static_cast<Segmentation::Type>(frame.data().at(i)));
  }
  return flattened;
}

// Computes the 3D semantic segmentation metrics.
void Compute(const std::string& pd_str, const std::string& gt_str) {
  MetricsMeanIOU miou = MetricsMeanIOU(
      {Segmentation::TYPE_VEHICLE, Segmentation::TYPE_PEDESTRIAN});
  SegmentationFrameList pd_frames;
  if (!pd_frames.ParseFromString(pd_str)) {
    std::cerr << "Failed to parse predictions.";
    return;
  }
  SegmentationFrameList gt_frames;
  if (!gt_frames.ParseFromString(gt_str)) {
    std::cerr << "Failed to parse ground truths.";
    return;
  }
  if (gt_frames.frames().size() != pd_frames.frames().size()) {
    std::cerr << "Two files contain different numbers of frames.";
    return;
  }
  for (size_t i = 0; i != gt_frames.frames().size(); ++i) {
    if (gt_frames.frames()[i].has_segmentation_labels() &&
        pd_frames.frames()[i].has_segmentation_labels()) {
      MatrixInt32 gt_matrix, pd_matrix;
      gt_matrix.ParseFromString(gt_frames.frames()[i]
                                    .segmentation_labels()
                                    .segmentation_label_compressed());
      pd_matrix.ParseFromString(pd_frames.frames()[i]
                                    .segmentation_labels()
                                    .segmentation_label_compressed());
      miou.Update(Flatten(pd_matrix), Flatten(gt_matrix));
    }
  }
  std::cout << "miou=" << miou.ComputeMeanIOU() << std::endl;
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
