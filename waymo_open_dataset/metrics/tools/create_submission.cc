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

// A simple commandline to create submission in leaderboard format.
// This is written in c++ as it is very slow to do this in plain python.
//
// Example usage:
// create_submission  --input_filenames='/tmp/preds.bin' \
// --output_filename='/tmp/my_model' \
// --submission_filename='waymo_open_dataset/metrics/tools/submission.txtpb'
//
// Example code to generate /tmp/preds.bin:
// See example_code_to_create_a_prediction_file below.
//
// For python, it should be similar but just python proto interface.

#include <cmath>
#include <fstream>
#include <iostream>
#include <streambuf>
#include <string>

#include <glog/logging.h>
#include "google/protobuf/text_format.h"
#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/strings/str_cat.h"
#include "absl/time/time.h"
#include "waymo_open_dataset/common/integral_types.h"
#include "waymo_open_dataset/dataset.pb.h"
#include "waymo_open_dataset/label.pb.h"
#include "waymo_open_dataset/protos/metrics.pb.h"
#include "waymo_open_dataset/protos/submission.pb.h"

ABSL_FLAG(std::vector<std::string>, input_filenames, {},
          "Comma separated list of files that contains "
          "car.open_dataset.Objects proto.");
ABSL_FLAG(std::string, output_filename, "", "The output filename.");
ABSL_FLAG(waymo::open_dataset::int32, num_shards, 4, "Number of sharded files.");
ABSL_FLAG(std::string, submission_filename, "",
          "Filename of a submission file that has a text proto of the "
          "Submission proto.");

namespace waymo {
namespace open_dataset {
namespace {
void example_code_to_create_a_prediction_file() {
  waymo::open_dataset::Objects objects;
  auto* o = objects.add_objects();

  // The following 3 fields are used to uniquely identify a frame a prediction
  // is predicted at. Make sure you set them to values exactly the same as what
  // we provided in the raw data. Otherwise your prediction is considered as a
  // false negative.
  o->set_context_name(
      "context_name for the prediction. See Frame::context::name in "
      "dataset.proto.");
  // The frame timestamp for the prediction. See Frame::timestamp_micros in
  // dataset.proto.
  constexpr int64 kInvalid = -1;
  o->set_frame_timestamp_micros(kInvalid);
  // This is only needed for 2D detection or tracking tasks.
  // Set it to the camera name the prediction is for.
  o->set_camera_name(waymo::open_dataset::CameraName::FRONT);

  // Populating box and score.
  auto* box = o->mutable_object()->mutable_box();
  box->set_center_x(0);
  box->set_center_y(0);
  box->set_center_z(0);
  box->set_length(0);
  box->set_width(0);
  box->set_height(0);
  box->set_heading(0);
  // This must be within [0.0, 1.0].
  o->set_score(0.5);
  // For tracking, this must be set and it must be unique for each tracked
  // sequence.
  o->mutable_object()->set_id("unique object tracking ID");
  // Use correct type.
  o->mutable_object()->set_type(Label::TYPE_PEDESTRIAN);

  // Add more objects here.

  // Write objects to a file.
  std::string objects_str = objects.SerializeAsString();
  std::ofstream of("your_filename");
  of << objects_str;
  of.close();
}

void ValidateFlags() {
  const std::vector<std::string> input_filenames =
      absl::GetFlag(FLAGS_input_filenames);
  if (input_filenames.empty()) {
    LOG(FATAL) << "--input_filenames must be set.";
  }
  const std::string output_filename = absl::GetFlag(FLAGS_output_filename);
  if (output_filename.empty()) {
    LOG(FATAL) << "--output_filename must be set.";
  }
  const int32 num_shards = absl::GetFlag(FLAGS_num_shards);
  if (num_shards <= 0) {
    LOG(FATAL) << "--num_shards must be > 0.";
  }
  const std::string submission_filename =
      absl::GetFlag(FLAGS_submission_filename);
  if (submission_filename.empty()) {
    LOG(FATAL) << "--submission_filename must be set. See submission.txtpb as "
                  "an example.";
  }
}

// Read all content in a file to a string.
std::string ReadFileToString(const std::string& filename) {
  std::ifstream s(filename.c_str());
  CHECK(s.is_open()) << filename << " does not exist.";
  const std::string content((std::istreambuf_iterator<char>(s)),
                            std::istreambuf_iterator<char>());
  s.close();
  return content;
}

void WriteStringToFile(const std::string& filename,
                       const std::string& content) {
  std::ofstream o(filename.c_str());
  o << content;
  o.close();
}

void Run() {
  ValidateFlags();
  const std::vector<std::string> input_filenames =
      absl::GetFlag(FLAGS_input_filenames);
  std::vector<Objects> objects_all(input_filenames.size());
  int i = 0;
  for (const auto& name : input_filenames) {
    std::string content = ReadFileToString(name);
    if (content.back() == '\n') {
      content.pop_back();
    }
    CHECK(objects_all[i].ParseFromString(content));
    i++;
  }

  const std::string submission_content =
      ReadFileToString(absl::GetFlag(FLAGS_submission_filename));
  Submission submission;
  CHECK(google::protobuf::TextFormat::ParseFromString(submission_content, &submission))
      << "Failed to parse " << submission_content;
  CHECK(!submission.unique_method_name().empty() &&
        !submission.account_name().empty())
      << "unique_method_name and account_name must be set in "
         "--submission_filename.";

  std::vector<Submission> submissions(absl::GetFlag(FLAGS_num_shards),
                                      submission);

  for (const auto& objects : objects_all) {
    for (int i = 0; i < objects.objects_size(); ++i) {
      auto o = objects.objects(i);
      if (o.score() < 0.03) {
        continue;
      }
      // Sanity checks.
      if (std::isnan(o.object().box().length()) ||
          std::isnan(o.object().box().width()) ||
          std::isnan(o.object().box().height()) ||
          std::isnan(o.object().box().center_x()) ||
          std::isnan(o.object().box().center_y()) ||
          std::isnan(o.object().box().center_z()) ||
          std::isnan(o.object().box().heading())) {
        LOG(INFO) << o.DebugString();
        continue;
      }
      o.mutable_object()->mutable_box()->set_length(
          std::max(objects.objects(i).object().box().length(), 1e-6));
      o.mutable_object()->mutable_box()->set_width(
          std::max(objects.objects(i).object().box().width(), 1e-6));
      o.mutable_object()->mutable_box()->set_height(
          std::max(objects.objects(i).object().box().height(), 1e-6));

      *submissions[i % submissions.size()]
           .mutable_inference_results()
           ->add_objects() = o;
    }
  }

  for (int i = 0; i < submissions.size(); ++i) {
    WriteStringToFile(absl::StrCat(absl::GetFlag(FLAGS_output_filename), i),
                      submissions[i].SerializeAsString());
  }
}
}  // namespace
}  // namespace open_dataset
}  // namespace waymo

int main(int argc, char** argv) {
  absl::ParseCommandLine(argc, argv);
  waymo::open_dataset::Run();
  return 0;
}
