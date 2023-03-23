/* Copyright 2023 The Waymo Open Dataset Authors.

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
// cc_library does not have a header, and clang-format is not able to find an
// include ordering consistent with the style guide.

// clang-format off
#include <string>
#include <vector>


#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/op_requires.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/errors.h"
#include "waymo_open_dataset/label.pb.h"
#include "waymo_open_dataset/metrics/matcher.h"
#include "waymo_open_dataset/protos/metrics.pb.h"
// clang-format on

namespace tf = tensorflow;

namespace tensorflow {
namespace {
namespace co = ::waymo::open_dataset;

// Returns true if the box has nonzero extents.
// Inputs to the kernel is a zero-padded tensor of boxes, and we do not need
// to bother trying to match these invalid boxes. They already get discarded by
// ComputeIoU2d and ComputeIoU3d; this filtering just avoids extra work and
// logging.
bool IsValidBox(const co::Label::Box& box,
                co::Label::Box::Type box_type) {
  switch (box_type) {
    case co::Label::Box::TYPE_3D:
      return box.width() > 0 && box.length() > 0 && box.height() > 0;
    case co::Label::Box::TYPE_2D:
    case co::Label::Box::TYPE_AA_2D:
      return box.width() > 0 && box.length() > 0;
    case co::Label::Box::TYPE_UNKNOWN:
      LOG(FATAL) << "Invalid (unknown) box type. ";
      return false;
  }
}

co::Object ParseBoxTensor(const tf::Tensor& input, int id) {
  int num_elements = input.NumElements();
  CHECK(num_elements == 7 || num_elements == 5 || num_elements == 4)
      << "Wrong size: " << input.shape().DebugString();
  co::Object output;
  output.set_score(1.0);
  co::Label* label = output.mutable_object();
  co::Label::Box* box = label->mutable_box();
  label->set_type(co::Label::TYPE_VEHICLE);
  label->set_id(absl::StrCat(id));
  const auto input_data = input.unaligned_flat<float>();
  if (num_elements == 7) {
    box->set_center_x(input_data(0));
    box->set_center_y(input_data(1));
    box->set_center_z(input_data(2) + input_data(5) / 2.0);
    box->set_length(input_data(3));
    box->set_width(input_data(4));
    box->set_height(input_data(5));
    box->set_heading(input_data(6));
  } else if (num_elements == 5) {
    box->set_center_x(input_data(0));
    box->set_center_y(input_data(1));
    box->set_length(input_data(2));
    box->set_width(input_data(3));
    box->set_heading(input_data(4));
  } else {
    box->set_center_x(input_data(0));
    box->set_center_y(input_data(1));
    box->set_length(input_data(2));
    box->set_width(input_data(3));
  }
  return output;
}

struct MatchResult {
  int prediction_index;
  int ground_truth_index;
  float iou;
  float longitudinal_affinity;
};

std::vector<MatchResult> MatchBoxes(
    const std::vector<co::Object> predictions,
    const std::vector<co::Object>& ground_truths,
    const co::Config& config) {
  auto matcher = co::Matcher::Create(config);
  CHECK(matcher != nullptr);
  matcher->SetPredictions(predictions);
  matcher->SetGroundTruths(ground_truths);
  std::vector<int> valid_prediction_index;
  valid_prediction_index.reserve(predictions.size());
  for (int i = 0; i < predictions.size(); ++i) {
    if (IsValidBox(predictions[i].object().box(), config.box_type()))
      valid_prediction_index.push_back(i);
  }

  std::vector<int> valid_ground_truth_index;
  valid_ground_truth_index.reserve(ground_truths.size());
  for (int i = 0; i < ground_truths.size(); ++i) {
    if (IsValidBox(ground_truths[i].object().box(), config.box_type())) {
      valid_ground_truth_index.push_back(i);
    }
  }
  std::vector<int> prediction_index;
  matcher->SetPredictionSubset(valid_prediction_index);
  matcher->SetGroundTruthSubset(valid_ground_truth_index);
  matcher->Match(&prediction_index, nullptr);
  CHECK_EQ(matcher->prediction_subset().size(), prediction_index.size());
  std::vector<MatchResult> output;
  for (int i = 0; i < prediction_index.size(); ++i) {
    if (prediction_index[i] == -1.0) continue;  // Object was not matched.
    int pd_idx = matcher->prediction_subset()[i];
    int gt_idx = matcher->ground_truth_subset()[prediction_index[i]];
    output.push_back({
        .prediction_index = pd_idx,
        .ground_truth_index = gt_idx,
        .iou = matcher->IoU(pd_idx, gt_idx),
        .longitudinal_affinity = matcher->LongitudinalAffinity(pd_idx, gt_idx),
    });
  }
  return output;
}

std::vector<std::vector<co::Object>> ParseBoxesTensor(
    const tf::Tensor& batched_inputs) {
  int batch_size = batched_inputs.shape().dim_size(0);
  int num_boxes = batched_inputs.shape().dim_size(1);
  std::vector<std::vector<co::Object>> output;
  output.reserve(batch_size);
  for (int batch_idx = 0; batch_idx < batch_size; ++batch_idx) {
    tf::Tensor inputs = batched_inputs.SubSlice(batch_idx);
    std::vector<co::Object> objects;
    objects.reserve(num_boxes);
    for (int box_idx = 0; box_idx < num_boxes; ++box_idx) {
      objects.push_back(ParseBoxTensor(inputs.SubSlice(box_idx), box_idx));
    }
    output.push_back(objects);
  }
  return output;
}

class OpenDatasetMatcherOp : public tf::OpKernel {
 public:
  explicit OpenDatasetMatcherOp(tf::OpKernelConstruction* ctx)
      : tf::OpKernel(ctx) {
    std::string config_str;
    OP_REQUIRES_OK(ctx, ctx->GetAttr("config", &config_str));
    OP_REQUIRES(ctx, config_.ParseFromString(config_str),
                errors::InvalidArgument(
                    "Could not parse input proto config from string."));
  }

  void Compute(tf::OpKernelContext* ctx) override {
    const tf::Tensor& predictions = ctx->input(0);
    const tf::Tensor& ground_truths = ctx->input(1);
    // Check that the dimensions of the two inputs are valid.
    OP_REQUIRES(ctx, predictions.dims() == 3,
                errors::InvalidArgument("Not a batched vector of boxes: ",
                                        predictions.shape().DebugString()));
    OP_REQUIRES(ctx, ground_truths.dims() == 3,
                errors::InvalidArgument("Not a batched vector of boxes"));
    OP_REQUIRES(ctx, predictions.dim_size(2) == ground_truths.dim_size(2),
                errors::InvalidArgument("Box dimensions are incompatible: ",
                                        ground_truths.shape().DebugString(),
                                        predictions.shape().DebugString()));
    OP_REQUIRES(ctx, ground_truths.dim_size(0) == predictions.dim_size(0),
                errors::InvalidArgument("Batch sizes don't match: ",
                                        predictions.shape().DebugString(),
                                        ground_truths.shape().DebugString()));
    {
      using Box = co::Label::Box;
      int dim_size = predictions.dim_size(2);
      auto box_type = config_.box_type();
      OP_REQUIRES(
          ctx,
          (box_type == Box::TYPE_AA_2D && dim_size == 4) ||
              (box_type == Box::TYPE_2D && (dim_size == 5 || dim_size == 7)) ||
              (box_type == Box::TYPE_3D && dim_size == 7),
          errors::InvalidArgument("Box dim=", dim_size,
                                  ", but requested matching was ",
                                  Box::Type_Name(box_type)));
    }
    OP_REQUIRES(
        ctx,
        predictions.dtype() == tf::DT_FLOAT &&
            ground_truths.dtype() == tf::DT_FLOAT,
        errors::InvalidArgument("Invalid input dtypes: [",
                                tf::DataTypeString(predictions.dtype()), ", ",
                                tf::DataTypeString(ground_truths.dtype()),
                                "], expected float32."));
    const int batch_size = predictions.dim_size(0);
    const int num_predicted_boxes = predictions.dim_size(1);
    const int num_ground_truth_boxes = ground_truths.dim_size(1);

    // Parse input boxes and run matching.
    std::vector<std::vector<co::Object>> batch_boxes_pd =
        ParseBoxesTensor(predictions);
    std::vector<std::vector<co::Object>> batch_boxes_gt =
        ParseBoxesTensor(ground_truths);
    std::vector<std::vector<MatchResult>> all_match_results;
    int num_matches = 0;
    for (int idx_batch = 0; idx_batch < batch_size; ++idx_batch) {
      all_match_results.push_back(MatchBoxes(
          batch_boxes_pd[idx_batch], batch_boxes_gt[idx_batch], config_));
      num_matches += all_match_results.back().size();
    }

    // Allocate and fill output tensors.
    tf::Tensor* out_predicted_idx = nullptr;
    OP_REQUIRES_OK(ctx,
                   ctx->allocate_output(0, {num_matches}, &out_predicted_idx));
    tf::Tensor* out_ground_truth_idx = nullptr;
    OP_REQUIRES_OK(
        ctx, ctx->allocate_output(1, {num_matches}, &out_ground_truth_idx));
    tf::Tensor* out_iou = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(2, {num_matches}, &out_iou));
    tf::Tensor* out_la = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(3, {num_matches}, &out_la));
    int idx = 0;
    for (int batch_idx = 0; batch_idx < batch_size; ++batch_idx) {
      for (const MatchResult& match : all_match_results[batch_idx]) {
        out_predicted_idx->flat<int32_t>()(idx) =
            match.prediction_index + batch_idx * num_predicted_boxes;
        out_ground_truth_idx->flat<int32_t>()(idx) =
            match.ground_truth_index + batch_idx * num_ground_truth_boxes;
        out_iou->flat<float>()(idx) = match.iou;
        out_la->flat<float>()(idx) = match.longitudinal_affinity;
        idx++;
      }
    }
  }

 private:
  co::Config config_;
};

REGISTER_KERNEL_BUILDER(Name("Match").Device(DEVICE_CPU),
                        OpenDatasetMatcherOp);

}  // namespace
}  // namespace tensorflow
