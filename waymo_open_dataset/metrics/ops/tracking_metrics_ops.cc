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

#include <memory>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/strings/escaping.h"
#include "absl/types/optional.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/platform/cpu_info.h"
#include "waymo_open_dataset/label.pb.h"
#include "waymo_open_dataset/metrics/ops/utils.h"
#include "waymo_open_dataset/metrics/tracking_metrics.h"
#include "waymo_open_dataset/protos/breakdown.pb.h"
#include "waymo_open_dataset/protos/metrics.pb.h"

namespace tensorflow {
namespace {
namespace co = ::waymo::open_dataset;

// Defines the tracking metrics op that computes the tracking metrics.
// See metrics_ops.cc for detailed explanation of the op.
class TrackingMetricsOp final : public OpKernel {
 public:
  explicit TrackingMetricsOp(OpKernelConstruction* context)
      : OpKernel(context) {
    std::string config_str;
    OP_REQUIRES_OK(context, context->GetAttr("config", &config_str));
    OP_REQUIRES(context, config_.ParseFromString(config_str),
                errors::InvalidArgument("Failed to parse config from string: ",
                                        absl::CEscape(config_str)));
    OP_REQUIRES(
        context, config_.box_type() != co::Label::Box::TYPE_UNKNOWN,
        errors::InvalidArgument("Unknown box type ",
                                co::Label::Box::Type_Name(config_.box_type())));
  }

  void Compute(OpKernelContext* context) override {
    InputTensors input;

    OP_REQUIRES_OK(context,
                   context->input("prediction_bbox", &input.prediction_bbox));
    OP_REQUIRES_OK(context,
                   context->input("prediction_type", &input.prediction_type));
    OP_REQUIRES_OK(context,
                   context->input("prediction_score", &input.prediction_score));
    OP_REQUIRES_OK(context, context->input("prediction_frame_id",
                                           &input.prediction_frame_id));
    OP_REQUIRES_OK(context, context->input("prediction_sequence_id",
                                           &input.prediction_sequence_id));
    OP_REQUIRES_OK(context, context->input("prediction_object_id",
                                           &input.prediction_object_id));
    OP_REQUIRES_OK(context, context->input("prediction_overlap_nlz",
                                           &input.prediction_overlap_nlz));
    OP_REQUIRES_OK(
        context, context->input("ground_truth_bbox", &input.ground_truth_bbox));
    OP_REQUIRES_OK(
        context, context->input("ground_truth_type", &input.ground_truth_type));
    OP_REQUIRES_OK(context, context->input("ground_truth_frame_id",
                                           &input.ground_truth_frame_id));
    OP_REQUIRES_OK(context, context->input("ground_truth_sequence_id",
                                           &input.ground_truth_sequence_id));
    OP_REQUIRES_OK(context, context->input("ground_truth_object_id",
                                           &input.ground_truth_object_id));
    OP_REQUIRES_OK(context, context->input("ground_truth_difficulty",
                                           &input.ground_truth_difficulty));
    OP_REQUIRES_OK(context, context->input("ground_truth_speed",
                                           &input.ground_truth_speed));
    // Create output tensor.
    OutputTensors output = ComputeImpl(input, context);
    context->set_output(0, output.mota);
    context->set_output(1, output.motp);
    context->set_output(2, output.miss);
    context->set_output(3, output.mismatch);
    context->set_output(4, output.fp);
    context->set_output(5, output.score_cutoff);
    context->set_output(6, output.breakdown);
  }

 private:
  // Wrapper of all inputs.
  struct InputTensors {
    const Tensor* prediction_bbox = nullptr;
    const Tensor* prediction_type = nullptr;
    const Tensor* prediction_score = nullptr;
    const Tensor* prediction_frame_id = nullptr;
    const Tensor* prediction_sequence_id = nullptr;
    const Tensor* prediction_object_id = nullptr;
    const Tensor* prediction_overlap_nlz = nullptr;

    const Tensor* ground_truth_bbox = nullptr;
    const Tensor* ground_truth_type = nullptr;
    const Tensor* ground_truth_frame_id = nullptr;
    const Tensor* ground_truth_sequence_id = nullptr;
    const Tensor* ground_truth_object_id = nullptr;
    const Tensor* ground_truth_difficulty = nullptr;
    const Tensor* ground_truth_speed = nullptr;
  };

  // Wrapper of all outputs.
  struct OutputTensors {
    OutputTensors(int num_breakdowns)
        : mota(DT_FLOAT, {num_breakdowns}),
          motp(DT_FLOAT, {num_breakdowns}),
          miss(DT_FLOAT, {num_breakdowns}),
          mismatch(DT_FLOAT, {num_breakdowns}),
          fp(DT_FLOAT, {num_breakdowns}),
          score_cutoff(DT_FLOAT, {num_breakdowns}),
          breakdown(DT_UINT8, {num_breakdowns, 3}) {}
    // One tensor for each field in TrackingMetrics proto.
    Tensor mota;
    Tensor motp;
    Tensor miss;
    Tensor mismatch;
    Tensor fp;
    Tensor score_cutoff;
    Tensor breakdown;

    // Creates an output struct from a vector of tracking metrics.
    static OutputTensors Create(
        const co::Config& config,
        const std::vector<co::TrackingMetrics>& metrics) {
      const int num_breakdowns = metrics.size();

      OutputTensors output(num_breakdowns);
      for (int i = 0; i < num_breakdowns; ++i) {
        output.mota.vec<float>()(i) = metrics[i].mota();
        output.motp.vec<float>()(i) = metrics[i].motp();
        output.miss.vec<float>()(i) = metrics[i].miss();
        output.mismatch.vec<float>()(i) = metrics[i].mismatch();
        output.fp.vec<float>()(i) = metrics[i].fp();
        output.score_cutoff.vec<float>()(i) = metrics[i].score_cutoff();

        output.breakdown.matrix<uint8>()(i, 0) =
            metrics[i].breakdown().generator_id();
        output.breakdown.matrix<uint8>()(i, 1) = metrics[i].breakdown().shard();
        output.breakdown.matrix<uint8>()(i, 2) =
            metrics[i].breakdown().difficulty_level();
      }
      return output;
    }
  };

  // Computes the tracking metrics.
  OutputTensors ComputeImpl(const InputTensors& input,
                            OpKernelContext* context) {
    LOG(INFO) << "Computing tracking metrics for "
              << input.prediction_bbox->dim_size(0) << " predicted boxes.";
    LOG(INFO) << "Parsing prediction "
              << input.prediction_bbox->shape().DebugString()
              << input.prediction_frame_id->shape();

    // Map of sequence ids to (map of frame ids to list of objects in that
    // frame).
    absl::flat_hash_map<
        std::string,
        absl::flat_hash_map<waymo::open_dataset::int64, std::vector<co::Object>>>
        pds_map = co::ParseObjectGroupedBySequenceFromTensors(
            /*bbox=*/*input.prediction_bbox,
            /*type=*/*input.prediction_type,
            /*frame_id=*/*input.prediction_frame_id,
            /*sequence_id=*/*input.prediction_sequence_id,
            /*object_id=*/*input.prediction_object_id,
            /*score=*/*input.prediction_score,
            /*overlap_nlz=*/*input.prediction_overlap_nlz,
            /*detection_difficulty=*/absl::nullopt,
            /*tracking_difficulty=*/absl::nullopt,
            /*object_speed=*/absl::nullopt);

    LOG(INFO) << "Parsing ground truth "
              << input.ground_truth_bbox->shape().DebugString()
              << input.ground_truth_frame_id->shape();
    // Map of sequence ids to (map of frame ids to list of objects in that
    // frame).
    absl::flat_hash_map<
        std::string,
        absl::flat_hash_map<waymo::open_dataset::int64, std::vector<co::Object>>>
        gts_map = co::ParseObjectGroupedBySequenceFromTensors(
            /*bbox=*/*input.ground_truth_bbox,
            /*type=*/*input.ground_truth_type,
            /*frame_id=*/*input.ground_truth_frame_id,
            /*sequence_id=*/*input.ground_truth_sequence_id,
            /*object_id=*/*input.ground_truth_object_id,
            /*score=*/absl::nullopt,
            /*overlap_nlz=*/absl::nullopt,
            /*detection_difficulty=*/*input.ground_truth_difficulty,
            /*tracking_difficulty=*/absl::nullopt,
            /*object_speed=*/*input.ground_truth_speed);

    std::set<std::string> sequence_ids;
    for (const auto& kv : pds_map) {
      sequence_ids.insert(kv.first);
    }
    for (const auto& kv : gts_map) {
      sequence_ids.insert(kv.first);
    }
    // gts and pds are aligned so that all their vectors are of the same size.
    // Format is vector of sequences -> vector of frames -> vector of objects.
    std::vector<std::vector<std::vector<co::Object>>> gts;
    std::vector<std::vector<std::vector<co::Object>>> pds;
    for (const auto& sequence_id : sequence_ids) {
      std::set<int64> frame_ids;
      for (const auto& kv : pds_map[sequence_id]) {
        frame_ids.insert(kv.first);
      }
      for (const auto& kv : gts_map[sequence_id]) {
        frame_ids.insert(kv.first);
      }
      std::vector<std::vector<co::Object>> pds_frames_of_objects;
      std::vector<std::vector<co::Object>> gts_frames_of_objects;
      for (const int64 id : frame_ids) {
        pds_frames_of_objects.push_back(std::move(pds_map[sequence_id][id]));
        gts_frames_of_objects.push_back(std::move(gts_map[sequence_id][id]));
      }

      pds.push_back(std::move(pds_frames_of_objects));
      gts.push_back(std::move(gts_frames_of_objects));
    }

    // Use threads to compute per sequence measurements.
    std::vector<std::vector<co::TrackingMeasurements>> measurements(pds.size());
    co::Config config = config_;
    if (config.score_cutoffs_size() == 0) {
      config = co::EstimateScoreCutoffs(config, pds, gts);
    }
    {
      tensorflow::thread::ThreadPool pool(
          context->env(), tensorflow::ThreadOptions(),
          "ComputeTrackingMetricsPool", port::MaxParallelism(),
          /*low_latency_hint=*/false);
      for (int i = 0, sz = pds.size(); i < sz; ++i) {
        pool.Schedule([&config, &pds, &gts, &measurements, i]() {
          measurements[i] =
              co::ComputeTrackingMeasurements(config, pds[i], gts[i]);
        });
      }
    }

    const std::vector<co::TrackingMetrics> metrics =
        co::ComputeTrackingMetrics(config, measurements);
    LOG(INFO) << "Done with computing tracking metrics.";
    return OutputTensors::Create(config, metrics);
  }

  co::Config config_;
};

REGISTER_KERNEL_BUILDER(Name("TrackingMetrics").Device(DEVICE_CPU),
                        TrackingMetricsOp);

}  // namespace
}  // namespace tensorflow
