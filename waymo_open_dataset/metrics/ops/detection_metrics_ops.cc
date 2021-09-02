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
#include "waymo_open_dataset/metrics/detection_metrics.h"
#include "waymo_open_dataset/metrics/ops/utils.h"
#include "waymo_open_dataset/protos/breakdown.pb.h"
#include "waymo_open_dataset/protos/metrics.pb.h"

namespace tensorflow {
namespace {
namespace co = ::waymo::open_dataset;

// Defines the detection metrics op that computes the detection metrics.
// See metrics_ops.cc for detailed explanation of the op.
class DetectionMetricsOp final : public OpKernel {
 public:
  explicit DetectionMetricsOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    std::string config_str;
    OP_REQUIRES_OK(ctx, ctx->GetAttr("config", &config_str));
    OP_REQUIRES(ctx, config_.ParseFromString(config_str),
                errors::InvalidArgument("Failed to parse config from string: ",
                                        absl::CEscape(config_str)));
    OP_REQUIRES(
        ctx, config_.box_type() != co::Label::Box::TYPE_UNKNOWN,
        errors::InvalidArgument("Unknown box type ",
                                co::Label::Box::Type_Name(config_.box_type())));
  }

  void Compute(OpKernelContext* ctx) override {
    InputTensors input;

    OP_REQUIRES_OK(ctx, ctx->input("prediction_bbox", &input.prediction_bbox));
    OP_REQUIRES_OK(ctx, ctx->input("prediction_type", &input.prediction_type));
    OP_REQUIRES_OK(ctx,
                   ctx->input("prediction_score", &input.prediction_score));
    OP_REQUIRES_OK(
        ctx, ctx->input("prediction_frame_id", &input.prediction_frame_id));
    OP_REQUIRES_OK(ctx, ctx->input("prediction_overlap_nlz",
                                   &input.prediction_overlap_nlz));
    OP_REQUIRES_OK(ctx,
                   ctx->input("ground_truth_bbox", &input.ground_truth_bbox));
    OP_REQUIRES_OK(ctx,
                   ctx->input("ground_truth_type", &input.ground_truth_type));
    OP_REQUIRES_OK(
        ctx, ctx->input("ground_truth_frame_id", &input.ground_truth_frame_id));
    OP_REQUIRES_OK(ctx, ctx->input("ground_truth_difficulty",
                                   &input.ground_truth_difficulty));
    OP_REQUIRES_OK(ctx, ctx->input("ground_truth_speed",
                                   &input.ground_truth_speed));
    OutputTensors output = ComputeImpl(input, ctx);
    ctx->set_output(0, output.average_precision);
    ctx->set_output(1, output.average_precision_ha_weighted);
    ctx->set_output(2, output.precision_recall);
    ctx->set_output(3, output.precision_recall_ha_weighted);
    ctx->set_output(4, output.breakdown);
  }

 private:
  // Wrapper of all inputs.
  struct InputTensors {
    const Tensor* prediction_bbox = nullptr;
    const Tensor* prediction_type = nullptr;
    const Tensor* prediction_score = nullptr;
    const Tensor* prediction_frame_id = nullptr;
    const Tensor* prediction_overlap_nlz = nullptr;

    const Tensor* ground_truth_bbox = nullptr;
    const Tensor* ground_truth_type = nullptr;
    const Tensor* ground_truth_frame_id = nullptr;
    const Tensor* ground_truth_difficulty = nullptr;
    const Tensor* ground_truth_speed = nullptr;
  };

  // Wrapper of all outputs.
  struct OutputTensors {
    OutputTensors(int num_breakdowns, int num_score_cutoffs)
        : average_precision(DT_FLOAT, {num_breakdowns}),
          average_precision_ha_weighted(DT_FLOAT, {num_breakdowns}),
          precision_recall(DT_FLOAT, {num_breakdowns, num_score_cutoffs, 2}),
          precision_recall_ha_weighted(DT_FLOAT,
                                       {num_breakdowns, num_score_cutoffs, 2}),
          breakdown(DT_UINT8, {num_breakdowns, 3}) {}
    Tensor average_precision;
    Tensor average_precision_ha_weighted;
    Tensor precision_recall;
    Tensor precision_recall_ha_weighted;
    Tensor breakdown;

    // Creates an output struct from a vector of detection metrics.
    static OutputTensors Create(
        const co::Config& config,
        const std::vector<co::DetectionMetrics>& metrics) {
      const int num_breakdowns = metrics.size();
      const int num_score_cutoffs = config.score_cutoffs_size();

      OutputTensors output(num_breakdowns, num_score_cutoffs);
      for (int i = 0; i < num_breakdowns; ++i) {
        output.average_precision.vec<float>()(i) =
            metrics[i].mean_average_precision();
        output.average_precision_ha_weighted.vec<float>()(i) =
            metrics[i].mean_average_precision_ha_weighted();
        output.breakdown.matrix<uint8>()(i, 0) =
            metrics[i].breakdown().generator_id();
        output.breakdown.matrix<uint8>()(i, 1) = metrics[i].breakdown().shard();
        output.breakdown.matrix<uint8>()(i, 2) =
            metrics[i].breakdown().difficulty_level();

        CHECK_EQ(metrics[i].precisions_size(), num_score_cutoffs);
        CHECK_EQ(metrics[i].recalls_size(), num_score_cutoffs);
        CHECK_EQ(metrics[i].precisions_ha_weighted_size(), num_score_cutoffs);
        CHECK_EQ(metrics[i].recalls_ha_weighted_size(), num_score_cutoffs);
        for (int j = 0; j < num_score_cutoffs; ++j) {
          output.precision_recall.tensor<float, 3>()(i, j, 0) =
              metrics[i].precisions(j);
          output.precision_recall.tensor<float, 3>()(i, j, 1) =
              metrics[i].recalls(j);
          output.precision_recall_ha_weighted.tensor<float, 3>()(i, j, 0) =
              metrics[i].precisions_ha_weighted(j);
          output.precision_recall_ha_weighted.tensor<float, 3>()(i, j, 1) =
              metrics[i].recalls_ha_weighted(j);
        }
      }
      return output;
    }
  };

  // Computes the detection metrics.
  OutputTensors ComputeImpl(const InputTensors& input, OpKernelContext* ctx) {
    LOG(INFO) << "Computing detection metrics for "
              << input.prediction_bbox->dim_size(0) << " predicted boxes.";
    LOG(INFO) << "Parsing prediction "
              << input.prediction_bbox->shape().DebugString()
              << input.prediction_frame_id->shape();
    absl::flat_hash_map<waymo::open_dataset::int64, std::vector<co::Object>>
        pds_map = co::ParseObjectFromTensors(
            *input.prediction_bbox, *input.prediction_type,
            *input.prediction_frame_id, *input.prediction_score,
            *input.prediction_overlap_nlz, absl::nullopt, absl::nullopt,
            absl::nullopt);
    LOG(INFO) << "Parsing ground truth "
              << input.ground_truth_bbox->shape().DebugString()
              << input.ground_truth_frame_id->shape();

    absl::flat_hash_map<waymo::open_dataset::int64, std::vector<co::Object>>
        gts_map = co::ParseObjectFromTensors(
            *input.ground_truth_bbox, *input.ground_truth_type,
            *input.ground_truth_frame_id, absl::nullopt, absl::nullopt,
            *input.ground_truth_difficulty, absl::nullopt,
            *input.ground_truth_speed);
    std::set<int64> frame_ids;
    for (const auto& kv : pds_map) {
      frame_ids.insert(kv.first);
    }
    for (const auto& kv : gts_map) {
      frame_ids.insert(kv.first);
    }
    std::vector<std::vector<co::Object>> gts;
    std::vector<std::vector<co::Object>> pds;
    for (const int64 id : frame_ids) {
      pds.push_back(std::move(pds_map[id]));
      gts.push_back(std::move(gts_map[id]));
    }

    // Ensure there is at least a single frame to run with or else the output
    // tensors will be empty.
    if (pds.empty() && gts.empty()) {
      pds.push_back({});
      gts.push_back({});
    }

    std::vector<std::vector<co::DetectionMeasurements>> measurements(
        pds.size());
    co::Config config = config_;
    if (config.score_cutoffs_size() == 0) {
      config = co::EstimateScoreCutoffs(config, pds, gts);
    }

    {
      tensorflow::thread::ThreadPool pool(
          ctx->env(), tensorflow::ThreadOptions(),
          "ComputeDetectionMetricsPool", port::MaxParallelism(),
          /*low_latency_hint=*/false);
      for (int i = 0, sz = pds.size(); i < sz; ++i) {
        pool.Schedule([&config, &pds, &gts, &measurements, i]() {
          measurements[i] =
              co::ComputeDetectionMeasurements(config, pds[i], gts[i]);
        });
      }
    }

    const std::vector<co::DetectionMetrics> metrics =
        co::ComputeDetectionMetrics(config, measurements);
    LOG(INFO) << "Done with computing detection metrics.";
    return OutputTensors::Create(config, metrics);
  }

  co::Config config_;
};

REGISTER_KERNEL_BUILDER(Name("DetectionMetrics").Device(DEVICE_CPU),
                        DetectionMetricsOp);

}  // namespace
}  // namespace tensorflow
