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
#include "waymo_open_dataset/metrics/config_util.h"
#include "waymo_open_dataset/metrics/motion_metrics.h"
#include "waymo_open_dataset/metrics/ops/utils.h"
#include "waymo_open_dataset/protos/breakdown.pb.h"
#include "waymo_open_dataset/protos/metrics.pb.h"
#include "waymo_open_dataset/protos/motion_metrics.pb.h"

namespace tensorflow {
namespace {
namespace co = ::waymo::open_dataset;

// Defines the motion metrics op that computes the motion metrics.
// See metrics_ops.cc for detailed explanation of the op.
class MotionMetricsOp final : public OpKernel {
 public:
  explicit MotionMetricsOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    std::string config_str;
    OP_REQUIRES_OK(ctx, ctx->GetAttr("config", &config_str));
    OP_REQUIRES(ctx, config_.ParseFromString(config_str),
                errors::InvalidArgument("Failed to parse config from string: ",
                                        absl::CEscape(config_str)));
  }

  void Compute(OpKernelContext* ctx) override {
    InputTensors input;

    OP_REQUIRES_OK(
        ctx, ctx->input("prediction_trajectory", &input.prediction_trajectory));
    OP_REQUIRES_OK(ctx,
                   ctx->input("prediction_score", &input.prediction_score));
    OP_REQUIRES_OK(ctx, ctx->input("ground_truth_trajectory",
                                   &input.ground_truth_trajectory));
    OP_REQUIRES_OK(
        ctx, ctx->input("ground_truth_is_valid", &input.ground_truth_is_valid));
    OP_REQUIRES_OK(ctx, ctx->input("prediction_ground_truth_indices",
                                   &input.prediction_ground_truth_indices));
    OP_REQUIRES_OK(ctx,
                   ctx->input("prediction_ground_truth_indices_mask",
                              &input.prediction_ground_truth_indices_mask));
    OP_REQUIRES_OK(ctx, ctx->input("object_type", &input.object_type));
    OP_REQUIRES_OK(ctx, ctx->input("object_id", &input.object_id));
    OP_REQUIRES_OK(ctx, ctx->input("scenario_id", &input.scenario_id));

    OutputTensors output = ComputeImpl(input, ctx);
    ctx->set_output(0, output.min_ade);
    ctx->set_output(1, output.min_fde);
    ctx->set_output(2, output.miss_rate);
    ctx->set_output(3, output.overlap_rate);
    ctx->set_output(4, output.mean_average_precision);
  }

 private:
  // Wrapper of all inputs.
  struct InputTensors {
    const Tensor* prediction_trajectory = nullptr;
    const Tensor* prediction_score = nullptr;
    const Tensor* ground_truth_trajectory = nullptr;
    const Tensor* ground_truth_is_valid = nullptr;
    const Tensor* prediction_ground_truth_indices = nullptr;
    const Tensor* prediction_ground_truth_indices_mask = nullptr;
    const Tensor* object_type = nullptr;
    const Tensor* object_id = nullptr;
    const Tensor* scenario_id = nullptr;
  };

  // Wrapper of all outputs.
  struct OutputTensors {
    OutputTensors(int num_breakdowns)
        : min_ade(DT_FLOAT, {num_breakdowns}),
          min_fde(DT_FLOAT, {num_breakdowns}),
          miss_rate(DT_FLOAT, {num_breakdowns}),
          overlap_rate(DT_FLOAT, {num_breakdowns}),
          mean_average_precision(DT_FLOAT, {num_breakdowns}) {}
    Tensor min_ade;
    Tensor min_fde;
    Tensor miss_rate;
    Tensor overlap_rate;
    Tensor mean_average_precision;
  };

  // Creates an output struct from a vector of motion metrics.
  OutputTensors CreateOutput(const co::MotionMetrics& metrics) {
    // Compute the metrics values from the accumulated statistics.
    const std::vector<co::Track::ObjectType> expected_types = {
        co::Track::TYPE_VEHICLE, co::Track::TYPE_PEDESTRIAN,
        co::Track::TYPE_CYCLIST};
    const int num_breakdowns =
        ::waymo::open_dataset::GetBreakdownNamesFromMotionConfig(config_).size();

    std::map<std::pair<int, int>, int> indices;

    for (int i = 0; i < metrics.metrics_bundles().size(); ++i) {
      int object_type = -1;
      if (metrics.metrics_bundles(i).has_object_filter()) {
        object_type =
            static_cast<int>(metrics.metrics_bundles(i).object_filter());
      }
      indices[{object_type, metrics.metrics_bundles(i).measurement_step()}] = i;
    }

    OutputTensors output(num_breakdowns);
    int i = 0;

    for (auto type : expected_types) {
      for (const auto& step : config_.step_configurations()) {
        auto found =
            indices.find({static_cast<int>(type), step.measurement_step()});
        if (found != indices.end()) {
          const auto bundle = metrics.metrics_bundles(found->second);
          output.min_ade.vec<float>()(i) = bundle.min_ade();
          output.min_fde.vec<float>()(i) = bundle.min_fde();
          output.miss_rate.vec<float>()(i) = bundle.miss_rate();
          output.overlap_rate.vec<float>()(i) = bundle.overlap_rate();
          output.mean_average_precision.vec<float>()(i) =
              bundle.mean_average_precision();
        } else {
          output.min_ade.vec<float>()(i) = -1.0;
          output.min_fde.vec<float>()(i) = -1.0;
          output.miss_rate.vec<float>()(i) = -1.0;
          output.overlap_rate.vec<float>()(i) = -1.0;
          output.mean_average_precision.vec<float>()(i) = -1.0;
        }
        ++i;
      }
    }
    return output;
  }

  // Computes the detection metrics.
  OutputTensors ComputeImpl(const InputTensors& input, OpKernelContext* ctx) {
    LOG(INFO) << "Computing motion metrics for "
              << input.scenario_id->dim_size(0) << " trajectories.";

    LOG(INFO) << "Parsing prediction "
              << input.prediction_trajectory->shape().DebugString()
              << input.scenario_id->shape();

    LOG(INFO) << "Parsing ground truth "
              << input.ground_truth_trajectory->shape().DebugString()
              << input.scenario_id->shape();

    absl::flat_hash_map<std::string,
                        std::pair<co::Scenario, co::ScenarioPredictions>>
        gts_pds_map = co::ParseScenarioAndPredictonsFromTensors(
            *input.prediction_trajectory, *input.prediction_score,
            *input.ground_truth_trajectory, *input.ground_truth_is_valid,
            *input.prediction_ground_truth_indices,
            *input.prediction_ground_truth_indices_mask, *input.object_type,
            *input.object_id, *input.scenario_id);

    co::MotionMetricsConfig metrics_config = config_;

    std::vector<std::pair<co::Scenario, co::ScenarioPredictions>> gts_pds(
        gts_pds_map.size());
    for (const auto& gt_pd : gts_pds_map) {
      gts_pds.push_back(std::move(gt_pd.second));
    }

    std::vector<std::pair<co::Status, co::BucketedMetricsStats>> stats(
        gts_pds.size());
    {
      tensorflow::thread::ThreadPool pool(
          ctx->env(), tensorflow::ThreadOptions(),
          "ComputeDetectionMetricsPool", port::MaxParallelism(),
          /*low_latency_hint=*/false);
      for (int i = 0, sz = gts_pds.size(); i < sz; ++i) {
        pool.Schedule([&metrics_config, &gts_pds, &stats, i]() {
          co::BucketedMetricsStats temp_stats;
          co::Status status = co::ComputeMetricsStats(
              metrics_config, gts_pds[i].second, gts_pds[i].first, &temp_stats);
          stats[i] = {status, temp_stats};
        });
      }
    }

    co::BucketedMetricsStats total_stats;
    for (const auto& s : stats) {
      CHECK(s.first.ok()) << s.first.message();
      total_stats.Accumulate(s.second);
    }

    co::MotionMetrics metrics = co::ComputeMotionMetrics(&total_stats);
    return CreateOutput(metrics);
  }

  co::MotionMetricsConfig config_;
};

REGISTER_KERNEL_BUILDER(Name("MotionMetrics").Device(DEVICE_CPU),
                        MotionMetricsOp);

}  // namespace
}  // namespace tensorflow
