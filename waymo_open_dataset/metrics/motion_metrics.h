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

#ifndef WAYMO_OPEN_DATASET_METRICS_MOTION_METRICS_H_
#define WAYMO_OPEN_DATASET_METRICS_MOTION_METRICS_H_

#include "absl/container/flat_hash_map.h"
#include "waymo_open_dataset/common/status.h"
#include "waymo_open_dataset/protos/metrics.pb.h"
#include "waymo_open_dataset/protos/motion_metrics.pb.h"
#include "waymo_open_dataset/protos/motion_submission.pb.h"
#include "waymo_open_dataset/protos/scenario.pb.h"

namespace waymo {
namespace open_dataset {

// A single precision/recall value.
struct PrSample {
  float recall;
  float precision;
};

// A container for accumulated measurements.
struct Accumulator {
  double total = 0.0;
  int num_measurements = 0;

  // Add a new measurement.
  void Add(double value) {
    total += value;
    num_measurements += 1;
  }

  // Accumulate another set of measurements into this one.
  void Accumulate(const Accumulator& other) {
    total += other.total;
    num_measurements += other.num_measurements;
  }

  // Returns the mean of all measurements. Returns zero if there are no
  // measurements.
  double Mean() const {
    return num_measurements == 0 ? 0.0 : (total / num_measurements);
  }
};

// A single sample for computing the mean average precision.
struct PredictionSample {
  PredictionSample(float confidence_level, bool is_true_positive)
      : confidence(confidence_level), true_positive(is_true_positive) {}
  float confidence;
  bool true_positive;
};

// A set of stats for trajectory predictions.
struct PredictionStats {
  // The set of prediction results.
  std::vector<PredictionSample> samples;

  // The total number of ground truth trajectories.
  int num_trajectories = 0;

  // Accumulate the stats from another PredictionStats.
  void Accumulate(const PredictionStats& prediction_stats);
};

// Stats to compute mean average precision metrics.
struct MeanAveragePrecisionStats {
  // The set of prediction stats for each trajectory bucket.
  std::vector<PredictionStats> pr_buckets;

  // Accumulate the stats from another MeanAveragePrecision object.
  void Accumulate(const MeanAveragePrecisionStats& mean_ap_stats);
};

// Stats for metrics computations. Each MetricStats object contains stats
// measured up to a given prediction time step as if that were the last step in
// the trajectory. Measuring at different times provides an understanding of how
// metrics vary according to trajectory length.
struct MetricsStats {
  // The accumulated values for the minimum average displacement (minADE). For
  // each object, the average displacement error (ADE) in meters up to the
  // measurement time step is computed for all trajectory predictions for that
  // object. The value with the minimum error is kept (minADE). The resulting
  // values are accumulated for all predicted objects in all scenarios. The
  // mean of all measurements in the accumulator is the average minADE.
  Accumulator min_average_displacement;

  // The accumulated values for the minimum displacement at that time step
  // (minFDE) in meters. For each object the error for a given trajectory at the
  // measurement time step is computed for all trajectory predictions for that
  // objects. The value with the minimum error is kept. The resulting values are
  // accumulated for all predicted objects over all scenarios. The mean of all
  // measurements in the accumulator is the average minFDE.
  Accumulator min_final_displacement;

  // The miss rate is calculated by computing the displacement from ground truth
  // at the measurement time step. If the displacement is greater than the miss
  // rate threshold it is considered a miss. Measurements are accumulated for
  // all predicted objects across all scenarios. The mean of all measurements in
  // the accumulator is equal to the overall miss rate.
  Accumulator miss_rate;

  // The accumulated overlap rate. Overlaps are detected as any intersection
  // of the bounding boxes of the highest confidence predicted object trajectory
  // with any other valid object at the same time step for time steps up to the
  // measurement time step. Only objects that were valid at the prediction time
  // step are considered. If one or more overlaps occur up to the evaluation
  // times it is considered a single overlap measurement. Measurements are
  // accumulated for all predicted objects for all scenarios. The mean of all
  // measurements is equal to the overall overlap rate.
  Accumulator overlap_rate;

  // The mean average precision stats. The mAP metric is computed by
  // accumulating true and false positive measurements based on thresholding the
  // FDE at the measurement time step. The measurements are separated into
  // buckets based on the trajectory shape. The mean average precision of each
  // bucket is computed as described in "The PASCAL Visual Object Classes (VOC)
  // Challenge" (Everingham, 2009, p. 11). using the newer method that includes
  // all samples in the computation consistent with the current PASCAL challenge
  // metrics. The mean of the mAP value across all trajectory shape buckets is
  // equal to the overall mAP value.
  MeanAveragePrecisionStats mean_average_precision;

  // Accumulate stats from another MetricsStats object.
  void Accumulate(const MetricsStats& metrics_stats);
};

// A set of accumulated metrics measurements separated by object type and
// measurement time step. Metric stats are measured up to a given
// prediction time step as if that were the last step in the trajectory.
// Measuring at different times provides an data about of how metrics vary
// according to trajectory length.
struct BucketedMetricsStats {
  // The metrics stats indexed by object type and by time step.
  // The outer map is keyed by object type and the inner map is keyed by the
  // predicted measurement time step.
  std::map<Track::ObjectType, std::map<int, MetricsStats>> stats;

  // Accumulate stats from another BucketedMetricsStats object.
  void Accumulate(const BucketedMetricsStats& metrics_stats);

  // Accumulate stats across all object types stored in this object. Returns
  // a map of MetricsStats keyed by the measurement step.
  std::map<int, MetricsStats> AccumulateAcrossTypes();
};

// -------- Functions.

// Computes the P/R curve for the given set of samples. After this call, the
// samples will be sorted by confidence. The total_gt_count parameter is the
// total possible number of true positives.
std::vector<PrSample> ComputePrCurve(
    std::vector<PredictionSample>* prediction_samples, int total_gt_count);

// Computes the mean average precision metric for a set of samples.
// All samples in prediction_samples will be sorted by confidence.
double ComputeMeanAveragePrecision(PredictionStats* prediction_samples);

// Computes the mean average precision averaged across all buckets.
// The samples vectors within the stats parameter will be sorted by confidence.
double ComputeMapMetric(MeanAveragePrecisionStats* stats);

// --------- Main API.

// Get the default config for the metrics server.
MotionMetricsConfig GetChallengeConfig();

// Computes metrics stats broken down by object type and measurement time step
// for a single ScenarioPredictions proto.
// Metric stats are measured up to a given prediction time step as if that were
// the last step in the trajectory. Measuring at different times provides an
// understanding of how metrics vary according to trajectory length.
// Args:
//   config - The metrics computation configuration.
//   predictions - The input submission proto containing model predictions for a
//      single scenario.
//   scenario - The scenario proto containing the data used to generate the
//      given submission proto.
//   stats - Return value container.
Status ComputeMetricsStats(const MotionMetricsConfig& config,
                           const ScenarioPredictions& predictions,
                           const Scenario& scenario,
                           BucketedMetricsStats* result);

// Computes the metrics values for each combination of object type and
// measurement time step. Returns a MotionMetrics proto. All precision samples
// vectors in total_stats will be sorted by confidence.
MotionMetrics ComputeMotionMetrics(BucketedMetricsStats* total_stats);

// Computes output metrics from a set of predictions and corresponding
// scenarios.
// Args:
//   config - The metrics computation configuration.
//   predictions - The predictions keyed by scenario ID.
//   scenarios - The scenarios keyed by scenario ID.
//   metrics - Computed metric values.
Status ComputeMotionMetrics(
    const MotionMetricsConfig& config,
    const absl::flat_hash_map<std::string, ScenarioPredictions>& predictions,
    const absl::flat_hash_map<std::string, Scenario>& scenarios,
    MotionMetrics* metrics);

}  // namespace open_dataset
}  // namespace waymo

#endif  // WAYMO_OPEN_DATASET_METRICS_MOTION_METRICS_H_
