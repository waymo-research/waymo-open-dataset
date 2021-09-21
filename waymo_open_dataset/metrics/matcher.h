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

#ifndef WAYMO_OPEN_DATASET_METRICS_MATCHER_H_
#define WAYMO_OPEN_DATASET_METRICS_MATCHER_H_

#include <math.h>

#include <memory>
#include <vector>

#include <glog/logging.h>
#include "waymo_open_dataset/label.pb.h"
#include "waymo_open_dataset/metrics/iou.h"
#include "waymo_open_dataset/protos/metrics.pb.h"

namespace waymo {
namespace open_dataset {

// Matcher interface.
// The interface is designed in a way such that we can perform multiple match
// calls on different subsets of ground truths and predictions multiple times.
// This can be useful to compute various metric breakdowns while caching results
// of some expensive operations such as IoU computations.
//
// It is recommended to create one Matcher instance per frame and reuse that
// Matcher instance for all breakdowns such that expensive operations (e.g IoU)
// for the same object can be cached.
//
// Coding convention: We put prediction related logic first whenever we need to
// deal with both prediction and ground_truth.
class Matcher {
 public:
  explicit Matcher(const std::vector<float>& iou_thresholds,
                   Label::Box::Type box_type)
      : iou_thresholds_(iou_thresholds), box_type_(box_type) {}

  virtual ~Matcher() = default;

  // Creates a matcher instance based on the matcher type.
  static std::unique_ptr<Matcher> Create(
      MatcherProto_Type matcher_type, const std::vector<float>& iou_thresholds,
      Label::Box::Type box_type);

  static std::unique_ptr<Matcher> Create(const Config& config) {
    const std::vector<float> iou_thresholds(config.iou_thresholds().begin(),
                                            config.iou_thresholds().end());
    return Create(config.matcher_type(), iou_thresholds, config.box_type());
  }

  // Sets all the predictions. The index of each element in the provided
  // predictions is used to refer to the prediction when calling Match below.
  // Note: the `predictions` provided here must be valid for all the Match calls
  // that operates on it.
  void SetPredictions(const std::vector<Object>& predictions) {
    predictions_ = &predictions;
    iou_caches_.clear();
  }

  // Sets all the ground truths. The index of each element in the provided
  // ground truths is used to refer to the ground truth when calling Match
  // below.
  // Note: the `ground_truths` provided here must be valid for all the Match
  // calls that operates on it.
  void SetGroundTruths(const std::vector<Object>& ground_truths) {
    ground_truths_ = &ground_truths;
    iou_caches_.clear();
  }

  // Sets the subset of predictions to be considered for the future Match
  // calls.
  // The subset vector does not need to be valid after this call.
  void SetPredictionSubset(const std::vector<int>& subset) {
    prediction_subset_ = subset;
  }

  // Sets the subset of ground truths to be considered for the future Match
  // calls.
  // The subset vector does not need to be valid after this call.
  void SetGroundTruthSubset(const std::vector<int>& subset) {
    ground_truth_subset_ = subset;
  }

  // Sets a custom IOU calculation function to replace the default function.
  void SetCustomIoUComputeFunc(ComputeIoUFunc custom_iou_func) {
    custom_iou_func_ = custom_iou_func;
  }

  // Accessors.
  const std::vector<Object>& predictions() const {
    CHECK(predictions_ != nullptr);
    return *predictions_;
  }
  const std::vector<Object>& ground_truths() const {
    CHECK(ground_truths_ != nullptr);
    return *ground_truths_;
  }
  const std::vector<int>& prediction_subset() const {
    return prediction_subset_;
  }
  const std::vector<int>& ground_truth_subset() const {
    return ground_truth_subset_;
  }

  // Validates the prediction_index.
  void ValidPredictionIndex(int prediction_index) const {
    CHECK_GE(prediction_index, 0);
    CHECK_LT(prediction_index, predictions().size());
  }

  // Validates the ground_truth_index.
  void ValidGroundTruthIndex(int ground_truth_index) const {
    CHECK_GE(ground_truth_index, 0);
    CHECK_LT(ground_truth_index, ground_truths().size());
  }

  // Performs the match operation.
  //
  // Sets the matching results in prediction_matches (if not null) and
  // ground_truth_matches (if not null).
  //
  // If not null, prediction_matches's size will be the same as the size of
  // prediction_subset. The (i, *predictions[i]) matches prediction_subset()[i]
  // and ground_truth_subset()[*predictions[i]].
  //
  // If not null, ground_truth_matches's size will be the same as the size of
  // ground_truth_subset. The (i, *ground_truth_matches[i]) matches
  // ground_truth_subset()[i] and prediction_subset()[*ground_truth_matches[i]].
  //
  // If a prediction or ground truth is not matched, its corresponding
  // matching entry is set to -1.
  //
  // Requires: Set{Predictions,GroundTruths} and
  // Set{Prediction,GroundTruth}Subset must be called at least once.
  virtual void Match(std::vector<int>* prediction_matches,
                     std::vector<int>* ground_truth_matches) = 0;

  // Returns true if the prediction can match the ground truth.
  bool CanMatch(int prediction_index, int ground_truth_index) const;

  // Computes IoU of a pair of prediction and ground truth.
  // The result is cached.
  // Return value is within [0.0, 1.0].
  virtual float IoU(int prediction_index, int ground_truth_index) const;

  // Returns a quantized value of the IoU.
  // Returned value is within [0, kMaxIoU].
  int QuantizedIoU(int prediction_index, int ground_truth_index) const {
    return std::round(IoU(prediction_index, ground_truth_index) * kMaxIoU);
  }

 private:
  const std::vector<float> iou_thresholds_;
  const Label::Box::Type box_type_;

  const std::vector<Object>* predictions_ = nullptr;    // Not owned.
  const std::vector<Object>* ground_truths_ = nullptr;  // Not owned.
  // A subset of predictions_ above. Eache element is an index to predictions_.
  std::vector<int> prediction_subset_;
  // A subset of ground_truths_ above. Eache element is an index to
  // ground_truths_.
  std::vector<int> ground_truth_subset_;

  // If set, will use to calculate the iou instead of the default one.
  ComputeIoUFunc custom_iou_func_ = nullptr;

  // The [i][j] element caches the IoU score between the i-th prediction and the
  // j-th ground truth.
  // The cache entry is not populated if its cell value is < 0.
  // Note that if the IoU score is smaller than the threshold specified in the
  // config, it is set to 0.
  mutable std::vector<std::vector<float>> iou_caches_;
};

// The Hungarian algorithm based matching.
// https://en.wikipedia.org/wiki/Hungarian_algorithm
// This class is not threadsafe.
class HungarianMatcher : public Matcher {
 public:
  explicit HungarianMatcher(const std::vector<float>& iou_thresholds,
                            Label::Box::Type box_type)
      : Matcher(iou_thresholds, box_type) {}

  ~HungarianMatcher() override = default;

  // This implements the Hungarian based matching which maximizes the sum of
  // IoUs of all matched pairs.
  void Match(std::vector<int>* prediction_matches,
             std::vector<int>* ground_truth_matches) override;
};

// A heuristic matching algorithm:
//   1. Sort predictions by scores in descending order.
//   2. For each prediction, match it with the groudtruth that:
//     - Has not yet matched with other predictions.
//     - Has IoU with the prediction greater than the threshold.
//     - Has the highest IoU with the prediction among remaining groundtruths.
//
// This is the same method employed by COCO api.
class ScoreFirstMatcher : public Matcher {
 public:
  explicit ScoreFirstMatcher(const std::vector<float>& iou_thresholds,
                             Label::Box::Type box_type)
      : Matcher(iou_thresholds, box_type) {}

  ~ScoreFirstMatcher() override = default;

  void Match(std::vector<int>* prediction_matches,
             std::vector<int>* ground_truth_matches) override;
};

// Test only matcher. It overrides IoU computation such that it is easy to set
// up any similarity matrix.
class TEST_HungarianMatcher : public HungarianMatcher {
 public:
  explicit TEST_HungarianMatcher(const std::vector<float>& iou_thresholds,
                                 Label::Box::Type box_type)
      : HungarianMatcher(iou_thresholds, box_type) {}

  ~TEST_HungarianMatcher() override = default;

  // Override IoU computation results.
  void SetIoU(const std::vector<std::vector<float>>& iou) { iou_ = iou; }

 private:
  float IoU(int prediction_index, int ground_truth_index) const override {
    return iou_[prediction_index][ground_truth_index];
  }

  std::vector<std::vector<float>> iou_;
};

}  // namespace open_dataset
}  // namespace waymo

#endif  // WAYMO_OPEN_DATASET_METRICS_MATCHER_H_
