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

#include "waymo_open_dataset/metrics/test_utils.h"

#include "waymo_open_dataset/label.pb.h"
#include "waymo_open_dataset/protos/breakdown.pb.h"
#include "waymo_open_dataset/protos/metrics.pb.h"

namespace waymo {
namespace open_dataset {

Label::Box BuildAA2dBox(double center_x, double center_y, double length,
                        double width) {
  Label::Box box;
  box.set_center_x(center_x);
  box.set_center_y(center_y);
  box.set_length(length);
  box.set_width(width);
  return box;
}

Label::Box BuildBox2d(double center_x, double center_y, double length,
                      double width, double heading) {
  Label::Box box = BuildAA2dBox(center_x, center_y, length, width);
  box.set_heading(heading);
  return box;
}

Label::Box BuildBox3d(double center_x, double center_y, double center_z,
                      double length, double width, double height,
                      double heading) {
  Label::Box box = BuildBox2d(center_x, center_y, length, width, heading);
  box.set_center_z(center_z);
  box.set_height(height);
  return box;
}

Polygon2dProto BuildNLZ(const std::vector<double>& x,
                        const std::vector<double>& y) {
  Polygon2dProto nlz;
  for (int i = 0, sz = x.size(); i < sz; ++i) {
    nlz.add_x(x[i]);
    nlz.add_y(y[i]);
  }
  return nlz;
}

Config BuildDefaultConfig() {
  Config config;
  config.mutable_score_cutoffs()->Reserve(10);
  for (int i = 0; i < 10; ++i) {
    config.add_score_cutoffs(0.1 * i);
  }
  config.add_breakdown_generator_ids(Breakdown::ONE_SHARD);
  config.add_difficulties();

  config.set_matcher_type(MatcherProto::TYPE_HUNGARIAN_TEST_ONLY);
  for (int i = 0; i <= Label::Type_MAX; ++i) {
    config.add_iou_thresholds(0.5);
  }
  config.set_box_type(Label::Box::TYPE_3D);
  return config;
}

}  // namespace open_dataset
}  // namespace waymo
