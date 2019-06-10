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

#ifndef WAYMO_OPEN_DATASET_METRICS_TEST_UTILS_H_
#define WAYMO_OPEN_DATASET_METRICS_TEST_UTILS_H_

#include <vector>

#include "waymo_open_dataset/label.pb.h"
#include "waymo_open_dataset/protos/metrics.pb.h"

// Some helper functions for testing.

namespace waymo {
namespace open_dataset {

// Builds an AA 2d box.
Label::Box BuildAA2dBox(double center_x, double center_y, double length,
                        double width);

// Builds a 2d box.
Label::Box BuildBox2d(double center_x, double center_y, double length,
                      double width, double heading);

// Builds a 3d box.
Label::Box BuildBox3d(double center_x, double center_y, double center_z,
                      double length, double width, double height,
                      double heading);

// Builds an NLZ.
Polygon2dProto BuildNLZ(const std::vector<double>& x,
                        const std::vector<double>& y);

// Builds a default metrics config used in unit tests.
Config BuildDefaultConfig();

}  // namespace open_dataset
}  // namespace waymo

#endif  // WAYMO_OPEN_DATASET_METRICS_TEST_UTILS_H_
