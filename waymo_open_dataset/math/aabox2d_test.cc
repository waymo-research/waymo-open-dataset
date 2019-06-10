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

#include "waymo_open_dataset/math/aabox2d.h"

#include <gtest/gtest.h>
#include "absl/strings/str_format.h"

namespace waymo {
namespace open_dataset {

namespace {

std::string ToString(const AABox2d<double>& box) {
  double minx, miny;
  box.GetMin(&minx, &miny);
  double maxx, maxy;
  box.GetMax(&maxx, &maxy);
  return ::absl::StrFormat("%.2f %.2f - %.2f %.2f", minx, miny, maxx, maxy);
}

void ExpectOverlap(bool expected, const AABox2d<double>& boxa,
                   const AABox2d<double>& boxb) {
  EXPECT_EQ(expected, boxa.Overlap(boxb)) << ToString(boxa) << " vs. "
                                          << ToString(boxb);
}

}  // namespace

TEST(AABox2dTest, SimpleOverlap) {
  const int n = 1;
  for (int ii = -n; ii <= n; ++ii) {
    for (int jj = -n; jj <= n; ++jj) {
      const AABox2d<double> a(ii, jj, 0.4, 0.4);
      const AABox2d<double> b(ii, jj, 0.0, 0.0);

      ExpectOverlap(true, a, b);
      ExpectOverlap(true, b, a);

      const AABox2d<double> c(ii - 1, jj, 0.4, 0.4);
      ExpectOverlap(false, a, c);
      ExpectOverlap(false, c, a);
      ExpectOverlap(false, b, c);
      ExpectOverlap(false, c, b);

      const AABox2d<double> d(ii - 1, jj, 0.0, 0.0);
      ExpectOverlap(false, a, d);
      ExpectOverlap(false, d, a);
    }
  }
}

TEST(AABox2dTest, ComputeIntersectionArea) {
  const int n = 1;
  for (int ii = -n; ii <= n; ++ii) {
    for (int jj = -n; jj <= n; ++jj) {
      const AABox2d<double> a(ii, jj, 0.4, 0.4);
      const AABox2d<double> b(ii, jj, 0.0, 0.0);

      EXPECT_EQ(a.ComputeIntersectionArea(b), 0.0);
      EXPECT_EQ(b.ComputeIntersectionArea(a), 0.0);

      const AABox2d<double> c(ii - 1, jj, 0.4, 0.4);
      EXPECT_EQ(a.ComputeIntersectionArea(c), 0.0);
      EXPECT_EQ(c.ComputeIntersectionArea(a), 0.0);
      EXPECT_EQ(b.ComputeIntersectionArea(c), 0.0);
      EXPECT_EQ(c.ComputeIntersectionArea(b), 0.0);

      const AABox2d<double> d(ii - 1, jj, 0.6, 0.6);
      EXPECT_EQ(a.ComputeIntersectionArea(d), 0.0);
      EXPECT_EQ(d.ComputeIntersectionArea(a), 0.0);

      const AABox2d<double> e(ii - 1, jj, 0.8, 0.8);
      EXPECT_DOUBLE_EQ(a.ComputeIntersectionArea(e), 0.16);
      EXPECT_DOUBLE_EQ(e.ComputeIntersectionArea(a), 0.16);

      const AABox2d<double> f(ii, jj - 1, 0.8, 0.8);
      EXPECT_DOUBLE_EQ(a.ComputeIntersectionArea(f), 0.16);
      EXPECT_DOUBLE_EQ(f.ComputeIntersectionArea(a), 0.16);
    }
  }
}

}  // namespace open_dataset
}  // namespace waymo
