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

#ifndef WAYMO_OPEN_DATASET_MATH_AABOX2D_H_
#define WAYMO_OPEN_DATASET_MATH_AABOX2D_H_

#include <algorithm>
#include <utility>
#include <cmath>
#include <cstdlib>
#include <vector>

namespace waymo {
namespace open_dataset {

// An axis aligned 2d box (rectangle).
// Parameterized internally as center and half offsets.
template <class T>
class AABox2d {
 public:
  AABox2d() : cx_(0),
              cy_(0),
              hx_(0),
              hy_(0) {
  }

  AABox2d(T cx, T cy, T hx, T hy)
      : cx_(cx), cy_(cy), hx_(hx), hy_(hy) {
  }

  // Accessors for the center and half offsets.
  T cx() const { return cx_; }
  T cy() const { return cy_; }
  T hx() const { return hx_; }
  T hy() const { return hy_; }

  void set_cx(T cx) { cx_ = cx; }
  void set_cy(T cy) { cy_ = cy; }
  void set_hx(T hx) { hx_ = hx; }
  void set_hy(T hy) { hy_ = hy; }

  void Init(T minx, T miny, T maxx, T maxy) {
    cx_ = (minx + maxx) / 2;
    hx_ = (maxx - minx) / 2;
    cy_ = (miny + maxy) / 2;
    hy_ = (maxy - miny) / 2;
  }

  void GetMin(T* minx, T* miny) const {
    *minx = cx_ - hx_;
    *miny = cy_ - hy_;
  }

  void GetMax(T* maxx, T* maxy) const {
    *maxx = cx_ + hx_;
    *maxy = cy_ + hy_;
  }

  // Returns the top left corner and width and height.
  void GetXYWH(T* x, T* y, T* w, T* h) const {
    *x = cx_ - hx_;
    *y = cy_ - hy_;
    *w = hx_ + hx_;
    *h = hy_ + hy_;
  }

  bool Overlap(const AABox2d& other) const {
    return OverlapAxis(cx_, hx_, other.cx_, other.hx_) &&
        OverlapAxis(cy_, hy_, other.cy_, other.hy_);
  }

  T ComputeIntersectionArea(const AABox2d& other) const {
    if (!Overlap(other)) return T{0};
    const T left = std::max(cx_ - hx_, other.cx_ - other.hx_);
    const T right = std::min(cx_ + hx_, other.cx_ + other.hx_);
    const T top = std::max(cy_ - hy_, other.cy_ - other.hy_);
    const T bottom = std::min(cy_ + hy_, other.cy_ + other.hy_);
    return (left - right) * (top - bottom);
  }

  void Union(const AABox2d& other) {
    T minx, miny, maxx, maxy;
    T ominx, ominy, omaxx, omaxy;
    GetMin(&minx, &miny);
    other.GetMin(&ominx, &ominy);
    GetMax(&maxx, &maxy);
    other.GetMax(&omaxx, &omaxy);
    if (ominx < minx) {
      minx = ominx;
    }
    if (omaxx > maxx) {
      maxx = omaxx;
    }
    if (ominy < miny) {
      miny = ominy;
    }
    if (omaxy > maxy) {
      maxy = omaxy;
    }
    Init(minx, miny, maxx, maxy);
  }

  // Translates the box by dx, dy.
  void Translate(T dx, T dy) {
    cx_ += dx;
    cy_ += dy;
  }

  // Returns true if x, y is inside the box.
  bool Inside(T x, T y) const {
    return std::abs(x - cx_) <= hx_ && std::abs(y - cy_) <= hy_;
  }

  // Return the square of the min distance to the given point from any points on
  // the perimeter of the rectangle. If the given point is inside the rectangle,
  // the returned distance is zero.
  double DistanceSqrToPoint(T x, T y) const {
    T minx, miny, maxx, maxy;
    GetMin(&minx, &miny);
    GetMax(&maxx, &maxy);
    T dx = std::max<T>(0, std::max<T>(x - maxx, minx - x));
    T dy = std::max<T>(0, std::max<T>(y - maxy, miny - y));
    return dx * dx + dy * dy;
  }

  // Shallow copying of this class is OK.
 protected:
  // Returns true if there is an axis overlap.
  bool OverlapAxis(T c1, T h1, T c2, T h2) const {
    return std::abs(c1 - c2) <= h1 + h2;
  }
  T cx_;
  T cy_;  // Center.
  T hx_;
  T hy_;  // Half offsets.
};

// Axis aligned box collection based on axis aligned boxes.
template <class T>
class AABox2dTree {
 public:
  AABox2dTree() { }
  ~AABox2dTree() { }
  // Pass in a vector of bounding boxes and the tree will
  // clone the boxes and build an acceleration structure around it.
  // We guarantee that the index order of the boxes do not change.
  // i.e. boxes[k] has an index k.
  void BuildTree(const std::vector<AABox2d<T> >& boxes) {
    boxes_.resize(boxes.size());
    for (int i = 0; i < boxes.size(); i++) {
      boxes_[i] = boxes[i];
    }
    nodes_.clear();
    AABoxNode node;
    nodes_.push_back(node);
    std::vector<int> idx(boxes.size());
    for (int i = 0; i < boxes.size(); i++) {
      idx[i] = i;
    }
    BuildTreeRecursive(0, 0, idx);
  }
  // When you pass in the index of a box inside the tree, it will
  // return the indices of all other boxes that overlap it.
  void CollectOverlaps(int index, std::vector<int>* hits) const {
    hits->clear();
    CollectOverlapsRecursive(0, index, hits);
  }

 protected:
  struct AABoxNode {
    AABoxNode() : is_leaf(true) {}
    ~AABoxNode() { }
    // If is leaf, the children point to other aabox nodes
    // otherwise it points to the boxes.
    std::vector<int> children;
    AABox2d<T> bounds;
    bool is_leaf;
  };
  void BuildTreeRecursive(int level, int node, const std::vector<int>& idx) {
    const int kMaxBoxesPerLeaf = 5;

    // The bounds of this node is the union of all the boxes.
    nodes_[node].bounds = boxes_[idx[0]];
    for (int i = 1; i < idx.size(); i++) {
      nodes_[node].bounds.Union(boxes_[idx[i]]);
    }

    // Terminating condition, just put everything in this node.
    if (idx.size() < kMaxBoxesPerLeaf) {
      nodes_[node].is_leaf = true;
      nodes_[node].children.resize(idx.size());
      for (int i = 0; i < idx.size(); i++) {
        nodes_[node].children[i] = idx[i];
      }
      return;
    }

    // Split by median X or Y depending on level.
    std::vector<std::pair<T, int> > sorted(idx.size());
    for (int i = 0; i < idx.size(); i++) {
      sorted[i].second = idx[i];
      if (level % 2 == 0) {
        sorted[i].first = boxes_[idx[i]].cx();
      } else {
        sorted[i].first = boxes_[idx[i]].cy();
      }
    }
    std::sort(sorted.begin(), sorted.end());
    std::vector<int> left, right;
    int mid = idx.size() / 2;
    for (int i = 0; i < idx.size(); i++) {
      if (i < mid) {
        left.push_back(sorted[i].second);
      } else {
        right.push_back(sorted[i].second);
      }
    }

    // Create two child nodes and recursively partition.
    nodes_[node].is_leaf = false;
    nodes_[node].children.clear();
    AABoxNode new_node;
    nodes_.push_back(new_node);
    nodes_[node].children.push_back(nodes_.size() - 1);
    BuildTreeRecursive(level + 1, nodes_.size() - 1, left);
    nodes_.push_back(new_node);
    nodes_[node].children.push_back(nodes_.size() - 1);
    BuildTreeRecursive(level + 1, nodes_.size() - 1, right);
  }
  void CollectOverlapsRecursive(int node, int index,
                                std::vector<int>* hits) const {
    const AABoxNode& n = nodes_[node];
    const AABox2d<T>& box = boxes_[index];
    if (!box.Overlap(n.bounds)) return;

    if (n.is_leaf) {
      for (int i = 0; i < n.children.size(); i++) {
        if (n.children[i] != index &&
            box.Overlap(boxes_[n.children[i]])) {
          hits->push_back(n.children[i]);
        }
      }
    } else {
      CollectOverlapsRecursive(n.children[0], index, hits);
      CollectOverlapsRecursive(n.children[1], index, hits);
    }
  }

  std::vector<AABox2d<T> > boxes_;
  std::vector<AABoxNode> nodes_;
};

}  // namespace open_dataset
}  // namespace waymo

#endif  // WAYMO_OPEN_DATASET_MATH_AABOX2D_H_
