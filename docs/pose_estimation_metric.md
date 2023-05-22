# Metrics for the Pose Estimation Challenge

## Supported metrics

We provide a python library to compute and report on the result page a number of
different metrics. The challenge participants may want to inspect different
subsets of them by selecting a group of keypoints and thresholds.

- [Pose Estimation Metric (**PEM**)](#PEM), a new metric created specifically
  for the Pose Estimation challenge, in meters (lower is better). It is used to
  rank the leaderboard.
- <a name="mpjpe">Mean Per Joint Position Error (**MPJPE**)</a>, in meters
  (lower is better). Useful to measure the quality of matched keypoints in easy to
  interpret units.
- <a name="pck">Percentage Of Correct Keypoints (**PCK**)</a>. The ratio of the
  number of keypoints close to ground truth to the total number of keypoints. We
  use thresholds `(0.05, 0.1, 0.2, 0.3, 0.4, 0.5)` relative to bounding
  box scales, for example for a human-like 1x1x2m box, the box's scale will be
  $`(1 \cdot 1 \cdot 2)^\frac{1}{3} = 1.26`$, so the $`0.20`$ threshold of this
  scale will be 25cm and keypoints with errors less than 25cm will be considered
  correct. The metric takes values in the `[0, 1]` range (higher is better).
  Useful to understand the distribution of errors.
- Precision of keypoints visibility (**Precision**). Values are
  in the `[0, 1]` range (higher is better). Useful to gauge the precision of the
  keypoint visibility classification and number of false positive detections.
- Recall of keypoints visibility (**Recall**). Values are in the `[0, 1]` range
  (higher is better). Useful to gauge recall of the keypoint visibility
  classification and number of false negative detections.
- <a name="oks">Precision at Object Keypoint Similarity (**OKS**)</a>
  for different thresholds. Values are in the `[0, 1]` range (higher is better).
  The OKS measures the distance between predicted and ground-truth keypoints
  relative to a scale, specific for each keypoint type. For example,
  the scale for hips is larger than the scale of wrists. Thus a 5mm error for
  hips will result in larger Precision OKS values for hips compared with the
  same 5mm error for wrists. The OKS metric can be used to evaluate the accuracy
  of 2D keypoint detectors in a consistent and standardized way. By using OKS as
  a point of comparison, participants can gain insights into the quality of
  their 3D keypoint detectors relative to state-of-the-art 2D keypoint
  detectors.
- Average Precision at OKS (**OKS_AP**), averaged over
  `[.5, .55, .60, .65, .70, .75, .80, 0.95]` thresholds. Values are in the
   `[0, 1]` range (higher is better).


NOTE: All auxiliary metrics for locations of keypoints (MPJPE, PCK, OKS)
take into account only matched keypoints and provided only for information
purposes.


## PEM

The set of well established metrics such as [MPJPE](#mpjpe), [PCK](#pck) or
[OKS](#oks) provide valuable insights on quality of a keypoint localization
method, but they do not take into account specifics of the partially labeled
data and ignore quality of the object detection. In order to rank submissions
for the challenge we introducing a new single metric called Pose Estimation
Metric (**PEM**), which is

- easily interpretable
- sensitive to
  - keypoint localization error and visibility classification accuracy
  - number of false positive and false negative object detections
- and not sensitive to
  - Intersection over Union (**IoU**) of object detection to avoid a strong
  dependency on 3D box accuracy.

The PEM is a weighted sum of the [MPJPE](#mpjpe) over visible
[matched](#object-matching-algorithm) keypoints and a penalty for
unmatched keypoints (aka `mismatch_penalty`), expressed in meters.

We compute the PEM on a set of candidate pairs of predicted and ground truth
objects, for which at least one predicted keypoint is within a distance
threshold constant $`C`$ from the ground truth box. The final object assignment
is selected using the Hungarian method to minimize:

$$\textbf{PEM}(Y,\hat{Y}) = \frac{\sum_{i\in M}\left\|y_{i} -
\hat{y}_{i}\right\|_2 + C|U|}{|M| + |U|}$$

where $`M`$ - a set of indices of matched keypoints, $`U`$ - a set of indices of
unmatched keypoints (ground truth keypoints without matching predicted keypoints
or predicted keypoints for unmatched objects); Sets
$`Y= \left\{y_i\right\}_{i \in M}`$ and
$`\hat{Y} = \left\{\hat{y}_i\right\}_{i \in M}`$ are ground truth
and predicted 3D coordinates of keypoints; $`C=0.25`$ - a constant penalty for
an unmatched keypoint.


## Object Matching Algorithm


The [Pose Estimation challenge](https://waymo.com/open/challenges/2023/pose-estimation/)
requires participants to provide keypoints for all human objects in a scene. To
evaluate the performance of the predictions, the evaluation service uses one of
the provided matching algorithms to automatically find correspondence between
predicted (**PR**) and ground truth (**GT**) objects. The matching algorithm
outputs three sets of objects:


 - true positives (**TP**), which are pairs of a GT object and its corresponding PR object
 - false positives (**FP**), which are PR objects without a corresponding GT object
 - false negatives (**FN**), which are GT objects without a corresponding PR object

However, matching is complicated by the fact that not all GT objects in WOD have
visible keypoints. To address this, two kinds of GT objects are distinguished:

  - $`GT_i`$ - GT objects without any visible keypoints, which includes unlabeled
  or heavily occluded human objects.
  - $`GT_v`$ - GT boxes with at least one
  visible keypoint.

| ![a toy example to illustrate $`GT_v`$ and $`GT_i`$](images/pem_matching_fig.png) |
| :-: |
| Fig 1. A toy scene |

On the Fig. 1 you can see:

- Ground truth objects:
  - $`GT_i`$: $`G_0`$, $`G_1`$, $`G_3`$, $`G_5`$, $`G_7`$
  - $`GT_v`$: $`G_2`$, $`G_4`$, $`G_6`$, $`G_8`$, $`G_9`$
- Predicted objects:
  $`P_0`$, $`P_1`$, $`P_2`$, $`P_3`$, $`P_4`$, $`P_5`$, $`P_6`$, $`P_7`$

If a PR object corresponds to a $`GT_i`$ object, no penalty is assigned since the
MPJPE cannot be computed for such matches. Only matches between $`GT_v`$ objects and
PR objects are considered for the computation of the PEM metric.

Since computing the PEM metric for all possible matches between GT and PR is not
feasible for scenes with many objects, several heuristics are used to narrow
down the set of candidate matches. The official matching algorithm for the
challenge is the
[`MeanErrorMatcher`](src/waymo_open_dataset/metrics/python/keypoint_metrics.py),
which computes keypoint errors for each pair of candidate matches. It has two stages:

  1. When keypoints clearly fall in $`GT_i`$ objects (see criterion in
    [keypoint_metrics.py](src/waymo_open_dataset/metrics/python/keypoint_metrics.py)),
    remove them from considerations, without any penalties.
  2. For all remaining candidate GTv ground truth boxes and detections pairs,
     perform Hungarian matching that minimizes the PEM metric.
For the example on the Fig 1. stages of the matching algorithm should work like
this:

- stage #1:
    - Select pairs of GT and PR objects for which at least one PR keypoint is
      inside GT box enlarged by 25cm.
    - assume $`PEM(G_4, P_5) > C`$ and $`PEM(G_6, P_6) < C`$
    - should exclude: $`(G_0, P_0)`$, $`(G_1, P_1)`$, $`(G_3, P_3)`$,
      $`(G_5, P_5)`$ pairs.
- stage #2:
    - consider only GTv objects
    - compute errors for candidate pairs and populate the assignment error $`A`$
     (aka cost matrix): $`A_{k,j}=PEM(G_k, P_j)`$ for
     $`(G_2, P_2)`$, $`(G_4, P_5)`$, $`(G_6, P_6)`$, $`(G_8, P_7)`$,
     $`(G_9, P_7)`$ and set the rest of the 8x7 matrix $`A=\infty`$.
    - assuming $`PEM(G_9, P_7) < PEM(G_8, P_7)`$, the matching assignment should
     output the following pairs:
      $`(G_1, P_1)`$, $`(G_2, P_2)`$, $`(G_6, P_6)`$, $`(G_9, P_7)`$
- the final output of the matcher should be:
      $`(G_2, P_2)`$, $`(G_6, P_6)`$, $`(G_9, P_7)`$,
      $`(G_4, \emptyset)`$, $`(G_8, \emptyset)`$,
      $`(\emptyset, P_4)`$

For the PEM metric, each ground-truth box – GTV and GTi – can only be
associated with a maximum of 1 detection. To maximize your PEM scores, you are
responsible for removing duplicate detections.

NOTE: The WOD library also implements the [`CppMatcher`](src/waymo_open_dataset/metrics/python/keypoint_metrics.py)
which maximizes total Intersection over Union (IoU) between predicted and ground
truth boxes. However, this matcher requires all predictions to have bounding
boxes and is provided only as a reference.