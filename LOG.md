# History of code and dataset version updates.

## March 2025 Update
This update contains several changes to the metrics and new datasets.

End-To-End Driving Dataset (v 1.0.0):

* Released the training set for the E2E driving dataset. This includes camera
images and high-level commands for the ego-vehicle. More information on the
dataset and the E2E driving challenge can be found on the
[website](https://waymo.com/open/data/e2e/). The dataset proto format can be
found in [this repository](https://github.com/waymo-research/waymo-open-dataset/tree/master/src/waymo_open_dataset/protos/end_to_end_driving_data.proto).

Codebase:

* Released tutorial for the End-To-End driving challenge, along with the
[proto format definition for the submission](https://github.com/waymo-research/waymo-open-dataset/tree/master/src/waymo_open_dataset/protos/end_to_end_driving_submission.proto).
* Fixed an issue for Motion Prediction metrics on a few edge cases where box
heading could flip.
* Updated the Sim Agents metrics to allow for simulated dimensions and more
flexibility on scenario timing configuration. This allows the Scenario
Generation challenge to re-use the same metrics.
* Released a tutorial for the Scenario Generation challenge.
* For Sim Agents and Scenario Generation, introduced a filter for vehicles for
the time-to-collision metric, which made several assumption on the object being
a vehicle, so it was incorrectly computed for all objects.
* For Sim Agents and Scenario Generation, introduced a new traffic light
violation metric (also filtered for vehicles), triggered when a vehicle passes
a red light at an intersection.
* Temporarily disabled one of the occupancy flow metrics to allow update to TF
2.13 and forward (will re-enable soon).

## October 2024 Update
Motion dataset (v 1.3.0):

* Improved the time alignment of traffic light information.

## April 2024 Update

* The Rules have been updated to allow training (including pre-training,
co-training or fine-tuning models) using frozen, pre-trained weights from
publicly available open source models for submissions to the Challenges.
* Added a new sets of fields (which are now required, or the server
will return an error) in the submission metadatas to track how participants
generated their submissions. We updated the tutorials to reflect this change,
check out the new fields in the submission proto files for motion, sim agents
and occupancy flow.

## March 2024 Update
This update contains several changes/addition to the datasets:

Perception dataset (v1.4.3 and v2.0.1):

* We made improvements in the 3D semantic segmentation ground truth labels,
especially for the class of motorcyclist.

Motion dataset (v1.2.1):

* The 1.2.1 WOMD release now provides camera data, including front,
front-left, front-right, side-left, side-right, rear-left, rear-right,
and rear sensors. Similar to the Lidar data, the camera data of the training,
validation and testing sets cover the first 1 second of each of the 9 second
windows. Instead of releasing raw camera images, we release the image tokens and
image embedding extracted from a pre-trained VQ-GAN model.
* The initial release of the WOMD camera data contained misalignment between
LiDAR data and roadgraph inputs for some frames. The 1.2.1 release provides
new timestamps for the lidar data with an updated pose transformation matrix
per time step.

We also provide the following changes to the code supporting the challenges.

Motion prediction:

* We have improved the logic behind the behavior bucketing used for mAP.

Sim Agents:

* We have improved the quality of the kinematic metrics by using smoother
estimates of speeds and accelerations.
* We have fixed an edge case for offroad computation with over-passes.
* We have re-calibratred the metric configuration and composite metrics weights.
* We report simulated collision and offroad rates (not likelihoods).

## December 2023 Update
We released v1.6.1 version of the pip package with fixes for the WOSAC metrics:

* Fixing a bug in validity checking for collision and offroad.
* Modifying the behaviour of collision/offroad checking when invalid.

## August 2023 Update
We released a large-scale object-centric asset dataset containing over 1.2M
images and lidar observations of two major categories (vehicles and pedestrians)
from the Perception Dataset (v2.0.0).

* Extracted perception objects from multi-sensor data: all five cameras and
the top lidar.
* Lidar features include 3D point cloud sequences that support 3D object
shape reconstruction. We additionally provide refined box pose through point
cloud shape registration for all vehicle objects.
* Camera features include sequences of camera patches from the
`most_visible_camera`, projected lidar returns on the corresponding camera,
per-pixel camera rays information, and auto-labeled 2D panoptic segmentation
that supports object NeRF reconstruction.
* Added a [tutorial](tutorial/tutorial_object_asset.ipynb) and supporting code.

## March 2023 Update
This major update includes supporting code to four challenges at waymo.com/open,
and dataset updates to both the Perception and Motion Datasets.

v2.0.0 of the Perception Dataset
 * Introduced the dataset in modular format, enabling users to selectively
 download only the components they need.
 * Includes all features in v1.4.2 of the Perception Dataset except maps.
 * Added a [tutorial](tutorial/tutorial_v2.ipynb) and supporting code.

v1.4.2 of the Perception Dataset
 * For the 2D video panoptic segmentation labels, added a mask to indicate the
 number of cameras covering each pixel.
 * Added 3D map data as polylines or polygons.

v1.2.0 of the Motion Dataset

* Added Lidar data for the training set (first 1s of each 9s windows), and
the corresponding [tutorial](tutorial/tutorial_womd_lidar.ipynb) and supporting code.
 * Added driveway entrances to the map data. Adjusted some road edge boundary
 height estimates.
 * Increased the max number of map points in tf_examples to 30k and reduced
 sampling to 1.0m to increase map coverage, so the coverage equalizes that of
 the dataset in scenario proto format. Added conversion code from scenario
 proto format to tf_examples format.

Added supporting code for the four 2023 Waymo Open Dataset Challenges
 * Sim Agents Challenge, with a [tutorial](tutorial/tutorial_sim_agents.ipynb)
 * Pose Estimation Challenge, with a [tutorial](tutorial/tutorial_keypoints.ipynb)
 * 2D Video Panoptic Segmentation Challenge, with a [tutorial](tutorial/tutorial_2d_pvps.ipynb)
 * Motion Prediction Challenge, with a [tutorial](tutorial/tutorial_motion.ipynb)

## December 2022 Update
We released v1.4.1 of the Perception dataset.

* Improved the quality of the 2D video panoptic segmentation labels.

## June 2022 Update
We released v1.4.0 of the Perception dataset.

* Added 2D video panoptic segmentation labels and supporting code.

## May 2022 Update (part 2)
 * Released a [tutorial](tutorial/tutorial_camera_only.ipynb) for the 3D Camera-Only Detection Challenge.
 * Added support for computing 3D-LET-APL in Python metrics ops. See
 `Compute Metrics` in the
 [tutorial](tutorial/tutorial_camera_only.ipynb).
 * Fixed a bug in the metrics implementation for the Occupancy and Flow
 Challenge.

## May 2022 Update
We released v1.3.2 of the Perception dataset to improve the quality and accuracy
of the labels.
 * Updated 3D semantic segmentation labels, for better temporal consistency and
 to fix mislabeled points.
 * Updated 2D key point labels to fix image cropping issues.
 * Added `num_top_lidar_points_in_box` in [dataset.proto](src/waymo_open_dataset/dataset.proto) for the 3D Camera-Only Detection Challenge.

## April 2022 Update
We released v1.3.1 of the Perception dataset to support the 2022 Challenges and
have updated this repository accordingly.

 * Added metrics (LET-3D-APL and LET-3D-AP) for the 3D Camera-Only Detection
 Challenge.
 * Added 80 segments of 20-second camera imagery, as a test set for the 3D
 Camera-Only Detection Challenge.
 * Added z-axis speed and acceleration in
 [lidar label metadata](src/waymo_open_dataset/label.proto#L53-L60).
 * Fixed some inconsistencies in `projected_lidar_labels` in
 [dataset.proto](src/waymo_open_dataset/dataset.proto).
 * Updated the default configuration for the Occupancy and Flow Challenge,
 switching from aggregate waypoints to [subsampled waypoints](src/waymo_open_dataset/protos/occupancy_flow_metrics.proto#L38-L55).
 * Updated the [tutorial](tutorial/tutorial_3d_semseg.ipynb) for 3D Semantic Segmentation Challenge with more
 detailed instructions.

## March 2022 Update

We released v1.3.0 of the Perception dataset and the 2022 challenges. We have
updated this repository to add support for the new labels and the challenges.
 * Added 3D semantic segmentation labels, tutorial, and metrics.
 * Added 2D and 3D keypoint labels, tutorial, and metrics.
 * Added correspondence between 2D (camera) and 3D (lidar) labels (pedestrian
 only).
 * Added tutorial and utilities for Occupancy Flow Prediction Challenge.
 * Added the soft mAP metric for Motion Prediction Challenge.

## September 2021 Update

We released v1.1 of the Motion dataset to include lane connectivity information.
To read more on the technical details, please read [lane_neighbors_and_boundaries.md](docs/lane_neighbors_and_boundaries.md).
 * Added lane connections. Each lane has a list of lane IDs that enter or exit
 the lane.
 * Added lane boundaries. Each lane has a list of left and right boundary
 features associated with the lane and the segment of the lane where the
 boundary is active.
 * Added lane neighbors. Each lane has a list of left and right neighboring
 lanes. These are lanes an agent may make a lane change into.
 * Improved timestamp precision.
 * Improved stop sign Z values.

## March 2021 Update

We expanded the Waymo Open Dataset to also include a Motion dataset comprising
object trajectories and corresponding 3D maps for over 100,000 segments.
We have updated this repository to add support for this new dataset.

Additionally, we added instructions and examples for the real-time detection
challenges. Please follow these [Instructions](src/waymo_open_dataset/latency/README.md).