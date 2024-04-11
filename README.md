# Waymo Open Dataset

We have released the Waymo Open Dataset publicly to aid the research community in making advancements in machine perception and autonomous driving technology.

The Waymo Open Dataset is composed of two datasets - the Perception dataset with high resolution sensor data and labels for 2,030 scenes, and the Motion dataset with object trajectories and corresponding 3D maps for 103,354 scenes.

## License

This code repository (excluding
[`src/waymo_open_dataset/wdl_limited`](src/waymo_open_dataset/wdl_limited)
folder) is licensed under the Apache License, Version 2.0. The code appearing in
[`src/waymo_open_dataset/wdl_limited`](src/waymo_open_dataset/wdl_limited) is
licensed under terms appearing therein.  The Waymo Open Dataset itself is
licensed under separate terms. Please
visit [https://waymo.com/open/terms/](https://waymo.com/open/terms/) for
details.  Code located in each of the subfolders located at
[`src/waymo_open_dataset/wdl_limited`](src/waymo_open_dataset/wdl_limited) is
licensed under (a) a BSD 3-clause copyright license and (b) an additional
limited patent license. Each limited
patent license is applicable only to code under the respective `wdl_limited`
subfolder, and is licensed for use only with the use case laid out in such
license in connection with the Waymo Open Dataset, as authorized by and in
compliance with the Waymo Dataset License Agreement for Non-Commercial Use. See
[wdl_limited/camera/](src/waymo_open_dataset/wdl_limited/camera),
[wdl_limited/camera_segmentation/](src/waymo_open_dataset/wdl_limited/camera_segmentation),
[wdl_limited/sim_agents_metrics/](src/waymo_open_dataset/wdl_limited/sim_agents_metrics),
respectively, for details.

## April 2024 Update
The Rules have been updated to allow training (including pre-training, co-training or fine-tuning models) using frozen, pre-trained weights from publicly available open source models for submissions to the Challenges. We also added a new sets of fields (which are now required, or the server will return an error) in the submission metadatas to track how participants generated their submissions.
We updated the tutorials to reflect this change, check out the new fields in the submission proto files
for motion, sim agents and occupancy flow.

## March 2024 Update
This update contains several changes/addition to the datasets:

Perception dataset (v1.4.3 and v2.0.1):
- We made improvements in the 3D semantic segmentation ground truth labels, especially for the class of motorcyclist.

Motion dataset (v1.2.1):
- The 1.2.1 WOMD release now provides camera data, including front, front-left, front-right, side-left, side-right, rear-left, rear-right, and rear sensors. Similar to the Lidar data, the camera data of the training, validation and testing sets cover the first 1 second of each of the 9 second windows. Instead of releasing raw camera images, we release the image tokens and image embedding extracted from a pre-trained VQ-GAN model.
- The initial release of the WOMD camera data contained misalignment between LiDAR data and roadgraph inputs for some frames. The 1.2.1 release provides new timestamps for the lidar data with an updated pose transformation matrix per time step.

We also provide the following changes to the code supporting the challenges.

Motion prediction:
- We have improved the logic behind the behavior bucketing used for mAP.

Sim Agents:
- We have improved the quality of the kinematic metrics by using smoother estimates of speeds and accelerations.
- We have fixed an edge case for offroad computation with over-passes.
- We have re-calibratred the metric configuration and composite metrics weights.
- We report simulated collision and offroad rates (not likelihoods).


## December 2023 Update
We released v1.6.1 version of the pip package with fixes for the WOSAC metrics:
- Fixing a bug in validity checking for collision and offroad.
- Modifying the behaviour of collision/offroad checking when invalid.


## August 2023 Update
We released a large-scale object-centric asset dataset containing over 1.2M images and lidar observations of two major categories (vehicles and pedestrians) from the Perception Dataset (v2.0.0).

 * Extracted perception objects from multi-sensor data: all five cameras and the top lidar.
 * Lidar features include 3D point cloud sequences that support 3D object shape reconstruction. We additionally provide refined box pose through point cloud shape registration for all vehicle objects.
 * Camera features include sequences of camera patches from the `most_visible_camera`, projected lidar returns on the corresponding camera, per-pixel camera rays information, and auto-labeled 2D panoptic segmentation that supports object NeRF reconstruction.
 * Added a [tutorial](tutorial/tutorial_object_asset.ipynb) and supporting code.


## March 2023 Update
This major update includes supporting code to four challenges at waymo.com/open, and dataset updates to both the Perception and Motion Datasets.

v2.0.0 of the Perception Dataset
 - Introduced the dataset in modular format, enabling users to selectively download only the components they need.
 - Includes all features in v1.4.2 of the Perception Dataset except maps.
 - Added a [tutorial](tutorial/tutorial_v2.ipynb) and supporting code.

v1.4.2 of the Perception Dataset
 - For the 2D video panoptic segmentation labels, added a mask to indicate the number of cameras covering each pixel.
 - Added 3D map data as polylines or polygons.

v1.2.0 of the Motion Dataset
 - Added Lidar data for the training set (first 1s of each 9s windows), and the corresponding [tutorial](tutorial/tutorial_womd_lidar.ipynb) and supporting code.
 - Added driveway entrances to the map data. Adjusted some road edge boundary height estimates.
 - Increased the max number of map points in tf_examples to 30k and reduced sampling to 1.0m to increase map coverage, so the coverage equalizes that of the dataset in scenario proto format. Added conversion code from scenario proto format to tf_examples format.

Added supporting code for the four 2023 Waymo Open Dataset Challenges
 - Sim Agents Challenge, with a [tutorial](tutorial/tutorial_sim_agents.ipynb)
 - Pose Estimation Challenge, with a [tutorial](tutorial/tutorial_keypoints.ipynb)
 - 2D Video Panoptic Segmentation Challenge, with a [tutorial](tutorial/tutorial_2d_pvps.ipynb)
 - Motion Prediction Challenge, with a [tutorial](tutorial/tutorial_motion.ipynb)

## December 2022 Update
We released v1.4.1 of the Perception dataset.
- Improved the quality of the 2D video panoptic segmentation labels.

## June 2022 Update
We released v1.4.0 of the Perception dataset.
 - Added 2D video panoptic segmentation labels and supporting code.

## May 2022 Update (part 2)
 - Released a [tutorial](tutorial/tutorial_camera_only.ipynb) for the 3D Camera-Only Detection Challenge.
 - Added support for computing 3D-LET-APL in Python metrics ops. See `Compute Metrics` in the [tutorial](tutorial/tutorial_camera_only.ipynb).
 - Fixed a bug in the metrics implementation for the Occupancy and Flow Challenge.

## May 2022 Update
We released v1.3.2 of the Perception dataset to improve the quality and accuracy of the labels.
 - Updated 3D semantic segmentation labels, for better temporal consistency and to fix mislabeled points.
 - Updated 2D key point labels to fix image cropping issues.
 - Added `num_top_lidar_points_in_box` in [dataset.proto](src/waymo_open_dataset/dataset.proto) for the 3D Camera-Only Detection Challenge.

## April 2022 Update
We released v1.3.1 of the Perception dataset to support the 2022 Challenges and have updated this repository accordingly.
 - Added metrics (LET-3D-APL and LET-3D-AP) for the 3D Camera-Only Detection Challenge.
 - Added 80 segments of 20-second camera imagery, as a test set for the 3D Camera-Only Detection Challenge.
 - Added z-axis speed and acceleration in [lidar label metadata](src/waymo_open_dataset/label.proto#L53-L60).
 - Fixed some inconsistencies in `projected_lidar_labels` in [dataset.proto](src/waymo_open_dataset/dataset.proto).
 - Updated the default configuration for the Occupancy and Flow Challenge, switching from aggregate waypoints to [subsampled waypoints](src/waymo_open_dataset/protos/occupancy_flow_metrics.proto#L38-L55).
 - Updated the [tutorial](tutorial/tutorial_3d_semseg.ipynb) for 3D Semantic Segmentation Challenge with more detailed instructions.

## March 2022 Update

We released v1.3.0 of the Perception dataset and the 2022 challenges. We have updated this repository to add support for the new labels and the challenges.
 - Added 3D semantic segmentation labels, tutorial, and metrics.
 - Added 2D and 3D keypoint labels, tutorial, and metrics.
 - Added correspondence between 2D (camera) and 3D (lidar) labels (pedestrian only).
 - Added tutorial and utilities for Occupancy Flow Prediction Challenge.
 - Added the soft mAP metric for Motion Prediction Challenge.

## September 2021 Update

We released v1.1 of the Motion dataset to include lane connectivity information. To read more on the technical details, please read [lane_neighbors_and_boundaries.md](docs/lane_neighbors_and_boundaries.md).
 - Added lane connections. Each lane has a list of lane IDs that enter or exit the lane.
 - Added lane boundaries.  Each lane has a list of left and right boundary features associated with the lane and the segment of the lane where the boundary is active.
 - Added lane neighbors. Each lane has a list of left and right neighboring lanes.  These are lanes an agent may make a lane change into.
 - Improved timestamp precision.
 - Improved stop sign Z values.

## March 2021 Update

We expanded the Waymo Open Dataset to also include a Motion dataset comprising object trajectories and corresponding 3D maps for over 100,000 segments. We have updated this repository to add support for this new dataset.

Additionally, we added instructions and examples for the real-time detection challenges. Please follow these [Instructions](src/waymo_open_dataset/latency/README.md).

## Website

To read more about the dataset and access it, please visit [https://www.waymo.com/open](https://www.waymo.com/open).

## Contents

This code repository contains:

* Definition of the dataset format
* Evaluation metrics
* Helper functions in TensorFlow to help with building models

## Citation
### for Perception dataset
@InProceedings{Sun_2020_CVPR,
  author = {Sun, Pei and Kretzschmar, Henrik and Dotiwalla, Xerxes and Chouard, Aurelien and Patnaik, Vijaysai and Tsui, Paul and Guo, James and Zhou, Yin and Chai, Yuning and Caine, Benjamin and Vasudevan, Vijay and Han, Wei and Ngiam, Jiquan and Zhao, Hang and Timofeev, Aleksei and Ettinger, Scott and Krivokon, Maxim and Gao, Amy and Joshi, Aditya and Zhang, Yu and Shlens, Jonathon and Chen, Zhifeng and Anguelov, Dragomir},
  title = {Scalability in Perception for Autonomous Driving: Waymo Open Dataset},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  month = {June},
  year = {2020}
}

### for Motion dataset
@InProceedings{Ettinger_2021_ICCV,
  author={Ettinger, Scott and Cheng, Shuyang and Caine, Benjamin and Liu, Chenxi and Zhao, Hang and Pradhan, Sabeek and Chai, Yuning and Sapp, Ben and Qi, Charles R. and Zhou, Yin and Yang, Zoey and Chouard, Aur\'elien and Sun, Pei and Ngiam, Jiquan and Vasudevan, Vijay and McCauley, Alexander and Shlens, Jonathon and Anguelov, Dragomir},
  title={Large Scale Interactive Motion Forecasting for Autonomous Driving: The Waymo Open Motion Dataset},
  booktitle= Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
  month={October},
  year={2021},
  pages={9710-9719}
}

@InProceedings{Kan_2024_icra,
  author={Chen, Kan and Ge, Runzhou and Qiu, Hang and Ai-Rfou, Rami and Qi, Charles R. and Zhou, Xuanyu and Yang, Zoey and Ettinger, Scott and Sun, Pei and Leng, Zhaoqi and Mustafa, Mustafa and Bogun, Ivan and Wang, Weiyue and Tan, Mingxing and Anguelov, Dragomir},
  title={WOMD-LiDAR: Raw Sensor Dataset Benchmark for Motion Forecasting},
  month={May},
  booktitle= Proceedings of the IEEE International Conference on Robotics and Automation (ICRA)},
  year={2024}
}

## Dataset Metadata
The following table is necessary for this dataset to be indexed by search
engines such as <a href="https://g.co/datasetsearch">Google Dataset Search</a>.
<div itemscope itemtype="http://schema.org/Dataset">
<table>
  <tr>
    <th>property</th>
    <th>value</th>
  </tr>
  <tr>
    <td>name</td>
    <td><code itemprop="name">Waymo Open Dataset: An autonomous driving dataset</code></td>
  </tr>
  <tr>
    <td>alternateName</td>
    <td><code itemprop="alternateName">Waymo Open Dataset</code></td>
  </tr>
  <tr>
    <td>url</td>
    <td><code itemprop="url">https://github.com/waymo-research/waymo-open-dataset</code></td>
  </tr>
  <tr>
    <td>sameAs</td>
    <td><code itemprop="sameAs">https://github.com/waymo-research/waymo-open-dataset</code></td>
  </tr>
    <tr>
    <td>sameAs</td>
    <td><code itemprop="sameAs">https://www.waymo.com/open</code></td>
  </tr>
  <tr>
    <td>description</td>
    <td><code itemprop="description">The Waymo Open Dataset is comprised of high-resolution sensor data collected by autonomous vehicles operated by the Waymo Driver in a wide variety of conditions. We’re releasing this dataset publicly to aid the research community in making advancements in machine perception and self-driving technology.</code></td>
  </tr>
  <tr>
    <td>provider</td>
    <td>
      <div itemscope itemtype="http://schema.org/Organization" itemprop="provider">
        <table>
          <tr>
            <th>property</th>
            <th>value</th>
          </tr>
          <tr>
            <td>name</td>
            <td><code itemprop="name">Waymo</code></td>
          </tr>
          <tr>
            <td>sameAs</td>
            <td><code itemprop="sameAs">https://en.wikipedia.org/wiki/Waymo</code></td>
          </tr>
        </table>
      </div>
    </td>
  </tr>
  <tr>
    <td>license</td>
    <td>
      <div itemscope itemtype="http://schema.org/CreativeWork" itemprop="license">
        <table>
          <tr>
            <th>property</th>
            <th>value</th>
          </tr>
          <tr>
            <td>name</td>
            <td><code itemprop="name">Waymo Dataset License Agreement for Non-Commercial Use (August 2019)</code></td>
          </tr>
          <tr>
            <td>url</td>
            <td><code itemprop="url">https://waymo.com/open/terms/</code></td>
          </tr>
        </table>
      </div>
    </td>
  </tr>
</table>
</div>
