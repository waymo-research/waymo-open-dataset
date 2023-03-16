# Waymo Open Dataset

We have released the Waymo Open Dataset publicly to aid the research community in making advancements in machine perception and autonomous driving technology.

The Waymo Open Dataset is composed of two datasets - the Perception dataset with high resolution sensor data and labels for 2,030 scenes, and the Motion dataset with object trajectories and corresponding 3D maps for 103,354 scenes.

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
 - Added `num_top_lidar_points_in_box` in [dataset.proto](waymo_open_dataset/dataset.proto) for the 3D Camera-Only Detection Challenge.

## April 2022 Update
We released v1.3.1 of the Perception dataset to support the 2022 Challenges and have updated this repository accordingly.
 - Added metrics (LET-3D-APL and LET-3D-AP) for the 3D Camera-Only Detection Challenge.
 - Added 80 segments of 20-second camera imagery, as a test set for the 3D Camera-Only Detection Challenge.
 - Added z-axis speed and acceleration in [lidar label metadata](waymo_open_dataset/label.proto#L53-L60).
 - Fixed some inconsistencies in `projected_lidar_labels` in [dataset.proto](waymo_open_dataset/dataset.proto).
 - Updated the default configuration for the Occupancy and Flow Challenge, switching from aggregate waypoints to [subsampled waypoints](waymo_open_dataset/protos/occupancy_flow_metrics.proto#L38-L55).
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

We expanded the Waymo Open Dataset to also include a Motion dataset comprising object trajectories and corresponding 3D maps for over 100,000 segments. We have updated this repository to add support for this new dataset. Please refer to the [Quick Start](docs/quick_start.md).

Additionally, we added instructions and examples for the real-time detection challenges. Please follow these [Instructions](waymo_open_dataset/latency/README.md).

## Website

To read more about the dataset and access it, please visit [https://www.waymo.com/open](https://www.waymo.com/open).

## Contents

This code repository contains:

* Definition of the dataset format
* Evaluation metrics
* Helper functions in TensorFlow to help with building models

Please refer to the [Quick Start](docs/quick_start.md).

## License
This code repository (excluding third_party) is licensed under the Apache License, Version 2.0.  Code appearing in third_party is licensed under terms appearing therein.

The Waymo Open Dataset itself is licensed under separate terms. Please visit [https://waymo.com/open/terms/](https://waymo.com/open/terms/) for details.  Code located at third_party/camera is licensed under a BSD 3-clause copyright license + an additional limited patent license applicable only when the code is used to process data from the Waymo Open Dataset as authorized by and in compliance with the Waymo Dataset License Agreement for Non-Commercial Use.  See third_party/camera for details.

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
