# Waymo Open Dataset

The Waymo Open Dataset was first launched in August 2019 with a perception dataset comprising high resolution sensor data and labels for 1,950 segments. We have released the Waymo Open Dataset publicly to aid the research community in making advancements in machine perception and autonomous driving technology.

## September 2021 Update

We released v1.1 of the motion dataset to include lane connectivity information. To read more on the technical details, please read [lane_neighbors_and_boundaries.md](docs/lane_neighbors_and_boundaries.md).
 - Added lane connections. Each lane has a list of lane IDs that enter or exit the lane.
 - Added lane boundaries.  Each lane has a list of left and right boundary features associated with the lane and the segment of the lane where the boundary is active.
 - Added lane neighbors. Each lane has a list of left and right neighboring lanes.  These are lanes an agent may make a lane change into.
 - Improved timestamp precision.
 - Improved stop sign Z values.

## March 2021 Update

We expanded the Waymo Open Dataset to also include a motion dataset comprising object trajectories and corresponding 3D maps for over 100,000 segments. We have updated this repository to add support for this new dataset. Please refer to the [Quick Start](docs/quick_start.md).

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
@inproceedings{sun2020scalability,
  title={Scalability in perception for autonomous driving: Waymo open dataset},
  author={Sun, Pei and Kretzschmar, Henrik and Dotiwalla, Xerxes and Chouard, Aurelien and Patnaik, Vijaysai and Tsui, Paul and Guo, James and Zhou, Yin and Chai, Yuning and Caine, Benjamin and others},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={2446--2454},
  year={2020}
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
