# Waymo Open Dataset

The Waymo Open Dataset is a collection of datasets and evaluation code that we
have released publicly to aid the research community in making advancements in
machine perception and autonomous driving technology.

The Waymo Open Dataset includes three datasets:

- The Perception dataset, with high resolution sensor data and labels for various tasks.
- The Motion dataset, with object trajectories and corresponding 3D maps for 103,354 scenes.
- The End-To-End Driving dataset, with camera data and high-level commands.

In this codebase, we provide evaluation code to support several tasks based on
these three dataset.

See [release history](src/waymo_open_dataset/LOG.md)
for a detailed list of changes.

More information can be found on the [Waymo Open Dataset website](https://waymo.com/open/).

## License

This code repository (excluding
[`src/waymo_open_dataset/wdl_limited`](src/waymo_open_dataset/wdl_limited)
folder) is licensed under the Apache License, Version 2.0. The code appearing in
[`src/waymo_open_dataset/wdl_limited`](src/waymo_open_dataset/wdl_limited) is
licensed under terms appearing therein. The Waymo Open Dataset itself is
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
