# Quick Start

This Quick Start contains installation instructions for the Open Dataset codebase. Refer to the [Colab tutorial](https://colab.sandbox.google.com/github/waymo-research/waymo-open-dataset/blob/r1.0/tutorial/tutorial.ipynb) for a quick demo of the installation and data format.

## System Requirements
* g++ 5 or higher.
* TensorFlow 1.14.0
* TensorFlow 1.15.0 (the upcoming tf release is supported in r1.0-tf.1.15
  branch)

The code has two main parts. One is a utility written in C++ to compute the evaluation metrics. The other part is a set of [TensorFlow](https://www.tensorflow.org/) functions in Python to help with model training.

First, download the code and enter the base directory.
``` bash
git clone https://github.com/waymo-research/waymo-open-dataset.git waymo-od
cd waymo-od
git checkout remotes/origin/r1.0
```

We use the [Bazel](https://www.bazel.build/) build system. These commands should
install it in most cases. Please see these
[instructions](https://docs.bazel.build/versions/master/install.html) for other
ways to install Bazel. We assume you have Python installed.
``` bash
sudo apt-get install --assume-yes pkg-config zip g++ zlib1g-dev unzip python3 python3-pip
BAZEL_VERSION=0.28.0
wget https://github.com/bazelbuild/bazel/releases/download/0.28.0/bazel-${BAZEL_VERSION}-installer-linux-x86_64.sh
sudo bash bazel-${BAZEL_VERSION}-installer-linux-x86_64.sh
sudo apt install build-essential
```

Configure .bazelrc. This command may output some warnings due to the differences
between Python 2 and Python 3, which you may safely ignore.
```
./configure.sh
```

To delete previous bazel outputs and reset internal caches, run the following
command:
```
bazel clean
```

## Metrics Computation
The core metrics library is written in C++, so it can be wrapped in
other languages or frameworks. It can compute detection metrics (mAP) and
tracking metrics (MOTA). See more information about the metrics on the
[website](https://waymo.com/open/next/).

We provide command line tools and TensorFlow ops to call the detection metrics
library to compute detection metrics. We will provide a similar wrapper for
tracking metrics library in the future.

### Command Line Tool
This tool does not require TensorFlow.

Run the metrics-related tests to verify they work as expected.

``` bash
bazel test waymo_open_dataset/metrics:all
```

This binary computes the metric values given a pair of prediction and
ground truth files.
``` bash
bazel build waymo_open_dataset/metrics/tools/compute_detection_metrics_main
bazel-bin/waymo_open_dataset/metrics/tools/compute_detection_metrics_main waymo_open_dataset/metrics/tools/fake_predictions.bin  waymo_open_dataset/metrics/tools/fake_ground_truths.bin
```

### A TensorFlow Op
A TensorFlow op is defined at metrics/ops/metrics_ops.cc. We provide a
Python wrapper of the op at metrics/ops/py_metrics_ops.py, and a tf.metrics
like implementation of the op at metrics/python/detection_metrics.py.

Install NumPy and TensorFlow and reconfigure .bazelrc.
``` bash
pip3 install numpy tensorflow
./configure.sh
```

We have configured our build system to work with TensorFlow 1.14.0. For a higher
version, you might need to update the proto version in WORKSPACE to match
your TensorFlow version.

Run TensorFlow metrics op related tests. They can serve as examples for usage.
``` bash
bazel build waymo_open_dataset/metrics/ops/...
bazel test waymo_open_dataset/metrics/ops/...
bazel test waymo_open_dataset/metrics/python/...
```

## Python Utilities

We provide a set of TensorFlow libraries in the utils directory to help with building models. Refer to the [Colab tutorial](https://colab.sandbox.google.com/github/waymo-research/waymo-open-dataset/blob/r1.0/tutorial/tutorial.ipynb)
for examples of their usage.

``` bash
bazel test waymo_open_dataset/utils/...
```

## Use pre-compiled pip/pip3 packages
We only pre-compiled the package for Python 2.7, 3.5, 3.6. If you need the
lib for a different python version, follow steps in pip_pkg_scripts to build pip
package on your own.
``` bash
pip3 install upgrade --pip
pip3 install waymo-open-dataset==1.0.1 --user
```
