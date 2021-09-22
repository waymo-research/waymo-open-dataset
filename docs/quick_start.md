# Quick Start

This Quick Start contains installation instructions for the Open Dataset
codebase. Refer to the
[Colab tutorial](https://colab.sandbox.google.com/github/waymo-research/waymo-open-dataset/blob/master/tutorial/tutorial.ipynb)
for a quick demo of the installation and data format.

## Access

As a first step, please visit https://www.waymo.com/open to gain access to the Waymo Open Dataset.

## Use pre-compiled pip/pip3 packages for Linux

We only pre-compiled the package for Python 3.7, 3.8, 3.9 for Linux. If you need the lib for a different python version, follow steps in pip_pkg_scripts to build pip package on your own.

``` bash
pip3 install --upgrade pip
```

### tf 2.6.0.

``` bash
pip3 install waymo-open-dataset-tf-2-6-0 --user
```

### tf 2.5.0.

``` bash
pip3 install waymo-open-dataset-tf-2-5-0 --user
```

### tf 2.4.0.

``` bash
pip3 install waymo-open-dataset-tf-2-4-0 --user
```

## Compile in our docker container
Follow instructions in [pip_pkg_scripts](../pip_pkg_scripts/README.md).

## Local compilation without docker system requirements

*   g++ 5 or higher.
*   python 3.7, 3.8, 3.9 with TensorFlow 2.4.0, 2.5.0, 2.6.0

The code has two main parts. One is a utility written in C++ to compute the
evaluation metrics. The other part is a set of
[TensorFlow](https://www.tensorflow.org/) functions in Python to help with model
training.

First, download the code and enter the base directory.
```bash
git clone https://github.com/waymo-research/waymo-open-dataset.git waymo-od
cd waymo-od
git checkout remotes/origin/master
```

We use the [Bazel](https://www.bazel.build/) build system. These commands should
install it in most cases. Please see these
[instructions](https://docs.bazel.build/versions/master/install.html) for other
ways to install Bazel. We assume you have Python installed.
```bash
sudo apt-get install --assume-yes pkg-config zip g++ zlib1g-dev unzip python3 python3-pip
BAZEL_VERSION=3.1.0
wget https://github.com/bazelbuild/bazel/releases/download/${BAZEL_VERSION}/bazel-${BAZEL_VERSION}-installer-linux-x86_64.sh
sudo bash bazel-${BAZEL_VERSION}-installer-linux-x86_64.sh
sudo apt install build-essential
```

Configure .bazelrc. `./configure.sh`

To delete previous bazel outputs and reset internal caches, run the following
command: `bazel clean`

## Metrics Computation

The core metrics library is written in C++, so it can be wrapped in other
languages or frameworks. It can compute detection metrics, tracking and motion prediction metrics. See more information about the metrics on the
[website](https://waymo.com/open/next/).

We provide command line tools and TensorFlow ops to call the detection metrics
library to compute detection metrics. We will provide a similar wrapper for
tracking metrics library in the future.

### Command Line Tool

This tool does not require TensorFlow.

Run the metrics-related tests to verify they work as expected.

```bash
bazel test waymo_open_dataset/metrics:all
```

This binary computes the metric values given a pair of prediction and ground
truth files.
```bash
bazel build
waymo_open_dataset/metrics/tools/compute_detection_metrics_main
bazel-bin/waymo_open_dataset/metrics/tools/compute_detection_metrics_main
waymo_open_dataset/metrics/tools/fake_predictions.bin
waymo_open_dataset/metrics/tools/fake_ground_truths.bin
```

### A TensorFlow Op

A TensorFlow op is defined at metrics/ops/metrics_ops.cc. We provide a Python
wrapper of the op at metrics/ops/py_metrics_ops.py, and a tf.metrics like
implementation of the op at metrics/python/detection_metrics.py.

Install NumPy and TensorFlow and reconfigure .bazelrc. `bash pip3 install numpy
tensorflow ./configure.sh`

Run TensorFlow metrics op related tests. They can serve as examples for usage.
```bash
bazel build waymo_open_dataset/metrics/ops/... bazel test
waymo_open_dataset/metrics/ops/... bazel test
waymo_open_dataset/metrics/python/...
```

## Python Utilities

We provide a set of TensorFlow libraries in the utils directory to help with
building models. Refer to the
[Perception Colab tutorial](https://colab.sandbox.google.com/github/waymo-research/waymo-open-dataset/blob/master/tutorial/tutorial.ipynb) and [Motion Prediction tutorial](https://colab.sandbox.google.com/github/waymo-research/waymo-open-dataset/blob/master/tutorial/tutorial_motion.ipynb)
for examples of their usage.

```bash
bazel test waymo_open_dataset/utils/...
```

## Submit to leaderboard (For Perception tasks only)

1.  Run inference and dump the predictions in protos/metrics.proto:Objects
    format. Example code can be found in
    metrics/tools/create_submission.cc:example_code_to_create_a_prediction_file.
    There is also a python version in metrics/tools/create_prediction_file_example.py.
    Assume the file you created is in /tmp/preds.bin.

2.  First modify metrics/tools/submission.txtpb to set the metadata information.
    Then run metrics/tools/create_submission to convert the file above to the
    submission proto by adding more metadata submission information. NOTE:
    follow [latency instruction](latency/README.md) to create the docker image
    and fill in the required fields.

```bash
mkdir /tmp/my_model
metrics/tools/create_submission  --input_filenames='/tmp/preds.bin' --output_filename='/tmp/my_model/model' --submission_filename='metrics/tools/submission.txtpb
```

You can try a submission by running the following to the validation server. It
should work. Make sure you change the fields in metrics/tools/submission.txtpb
before running the command.

```bash
mkdir /tmp/my_model
metrics/tools/create_submission  --input_filenames='metrics/tools/fake_predictions.bin' --output_filename='/tmp/my_model/model' --submission_filename='metrics/tools/submission.txtpb'
```

1.  Tar and gzip the file.

```bash
tar cvf /tmp/my_model.tar /tmp/my_model/
gzip /tmp/my_model.tar
```

1.  Upload to the eval server for the validation set first as there is no limit
    on how frequently you submit for validation set. You can use this to ensure
    that your submission is in the right format. Then submit against the test
    set. Every registered user can only submit 3 times per month for each task.
