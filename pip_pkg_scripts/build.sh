#!/bin/bash
# Copyright 2019 The Waymo Open Dataset Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

# This script uses custom-op docker, downloads code, builds and tests and then
# builds a pip package.

# Execute this in docker container: tensorflow/tensorflow:custom-op-ubuntu16
# docker pull tensorflow/tensorflow:custom-op-ubuntu16
# docker run -it tensorflow/tensorflow:custom-op-ubuntu16 /bin/bash

set -e

GITHUB_BRANCH=$1
PYTHON_VERSION=$2

# Install bazel 0.28.0
BAZEL_VERSION=0.28.0
wget https://github.com/bazelbuild/bazel/releases/download/0.28.0/bazel-${BAZEL_VERSION}-installer-linux-x86_64.sh
sudo bash bazel-${BAZEL_VERSION}-installer-linux-x86_64.sh
sudo apt install build-essential

git clone https://github.com/waymo-research/waymo-open-dataset.git waymo-od
cd waymo-od
git checkout remotes/origin/${GITHUB_BRANCH}

# Install tensorflow
pip install tensorflow==1.14.0
pip3 install tensorflow==1.14.0

pip3 install --upgrade auditwheel

export PIP_MANYLINUX2010="1"
./configure.sh ${PYTHON_VERSION}

bazel clean
bazel build ...
bazel test ...

rm /tmp/od/package/*
./pip_pkg_scripts/build_pip_pkg.sh /tmp/od/package ${PYTHON_VERSION}
# Comment the following line if you run this outside of the container.
./third_party/auditwheel.sh repair --plat manylinux2010_x86_64 -w /tmp/od/package /tmp/od/package/*
