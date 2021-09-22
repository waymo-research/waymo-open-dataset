#!/bin/bash
# Copyright 2021 The Waymo Open Dataset Authors. All Rights Reserved.
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

set -e -x

docker build --tag=open_dataset_pip -f pip_pkg_scripts/build.Dockerfile .
mkdir /tmp/artifacts | true
mkdir /tmp/artifacts_all | true

for pyv in 7 8 9; do
  for tfv in 5 6; do
    docker run --mount type=bind,source=/tmp/artifacts,target=/tmp/artifacts -e "PYTHON_VERSION=3" -e "PYTHON_MINOR_VERSION=${pyv}" -e "PIP_MANYLINUX2010=1" -e "TF_VERSION=2.${tfv}.0" open_dataset_pip
    cp -f /tmp/artifacts/* /tmp/artifacts_all
  done
done

