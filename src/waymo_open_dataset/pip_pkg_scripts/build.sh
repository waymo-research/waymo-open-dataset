#!/bin/bash
# Copyright 2023 The Waymo Open Dataset Authors.
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

# See README.md for instructions to use this script.

set -e -x

bazelisk test -c opt ... --test_output=all
bazelisk build -c opt //waymo_open_dataset/pip_pkg_scripts:wheel_manylinux
cp bazel-out/k8-opt/bin/waymo_open_dataset/pip_pkg_scripts/auditwheel_manylinux/* /tmp/wod
