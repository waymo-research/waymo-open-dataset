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

# See README.md for instructions to use this script.

set -e -x

# Override the following env variables if necessary.
export PYTHON_VERSION="${PYTHON_VERSION:-3}"
export PYTHON_MINOR_VERSION="${PYTHON_MINOR_VERSION}"
export PIP_MANYLINUX2010="${PIP_MANYLINUX2010:-0}"
export TF_VERSION="${TF_VERSION:-2.5.0}"

if [[ -z "${PYTHON_MINOR_VERSION}" ]]; then
  PYTHON="python${PYTHON_VERSION}"
else
  PYTHON="python${PYTHON_VERSION}.${PYTHON_MINOR_VERSION}"
fi

curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
$PYTHON get-pip.py --user

PIP="$PYTHON -m pip"

${PIP} install --upgrade setuptools --user
${PIP} install --upgrade tensorflow=="${TF_VERSION}" --user

./configure.sh

bazel clean
bazel build ...
bazel test ... --test_output=all

DST_DIR="/tmp/artifacts"
rm -rf "$DST_DIR" || true
./pip_pkg_scripts/build_pip_pkg.sh "$DST_DIR" ${PYTHON_VERSION}
# Comment the following line if you run this outside of the container.
if [[ "${PIP_MANYLINUX2010}" == "1" ]]; then
  find "$DST_DIR" -name *.whl | xargs ./third_party/auditwheel.sh repair --plat manylinux2010_x86_64 -w "$DST_DIR"
fi

