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

# This file is modified from configure.sh in
# https://github.com/tensorflow/custom-op.
# This script writes to .bazelrc to tensorflow libs.

export PYTHON_VERSION="${PYTHON_VERSION:-3}"
export PYTHON_MINOR_VERSION="${PYTHON_MINOR_VERSION}"
export PIP_MANYLINUX2010="${PIP_MANYLINUX2010:-0}"

if [[ -z "${PYTHON_MINOR_VERSION}" ]]; then
  PYTHON="python${PYTHON_VERSION}"
else
  PYTHON="python${PYTHON_VERSION}.${PYTHON_MINOR_VERSION}"
fi
PIP="$PYTHON -m pip"
update-alternatives --install /usr/bin/python3 python3 "/usr/bin/$PYTHON" 1

function write_to_bazelrc() {
  echo "$1" >> .bazelrc
}

function write_action_env_to_bazelrc() {
  write_to_bazelrc "build --action_env $1=\"$2\""
}

# Remove .bazelrc if it already exist
[ -e .bazelrc ] && rm .bazelrc

write_to_bazelrc "build -c opt"
write_to_bazelrc 'build --cxxopt="-std=c++11"'
write_to_bazelrc 'build --auto_output_filter=subpackages'
write_to_bazelrc 'build --copt="-Wall" --copt="-Wno-sign-compare"'
write_to_bazelrc 'build --linkopt="-lrt -lm"'

TF_NEED_CUDA=0
# Check if it's installed
TF_CFLAGS=""
TF_LFLAGS=""
if ${PIP} list | grep "tensorflow \|tensorflow-gpu\|tensorflow-cpu\|tf-nightly" >/dev/null ; then
  echo 'Using installed tensorflow'
  TF_CFLAGS=( $(${PYTHON} -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))') )
  TF_LFLAGS="$(${PYTHON} -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))')"
  if [[ -z ${TF_VERSION} ]]; then
    export TF_VERSION=$(${PYTHON} -c 'import tensorflow as tf; print(tf.__version__)')
  fi
else
  echo 'Tensorflow is not installed. Code still works.'
fi

write_action_env_to_bazelrc "TF_HEADER_DIR" ${TF_CFLAGS:2}
SHARED_LIBRARY_DIR=${TF_LFLAGS:2}
SHARED_LIBRARY_NAME=$(echo $TF_LFLAGS | rev | cut -d":" -f1 | rev)
if ! [[ $TF_LFLAGS =~ .*:.* ]]; then
  if [[ "$(uname)" == "Darwin" ]]; then
    SHARED_LIBRARY_NAME="libtensorflow_framework.dylib"
  else
    SHARED_LIBRARY_NAME="libtensorflow_framework.so"
  fi
fi
write_action_env_to_bazelrc "TF_SHARED_LIBRARY_DIR" ${SHARED_LIBRARY_DIR}
write_action_env_to_bazelrc "TF_SHARED_LIBRARY_NAME" ${SHARED_LIBRARY_NAME}
write_action_env_to_bazelrc "TF_NEED_CUDA" ${TF_NEED_CUDA}
write_to_bazelrc "build:manylinux2010 --crosstool_top=//third_party/toolchains/preconfig/ubuntu16.04/gcc7_manylinux2010-nvcc-cuda10.0:toolchain"

if [[ "$PIP_MANYLINUX2010" == "1" ]]; then
  write_to_bazelrc "build --config=manylinux2010"
  write_to_bazelrc "test --config=manylinux2010"
fi

export TF_VERSION="${TF_VERSION:-2.5.0}"
export TF_VERSION_UNDERSCORE=$(echo $TF_VERSION | sed 's/\./_/g')
export TF_VERSION_DASH=$(echo $TF_VERSION | sed 's/\./-/g')

cat WORKSPACE.in | sed "s/TF_VERSION/${TF_VERSION_UNDERSCORE}/" > WORKSPACE
cat pip_pkg_scripts/setup.py.in | sed "s/TF_VERSION/${TF_VERSION_DASH}/" > pip_pkg_scripts/setup.py

if [[ ${TF_VERSION} == '1.14.0' ]]; then
  write_to_bazelrc 'build --cxxopt="-D_GLIBCXX_USE_CXX11_ABI=1"'
else
  write_to_bazelrc 'build --cxxopt="-D_GLIBCXX_USE_CXX11_ABI=0"'
fi

