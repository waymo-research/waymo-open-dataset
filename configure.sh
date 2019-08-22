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
write_to_bazelrc 'build --cxxopt="-D_GLIBCXX_USE_CXX11_ABI=0"'
write_to_bazelrc 'build --auto_output_filter=subpackages'
write_to_bazelrc 'build --copt="-Wall" --copt="-Wno-sign-compare"'
write_to_bazelrc 'build --incompatible_bzl_disallow_load_after_statement=false'
write_to_bazelrc 'query --incompatible_bzl_disallow_load_after_statement=false'

TF_NEED_CUDA=0
# Check if it's installed
TF_CFLAGS=""
TF_LFLAGS=""
if [[ $(pip3 show tensorflow) == *tensorflow* ]] || [[ $(pip3 show tf-nightly) == *tf-nightly* ]] ; then
  echo 'Using installed tensorflow'
  TF_CFLAGS=( $(python3 -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))') )
  TF_LFLAGS="$(python3 -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))')"
else
  echo 'CPU Tensorflow is not installed'
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
