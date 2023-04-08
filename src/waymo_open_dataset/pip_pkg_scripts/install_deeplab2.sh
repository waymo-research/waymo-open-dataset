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

srcs=(
  '__init__.py'
  'data/dataset.py'
  'data/__init__.py'
  'data/ade20k_constants.py'
  'data/waymo_constants.py'
  'evaluation/__init__.py'
  'evaluation/segmentation_and_tracking_quality.py'
)
base_url='https://raw.githubusercontent.com/google-research/deeplab2/main'
src_dir='/content/deeplab2/src'

set -x -e

mkdir -p "${src_dir}/deeplab2/data" "${src_dir}/deeplab2/evaluation"
for path in "${srcs[@]}"; do
  wget "${base_url}/${path}" -O "${src_dir}/deeplab2/${path}"
done

cat << EOF > "${src_dir}/setup.py"
from setuptools import setup, find_packages

setup(name='deeplab2-for-wod',
      version='0.0.1',
      description='A minimal copy of DeepLab2 for Waymo Open Dataset',
      url='https://github.com/google-research/deeplab2',
      author='Waymo Open Dataset Authors',
      author_email='open-dataset@waymo.com',
      license='Apache License 2.0',
      packages=find_packages(),
      zip_safe=False
)
EOF

cd "${src_dir}" && pip install .