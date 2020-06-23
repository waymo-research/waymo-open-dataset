#!/usr/bin/env bash
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

set -e

PLATFORM="$(uname -s | tr 'A-Z' 'a-z')"

PIP_FILE_PREFIX="pip_pkg_scripts/"

export PYTHON_VERSION="${PYTHON_VERSION:-3}"
export PYTHON_MINOR_VERSION="${PYTHON_MINOR_VERSION}"

if [[ -z "${PYTHON_MINOR_VERSION}" ]]; then
  PYTHON="python${PYTHON_VERSION}"
else
  PYTHON="python${PYTHON_VERSION}.${PYTHON_MINOR_VERSION}"
fi

function main() {
  DEST=${1}
  if [[ -z "${DEST}" ]]; then
    echo "No destination directory provided."
    exit 1
  fi

  # Create the directory, then do dirname on a non-existent file inside it to
  # give us an absolute paths with tilde characters resolved to the destination
  # directory.
  mkdir -p "${DEST}"
  DEST=$(readlink -f "${DEST}")
  echo "=== destination directory: ${DEST}"

  TMPDIR=$(mktemp -d -t tmp.XXXXXXXXXX)

  echo $(date) : "=== Using tmpdir: ${TMPDIR}"

  echo "=== Copy Waymo Open Dataset files"

  cp ${PIP_FILE_PREFIX}setup.py "${TMPDIR}"
  cp ${PIP_FILE_PREFIX}MANIFEST.in "${TMPDIR}"
  cp LICENSE "${TMPDIR}"
  rsync -avm -L --exclude="*_test.py" waymo_open_dataset "${TMPDIR}"
  rsync -avm -L --exclude="*_test.py" third_party/camera "${TMPDIR}/waymo_open_dataset"
  rsync -avm -L  --include="*.so" --include="*_pb2.py" \
    --exclude="*.runfiles" --exclude="*_obj" --include="*/" --exclude="*" \
    bazel-bin/waymo_open_dataset "${TMPDIR}"
  rsync -avm -L  --include="*.so" --include="*_pb2.py" \
    --exclude="*.runfiles" --exclude="*_obj" --include="*/" --exclude="*" \
    bazel-bin/third_party/camera "${TMPDIR}/waymo_open_dataset"

  pushd ${TMPDIR}
  echo $(date) : "=== Building wheel"

  ${PYTHON} setup.py bdist_wheel > /dev/null
  cp dist/*.whl "${DEST}"
  popd
  rm -rf ${TMPDIR}
  echo $(date) : "=== Output wheel file is in: ${DEST}"
}

main "$@"
