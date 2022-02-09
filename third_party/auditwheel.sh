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

function parse_bazelrc_value() {
  awk -F= '/'$1'/ {print(substr($2, 2, length($2)-2))}' .bazelrc
}

TF_SHARED_LIBRARY_NAME="$(parse_bazelrc_value TF_SHARED_LIBRARY_NAME)"
TF_SHARED_LIBRARY_DIR="$(parse_bazelrc_value TF_SHARED_LIBRARY_DIR)"

POLICY_JSON=$(find / -name policy.json)

sed -i "s/libresolv.so.2\"/libresolv.so.2\", \"$TF_SHARED_LIBRARY_NAME\"/g" $POLICY_JSON

cat $POLICY_JSON

export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$TF_SHARED_LIBRARY_DIR"
auditwheel $@
