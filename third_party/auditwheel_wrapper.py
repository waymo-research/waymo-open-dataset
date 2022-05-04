# Copyright 2022 The Waymo Open Dataset Authors. All Rights Reserved.
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
"""Monkey patch auditwheel to not ship tensorflow in pypi wheels.

Inspired by https://stackoverflow.com/a/69521879.
"""
import os
import re
import sys

from auditwheel import main
from auditwheel import policy


def _parse_bazelrc():
  variables = {}
  with open('.bazelrc') as f:
    for line in f:
      m = re.search(r'(?P<variable>[^\s=]+)=\"(?P<value>[^\"]+)\"', line)
      if m is None:
        continue
      variables[m.groupdict()['variable']] = m.groupdict()['value']
  return variables


if __name__ == '__main__':
  bazel_vars = _parse_bazelrc()
  if 'LD_LIBRARY_PATH' in os.environ:
    os.environ['LD_LIBRARY_PATH'] += ':' + bazel_vars['TF_SHARED_LIBRARY_DIR']
  else:
    os.environ['LD_LIBRARY_PATH'] = bazel_vars['TF_SHARED_LIBRARY_DIR']
  for p in policy.load_policies():
    p['lib_whitelist'].append(bazel_vars['TF_SHARED_LIBRARY_NAME'])
  sys.exit(main.main())
