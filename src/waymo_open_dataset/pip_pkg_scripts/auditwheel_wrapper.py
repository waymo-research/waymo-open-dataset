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
"""Monkey patch auditwheel to not ship tensorflow in pypi wheels.

Inspired by https://stackoverflow.com/a/69521879.
"""
import os
import sys

from auditwheel import main
from auditwheel import policy
import tensorflow as tf


if __name__ == '__main__':
  print(' '.join(sys.argv))
  tf_dirs = ':'.join(tf.__path__)
  if 'LD_LIBRARY_PATH' in os.environ:
    os.environ['LD_LIBRARY_PATH'] += ':' + tf_dirs
  else:
    os.environ['LD_LIBRARY_PATH'] = tf_dirs
  for p in policy.load_policies():
    p['lib_whitelist'].append('libtensorflow_framework.so.2')
  sys.exit(main.main())
