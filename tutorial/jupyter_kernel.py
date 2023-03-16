# Copyright 2019 The Waymo Open Dataset Authors.
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
""""A custom Jupyter kernel with waymo_opend_dataset dependencies."""

from absl import app
from notebook import notebookapp


def main(_):
  instance = notebookapp.NotebookApp.instance()
  instance.open_browser = False
  instance.ip = "0.0.0.0"
  instance.port = 8888
  instance.port_retries = 0
  instance.allow_origin_pat = "https://colab\\.[^.]+\\.google.com"
  instance.allow_root = True
  instance.token = ""
  instance.disable_check_xsrf = True
  instance.initialize()
  instance.start()


if __name__ == "__main__":
  app.run(main)
