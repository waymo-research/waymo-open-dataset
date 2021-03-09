# Copyright 2021 The Waymo Open Dataset Authors. All Rights Reserved.
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
"""The WOD (Waymo Open Dataset) latency evaluation.

This script runs a user-submitted detection model on WOD data (extracted into
a set of numpy arrays), saves the outputs of those models as numpy arrays, and
records the latency of the detection model's inference. Both the input data and
the output detection results are stored in the following directory structure:
`<root_dir>/<context.name>/<timestamp_micros>/` with one such subdirectory for
every frame. The input data is stored in a series of numpy files where the
basename is the name of the field; see convert_frame_to_dict in
utils/frame_utils.py for a list of the field names. The detection results are
saved in three numpy files: boxes, scores, and classes. Finally, the latency
results are saved in a text file containing the latency values for each frame.

NOTE: This is a standalone script that does not depend on any TensorFlow,
PyTorch, or WOD code, or on bazel. It is intended to run in a docker container
submitted by the user, with the only requirements being a numpy installation and
a user-generated wod_latency_submission module.
"""
import argparse
import os
import time

import numpy as np

import wod_latency_submission


def process_example(input_dir, output_dir):
  """Process a single example, save its outputs, and return the latency.

  In particular, this function requires that the submitted model (the run_model
  function in the wod_latency_submission module) takes in a bunch of numpy
  arrays as keyword arguments, whose names are defined in
  wod_latency_submission.DATA_FIELDS. It also assumes that input_directory
  contains a bunch of .npy data files with basenames corresponding to the valid
  values of DATA_FIELDS. Thus, this function loads the required fields from the
  input_directory, runs and times the run_model function, saves the model's
  outputs (a 'boxes' N x 7 ndarray, a 'scores' N ndarray, and a 'classes' N
  ndarray) to the output directory, and returns the model's runtime in seconds.

  Args:
    input_dir: string directory name to find the input .npy data files
    output_dir: string directory name to save the model results to.

  Returns:
    float latency value of the run_model call, in seconds.
  """
  # Load all the data fields that the user requested.
  data = {
      field: np.load(os.path.join(input_dir, f'{field}.npy'))
      for field in wod_latency_submission.DATA_FIELDS
  }

  # Time the run_model function of the user's submitted module, with the data
  # fields passed in as keyword arguments.
  tic = time.perf_counter()
  output = wod_latency_submission.run_model(**data)
  toc = time.perf_counter()

  # Sanity check the output before saving.
  assert len(output) == 3
  assert set(output.keys()) == set(('boxes', 'scores', 'classes'))
  num_objs = output['boxes'].shape[0]
  assert output['scores'].shape[0] == num_objs
  assert output['classes'].shape[0] == num_objs

  # Save the outputs as numpy files.
  for k, v in output.items():
    np.save(os.path.join(output_dir, k), v)

  # Save the list of input fields in a text file.
  with open(os.path.join(output_dir, 'input_fields.txt'), 'w') as f:
    f.write('\n'.join(wod_latency_submission.DATA_FIELDS))

  # Return the elapsed time of the run_model call.
  return toc - tic


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--input_data_dir', type=str, required=True)
  parser.add_argument('--output_dir', type=str, required=True)
  parser.add_argument('--latency_result_file', type=str, required=True)
  args = parser.parse_args()

  # Run any user-specified initialization code for their submission.
  wod_latency_submission.initialize_model()

  latencies = []
  # Iterate through the subdirectories for each frame.
  for context_name in os.listdir(args.input_data_dir):
    context_dir = os.path.join(args.input_data_dir, context_name)
    if not os.path.isdir(context_dir):
      continue
    for timestamp_micros in os.listdir(context_dir):
      timestamp_dir = os.path.join(context_dir, timestamp_micros)
      if not os.path.isdir(timestamp_dir):
        continue

      out_dir = os.path.join(args.output_dir, context_name, timestamp_micros)
      os.makedirs(out_dir, exist_ok=True)
      print('Processing', context_name, timestamp_micros)
      latencies.append(process_example(timestamp_dir, out_dir))

  # Save all the latency values in a text file.
  with open(args.latency_result_file, 'w') as latency_file:
    latency_file.write('\n'.join(str(l) for l in latencies))
