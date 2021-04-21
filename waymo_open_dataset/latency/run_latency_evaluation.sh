#!/bin/bash
# Run the full latency evaluation pipeline (timing the functions and checking
# that the results match the submission proto) in a single shell script.

# Variables with the paths/commands for the binaries this script uses. They all
# use relative paths and assume that this script is run from the
# `waymo_open_dataset/latency` directory.
LATENCY_EVALUATOR_PATH="wod_latency_evaluator.py"
MAKE_OBJS_CMD="bazel run :make_objects_file_from_latency_results -- "
COMPARE_OBJS_TO_SUBMISSION_CMD="bazel run :compare_objects_file_to_submission_main -- "

# Process the arguments.
if [[ "$#" -ne 4 ]]; then
  echo "Needs four arguments: the data directory, the image name, the submission pb, and the output latency file."
  exit 1
fi
DATA_DIR=$1
IMAGE_NAME=$2
SUBMISSION_PB=$3
OUTPUT_LATENCY_FILE=$4

# Run the evaluation in a Docker container.
LOCAL_EVALUATOR_PATH=$(mktemp)
cp $LATENCY_EVALUATOR_PATH $LOCAL_EVALUATOR_PATH
DETECTION_OUTPUT_DIR=$(mktemp -d)
touch $OUTPUT_LATENCY_FILE
docker run --rm \
  --mount type=bind,source=$LOCAL_EVALUATOR_PATH,dst=/code/evaluator.py,readonly \
  --mount type=bind,source=$DATA_DIR,dst=/eval_data,readonly \
  --mount type=bind,source=$DETECTION_OUTPUT_DIR,dst=/eval_outputs \
  --mount type=bind,source=$OUTPUT_LATENCY_FILE,dst=/latency_output.txt \
  $IMAGE_NAME python /code/evaluator.py \
    --input_data_dir=/eval_data \
    --output_dir=/eval_outputs \
    --latency_result_file=/latency_output.txt

# Convert the detection results to an Objects file and compare against the
# submission.
OBJS_FILE=$(mktemp)
$MAKE_OBJS_CMD --results_dir $DETECTION_OUTPUT_DIR --output_file $OBJS_FILE
$COMPARE_OBJS_TO_SUBMISSION_CMD --latency_result_filename $OBJS_FILE --full_result_filenames $SUBMISSION_PB

# Clean up the outputs of the accuracy check.
sudo rm -rf $DETECTION_OUTPUT_DIR $OBJS_FILE
