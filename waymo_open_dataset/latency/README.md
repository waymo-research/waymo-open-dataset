# Real-time Detection Challenges

For the real-time 3D and 2D detection challenges, users need to submit not only their results in a Submission proto, but also an executable version of their model that can be timed on an NVIDIA Tesla V100 GPU. This directory contains the code that will be used to evaluate submitted models, along with example submissions.

## User Interface

Users will submit docker images that contain their models. Docker is used to abstract over dependency management, including differing versions of CUDA drivers. As such, users can use arbitrary versions of TensorFlow or PyTorch for their submissions with ease. In particular, we have tested TensorFlow 2.3.0 and 2.4.1, PyTorch 1.3 and 1.7.1, CUDA 10.0 and 11.0, and cuDNN 7 and 8, but other versions, especially older versions, should likely work with little issue. The only required dependency that users must have installed is numpy.

User-submitted models will take the form of a Python module named `wod_latency_submission` that can be imported from our evaluation script (i.e. it is in the image's PYTHONPATH or pip-installed). The module must contain the following elements:

* `initialize_model`: A function that takes no arguments that loads and initializes the user's model. Will be run at the beginning of the evaluation script, before any data is passed in to the model.
* `run_model`: A function that takes in various numpy ndarrays whose names match those of the data formats (see below). This function runs the model inference on the passed-in data and returns a dictionary from string to numpy ndarray with the following key-value pairs:
  * `boxes`: N x 7 float32 array with the center x, center y, center z, length, width, height, and heading for each detection box. **NOTE**: If you are participating in the 2D object detection challenge instead, this should be a N x 4 float32 array with the center x, center y, length, and width for each detection box instead.
  * `scores`: N length float32 array with the confidence scores in `[0, 1]` for each detection box
  * `classes`: N length uint8 array with the type IDs for each detection box.
* `DATA_FORMATS`: A list of strings indicating which data formats the model requires. See below for more details.

## Data Formats

Converting from Frame protos to usable point clouds/images can be non-trivially expensive (involving various unzippings and transforms) and does not reflect a workflow that would realistically be present in an autonomous driving scenario. Thus, our evaluation of submitted models does not time the conversion from Frame proto to tensor. Instead, we have pre-extracted the dataset into numpy ndarrays. The keys, shapes, and data types are:

* `POSE`: 4x4 float32 array with the vehicle pose.
* `TIMESTAMP`: int64 scalar with the timestamp of the frame in microseconds.
* For each lidar:
  * `<LIDAR_NAME>_RANGE_IMAGE_FIRST_RETURN`: HxWx6 float32 array with the range image of the first return for this lidar. The six channels are range, intensity, elongation, x, y, and z. The x, y, and z values are in vehicle frame. Pixels with range 0 are not valid points.
  * `<LIDAR_NAME>_RANGE_IMAGE_SECOND_RETURN`: HxWx6 float32 array with the range image of the first return for this lidar. Same channels as the first return range image.
  * `<LIDAR_NAME>_BEAM_INCLINATION`: H-length float32 array with the beam inclinations for each row of range image for this lidar.
  * `<LIDAR_NAME>_LIDAR_EXTRINSIC`: 4x4 float32 array with the 4x4 extrinsic matrix for this lidar.
  * `<LIDAR_NAME>_CAM_PROJ_FIRST_RETURN`: HxWx6 int64 array with the lidar point to camera image projections for the first return of this lidar. See the documentation for `RangeImage.camera_projection_compressed` in [dataset.proto](https://github.com/waymo-research/waymo-open-dataset/blob/eb7d74d1e11f40f5f8485ae8e0dc71f0944e8661/waymo_open_dataset/dataset.proto#L181-L200) for details.
  * `<LIDAR_NAME>_CAM_PROJ_SECOND_RETURN`: HxWx6 float32 array with the lidar point to camera image projections for the first return of this lidar. See the documentation for `RangeImage.camera_projection_compressed` in [dataset.proto](https://github.com/waymo-research/waymo-open-dataset/blob/eb7d74d1e11f40f5f8485ae8e0dc71f0944e8661/waymo_open_dataset/dataset.proto#L181-L200) for details.
  * (top lidar only) `TOP_RANGE_IMAGE_POSE`: HxWx6 float32 array with the transform from vehicle frame ot global frame for every pixel in the range image for the TOP lidar. See the documentation for `RangeImage.range_image_pose_compressed` in [dataset.proto](https://github.com/waymo-research/waymo-open-dataset/blob/eb7d74d1e11f40f5f8485ae8e0dc71f0944e8661/waymo_open_dataset/dataset.proto#L202-L217) for details.
* For each camera:
  * `<CAMERA_NAME>_IMAGE`: HxWx3 uint8 array with the RGB image from this camera.
  * `<CAMERA_NAME>_INTRINSIC`: 9 float32 array with the intrinsics of this camera. See the documentation for `CameraCalibration.intrinsic` in [dataset.proto](https://github.com/waymo-research/waymo-open-dataset/blob/eb7d74d1e11f40f5f8485ae8e0dc71f0944e8661/waymo_open_dataset/dataset.proto#L91-L100) for details.
  * `<CAMERA_NAME>_EXTRINSIC`: 4x4 float32 array with the 4x4 extrinsic matrix for this camera.
  * `<CAMERA_NAME>_WIDTH`: int64 scalar with the width of this camera image.
  * `<CAMERA_NAME>_HEIGHT`: int64 scalar with the height of this camera image.
  * `<CAMERA_NAME>_POSE`: 4x4 float32 array with the vehicle pose at the timestamp of this camera image.
  * `<CAMERA_NAME>_POSE_TIMESTAMP`: float32 scalar with the timestamp in seconds for the image (i.e. the timestamp that `<CAMERA_NAME>_POSE` is valid at).
  * `<CAMERA_NAME>_ROLLING_SHUTTER_DURATION`: float32 scalar with the duration of the rolling shutter in seconds. See the documentation for `CameraImage.shutter in [dataset.proto](https://github.com/waymo-research/waymo-open-dataset/blob/eb7d74d1e11f40f5f8485ae8e0dc71f0944e8661/waymo_open_dataset/dataset.proto#L268-L283) for details.
  * `<CAMERA_NAME>_ROLLING_SHUTTER_DIRECTION`: int64 scalar with the direction of the rolling shutter, expressed as the int value of a `CameraCalibration.RollingShutterReadOutDirection` enum.
  * `<CAMERA_NAME>_CAMERA_TRIGGER_TIME`: float32 scalar with the time when the camera was triggered.
  * `<CAMERA_NAME>_CAMERA_READOUT_DONE_TIME`: float32 scalar with the time when the last readout finished. The difference between this and the trigger time includes the exposure time and the actual sensor readout time.

See the `LaserName.Name` and `CameraName.Name` enums in [dataset.proto](https://github.com/waymo-research/waymo-open-dataset/blob/eb7d74d1e11f40f5f8485ae8e0dc71f0944e8661/waymo_open_dataset/dataset.proto#L48-L69) for the valid lidar and camera name strings.

To request a field from the previous frame, add `_1` to the end of the field name; for example, `TOP_RANGE_IMAGE_FIRST_RETURN_1` is the range image for the top lidar from the previous frame. Likewise, to request a field from two frames ago, add `_2` to the end of the field name (e.g. `TOP_RANGE_IMAGE_FIRST_RETURN_2`). Note that only two previous frames (in addition to the current frame, which does not require a subscript) can be requested.

Users specify which of these arrays they would like their models to receive using the `DATA_FIELDS` list of strings in their `wod_latency_submission` module. The requested arrays will then be passed to `run_model` as keyword arguments with the original names (e.g. `TOP_RANGE_IMAGE_FIRST_RETURN`).

Note that the `convert_frame_to_dict` function in [utils/frame_utils.py](https://github.com/waymo-research/waymo-open-dataset/blob/ae21353bf721bf36e197654e67d482b5619a2302/waymo_open_dataset/utils/frame_utils.py#L215) will convert a Frame proto into a dictionary with the same keys and values defined above. However, it will not add the `_1` or `_2` suffix to the keys from earlier frames for multi-frame input.

## Examples

These examples all show different ways of making Docker images that contain submission models that comply with the input and output formats for the challenges. The repo contains a TensorFlow and a PyTorch example for each of the real-time detection challenges (2D and 3D).

* `tensorflow/from_saved_model`: Contains a basic PointPillars model that is executed from a [SavedModel](https://www.tensorflow.org/guide/saved_model) for the 3D detection challenge.
* `pytorch/from_saved_model`: Contains a PV-RCNN model that loads saved weights and runs model inference using the input/output formats specified above for the 3D detection challenge.
* `tensorflow/multiframe`: Contains a basic PointNet model that shows how to use multiple frames as input.
* `2d_challenge/tensorflow`: Contains a pretrained EfficientDet model loaded from the [TF Model Zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md) that outputs detections for the 2D detection challenge.
* `2d_challenge/pytorch`: Contains a pretrained Faster R-CNN model loaded from [torchvision](https://pytorch.org/vision/0.8/models.html#faster-r-cnn) that outputs detections for the 2D detection challenge.

## Docker submission instructions for latency benchmark

Upload your docker image to a Bucket on Google Cloud Storage or to Google
Container/Artifact Registry and indicate resource location using
`docker_image_source` field of the submission proto.

Please note that the latency would only be benchmarked if a valid docker image
path is provided

### Uploading to Google Cloud Storage:
You can use any bucket configuration but to minimize any incurred costs it is
recommended to create a bucket with the following settings `Location: Region`,
`Region: us-west1 (Oregon)`

docker_image_source should contain full path to the submission image starting
with `gs://` protocol, i.e. `docker_image_source: "gs://example_bucket_name/example_folder/example_docker_image.tar.gz"`

You can produce a compatible Docker image via the following command:

```bash
docker save --output="example_docker_image.tar" ID_OF_THE_IMAGE
```

More info on docker save is available at [official documentation](https://docs.docker.com/engine/reference/commandline/save/)

To improve container size and upload time you can additionally compress the
image:

```bash
gzip example_docker_image.tar
```

To upload image to the bucket you can use Web interface of Google Cloud Storage
or `gsutil` command:

```bash
gsutil cp example_docker_image.tar.gz  gs://example_bucket_name/example_folder/
```

More info on `gsutil` command available at [official documentation](https://cloud.google.com/storage/docs/gsutil/commands/cp)

You will need to grant Waymo’s service account
`213834518535-compute@developer.gserviceaccount.com` permissions to read from
your submission bucket (for this reason we recommend use a separate bucket
just for submissions)

 * It is recommended to use a unique name for every submitted docker image as
   the evaluation server may fetch the submission with a delay from the actual
   submission time.
 * Please note that resulting docker images, even after compressions may be
   larger than 1GB and using Google Container Registry or Google Artifact
   Registry may be preferred.

### Using Google Container Registry or Google Artifact Registry

There is no difference between using Google Container Registry or Google
Artifact Registry

Google Container Registry and Google Artifact Registry use same underlying
bucket base storage system so the recommended bucket settings apply too
   `Location: Region`, `Region: us-west1 (Oregon)`

 * docker_image_source should contain full id of the submission image within
   *docker.pkg.dev doman, i.e.
   `docker_image_source: "us-west1-docker.pkg.dev/example-registry-name/example-folder/example-image@sha256:example-sha256-hash"`

You can upload Docker image via the following command:

```bash
docker tag example-image-hash us-west1-docker.pkg.dev/example-registry-name/example-folder/example-image
docker push us-west1-docker.pkg.dev/example-registry-name/example-folder/example-image
```

More info on docker save is available at [official documentation](https://cloud.google.com/artifact-registry/docs/docker/quickstart#add-image)

You will need to grant Waymo’s service account
`213834518535-compute@developer.gserviceaccount.com` permissions to read from
your Google Docker Registry and Google Artifact Registry (for this reason we
recommend use a separate repository just for submissions)

It is recommended to always submit using sha256 of the currently submitted image
and not using ‘latest’ as the evaluation server may fetch the submission with a
delay from the actual submission time.
