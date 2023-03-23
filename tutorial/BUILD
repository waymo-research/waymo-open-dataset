load("@wod_deps//:requirements.bzl", "requirement")

package(
    default_applicable_licenses = ["//waymo_open_dataset:license"],
    default_visibility = [
        "//visibility:public",
    ],
)

licenses(["notice"])

filegroup(
    name = "framedata",
    srcs = [
        "3d_point_cloud.png",
        "3d_semseg_test_set_frames.txt",
        "frame_with_keypoints.tfrecord",
        "frames",
        "frames_with_maps.tfrecord",
    ],
)

py_binary(
    name = "jupyter_kernel",
    srcs = ["jupyter_kernel.py"],
    data = [
        "tutorial_local.ipynb",
        ":framedata",
        "//waymo_open_dataset/metrics/tools:compute_detection_metrics_main",
        "//waymo_open_dataset/metrics/tools:fakedata",
        "//waymo_open_dataset/utils/testdata:motion_dataset_test_data",
    ],
    main = "jupyter_kernel.py",
    python_version = "PY3",
    deps = [
        requirement("matplotlib"),
        requirement("numpy"),
        requirement("tensorflow"),
        requirement("tensorflow_graphics"),
        "//waymo_open_dataset:dataset_proto_py_pb2",
        "//waymo_open_dataset:label_proto_py_pb2",
        "//waymo_open_dataset/metrics/ops:py_metrics_ops",
        "//waymo_open_dataset/metrics/python:config_util_py",
        "//waymo_open_dataset/metrics/python:keypoint_metrics",
        "//waymo_open_dataset/protos:breakdown_proto_py_pb2",
        "//waymo_open_dataset/protos:camera_segmentation_proto_py_pb2",
        "//waymo_open_dataset/protos:camera_segmentation_submission_proto_py_pb2",
        "//waymo_open_dataset/protos:keypoints_submission_proto_py_pb2",
        "//waymo_open_dataset/protos:map_proto_py_pb2",
        "//waymo_open_dataset/protos:metrics_proto_py_pb2",
        "//waymo_open_dataset/protos:motion_metrics_proto_py_pb2",
        "//waymo_open_dataset/protos:motion_submission_proto_py_pb2",
        "//waymo_open_dataset/protos:occupancy_flow_metrics_proto_py_pb2",
        "//waymo_open_dataset/protos:occupancy_flow_submission_proto_py_pb2",
        "//waymo_open_dataset/protos:scenario_proto_py_pb2",
        "//waymo_open_dataset/protos:submission_proto_py_pb2",
        "//waymo_open_dataset/utils:box_utils",
        "//waymo_open_dataset/utils:camera_segmentation_utils",
        "//waymo_open_dataset/utils:frame_utils",
        "//waymo_open_dataset/utils:keypoint_data",
        "//waymo_open_dataset/utils:keypoint_draw",
        "//waymo_open_dataset/utils:occupancy_flow_data",
        "//waymo_open_dataset/utils:occupancy_flow_grids",
        "//waymo_open_dataset/utils:occupancy_flow_metrics",
        "//waymo_open_dataset/utils:occupancy_flow_renderer",
        "//waymo_open_dataset/utils:occupancy_flow_vis",
        "//waymo_open_dataset/utils:plot_maps",
        "//waymo_open_dataset/utils:range_image_utils",
        "//waymo_open_dataset/utils:transform_utils",
        "//waymo_open_dataset/utils:womd_lidar_utils",
        "//waymo_open_dataset/utils/compression:delta_encoder",
        "//waymo_open_dataset/wdl_limited/camera_segmentation:camera_segmentation_metrics",
    ],
)

py_binary(
    name = "jupyter_kernel_v2",
    srcs = ["jupyter_kernel.py"],
    main = "jupyter_kernel.py",
    python_version = "PY3",
    deps = [
        requirement("matplotlib"),
        requirement("numpy"),
        requirement("tensorflow"),
        requirement("tensorflow_graphics"),
        "//waymo_open_dataset/v2",
    ],
)
