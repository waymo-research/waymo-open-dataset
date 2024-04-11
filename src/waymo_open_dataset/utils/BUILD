load("@wod_deps//:requirements.bzl", "requirement")

package(
    default_applicable_licenses = ["//waymo_open_dataset:license"],
    default_visibility = [
        "//visibility:public",
    ],
)

licenses(["notice"])

exports_files(["BUILD"])

# Py libs.

py_library(
    name = "box_utils",
    srcs = ["box_utils.py"],
    deps = [
        requirement('tensorflow'),
        ":transform_utils",
        "//waymo_open_dataset:label_proto_py_pb2",
    ],
)

py_library(
    name = "occupancy_flow_data",
    srcs = ["occupancy_flow_data.py"],
    deps = [
        requirement('tensorflow'),
        "//waymo_open_dataset/protos:scenario_proto_py_pb2",
    ],
)

py_library(
    name = "occupancy_flow_grids",
    srcs = ["occupancy_flow_grids.py"],
    deps = [
        requirement('tensorflow'),
        ":occupancy_flow_data",
        ":occupancy_flow_renderer",
        "//waymo_open_dataset/protos:occupancy_flow_metrics_proto_py_pb2",
        "//waymo_open_dataset/protos:scenario_proto_py_pb2",
    ],
)

py_library(
    name = "occupancy_flow_metrics",
    srcs = ["occupancy_flow_metrics.py"],
    deps = [
        requirement('tensorflow'),
        requirement('tensorflow_graphics'),
        ":occupancy_flow_grids",
        "//waymo_open_dataset/protos:occupancy_flow_metrics_proto_py_pb2",
    ],
)

py_library(
    name = "occupancy_flow_renderer",
    srcs = ["occupancy_flow_renderer.py"],
    deps = [
        requirement('numpy'),
        requirement('tensorflow'),
        ":occupancy_flow_data",
        "//waymo_open_dataset/protos:occupancy_flow_metrics_proto_py_pb2",
        "//waymo_open_dataset/protos:scenario_proto_py_pb2",
    ],
)

py_library(
    name = "occupancy_flow_test_util",
    srcs = ["occupancy_flow_test_util.py"],
    data = [
        "//waymo_open_dataset/utils/testdata:motion_dataset_test_data",
    ],
    deps = [
        requirement('tensorflow'),
        ":occupancy_flow_data",
        "//waymo_open_dataset/protos:occupancy_flow_metrics_proto_py_pb2",
    ],
)

py_library(
    name = "occupancy_flow_vis",
    srcs = ["occupancy_flow_vis.py"],
    deps = [
        requirement('tensorflow'),
        ":occupancy_flow_data",
    ],
)

py_library(
    name = "range_image_utils",
    srcs = ["range_image_utils.py"],
    deps = [
        requirement('numpy'),
        requirement('tensorflow'),
    ],
)

py_library(
    name = "transform_utils",
    srcs = ["transform_utils.py"],
    deps = [
        requirement('tensorflow'),
    ],
)

py_library(
    name = "test_utils",
    testonly = 1,
    srcs = ["test_utils.py"],
    data = [
        "//waymo_open_dataset/utils/testdata:motion_dataset_test_data",
    ],
    deps = [
        requirement('numpy'),
        requirement('tensorflow'),
        ":transform_utils",
        "//waymo_open_dataset/protos:scenario_proto_py_pb2",
    ],
)

py_library(
    name = "frame_utils",
    srcs = ["frame_utils.py"],
    deps = [
        requirement('numpy'),
        requirement('tensorflow'),
        ":range_image_utils",
        ":transform_utils",
        "//waymo_open_dataset:dataset_proto_py_pb2",
    ],
)

py_library(
    name = "keypoint_data",
    srcs = ["keypoint_data.py"],
    deps = [
        requirement('numpy'),
        requirement('tensorflow'),
        "//waymo_open_dataset:dataset_proto_py_pb2",
        "//waymo_open_dataset:label_proto_py_pb2",
        "//waymo_open_dataset/protos:box_proto_py_pb2",
        "//waymo_open_dataset/protos:keypoint_proto_py_pb2",
        "//waymo_open_dataset/protos:vector_proto_py_pb2",
    ],
)

py_library(
    name = "keypoint_draw",
    srcs = ["keypoint_draw.py"],
    deps = [
        requirement('immutabledict'),
        requirement('matplotlib'),
        requirement('numpy'),
        requirement('plotly'),
        ":keypoint_data",
        "//waymo_open_dataset:label_proto_py_pb2",
        "//waymo_open_dataset/protos:keypoint_proto_py_pb2",
    ],
)

py_library(
    name = "plot_maps",
    srcs = ["plot_maps.py"],
    deps = [
        requirement('pandas'),
        requirement('plotly'),
        "//waymo_open_dataset/protos:map_proto_py_pb2",
    ],
)

py_library(
    name = "keypoint_test_util",
    testonly = True,
    srcs = ["keypoint_test_util.py"],
    deps = [
        "//waymo_open_dataset:label_proto_py_pb2",
        "//waymo_open_dataset/protos:keypoint_proto_py_pb2",
        "//waymo_open_dataset/protos:vector_proto_py_pb2",
    ],
)

py_library(
    name = "camera_segmentation_utils",
    srcs = ["camera_segmentation_utils.py"],
    deps = [
        requirement('immutabledict'),
        requirement('numpy'),
        requirement('scikit-image'),
        requirement('tensorflow'),
        "//waymo_open_dataset:dataset_proto_py_pb2",
        "//waymo_open_dataset/protos:camera_segmentation_proto_py_pb2",
        "//waymo_open_dataset/v2/perception:segmentation",
    ],
)

py_library(
    name = "trajectory_utils",
    srcs = ["trajectory_utils.py"],
    deps = [
        requirement('tensorflow'),
        "//waymo_open_dataset/protos:scenario_proto_py_pb2",
    ],
)

py_library(
    name = "geometry_utils",
    srcs = ["geometry_utils.py"],
    deps = [
        requirement('numpy'),
        requirement('tensorflow'),
    ],
)

py_library(
    name = "womd_lidar_utils",
    srcs = ["womd_lidar_utils.py"],
    deps = [
        requirement('numpy'),
        requirement('tensorflow'),
        ":range_image_utils",
        ":transform_utils",
        "//waymo_open_dataset:dataset_proto_py_pb2",
        "//waymo_open_dataset/protos:compressed_lidar_proto_py_pb2",
        "//waymo_open_dataset/protos:scenario_proto_py_pb2",
        "//waymo_open_dataset/utils/compression:delta_encoder",
    ],
)

py_library(
    name = "womd_camera_utils",
    srcs = ["womd_camera_utils.py"],
    deps = [
        requirement('numpy'),
        "//waymo_open_dataset/protos:scenario_proto_py_pb2",
    ],
)

py_library(
    name = "example_utils",
    srcs = ["example_utils.py"],
    deps = [
        requirement('numpy'),
        requirement('tensorflow'),
    ],
)

# Tests
py_test(
    name = "box_utils_test",
    srcs = ["box_utils_test.py"],
    python_version = "PY3",
    deps = [
        requirement('tensorflow'),
        ":box_utils",
        ":test_utils",
        ":transform_utils",
    ],
)

py_test(
    name = "occupancy_flow_data_test",
    srcs = ["occupancy_flow_data_test.py"],
    python_version = "PY3",
    deps = [
        requirement('tensorflow'),
        ":occupancy_flow_data",
        ":occupancy_flow_test_util",
    ],
)

py_test(
    name = "occupancy_flow_grids_test",
    srcs = ["occupancy_flow_grids_test.py"],
    python_version = "PY3",
    deps = [
        requirement('tensorflow'),
        ":occupancy_flow_data",
        ":occupancy_flow_grids",
        ":occupancy_flow_renderer",
        ":occupancy_flow_test_util",
    ],
)

py_test(
    name = "occupancy_flow_metrics_test",
    srcs = ["occupancy_flow_metrics_test.py"],
    python_version = "PY3",
    deps = [
        requirement('numpy'),
        requirement('tensorflow'),
        ":occupancy_flow_grids",
        ":occupancy_flow_metrics",
        ":occupancy_flow_test_util",
    ],
)

py_test(
    name = "occupancy_flow_renderer_test",
    srcs = ["occupancy_flow_renderer_test.py"],
    python_version = "PY3",
    deps = [
        requirement('numpy'),
        requirement('tensorflow'),
        ":occupancy_flow_data",
        ":occupancy_flow_renderer",
        ":occupancy_flow_test_util",
    ],
)

py_test(
    name = "occupancy_flow_test_util_test",
    srcs = ["occupancy_flow_test_util_test.py"],
    python_version = "PY3",
    deps = [
        requirement('tensorflow'),
        ":occupancy_flow_test_util",
    ],
)

py_test(
    name = "occupancy_flow_vis_test",
    srcs = ["occupancy_flow_vis_test.py"],
    python_version = "PY3",
    deps = [
        requirement('tensorflow'),
        ":occupancy_flow_data",
        ":occupancy_flow_grids",
        ":occupancy_flow_test_util",
        ":occupancy_flow_vis",
    ],
)

py_test(
    name = "range_image_utils_test",
    srcs = ["range_image_utils_test.py"],
    python_version = "PY3",
    deps = [
        requirement('numpy'),
        requirement('tensorflow'),
        ":range_image_utils",
        ":test_utils",
    ],
)

py_test(
    name = "transform_utils_test",
    srcs = ["transform_utils_test.py"],
    python_version = "PY3",
    deps = [
        requirement('tensorflow'),
        ":transform_utils",
    ],
)

py_test(
    name = "keypoint_data_test",
    srcs = ["keypoint_data_test.py"],
    deps = [
        requirement('numpy'),
        requirement('tensorflow'),
        ":keypoint_data",
        ":keypoint_test_util",
        "//waymo_open_dataset:dataset_proto_py_pb2",
        "//waymo_open_dataset:label_proto_py_pb2",
    ],
)

py_test(
    name = "keypoint_draw_test",
    srcs = ["keypoint_draw_test.py"],
    deps = [
        requirement('matplotlib'),
        requirement('numpy'),
        requirement('plotly'),
        requirement('tensorflow'),
        ":keypoint_data",
        ":keypoint_draw",
        ":keypoint_test_util",
        "//waymo_open_dataset:label_proto_py_pb2",
        "//waymo_open_dataset/protos:keypoint_proto_py_pb2",
        "//waymo_open_dataset/protos:vector_proto_py_pb2",
    ],
)

py_test(
    name = "plot_maps_test",
    srcs = ["plot_maps_test.py"],
    deps = [
        requirement('absl-py'),
        ":plot_maps",
        "//waymo_open_dataset/protos:map_proto_py_pb2",
    ],
)

py_test(
    name = "camera_segmentation_utils_test",
    srcs = ["camera_segmentation_utils_test.py"],
    data = [
        "//waymo_open_dataset/utils/testdata:pvps_test_data",
    ],
    deps = [
        requirement('absl-py'),
        requirement('numpy'),
        requirement('tensorflow'),
        ":camera_segmentation_utils",
        "//waymo_open_dataset:dataset_proto_py_pb2",
        "//waymo_open_dataset/protos:camera_segmentation_proto_py_pb2",
        "//waymo_open_dataset/v2/perception/compat_v1:interfaces",
        "//waymo_open_dataset/v2/perception/compat_v1:segmentation",
    ],
)

py_test(
    name = "trajectory_utils_test",
    srcs = ["trajectory_utils_test.py"],
    python_version = "PY3",
    deps = [
        requirement('tensorflow'),
        ":test_utils",
        ":trajectory_utils",
    ],
)

py_test(
    name = "geometry_utils_test",
    srcs = ["geometry_utils_test.py"],
    python_version = "PY3",
    deps = [
        requirement('tensorflow'),
        ":geometry_utils",
    ],
)

py_test(
    name = "womd_lidar_utils_test",
    srcs = ["womd_lidar_utils_test.py"],
    data = ["//waymo_open_dataset/utils/testdata:motion_dataset_test_data"],
    python_version = "PY3",
    deps = [
        requirement('numpy'),
        requirement('tensorflow'),
        ":womd_lidar_utils",
        "//waymo_open_dataset:dataset_proto_py_pb2",
        "//waymo_open_dataset/protos:compressed_lidar_proto_py_pb2",
        "//waymo_open_dataset/protos:scenario_proto_py_pb2",
    ],
)

py_test(
    name = "womd_camera_utils_test",
    srcs = ["womd_camera_utils_test.py"],
    python_version = "PY3",
    deps = [
        requirement('numpy'),
        requirement('tensorflow'),
        ":womd_camera_utils",
        "//waymo_open_dataset/protos:camera_tokens_proto_py_pb2",
        "//waymo_open_dataset/protos:scenario_proto_py_pb2",
    ],
)

py_test(
    name = "example_utils_test",
    srcs = ["example_utils_test.py"],
    python_version = "PY3",
    deps = [
        requirement('absl-py'),
        requirement('tensorflow'),
        ":example_utils",
    ],
)
