# pytype
load("//waymo_open_dataset/utils:pytype.bzl", "pytype_strict_library")

# strict
load("//waymo_open_dataset/utils:pytype.bzl", "py_strict_test")

package(
    default_visibility = [
        "//visibility:public",
    ],
)

licenses(["notice"])  # Apache 2.0

exports_files(["BUILD"])

# Py libs.

pytype_strict_library(
    name = "box_utils",
    srcs = ["box_utils.py"],
    deps = [
        ":transform_utils",
        # Implicit tensorflow dependency.
        "//waymo_open_dataset:label_proto_py_pb2",
    ],
)

filegroup(
    name = "occupancy_flow_test_data",
    srcs = [
        "testdata/motion_data_one_example.tfrecord",
    ],
)

pytype_strict_library(
    name = "occupancy_flow_data",
    srcs = ["occupancy_flow_data.py"],
    deps = [
        # Implicit tensorflow dependency.
        "//waymo_open_dataset/protos:scenario_proto_py_pb2",
    ],
)

pytype_strict_library(
    name = "occupancy_flow_grids",
    srcs = ["occupancy_flow_grids.py"],
    deps = [
        ":occupancy_flow_data",
        ":occupancy_flow_renderer",
        # Implicit tensorflow dependency.
        "//waymo_open_dataset/protos:occupancy_flow_metrics_proto_py_pb2",
        "//waymo_open_dataset/protos:scenario_proto_py_pb2",
    ],
)

pytype_strict_library(
    name = "occupancy_flow_metrics",
    srcs = ["occupancy_flow_metrics.py"],
    deps = [
        ":occupancy_flow_grids",
        # Implicit tensorflow dependency.
        # Implicit tensorflow_graphics/image:transformer dependency.
        "//waymo_open_dataset/protos:occupancy_flow_metrics_proto_py_pb2",
    ],
)

pytype_strict_library(
    name = "occupancy_flow_renderer",
    srcs = ["occupancy_flow_renderer.py"],
    deps = [
        ":occupancy_flow_data",
        # Implicit numpy dependency.
        # Implicit tensorflow dependency.
        "//waymo_open_dataset/protos:occupancy_flow_metrics_proto_py_pb2",
        "//waymo_open_dataset/protos:scenario_proto_py_pb2",
    ],
)

pytype_strict_library(
    name = "occupancy_flow_test_util",
    srcs = ["occupancy_flow_test_util.py"],
    data = [
        ":occupancy_flow_test_data",
    ],
    deps = [
        ":occupancy_flow_data",
        # Implicit python proto dependency.
        # Implicit resources dependency.
        # Implicit tensorflow dependency.
        "//waymo_open_dataset/protos:occupancy_flow_metrics_proto_py_pb2",
    ],
)

pytype_strict_library(
    name = "occupancy_flow_vis",
    srcs = ["occupancy_flow_vis.py"],
    deps = [
        ":occupancy_flow_data",
        # Implicit tensorflow dependency.
    ],
)

pytype_strict_library(
    name = "range_image_utils",
    srcs = ["range_image_utils.py"],
    deps = [
        # Implicit numpy dependency.
        # Implicit tensorflow dependency.
    ],
)

pytype_strict_library(
    name = "transform_utils",
    srcs = ["transform_utils.py"],
    deps = [
        # Implicit tensorflow dependency.
    ],
)

pytype_strict_library(
    name = "test_utils",
    testonly = 1,
    srcs = ["test_utils.py"],
    deps = [
        ":transform_utils",
        # Implicit numpy dependency.
        # Implicit tensorflow dependency.
    ],
)

pytype_strict_library(
    name = "frame_utils",
    srcs = ["frame_utils.py"],
    deps = [
        ":range_image_utils",
        ":transform_utils",
        # Implicit numpy dependency.
        # Implicit tensorflow dependency.
        "//waymo_open_dataset:dataset_proto_py_pb2",
    ],
)

pytype_strict_library(
    name = "keypoint_data",
    srcs = ["keypoint_data.py"],
    deps = [
        # Implicit numpy dependency.
        # Implicit tensorflow dependency.
        "//waymo_open_dataset:dataset_proto_py_pb2",
        "//waymo_open_dataset:label_proto_py_pb2",
        "//waymo_open_dataset/protos:keypoint_proto_py_pb2",
    ],
)

pytype_strict_library(
    name = "keypoint_draw",
    srcs = ["keypoint_draw.py"],
    deps = [
        ":keypoint_data",
        # Implicit immutabledict dependency.
        # Implicit matplotlib dependency.
        # Implicit numpy dependency.
        # Implicit plotly dependency.
        "//waymo_open_dataset:label_proto_py_pb2",
        "//waymo_open_dataset/protos:keypoint_proto_py_pb2",
    ],
)

pytype_strict_library(
    name = "keypoint_test_util",
    testonly = True,
    srcs = ["keypoint_test_util.py"],
    deps = [
        "//waymo_open_dataset:label_proto_py_pb2",
        "//waymo_open_dataset/protos:keypoint_proto_py_pb2",
    ],
)

# Tests
py_strict_test(
    name = "box_utils_test",
    srcs = ["box_utils_test.py"],
    python_version = "PY3",
    deps = [
        ":box_utils",
        ":test_utils",
        ":transform_utils",
        # Implicit tensorflow dependency.
    ],
)

py_strict_test(
    name = "occupancy_flow_data_test",
    srcs = ["occupancy_flow_data_test.py"],
    python_version = "PY3",
    deps = [
        ":occupancy_flow_data",
        ":occupancy_flow_test_util",
        # Implicit tensorflow dependency.
    ],
)

py_strict_test(
    name = "occupancy_flow_grids_test",
    srcs = ["occupancy_flow_grids_test.py"],
    python_version = "PY3",
    deps = [
        ":occupancy_flow_data",
        ":occupancy_flow_grids",
        ":occupancy_flow_renderer",
        ":occupancy_flow_test_util",
        # Implicit tensorflow dependency.
    ],
)

py_strict_test(
    name = "occupancy_flow_metrics_test",
    srcs = ["occupancy_flow_metrics_test.py"],
    python_version = "PY3",
    deps = [
        ":occupancy_flow_grids",
        ":occupancy_flow_metrics",
        ":occupancy_flow_test_util",
        # Implicit numpy dependency.
        # Implicit tensorflow dependency.
    ],
)

py_strict_test(
    name = "occupancy_flow_renderer_test",
    srcs = ["occupancy_flow_renderer_test.py"],
    python_version = "PY3",
    deps = [
        ":occupancy_flow_data",
        ":occupancy_flow_renderer",
        ":occupancy_flow_test_util",
        # Implicit numpy dependency.
        # Implicit tensorflow dependency.
    ],
)

py_strict_test(
    name = "occupancy_flow_test_util_test",
    srcs = ["occupancy_flow_test_util_test.py"],
    python_version = "PY3",
    deps = [
        ":occupancy_flow_test_util",
        # Implicit tensorflow dependency.
    ],
)

py_strict_test(
    name = "occupancy_flow_vis_test",
    srcs = ["occupancy_flow_vis_test.py"],
    python_version = "PY3",
    deps = [
        ":occupancy_flow_data",
        ":occupancy_flow_grids",
        ":occupancy_flow_test_util",
        ":occupancy_flow_vis",
        # Implicit tensorflow dependency.
    ],
)

py_strict_test(
    name = "range_image_utils_test",
    srcs = ["range_image_utils_test.py"],
    python_version = "PY3",
    deps = [
        ":range_image_utils",
        ":test_utils",
        # Implicit numpy dependency.
        # Implicit tensorflow dependency.
    ],
)

py_strict_test(
    name = "transform_utils_test",
    srcs = ["transform_utils_test.py"],
    python_version = "PY3",
    deps = [
        ":transform_utils",
        # Implicit tensorflow dependency.
    ],
)

py_strict_test(
    name = "keypoint_data_test",
    srcs = ["keypoint_data_test.py"],
    deps = [
        ":keypoint_data",
        ":keypoint_test_util",
        # Implicit numpy dependency.
        # Implicit tensorflow dependency.
        "//waymo_open_dataset:dataset_proto_py_pb2",
        "//waymo_open_dataset:label_proto_py_pb2",
    ],
)

py_strict_test(
    name = "keypoint_draw_test",
    srcs = ["keypoint_draw_test.py"],
    deps = [
        ":keypoint_data",
        ":keypoint_draw",
        ":keypoint_test_util",
        # Implicit matplotlib dependency.
        # Implicit numpy dependency.
        # Implicit plotly dependency.
        # Implicit tensorflow dependency.
        "//waymo_open_dataset:label_proto_py_pb2",
        "//waymo_open_dataset/protos:keypoint_proto_py_pb2",
    ],
)
