load("@wod_deps//:requirements.bzl", "requirement")

package(
    default_applicable_licenses = ["//waymo_open_dataset:license"],
    default_visibility = [
        "//visibility:public",
    ],
)

licenses(["notice"])

exports_files(["BUILD"])

# ==== PYTHON LIBRARIES ====
py_library(
    name = "submission_specs",
    srcs = ["submission_specs.py"],
    deps = [
        "//waymo_open_dataset/protos:scenario_proto_py_pb2",
        "//waymo_open_dataset/protos:sim_agents_submission_proto_py_pb2",
    ],
)

py_library(
    name = "converters",
    srcs = ["converters.py"],
    deps = [
        requirement('tensorflow'),
        ":submission_specs",
        "//waymo_open_dataset/protos:scenario_proto_py_pb2",
        "//waymo_open_dataset/protos:sim_agents_submission_proto_py_pb2",
        "//waymo_open_dataset/utils:trajectory_utils",
    ],
)

py_library(
    name = "test_utils",
    srcs = ["test_utils.py"],
    data = [
        ":sim_agents_test_data",
    ],
    deps = [
        requirement('protobuf'),
        ":submission_specs",
        "//waymo_open_dataset/protos:scenario_proto_py_pb2",
        "//waymo_open_dataset/protos:sim_agents_metrics_proto_py_pb2",
        "//waymo_open_dataset/protos:sim_agents_submission_proto_py_pb2",
    ],
)

py_library(
    name = "visualizations",
    srcs = ["visualizations.py"],
    deps = [
        requirement('matplotlib'),
        requirement('numpy'),
        requirement('tensorflow'),
        "//waymo_open_dataset/protos:map_proto_py_pb2",
        "//waymo_open_dataset/protos:scenario_proto_py_pb2",
    ],
)

# ==== TEST DATA ====
filegroup(
    name = "sim_agents_test_data",
    srcs = [
        "testdata/test_config.textproto",
        "testdata/test_config_dependent_timesteps.textproto",
        "testdata/test_submission.binproto",
    ],
)

# ==== PYTHON TESTS ====
py_test(
    name = "submission_specs_test",
    srcs = ["submission_specs_test.py"],
    python_version = "PY3",
    deps = [
        requirement('tensorflow'),
        ":submission_specs",
        ":test_utils",
        "//waymo_open_dataset/protos:scenario_proto_py_pb2",
        "//waymo_open_dataset/protos:sim_agents_submission_proto_py_pb2",
        "//waymo_open_dataset/utils:test_utils",
    ],
)

py_test(
    name = "converters_test",
    srcs = ["converters_test.py"],
    python_version = "PY3",
    deps = [
        requirement('absl-py'),
        requirement('tensorflow'),
        ":converters",
        ":submission_specs",
        "//waymo_open_dataset/utils:test_utils",
    ],
)
