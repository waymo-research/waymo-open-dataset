package(
    default_applicable_licenses = ["//waymo_open_dataset:license"],
    default_visibility = [
        "//visibility:public",
    ],
)

filegroup(
    name = "challenge_config",
    srcs = [
        "challenge_config.textproto",
    ],
)

# ==== PYTHON LIBRARIES ====
py_library(
    name = "estimators",
    srcs = ["estimators.py"],
    deps = ["//waymo_open_dataset/protos:sim_agents_metrics_proto_py_pb2"],
)

py_library(
    name = "interaction_features",
    srcs = ["interaction_features.py"],
    deps = [
        ":trajectory_features",
        "//waymo_open_dataset/utils:box_utils",
        "//waymo_open_dataset/utils:geometry_utils",
    ],
)

py_library(
    name = "map_metric_features",
    srcs = ["map_metric_features.py"],
    deps = [
        ":trajectory_features",
        "//waymo_open_dataset/protos:map_proto_py_pb2",
        "//waymo_open_dataset/utils:box_utils",
        "//waymo_open_dataset/utils:geometry_utils",
    ],
)

py_library(
    name = "metric_features",
    srcs = ["metric_features.py"],
    deps = [
        ":interaction_features",
        ":map_metric_features",
        ":trajectory_features",
        "//waymo_open_dataset/protos:scenario_proto_py_pb2",
        "//waymo_open_dataset/protos:sim_agents_submission_proto_py_pb2",
        "//waymo_open_dataset/utils:trajectory_utils",
        "//waymo_open_dataset/utils/sim_agents:converters",
        "//waymo_open_dataset/utils/sim_agents:submission_specs",
    ],
)

py_library(
    name = "metrics",
    srcs = ["metrics.py"],
    data = [
        ":challenge_config",
    ],
    deps = [
        ":estimators",
        ":metric_features",
        "//waymo_open_dataset/protos:scenario_proto_py_pb2",
        "//waymo_open_dataset/protos:sim_agents_metrics_proto_py_pb2",
        "//waymo_open_dataset/protos:sim_agents_submission_proto_py_pb2",
    ],
)

py_library(
    name = "trajectory_features",
    srcs = ["trajectory_features.py"],
)

# ==== PYTHON TESTS ====
py_test(
    name = "estimators_test",
    srcs = ["estimators_test.py"],
    deps = [
        ":estimators",
        "//waymo_open_dataset/protos:sim_agents_metrics_proto_py_pb2",
    ],
)

py_test(
    name = "interaction_features_test",
    srcs = ["interaction_features_test.py"],
    deps = [
        ":interaction_features",
        "//waymo_open_dataset/utils:test_utils",
        "//waymo_open_dataset/utils/sim_agents:converters",
        "//waymo_open_dataset/utils/sim_agents:test_utils",
    ],
)

py_test(
    name = "map_metric_features_test",
    srcs = ["map_metric_features_test.py"],
    deps = [
        ":map_metric_features",
        "//waymo_open_dataset/utils:test_utils",
        "//waymo_open_dataset/utils/sim_agents:converters",
        "//waymo_open_dataset/utils/sim_agents:test_utils",
    ],
)

py_test(
    name = "metric_features_test",
    srcs = ["metric_features_test.py"],
    deps = [
        ":metric_features",
        "//waymo_open_dataset/utils:test_utils",
        "//waymo_open_dataset/utils/sim_agents:converters",
        "//waymo_open_dataset/utils/sim_agents:test_utils",
    ],
)

py_test(
    name = "metrics_test",
    srcs = ["metrics_test.py"],
    deps = [
        ":metrics",
        "//waymo_open_dataset/protos:sim_agents_metrics_proto_py_pb2",
        "//waymo_open_dataset/utils:test_utils",
        "//waymo_open_dataset/utils/sim_agents:test_utils",
    ],
)

py_test(
    name = "trajectory_features_test",
    srcs = ["trajectory_features_test.py"],
    python_version = "PY3",
    deps = [":trajectory_features"],
)
