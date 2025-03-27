load("@wod_deps//:requirements.bzl", "requirement")

package(
    default_applicable_licenses = ["//waymo_open_dataset:license"],
    default_visibility = [
        "//visibility:public",
    ],
)

filegroup(
    name = "challenge_config",
    srcs = [
        "challenge_2024_config.textproto",
        "challenge_2025_scenario_gen_config.textproto",
        "challenge_2025_sim_agents_config.textproto",
    ],
)

# ==== PYTHON LIBRARIES ====
py_library(
    name = "estimators",
    srcs = ["estimators.py"],
    deps = [
        requirement('numpy'),
        requirement('scikit-learn'),
        requirement('tensorflow'),
        requirement('tensorflow_probability'),
        "//waymo_open_dataset/protos:sim_agents_metrics_proto_py_pb2",
    ],
)

py_library(
    name = "interaction_features",
    srcs = ["interaction_features.py"],
    deps = [
        requirement('tensorflow'),
        ":trajectory_features",
        "//waymo_open_dataset/utils:box_utils",
        "//waymo_open_dataset/utils:geometry_utils",
    ],
)

py_library(
    name = "map_metric_features",
    srcs = ["map_metric_features.py"],
    deps = [
        requirement('tensorflow'),
        "//waymo_open_dataset/protos:map_proto_py_pb2",
        "//waymo_open_dataset/utils:box_utils",
        "//waymo_open_dataset/utils:geometry_utils",
    ],
)

py_library(
    name = "metric_features",
    srcs = ["metric_features.py"],
    deps = [
        requirement('tensorflow'),
        ":interaction_features",
        ":map_metric_features",
        ":traffic_light_features",
        ":trajectory_features",
        "//waymo_open_dataset/protos:map_proto_py_pb2",
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
        requirement('numpy'),
        requirement('protobuf'),
        requirement('tensorflow'),
        ":estimators",
        ":metric_features",
        ":trajectory_features",
        "//waymo_open_dataset/protos:scenario_proto_py_pb2",
        "//waymo_open_dataset/protos:sim_agents_metrics_proto_py_pb2",
        "//waymo_open_dataset/protos:sim_agents_submission_proto_py_pb2",
        "//waymo_open_dataset/utils/sim_agents:submission_specs",
    ],
)

py_library(
    name = "trajectory_features",
    srcs = ["trajectory_features.py"],
    deps = [
        requirement('numpy'),
        requirement('tensorflow'),
    ],
)

py_library(
    name = "traffic_light_features",
    srcs = ["traffic_light_features.py"],
    deps = [
        requirement('numpy'),
        requirement('tensorflow'),
        ":map_metric_features",
        "//waymo_open_dataset/protos:map_proto_py_pb2",
        "//waymo_open_dataset/utils:box_utils",
        "//waymo_open_dataset/utils:geometry_utils",
    ],
)

# ==== PYTHON TESTS ====
py_test(
    name = "estimators_test",
    srcs = ["estimators_test.py"],
    deps = [
        requirement('absl-py'),
        requirement('numpy'),
        requirement('tensorflow'),
        ":estimators",
        "//waymo_open_dataset/protos:sim_agents_metrics_proto_py_pb2",
    ],
)

py_test(
    name = "interaction_features_test",
    srcs = ["interaction_features_test.py"],
    deps = [
        requirement('absl-py'),
        requirement('numpy'),
        requirement('tensorflow'),
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
        requirement('tensorflow'),
        ":map_metric_features",
        "//waymo_open_dataset/utils:test_utils",
        "//waymo_open_dataset/utils/sim_agents:converters",
        "//waymo_open_dataset/utils/sim_agents:test_utils",
    ],
)

py_test(
    name = "metric_features_test",
    timeout = "long",
    srcs = ["metric_features_test.py"],
    exec_properties = {"mem": "28g"},
    shard_count = 8,
    deps = [
        requirement('absl-py'),
        requirement('tensorflow'),
        ":interaction_features",
        ":map_metric_features",
        ":metric_features",
        "//waymo_open_dataset/protos:sim_agents_submission_proto_py_pb2",
        "//waymo_open_dataset/utils:test_utils",
        "//waymo_open_dataset/utils/sim_agents:converters",
        "//waymo_open_dataset/utils/sim_agents:submission_specs",
        "//waymo_open_dataset/utils/sim_agents:test_utils",
    ],
)

py_test(
    name = "metrics_test",
    size = "enormous",
    timeout = "long",
    srcs = ["metrics_test.py"],
    exec_properties = {"mem": "28g"},
    shard_count = 8,
    deps = [
        requirement('absl-py'),
        requirement('tensorflow'),
        ":metric_features",
        ":metrics",
        "//waymo_open_dataset/protos:sim_agents_metrics_proto_py_pb2",
        "//waymo_open_dataset/protos:sim_agents_submission_proto_py_pb2",
        "//waymo_open_dataset/utils:test_utils",
        "//waymo_open_dataset/utils/sim_agents:converters",
        "//waymo_open_dataset/utils/sim_agents:submission_specs",
        "//waymo_open_dataset/utils/sim_agents:test_utils",
    ],
)

py_test(
    name = "trajectory_features_test",
    srcs = ["trajectory_features_test.py"],
    deps = [
        requirement('absl-py'),
        requirement('numpy'),
        requirement('tensorflow'),
        ":trajectory_features",
    ],
)

py_test(
    name = "traffic_light_features_test",
    srcs = ["traffic_light_features_test.py"],
    deps = [
        requirement('numpy'),
        requirement('tensorflow'),
        ":traffic_light_features",
        "//waymo_open_dataset/protos:map_proto_py_pb2",
        "//waymo_open_dataset/utils:test_utils",
        "//waymo_open_dataset/utils:trajectory_utils",
        "//waymo_open_dataset/utils/sim_agents:converters",
        "//waymo_open_dataset/utils/sim_agents:test_utils",
    ],
)
