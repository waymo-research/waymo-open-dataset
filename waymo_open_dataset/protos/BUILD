load("//tf:build_config.bzl", "all_proto_library")


package(
    default_visibility = [
        "//visibility:public",
    ],
)

licenses(["notice"])  # Apache 2.0

exports_files(["BUILD"])

all_proto_library(
    src = "breakdown.proto",
    deps = [
        "//waymo_open_dataset:label_proto",
    ],
)

all_proto_library(
    src = "map.proto",
    deps = [
        "//waymo_open_dataset:dataset_proto",
    ],
)

all_proto_library(
    src = "metrics.proto",
    deps = [
        ":breakdown_proto",
        ":scenario_proto",
        "//waymo_open_dataset:dataset_proto",
        "//waymo_open_dataset:label_proto",
    ],
)

all_proto_library(
    src = "motion_metrics.proto",
    deps = [
        ":scenario_proto",
    ],
)

all_proto_library(
    src = "motion_submission.proto",
)

all_proto_library(
    src = "scenario.proto",
    deps = [
        ":map_proto",
    ],
)

all_proto_library(
    src = "submission.proto",
    deps = [
        ":metrics_proto",
        "//waymo_open_dataset:label_proto",
    ],
)

