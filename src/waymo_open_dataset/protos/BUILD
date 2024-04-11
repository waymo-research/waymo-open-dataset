

load(":defs.bzl", "all_proto_library")

package(
    default_applicable_licenses = ["//waymo_open_dataset:license"],
    default_visibility = [
        "//visibility:public",
    ],
)

licenses(["notice"])

exports_files(["BUILD"])

all_proto_library(
    src = "breakdown.proto",
    deps = [
        "//waymo_open_dataset:label_proto",
    ],
)

all_proto_library(
    src = "map.proto",
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
    src = "segmentation_metrics.proto",
    deps = [
        ":segmentation_proto",
        "//waymo_open_dataset:dataset_proto",
    ],
)

all_proto_library(
    src = "segmentation_submission.proto",
    deps = [
        ":segmentation_metrics_proto",
    ],
)

all_proto_library(
    src = "camera_segmentation_metrics.proto",
    deps = [
        ":camera_segmentation_proto",
        "//waymo_open_dataset:dataset_proto",
    ],
)

all_proto_library(
    src = "camera_segmentation_submission.proto",
    deps = [
        ":camera_segmentation_metrics_proto",
    ],
)

all_proto_library(
    src = "motion_submission.proto",
)

all_proto_library(
    src = "occupancy_flow_metrics.proto",
)

all_proto_library(
    src = "occupancy_flow_submission.proto",
)

all_proto_library(
    src = "scenario.proto",
    deps = [
        ":camera_tokens_proto",
        ":compressed_lidar_proto",
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

all_proto_library(
    src = "vector.proto",
)

all_proto_library(
    src = "box.proto",
    deps = [":vector_proto"],
)

all_proto_library(
    src = "keypoint.proto",
    deps = [":vector_proto"],
)

all_proto_library(
    src = "segmentation.proto",
)

all_proto_library(
    src = "camera_segmentation.proto",
)

all_proto_library(
    src = "conversion_config.proto",
    deps = [
        ":scenario_proto",
    ],
)

all_proto_library(
    src = "sim_agents_submission.proto",
)

all_proto_library(
    src = "sim_agents_metrics.proto",
)

all_proto_library(
    src = "keypoints_metrics.proto",
)

all_proto_library(
    src = "keypoints_submission.proto",
    deps = [
        ":box_proto",
        ":keypoint_proto",
    ],
)

all_proto_library(
    src = "compressed_lidar.proto",
    deps = [
        "//waymo_open_dataset:dataset_proto",
    ],
)

all_proto_library(
    src = "camera_tokens.proto",
    deps = [
        "//waymo_open_dataset:dataset_proto",
    ],
)


