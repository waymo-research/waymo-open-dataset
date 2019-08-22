package(
    default_visibility = [
        "//visibility:public",
    ],
)

licenses(["notice"])  # Apache 2.0

exports_files(["BUILD"])

load("//tf:build_config.bzl", "all_proto_library")


all_proto_library(
    src = "metrics.proto",
    deps = [
        ":breakdown_proto",
        "//waymo_open_dataset:label_proto",
    ],
)


all_proto_library(
    src = "breakdown.proto",
    deps = [
        "//waymo_open_dataset:label_proto",
    ],
)
