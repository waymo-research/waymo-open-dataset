# Waymo open dataset

load("@rules_license//rules:license.bzl", "license")
load("//tf:build_config.bzl", "all_proto_library")



package(
    default_applicable_licenses = ["//waymo_open_dataset:license"],
    default_visibility = ["//visibility:public"],
)

license(
    name = "license",
    package_name = "waymo_open_dataset",
)

licenses(["notice"])

exports_files(["LICENSE"])

all_proto_library(
    src = "dataset.proto",
    deps = [
        ":label_proto",
        "//waymo_open_dataset/protos:map_proto",
        "//waymo_open_dataset/protos:vector_proto",
    ],
)



all_proto_library(
    src = "label.proto",
    deps = ["//waymo_open_dataset/protos:keypoint_proto"],
)


