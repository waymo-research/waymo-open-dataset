# Waymo open dataset

load("//tf:build_config.bzl", "all_proto_library")



package(
    default_visibility = ["//visibility:public"],
)

licenses(["notice"])

exports_files(["LICENSE"])

all_proto_library(
    src = "dataset.proto",
    deps = [":label_proto"],
)



all_proto_library(
    src = "label.proto",
    deps = ["//waymo_open_dataset/protos:keypoint_proto"],
)


