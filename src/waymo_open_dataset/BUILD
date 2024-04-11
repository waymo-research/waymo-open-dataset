# Waymo open dataset

load("@rules_license//rules:license.bzl", "license")
load("//waymo_open_dataset/protos:defs.bzl", "all_proto_library")

load("@rules_python//python:pip.bzl", "compile_pip_requirements")

package(
    default_applicable_licenses = ["//waymo_open_dataset:license"],
    default_visibility = ["//visibility:public"],
)

license(
    name = "license",
    package_name = "waymo_open_dataset",
    license_kinds = ["@rules_license//licenses/generic:notice"],
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

compile_pip_requirements(
    name = "requirements",
    extra_args = [
        "--resolver=backtracking",
        "--allow-unsafe",
    ],
    requirements_in = "requirements.in",
    requirements_txt = "requirements.txt",
    timeout = "moderate"
)
