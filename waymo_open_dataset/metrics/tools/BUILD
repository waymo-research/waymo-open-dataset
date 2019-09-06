package(
    default_visibility = [
        "//visibility:public",
    ],
)

licenses(["notice"])  # Apache 2.0

filegroup(
    name = "fakedata",
    srcs = [
        "fake_ground_truths.bin",
        "fake_predictions.bin",
    ],
)

cc_binary(
    name = "compute_detection_metrics_main",
    srcs = ["compute_detection_metrics_main.cc"],
    deps = [
        "//waymo_open_dataset:label_cc_proto",
        "//waymo_open_dataset/common:integral_types",
        "//waymo_open_dataset/metrics:config_util",
        "//waymo_open_dataset/metrics:detection_metrics",
        "//waymo_open_dataset/protos:breakdown_cc_proto",
        "//waymo_open_dataset/protos:metrics_cc_proto",
        "@com_google_absl//absl/strings",
    ],
)
