"""Install dependencies."""

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

def load_tf_version_specific_dependencies():
    """Load TF version specific dependencies."""

    http_archive(
        name = "zlib_archive",
        build_file = "//third_party:zlib.BUILD",
        sha256 = "c3e5e9fdd5004dcb542feda5ee4f0ff0744628baf8ed2dd5d66f8ca1197cb1a1",
        strip_prefix = "zlib-1.2.11",
        urls = [
            "http://mirror.tensorflow.org/zlib.net/zlib-1.2.11.tar.gz",
            "https://zlib.net/zlib-1.2.11.tar.gz",
        ],
    )

    native.bind(
        name = "zlib",
        actual = "@zlib_archive//:zlib",
    )

    # This proto version is the same as tensorflow 1.14.0. If you are using a
    # different tensorflow version, update these based on
    # https://github.com/tensorflow/tensorflow/blob/{YOUR_TF_VERSION}/tensorflow/workspace.bzl
    #
    # 5902e759108d14ee8e6b0b07653dac2f4e70ac73 is based on 3.7.1 with a fix for BUILD file.
    PROTOBUF_URLS = [
        "http://mirror.tensorflow.org/github.com/protocolbuffers/protobuf/archive/5902e759108d14ee8e6b0b07653dac2f4e70ac73.tar.gz",
        "https://github.com/protocolbuffers/protobuf/archive/5902e759108d14ee8e6b0b07653dac2f4e70ac73.tar.gz",
    ]

    PROTOBUF_SHA256 = "1c020fafc84acd235ec81c6aac22d73f23e85a700871466052ff231d69c1b17a"

    PROTOBUF_STRIP_PREFIX = "protobuf-5902e759108d14ee8e6b0b07653dac2f4e70ac73"

    # We need to import the protobuf library under the names com_google_protobuf
    # and com_google_protobuf_cc to enable proto_library support in bazel.
    http_archive(
        name = "com_google_protobuf",
        sha256 = PROTOBUF_SHA256,
        strip_prefix = PROTOBUF_STRIP_PREFIX,
        urls = PROTOBUF_URLS,
    )

    http_archive(
        name = "com_google_protobuf_cc",
        sha256 = PROTOBUF_SHA256,
        strip_prefix = PROTOBUF_STRIP_PREFIX,
        urls = PROTOBUF_URLS,
    )
