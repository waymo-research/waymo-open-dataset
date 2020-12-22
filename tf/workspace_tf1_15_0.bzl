"""Install dependencies."""

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

def load_tf_version_specific_dependencies():
    """Load TF version specific dependencies."""
    http_archive(
        name = "com_google_absl",
        sha256 = "56cd3fbbbd94468a5fff58f5df2b6f9de7a0272870c61f6ca05b869934f4802a",
        strip_prefix = "abseil-cpp-daf381e8535a1f1f1b8a75966a74e7cca63dee89",
        urls = [
            "https://github.com/abseil/abseil-cpp/archive/daf381e8535a1f1f1b8a75966a74e7cca63dee89.tar.gz",
        ],
    )

    http_archive(
        name = "zlib",
        build_file = "//third_party:zlib.BUILD",
        sha256 = "c3e5e9fdd5004dcb542feda5ee4f0ff0744628baf8ed2dd5d66f8ca1197cb1a1",
        strip_prefix = "zlib-1.2.11",
        urls = [
            "http://mirror.tensorflow.org/zlib.net/zlib-1.2.11.tar.gz",
            "https://zlib.net/zlib-1.2.11.tar.gz",
        ],
    )

    PROTOBUF_URLS = [
        "https://storage.googleapis.com/mirror.tensorflow.org/github.com/protocolbuffers/protobuf/archive/310ba5ee72661c081129eb878c1bbcec936b20f0.tar.gz",
        "https://github.com/protocolbuffers/protobuf/archive/310ba5ee72661c081129eb878c1bbcec936b20f0.tar.gz",
    ]

    PROTOBUF_SHA256 = "b9e92f9af8819bbbc514e2902aec860415b70209f31dfc8c4fa72515a5df9d59"

    PROTOBUF_STRIP_PREFIX = "protobuf-310ba5ee72661c081129eb878c1bbcec936b20f0"

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
