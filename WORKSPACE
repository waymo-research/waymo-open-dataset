"""Install dependencies."""

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
load("//tf:tf_configure.bzl", "tf_configure")

# abseil-cpp
http_archive(
    name = "com_google_absl",
    sha256 = "71d00d15fe6370220b6685552fb66e5814f4dd2e130f3836fc084c894943753f",
    strip_prefix = "abseil-cpp-7c7754fb3ed9ffb57d35fe8658f3ba4d73a31e72",
    urls = ["https://github.com/abseil/abseil-cpp/archive/7c7754fb3ed9ffb57d35fe8658f3ba4d73a31e72.zip"],  # 2019-03-14
)

# googletest
http_archive(
    name = "com_google_googletest",
    build_file_content = """
cc_library(
    name = "gtest",
    srcs = [
          "googletest/src/gtest-all.cc",
          "googlemock/src/gmock-all.cc",
    ],
    hdrs = glob([
        "**/*.h",
        "googletest/src/*.cc",
        "googlemock/src/*.cc",
    ]),
    includes = [
        "googlemock",
        "googletest",
        "googletest/include",
        "googlemock/include",
    ],
    linkopts = ["-pthread"],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "gtest_main",
    srcs = ["googlemock/src/gmock_main.cc"],
    linkopts = ["-pthread"],
    visibility = ["//visibility:public"],
    deps = [":gtest"],
)
""",
    sha256 = "58a6f4277ca2bc8565222b3bbd58a177609e9c488e8a72649359ba51450db7d8",
    strip_prefix = "googletest-release-1.8.0",
    urls = [
        "https://github.com/google/googletest/archive/release-1.8.0.tar.gz",
    ],
)

# gflags
http_archive(
    name = "com_github_gflags_gflags",
    sha256 = "19713a36c9f32b33df59d1c79b4958434cb005b5b47dc5400a7a4b078111d9b5",
    strip_prefix = "gflags-2.2.2",
    urls = [
        "https://github.com/gflags/gflags/archive/v2.2.2.zip",
    ],
)

# glog
http_archive(
    name = "com_google_glog",
    build_file = "//third_party:glog.BUILD",
    sha256 = "f28359aeba12f30d73d9e4711ef356dc842886968112162bc73002645139c39c",
    strip_prefix = "glog-0.4.0",
    urls = [
        "https://github.com/google/glog/archive/v0.4.0.tar.gz",
    ],
)

# openssl
http_archive(
    name = "openssl",
    build_file_content = """
config_setting(
    name = "darwin",
    values = {
        "cpu": "darwin_x86_64",
    },
)

cc_library(
    name = "crypto",
    srcs = ["libcrypto.a"],
    hdrs = glob(["include/openssl/*.h"]) + ["include/openssl/opensslconf.h"],
    includes = ["include"],
    linkopts = select({
        ":darwin": [],
        "//conditions:default": [
            "-lpthread",
            "-ldl",
        ],
    }),
    visibility = ["//visibility:public"],
)

cc_library(
    name = "ssl",
    srcs = ["libssl.a"],
    hdrs = glob(["include/openssl/*.h"]) + ["include/openssl/opensslconf.h"],
    includes = ["include"],
    visibility = ["//visibility:public"],
    deps = [":crypto"],
)

genrule(
    name = "openssl-build",
    srcs = glob(
        ["**/*"],
        exclude = ["bazel-*"],
    ),
    outs = [
        "libcrypto.a",
        "libssl.a",
        "include/openssl/opensslconf.h",
    ],
    cmd = \"\"\"
        OPENSSL_ROOT=$$(dirname $(location config))
        pushd $$OPENSSL_ROOT
            ./config
            make
        popd
        cp $$OPENSSL_ROOT/libcrypto.a $(location libcrypto.a)
        cp $$OPENSSL_ROOT/libssl.a $(location libssl.a)
        cp $$OPENSSL_ROOT/include/openssl/opensslconf.h $(location include/openssl/opensslconf.h)
    \"\"\",
)
""",
    sha256 = "f56dd7d81ce8d3e395f83285bd700a1098ed5a4cb0a81ce9522e41e6db7e0389",
    strip_prefix = "openssl-OpenSSL_1_1_0h",
    url = "https://github.com/openssl/openssl/archive/OpenSSL_1_1_0h.tar.gz",
)

# bazel-skylib
skylib_version = "0.8.0"

http_archive(
    name = "bazel_skylib",
    sha256 = "2ef429f5d7ce7111263289644d233707dba35e39696377ebab8b0bc701f7818e",
    type = "tar.gz",
    url = "https://github.com/bazelbuild/bazel-skylib/releases/download/{}/bazel-skylib.{}.tar.gz".format(skylib_version, skylib_version),
)

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

http_archive(
    name = "six_archive",
    build_file = "//third_party:six.BUILD",
    sha256 = "105f8d68616f8248e24bf0e9372ef04d3cc10104f1980f54d57b2ce73a5ad56a",
    strip_prefix = "six-1.10.0",
    urls = [
        "http://mirror.tensorflow.org/pypi.python.org/packages/source/s/six/six-1.10.0.tar.gz",
        "https://pypi.python.org/packages/source/s/six/six-1.10.0.tar.gz",
    ],
)

# Needed by Protobuf
bind(
    name = "six",
    actual = "@six_archive//:six",
)

bind(
    name = "zlib",
    actual = "@zlib_archive//:zlib",
)

# This proto version is the same as tensorflow 1.4.0. If you are using a
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

tf_configure(name = "local_config_tf")
