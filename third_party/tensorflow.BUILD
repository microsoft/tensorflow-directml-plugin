load("@rules_cc//cc:defs.bzl", "cc_library")
load("@rules_proto//proto:defs.bzl", "proto_library")

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "headers",
    hdrs = glob(["tensorflow/include/**/*"]),
    includes = ["tensorflow/include"],
)

cc_import(
    name = "linux_lib",
    shared_library = "tensorflow/libtensorflow_framework.so.2",
)

cc_library(
    name = "windows_lib",
    linkopts = ["external/tensorflow/tensorflow/python/_pywrap_tensorflow_internal.lib"],
    linkstatic = 1,
    alwayslink = 1,
)

alias(
    name = "lib",
    actual = select({
        "@bazel_tools//src/conditions:windows": ":windows_lib",
        "//conditions:default": ":linux_lib",
    }),
)

cc_proto_library(
    name = "resource_handle_cc_proto",
    deps = [":resource_handle_proto"],
)

proto_library(
    name = "resource_handle_proto",
    srcs = ["tensorflow/include/tensorflow/core/framework/resource_handle.proto"],
    strip_import_prefix = "tensorflow/include",
    deps = [
        ":tensor_shape_proto",
        ":types_proto",
        "@com_google_protobuf//:any_proto",
    ],
)

proto_library(
    name = "tensor_shape_proto",
    srcs = ["tensorflow/include/tensorflow/core/framework/tensor_shape.proto"],
    strip_import_prefix = "tensorflow/include",
)

proto_library(
    name = "types_proto",
    srcs = ["tensorflow/include/tensorflow/core/framework/types.proto"],
    strip_import_prefix = "tensorflow/include",
)

cc_proto_library(
    name = "xplane_cc_proto",
    deps = [":xplane_proto"],
)

proto_library(
    name = "xplane_proto",
    srcs = ["tensorflow/include/tensorflow/core/profiler/protobuf/xplane.proto"],
    strip_import_prefix = "tensorflow/include",
)

cc_proto_library(
    name = "graph_cc_proto",
    deps = [":graph_proto"],
)

proto_library(
    name = "graph_proto",
    srcs = ["tensorflow/include/tensorflow/core/framework/graph.proto"],
    strip_import_prefix = "tensorflow/include",
    deps = [
        ":function_proto",
        ":node_def_proto",
        ":versions_proto",
    ],
)

proto_library(
    name = "function_proto",
    srcs = ["tensorflow/include/tensorflow/core/framework/function.proto"],
    strip_import_prefix = "tensorflow/include",
    deps = [
        ":attr_value_proto",
        ":node_def_proto",
        ":op_def_proto",
    ],
)

proto_library(
    name = "op_def_proto",
    srcs = ["tensorflow/include/tensorflow/core/framework/op_def.proto"],
    strip_import_prefix = "tensorflow/include",
    deps = [
        ":attr_value_proto",
        ":full_type_proto",
        ":resource_handle_proto",
        ":types_proto",
    ],
)

proto_library(
    name = "node_def_proto",
    srcs = ["tensorflow/include/tensorflow/core/framework/node_def.proto"],
    strip_import_prefix = "tensorflow/include",
    deps = [
        ":attr_value_proto",
        ":full_type_proto",
    ],
)

proto_library(
    name = "versions_proto",
    srcs = ["tensorflow/include/tensorflow/core/framework/versions.proto"],
    strip_import_prefix = "tensorflow/include",
)

proto_library(
    name = "attr_value_proto",
    srcs = ["tensorflow/include/tensorflow/core/framework/attr_value.proto"],
    strip_import_prefix = "tensorflow/include",
    deps = [
        ":tensor_proto",
        ":tensor_shape_proto",
        ":types_proto",
    ],
)

proto_library(
    name = "full_type_proto",
    srcs = ["tensorflow/include/tensorflow/core/framework/full_type.proto"],
    strip_import_prefix = "tensorflow/include",
)

proto_library(
    name = "tensor_proto",
    srcs = ["tensorflow/include/tensorflow/core/framework/tensor.proto"],
    strip_import_prefix = "tensorflow/include",
    deps = [
        ":resource_handle_proto",
        ":tensor_shape_proto",
        ":types_proto",
    ],
)
