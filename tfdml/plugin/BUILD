load(
    "//tfdml:tfdml.bzl",
)

package(
    licenses = ["notice"],  # MIT
)

cc_library(
    name = "plugin_device",
    srcs = ["plugin_device.cc"],
    linkstatic = 1,
    visibility = ["//visibility:public"],
    deps = [
        "//tfdml/core:dml_runtime",
        "@directml//:directmlx",
        "@directx_headers",
        "@directx_headers//:directx_guids",
        "@directx_headers//:directx_winadapter",
        "@dml_redist//:headers",
        "@tensorflow//:headers",
    ],
    alwayslink = True,
)

cc_library(
    name = "plugin_kernel",
    srcs = ["plugin_kernel.cc"],
    linkstatic = 1,
    visibility = ["//visibility:public"],
    deps = [
        ":plugin_device",
        "//tfdml/core/kernels:dml_kernels",
        "@tensorflow//:headers",
    ],
    alwayslink = True,
)