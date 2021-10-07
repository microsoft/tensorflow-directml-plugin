package(
    licenses = ["notice"],  # Apache 2.0
)

cc_library(
    name = "dml_device_impl",
    srcs = ["dml_device_plugin.cc"],
    hdrs = [
        "dml_device_plugin.h",
    ],
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

exports_files(
    srcs = [
        "dml_device_plugin.h",
    ],
    visibility = ["//visibility:public"],
)
