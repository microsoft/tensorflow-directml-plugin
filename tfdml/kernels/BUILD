package(
    licenses = ["notice"],  # MIT
)

cc_library(
    name = "dml_kernels",
    srcs = [
        "dml_addn_op.cc",
        "dml_assign_variable_op.cc",
        "dml_concat_op.cc",
        "dml_gather_nd_op.cc",
        "dml_gather_op.cc",
    ],
    linkstatic = 1,
    visibility = ["//visibility:public"],
    deps = [
        "//tfdml/core:dml_runtime",
        "@com_google_absl//absl/cleanup",
    ],
    alwayslink = True,
)
