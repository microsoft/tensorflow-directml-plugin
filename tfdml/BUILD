cc_binary(
    name = "tfdml_plugin",
    features = ["windows_export_all_symbols"],
    linkshared = True,
    visibility = ["//visibility:public"],
    deps = [
        "//tfdml/core:plugin_device",
        "//tfdml/core:plugin_kernel",
    ],
)

config_setting(
    name = "linux_x86_64",
    values = {"cpu": "k8"},
    visibility = ["//visibility:public"],
)
