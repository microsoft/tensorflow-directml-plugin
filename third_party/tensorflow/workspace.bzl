"""Downloads and extracts the appropriate TensorFlow package for the python version"""

def _tensorflow_pip_package_impl(repository_ctx):
    is_windows = repository_ctx.os.name.lower().find("windows") != -1

    repository_ctx.download_and_extract(
        url = repository_ctx.attr.windows_url if is_windows else repository_ctx.attr.linux_url,
        sha256 = repository_ctx.attr.windows_sha256 if is_windows else repository_ctx.attr.linux_sha256,
        type = "zip",
    )

    repository_ctx.template(
        "BUILD.bazel",
        repository_ctx.attr.build_file,
    )

tensorflow_pip_package = repository_rule(
    implementation = _tensorflow_pip_package_impl,
    attrs = {
        "linux_url": attr.string(mandatory = True),
        "linux_sha256": attr.string(mandatory = True),
        "windows_url": attr.string(mandatory = True),
        "windows_sha256": attr.string(mandatory = True),
        "build_file": attr.label(mandatory = True),
    },
)
