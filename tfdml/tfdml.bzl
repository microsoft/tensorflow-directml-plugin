"""
Return the options to use for a C++ library or binary build.
Uses the ":optmode" config_setting to pick the options.
"""

def if_linux_x86_64(a, otherwise = []):
    return select({
        "//conditons:default": otherwise,
    })

def _get_transitive_headers(hdrs, deps):
    return depset(
        hdrs,
        transitive = [dep[CcInfo].compilation_context.headers for dep in deps],
    )

TransitiveHeadersInfo = provider(
    "A list of transitive headers.",
    fields = {
        "files": "list of strings",
    },
)

def _transitive_hdrs_impl(ctx):
    outputs = _get_transitive_headers([], ctx.attr.deps)
    return TransitiveHeadersInfo(files = outputs)

_transitive_hdrs = rule(
    attrs = {
        "deps": attr.label_list(
            allow_files = True,
            providers = [CcInfo],
        ),
    },
    implementation = _transitive_hdrs_impl,
)

def transitive_hdrs(name, deps = [], **kwargs):
    _transitive_hdrs(name = name + "_gather", deps = deps)
    native.filegroup(name = name, srcs = [":" + name + "_gather"])

def cc_header_only_library(name, deps = [], includes = [], extra_deps = [], **kwargs):
    _transitive_hdrs(name = name + "_gather", deps = deps)
    native.cc_library(
        name = name,
        srcs = [":" + name + "_gather"],
        hdrs = includes,
        deps = extra_deps,
        **kwargs
    )

def if_windows(a, otherwise = []):
    return select({
        "@bazel_tools//src/conditions:windows": a,
        "//conditions:default": otherwise,
    })

def if_not_windows(a):
    return select({
        "@bazel_tools//src/conditions:windows": [],
        "//conditions:default": a,
    })
