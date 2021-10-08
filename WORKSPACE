workspace(name = "org_tensorflow_directml_plugin")

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

# buildifier is written in Go and hence needs rules_go to be built.
http_archive(
    name = "io_bazel_rules_go",
    sha256 = "d1ffd055969c8f8d431e2d439813e42326961d0942bdf734d2c95dc30c369566",
    urls = [
        "https://mirror.bazel.build/github.com/bazelbuild/rules_go/releases/download/v0.24.5/rules_go-v0.24.5.tar.gz",
        "https://github.com/bazelbuild/rules_go/releases/download/v0.24.5/rules_go-v0.24.5.tar.gz",
    ],
)

load("@io_bazel_rules_go//go:deps.bzl", "go_register_toolchains", "go_rules_dependencies")

go_rules_dependencies()

go_register_toolchains()

http_archive(
    name = "bazel_gazelle",
    sha256 = "b85f48fa105c4403326e9525ad2b2cc437babaa6e15a3fc0b1dbab0ab064bc7c",
    urls = [
        "https://mirror.bazel.build/github.com/bazelbuild/bazel-gazelle/releases/download/v0.22.2/bazel-gazelle-v0.22.2.tar.gz",
        "https://github.com/bazelbuild/bazel-gazelle/releases/download/v0.22.2/bazel-gazelle-v0.22.2.tar.gz",
    ],
)

load("@bazel_gazelle//:deps.bzl", "gazelle_dependencies")

gazelle_dependencies()

# TODO: Switch to the official branch once the PR is merged
# https://github.com/bazelbuild/buildtools/pull/1011
http_archive(
    name = "com_github_patricevignola_buildtools",
    strip_prefix = "buildtools-c569bb395c453777b3f858b8b223c81385759811",
    url = "https://github.com/PatriceVignola/buildtools/archive/c569bb395c453777b3f858b8b223c81385759811.zip",
)

load("//third_party:version_check.bzl", "check_bazel_version_at_least")

check_bazel_version_at_least("3.1.0")

load("//tfdml:workspace.bzl", "tfdml_plugin_workspace")

tfdml_plugin_workspace()

load(
    "@bazel_toolchains//repositories:repositories.bzl",
    bazel_toolchains_repositories = "repositories",
)

bazel_toolchains_repositories()

load("@rules_proto//proto:repositories.bzl", "rules_proto_dependencies", "rules_proto_toolchains")

rules_proto_dependencies()

rules_proto_toolchains()
