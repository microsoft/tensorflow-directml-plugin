"""Describes a list of plugins and tools to download"""

load("//third_party/dml/redist:workspace.bzl", "dml_repository")
load("//third_party/pix:workspace.bzl", "pix_repository")
load("//third_party/tensorflow:workspace.bzl", "tensorflow_pip_package")
load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

def clean_dep(dep):
    return str(Label(dep))

def tfdml_plugin_workspace():
    """All external dependencies for TFDML builds"""
    http_archive(
        name = "bazel_toolchains",
        sha256 = "109a99384f9d08f9e75136d218ebaebc68cc810c56897aea2224c57932052d30",
        strip_prefix = "bazel-toolchains-94d31935a2c94fe7e7c7379a0f3393e181928ff7",
        urls = [
            "http://mirror.tensorflow.org/github.com/bazelbuild/bazel-toolchains/archive/94d31935a2c94fe7e7c7379a0f3393e181928ff7.tar.gz",
            "https://github.com/bazelbuild/bazel-toolchains/archive/94d31935a2c94fe7e7c7379a0f3393e181928ff7.tar.gz",
        ],
    )

    http_archive(
        name = "com_google_absl",
        build_file = clean_dep("//third_party:com_google_absl.BUILD"),
        sha256 = "35f22ef5cb286f09954b7cc4c85b5a3f6221c9d4df6b8c4a1e9d399555b366ee",
        strip_prefix = "abseil-cpp-997aaf3a28308eba1b9156aa35ab7bca9688e9f6",
        urls = [
            "http://mirror.tensorflow.org/github.com/abseil/abseil-cpp/archive/997aaf3a28308eba1b9156aa35ab7bca9688e9f6.tar.gz",
            "https://github.com/abseil/abseil-cpp/archive/997aaf3a28308eba1b9156aa35ab7bca9688e9f6.tar.gz",
        ],
    )

    # rules_proto defines abstract rules for building Protocol Buffers.
    http_archive(
        name = "rules_proto",
        sha256 = "558705624135d8c1d6653daa90a444a5d40706985b8b9aacbc88d0fa192cccbf",
        strip_prefix = "rules_proto-b9e633e71e2a79cc6bbf69bb9b265343a1c47039",
        urls = [
            "https://github.com/bazelbuild/rules_proto/archive/b9e633e71e2a79cc6bbf69bb9b265343a1c47039.tar.gz",
        ],
    )

    # To update the package version, query https://pypi.org/pypi/tf-nightly-cpu/json and search for the SHA and URL of
    # the desired date for the cp37 version
    tensorflow_pip_package(
        name = "tensorflow",
        build_file = clean_dep("//third_party:tensorflow.BUILD"),
        linux_url = "https://files.pythonhosted.org/packages/ed/ed/01ca3c3661b66d84d5038b9f5682ccf0503ca5f5c93a616db059766c7cd5/tf_nightly_cpu-2.7.0.dev20210904-cp37-cp37m-manylinux2010_x86_64.whl",
        linux_sha256 = "9f0fe4607dbe099883889562d73a6e2b56aade8744a9660bdc4de170a81fe58f",
        windows_url = "https://files.pythonhosted.org/packages/34/76/4ad2ba3a3e1d87539e6723e827b4dfc5a80b5630f28f4549ada61a44cf0a/tf_nightly_cpu-2.7.0.dev20210904-cp37-cp37m-win_amd64.whl",
        windows_sha256 = "1f87a6e0a230ec92c79fc76ed5fdd925f439be6597000d852d6ed813a790a0a9",
    )

    http_archive(
        name = "directx_headers",
        urls = [
            "https://github.com/microsoft/DirectX-Headers/archive/1e20e8e975c784f71c15260591e49d08b84a6573.tar.gz",
        ],
        sha256 = "37fa82e34f7f12bbbc7dd21e4e02112585174259699b526fb0236b0389de7fc9",
        strip_prefix = "DirectX-Headers-1e20e8e975c784f71c15260591e49d08b84a6573",
        build_file = clean_dep("//third_party:directx_headers.BUILD"),
    )

    http_archive(
        name = "directml",
        urls = [
            "https://github.com/microsoft/DirectML/archive/36a8fcbac70fecb9f451a4e617d48ad3780de6cb.tar.gz",
        ],
        sha256 = "a14ba4a2cdd8ea9ebff4043e6141601261262e1d722eb3ea15e15e8bcb59ef24",
        strip_prefix = "DirectML-36a8fcbac70fecb9f451a4e617d48ad3780de6cb",
        build_file = clean_dep("//third_party:directml.BUILD"),
    )

    # URL must point at the DirectML redistributable NuGet package.
    dml_repository(
        name = "dml_redist",
        url = "https://www.nuget.org/api/v2/package/Microsoft.AI.DirectML/1.7.0",
        sha256 = "77bd5de862c36f084c138ff3341936dca01bd21e58bfc57cb45118b116b1f9f4",
        build_file = "//third_party/dml/redist:BUILD.bazel",
    )

    pix_repository(
        name = "pix",
        package = "WinPixEventRuntime",
        version = "1.0.210209001",
        source = "https://api.nuget.org/v3/index.json",
        build_file = "//third_party/pix:BUILD.bazel",
    )
