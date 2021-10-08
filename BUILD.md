# Building TensorFlow-DirectML-Plugin <!-- omit in toc -->

This document contains instructions for producing private builds of tensorflow-directml-plugin.

- [Developer Mode](#developer-mode)
- [DirectX Development Files](#directx-development-files)
- [Recommended Build Environment](#recommended-build-environment)
- [Build Script](#build-script)
- [CI/PyPI Builds](#cipypi-builds)
- [Linux Wheels (Manylinux) and DirectX Libraries](#linux-wheels-manylinux-and-directx-libraries)
- [Detailed Instructions: Windows](#detailed-instructions-windows)
  - [Install Visual Studio 2019](#install-visual-studio-2019)
  - [Install Bazel](#install-bazel)
  - [Install Miniconda](#install-miniconda)
  - [Create Conda Build Environment](#create-conda-build-environment)
  - [Clone](#clone)
  - [Build](#build)
- [Detailed Instructions: Linux](#detailed-instructions-linux)
  - [Install Development Tools](#install-development-tools)
  - [Install Bazel](#install-bazel-1)
  - [Install Miniconda](#install-miniconda-1)
  - [Create Conda Build Environment](#create-conda-build-environment-1)
  - [Clone](#clone-1)
  - [Build](#build-1)

# Developer Mode

This repository may periodically reference in-development versions of DirectML for testing new features. For example, experimental APIs are added to `DirectMLPreview.h`, which may have breaking changes; once an API appears in `DirectML.h` it is immutable. A preview build of DirectML requires *developer mode* to be enabled or it will fail to load. This restriction is intended to avoid a long-term dependency on the preview library. **Packages on PyPI will only be released when the repository depends on a stable version of DirectML.**

You can determine if the current state of the repository references an in-development version of DirectML by inspecting `tensorflow/workspace.bzl`. If the package name is `Microsoft.AI.DirectML.Preview`, or the version ends with `-dev*`, then developer mode will be required. For example, the following snippet shows a dependency on DirectML Microsoft.AI.DirectML.Preview.1.5.0-dev20210406, which requires developer mode.

```
dml_repository(
    name = "dml_redist",
    package = "Microsoft.AI.DirectML.Preview",
    version = "1.5.0-dev20210406",
    source = "https://pkgs.dev.azure.com/ms/DirectML/_packaging/tensorflow-directml-plugin/nuget/v3/index.json",
    build_file = "//third_party/dml/redist:BUILD.bazel",
)
```

Developer mode is, as the name indicates, only intended to be used for development! It should not be used for any other purpose. To enable developer mode:

- **Windows**
  - [Toggle "Developer Mode" to "On"](https://docs.microsoft.com/en-us/windows/uwp/get-started/enable-your-device-for-development) in the "For developers" tab of the "Update & Security" section of the Settings app.

- **WSL**
  - Create a plain-text file in your Linux home directory, `~/.directml.conf`, with the contents `devmode = 1`.

# DirectX Development Files

The development headers and libraries used to build DirectX-based applications are included in the Windows SDK; however, this SDK is not available when building for Linux, and some of required APIs may not yet exist in public versions of the SDK. For these reasons, the DirectX development files are integrated a little differently in this project.

The header files for Direct3D, DXCore, and DirectML are downloaded automatically. This project does not use any DirectX headers included in the Windows 10 SDK *except* for dxgi1_6.h, which is part of Windows 10 SDK 10.0.17763.0 or newer.

The use of the redistributable DirectML library is governed by a separate license that is found as part of the package (found in `tensorflow-plugins/directml/DirectML_LICENSE.txt` when extracted).

# Recommended Build Environment

We've tested the following build environments, and we recommend using a Python environment manager like [Miniconda](https://docs.conda.io/en/latest/miniconda.html) to sandbox your builds. It may be possible to build in other environments that we have not tested.

- **Windows**
  - Visual Studio 2019 (16.x)
  - Windows 10 SDK 10.0.17763.0 or newer
  - Bazel 3.7.2
  - Python 3.7 environment

- **Linux**
  - Any glibc-based distro (we've tested Ubuntu 18.04+) with gcc/g++ packages
  - Bazel 3.7.2
  - Python 3.7 environment

# Build Script

A helper script, build.py, can be used to build this repository. This script is a thin wrapper around the bazel commands for configuring and building the Pluggable Device PyPI Package; you may use bazel directly if you prefer. Run `build.py --help` for a full list of options, or inspect this file to get a full list of the bazel commands it executes. 

# CI/PyPI Builds

We do not produce CI or PyPI yet.

# Linux Wheels (Manylinux) and DirectX Libraries

Building *portable* Linux binaries is tricky in comparison to Windows or MacOS. The [Manylinux](https://github.com/pypa/manylinux) project attempts to wrangle some of this complexity by providing standards for build environments to produce binaries that are widely usable across GLIBC-based Linux distros. Manylinux also provides container images that implement the various manylinux standards. 

Unfortunately, the official manylinux2010 image does not support pre-built versions of bazel (the build tool for TensorFlow); it's necessary to build bazel from source or set up a manylinux2010-compliant toolchain on a different host OS. The official TensorFlow CI builds use Ubuntu 16.04 as the base OS for its container-based builds, and TFDML CI builds take a similar approach by leveraging Google's [custom-op-ubuntu16](https://hub.docker.com/layers/tensorflow/tensorflow/custom-op-ubuntu16/images/sha256-f0676af86cb61665ae20c7935340b4073e325ccbad1f2bc7b904577dd6d511c0?context=explore) image.

The DirectX libraries bundled into the TFDML wheel are built using a different cross-compiling technique intended to support both glibc and musl, so they may not be recognized as manylinux2010 compliant. The DX libraries only run in WSL2 distros, so this should not have any real-world consequence, but the auditwheel tool may report issues with these binaries when scanning the TFDML Linux wheels.

You will need to build this repository from source if the manylinux wheels on PyPI do not work for your target WSL2 distro (e.g. Alpine Linux uses musl instead of glibc). PyPI does not support uploading Linux binary wheels unless they conform to the manylinux standards.

# Detailed Instructions: Windows

These instructions use Miniconda to sandbox your build environment. This isn't strictly necessary, and there are other ways to do this (e.g. virtual machines, containers), but for the purpose of this walk-through you will use a Python virtual environment to build TFDML.

## Install Visual Studio 2019

tensorflow-plugin-directml builds with either VS2017 or VS2019, but VS2019 is recommended. [Download](https://my.visualstudio.com/Downloads?q=visual%20studio%202019&wt.mc_id=o~msft~vscom~older-downloads) and install the community, professional, or enterprise edition.

Make sure you install the Windows 10 SDK as well, which should be version 10.0.17763.0 or newer.

## Install Bazel

### Install Bazel using Bazelisk (recommended)

A good habit when using Bazel is to use Bazelisk instead of manually installing the right version of Bazel. Bazelisk will automatically download and use the version of bazel from `.bazelversion`, so you don't have to worry about using the wrong version.
It also makes it easier to work on different projects that expect different versions of Bazel since you don't have to keep changing the path.

You can download Bazelisk through npm or golang:

```bash
npm install -g @bazel/bazelisk
```

```bash
go install github.com/bazelbuild/bazelisk@latest
```

### Install Bazel manually

Although installing Bazelisk is the recommended approach since it automatically detects the right version of Bazel to install, you can also install it manually. **Make sure that the version of Bazel that you install matches the content of .bazelversion.**

[Bazel](https://bazel.build/) is the build tool for TensorFlow. Bazel is distributed as an executable and there is no installer step.

1. Download [Bazel 3.7.2](https://github.com/bazelbuild/bazel/releases/download/3.7.2/bazel-3.7.2-windows-x86_64.exe).
2. Copy/rename `bazel-3.7.2-windows-x86_64.exe` to a versioned path, such as `C:\bazel\3.7.2\bazel.exe`. TensorFlow will expect bazel.exe on the `%PATH%`, so renaming the executable while retaining the version within the path is useful.

## Install Miniconda

Miniconda is a bundle that includes [Python](https://www.python.org/), a package and environment manager named [Conda](https://docs.conda.io/projects/conda/en/latest/), and a very small set of Python packages. It is a lighter-weight alternative to Anaconda, which contains hundreds of Python packages that aren't necessary for building. The rest of this document applies equally to Anaconda if you prefer.

Download the latest [Miniconda3 Windows 64-bit](https://docs.conda.io/en/latest/miniconda.html) installer. You can leave all the default settings, but take note of the installation location (you'll need it later). The examples in this doc will reference "c:\miniconda3" as the install location.

## Create Conda Build Environment

Launch a Miniconda prompt (appears as "Anaconda PowerShell Prompt (Miniconda3)" in Windows Start Menu). The examples below will use the PowerShell version, but you can use the CMD version if you prefer. This prompt is a thin wrapper around PowerShell/CMD that adds the Conda installation to the `PATH` environment variable and modifies the display prompt slightly.

After launching the prompt, create a new environment. The examples here use the name `tfdml`, but you can choose any name you like. The idea is to sandbox any packages and dependencies in this environment. This will create a separate copy of Python and its packages in a directory `C:\miniconda3\envs\tfdml`.

**IMPORTANT**: make sure to use Python 3.7, 3.8 or 3.9. TensorFlow 2 is not supported on older versions of Python.

```
(base) PS> conda create --name tfdml python=3.7
```

Next, activate the environment. Activating an environment will set up the `PATH` to use the correct version of Python and its packages.

```
(base) PS> conda activate tfdml
```

## Clone

Clone the repository to a location of your choosing. The examples here will assume you clone to `C:\src\tensorflow-directml-plugin` for the sake of brevity, but you may clone wherever you like.

```
PS> cd c:\src
PS> git clone https://github.com/microsoft/tensorflow-directml-plugin.git
```

## Build

Remember to activate your build environment whenever you need to build. Change your working directory to the clone location:

```
(base) PS> conda activate tfdml
(tfdml) PS> cd c:\src\tensorflow-directml-plugin
```

To produce the Python package run the following:

```
(tfdml) PS> python build.py
```

After the package is built you will find a wheel package under `<PATH_TO_CLONE>\..\tfdml_plugin_build\python_package` (e.g. `C:\src\tfdml_plugin_build\python_package` in these examples). You can run `pip install` on the output .whl file to install your locally built copy of TensorFlow-DirectML.

The build script has additional options you can experiment with. To see more details:

```
(tfdml) PS> python build.py --help
```

# Detailed Instructions: Linux

These instructions use Miniconda to sandbox your build environment. This isn't strictly necessary, and there are other ways to do this (e.g. virtual machines, containers), but for the purpose of this walk-through you will use a Python virtual environment to build TFDML.

## Install Development Tools

```bash
sudo apt install unzip gcc g++
```

## Install Bazel

### Install Bazel using Bazelisk (recommended)

A good habit when using Bazel is to use Bazelisk instead of manually installing the right version of Bazel. Bazelisk will automatically download and use the version of bazel from `.bazelversion`, so you don't have to worry about using the wrong version.
It also makes it easier to work on different projects that expect different versions of Bazel since you don't have to keep changing the path.

You can download Bazelisk through npm or golang:

```bash
npm install -g @bazel/bazelisk
```

```bash
go install github.com/bazelbuild/bazelisk@latest
```

### Install Bazel manually

Although installing Bazelisk is the recommended approach since it automatically detects the right version of Bazel to install, you can also install it manually. **Make sure that the version of Bazel that you install matches the content of .bazelversion.**

```bash
wget https://github.com/bazelbuild/bazel/releases/download/3.7.2/bazel-3.7.2-installer-linux-x86_64.sh
bash bazel-3.7.2-installer-linux-x86_64.sh --bin=$HOME/bin/bazel/3.7.2 --base=$HOME/.bazel
```

## Install Miniconda

Miniconda is a bundle that includes [Python](https://www.python.org/), a package and environment manager named [Conda](https://docs.conda.io/projects/conda/en/latest/), and a very small set of Python packages. It is a lighter-weight alternative to Anaconda, which contains hundreds of Python packages that aren't necessary for building. The rest of this document applies equally to Anaconda if you prefer.

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
```

## Create Conda Build Environment

Restart your shell if you configured conda to init in the .bashrc file. Otherwise, you can launch the base environment as follows:

```bash
eval "$($HOME/miniconda3/bin/conda shell.bash hook)" 
```

After launching the prompt, create a new environment. The examples here use the name `tfdml`, but you can choose any name you like. The idea is to sandbox any packages and dependencies in this environment. This will create a separate copy of Python and its packages.

**IMPORTANT**: make sure to use Python 3.7, 3.8 or 3.9. TensorFlow 2 is not supported on older versions of Python.

```
(base) ~$ conda create --name tfdml python=3.7
```

Next, activate the environment. Activating an environment will set up the `PATH` to use the correct version of Python and its packages.

```
(base) ~$ conda activate tfdml
```

## Clone

Clone the repository to a location of your choosing. The examples here will assume you clone to `~/src/tensorflow-directml-plugin` for the sake of brevity, but you may clone wherever you like.

```
(tfdml) :~$ cd ~/src
(tfdml) :~$ git clone https://github.com/microsoft/tensorflow-directml-plugin.git
```

## Build

Remember to activate your build environment whenever you need to build. To produce the Python package run the following:

```
(tfdml) :~$ cd ~/src/tensorflow-directml-plugin
(tfdml) :~$ python build.py -p -c release
```

After the package is built you will find a wheel package under `~/tfdml_plugin_build/python_package`. You can run `pip install` on the output .whl file to install your locally built copy of TensorFlow-DirectML-Plugin.

The build script has additional options you can experiment with. To see more details:

```
(tfdml) :~$ python build.py --help
```