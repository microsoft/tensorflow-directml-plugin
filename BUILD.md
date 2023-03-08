# Building TensorFlow-DirectML-Plugin <!-- omit in toc -->

This document contains instructions for producing private builds of tensorflow-directml-plugin.

- [Step 1: Setup Build Machine](#step-1-setup-build-machine)
  - [Install CMake](#install-cmake)
  - [Install Development Tools](#install-development-tools)
  - [Install Python/Miniconda](#install-pythonminiconda)
- [Step 2: Clone](#step-2-clone)
- [Step 3: Build](#step-3-build)
  - [Build with Helper Script](#build-with-helper-script)
  - [Build with Visual Studio IDE](#build-with-visual-studio-ide)
  - [Build with CMake Manually](#build-with-cmake-manually)
- [Development Details](#development-details)
  - [Developer Mode](#developer-mode)
  - [DirectX Development Files](#directx-development-files)
  - [Linux Wheels (Manylinux) and DirectX Libraries](#linux-wheels-manylinux-and-directx-libraries)

# Step 1: Setup Build Machine

Your machine must have a few tools installed before you can get started.

- CMake.
- A Python 3.8, 3.9, 3.10 or 3.11 environment with the `wheel` package installed.
- Build tools appropriate for your platform (MSVC on Windows, Clang on Linux).
- Linux: a glibc-based distro, like Ubuntu.

## Install CMake

CMake is used to generate and build this project on all platforms.

**Windows**:
- The easiest way to install is with the latest .msi installer on the [downloads page](https://cmake.org/download/).

**Linux**:
- Refer to [CMake's instructions](https://cmake.org/install/) to get a compatible release. The version of CMake available through your distro's package manager (e.g. `apt`) may not be new enough without adding a different upstream/repository! 

Example to download and extract pre-built CMake binaries to `/home/<username>/.cmake`:
```
wget https://github.com/Kitware/CMake/releases/download/v3.22.1/cmake-3.22.1-linux-x86_64.sh
mkdir ~/.cmake
bash cmake-3.22.1-linux-x86_64.sh --prefix=/home/<username>/.cmake --skip-license
```

## Install Development Tools

**Windows**:

- Download and install [Visual Studio 2022](https://visualstudio.microsoft.com/vs/) or [Visual Studio 2019](https://visualstudio.microsoft.com/vs/older-downloads/).
- You must have Windows 10 SDK 10.0.17763.0 or newer, which will be satisfied by installing VS2022 or VS2019.
- Note that Visual Studio comes with a copy of the Ninja build system (you don't need to install it separately).

**Linux**:

On Linux you must use Clang to compile. You'll also want to install the Ninja build system if using the build.py helper script.

```
sudo apt update
sudo apt install clang ninja-build
```

## Install Python/Miniconda

This project requires a Python interpreter to produce the TFDML plugin wheel. This environment must have the `wheel` package installed. On Windows you should have the `vswhere` package if you intend to use `build.py` with the Ninja generator.

Feel free to use any version of Python, but we recommend [Miniconda](https://docs.conda.io/en/latest/miniconda.html) to sandbox your project environments. Miniconda is a bundle that includes [Python](https://www.python.org/), a package and environment manager named [Conda](https://docs.conda.io/projects/conda/en/latest/), and a very small set of Python packages. It is a lighter-weight alternative to Anaconda, which contains hundreds of Python packages that aren't necessary for building. See the [user guide](https://docs.conda.io/projects/conda/en/latest/user-guide/getting-started.html) for more details if you're not familiar with its usage.

Once you've installed Miniconda you will want to create and activate a Python 3.8, 3.9, 3.10, or 3.11 environment with the required packages Use this activated environment when building this project.

**Windows**:

Download the latest [Miniconda3 Windows 64-bit](https://docs.conda.io/en/latest/miniconda.html) installer. You can leave all the default settings.

```
conda create --name tfdml_plugin python=3.7
conda activate tfdml_plugin
pip install wheel vswhere
```

**Linux**:

Download and run the installer script, then create a build environment.

```
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
# follow prompts to install Miniconda and activate it in your shell

conda create --name tfdml_plugin python=3.7
conda activate tfdml_plugin
pip install wheel
```

# Step 2: Clone

Clone the repository to a location of your choosing.

```
git clone https://github.com/microsoft/tensorflow-directml-plugin.git 
```

# Step 3: Build

CMake projects have three phases: configure, generate, and build. 

1. **Configure**: CMake locate the compilers/linkers, locate tools (e.g. Python), and set any build options (e.g. building with telemetry or not).
2. **Generate**: CMake will produce the build files for the desired generator (Visual Studio, Ninja, Makefiles, etc.).
3. **Build**: CMake will invoke the underlying build system to compile and link targets.

This project has a helper script, `build.py`, that combines all three steps into a single command line invocation. However, you can use CMake directly for more control.

## Build with Helper Script

The `build.py` script is a thin wrapper around CMake commands and handles configuring and building in one step.

The default settings will produce the Python wheel (debug build by default) in a folder name `build/` under the source tree:

```
python build.py
```

Run `build.py --help` for a full list of options! Especially useful parameters include `--config` (`-c`), `--target` (`-t`), and `--build_output` (`-o`).

By default, this script uses the "Ninja Multi-Config" generator for both Windows and Linux. You can run the script and change between release and debug builds without using separate build output directories.

Configuration will occur if the project hasn't yet been generated or the script detects a change in one of the build options. If you reuse an existing build output directory and change generators CMake will give you an error; you must manually delete the build directory in this case.

## Build with Visual Studio IDE

If you prefer to work within the Visual Studio IDE you can generate the project without building it:

```
python build.py --configure-only --generator "Visual Studio 17 2022"
```

You'll find the generated solution under `build/tensorflow-directml-plugin.sln`.

## Build with CMake Manually

You can use CMake directly if you prefer; this is especially useful if you want to use a different generator or toolset than what's supported by the build.py helper. On most platforms it's as simple as the following:

```
cd <path_to_repo>

# Configure and generate
cmake -S . -B build

# Build
cmake --build build --target tfdml_plugin_wheel
```

If you want to emulate the configuration performed by the `build.py` helper script you can use the `--show` option to see the command lines invoked. For example, the following shows the two CMake invocations to configure and build with VS2022:

```
python build.py -g "Visual Studio 17 2022" --show

- Configure: cmake -S S:\tensorflow-directml-plugin -B S:\tensorflow-directml-plugin\build -G "Visual Studio 17 2022" -DFETCHCONTENT_QUIET=OFF -DTFDML_TELEMETRY=OFF

- Build: cmake --build S:\tensorflow-directml-plugin\build --config Debug --target tfdml_plugin_wheel -- /m
```

**NOTE**: If you prefer to use the Ninja generator with MSVC on Windows you'll want to run CMake from within a VS developer command prompt. Search for `x64 Native Tools Command Prompt for VS 2022` in your start menu and launch it. Use this prompt to issue all CMake commands. This isn't necessary when using the VS generators, which are able to locate the MSVC tools automatically. The `build.py` helper script hides this extra step by invoking the VsDevCmd.bat before running any CMake commands:

```
python build.py -g "Ninja Multi-Config" --show

- Configure: "C:/Program Files (x86)/Microsoft Visual Studio/2019/Enterprise/Common7/Tools/VsDevCmd.bat" -arch=x64 -no_logo && cmake -S S:\tensorflow-directml-plugin -B S:\tensorflow-directml-plugin\build -G "Ninja Multi-Config" -DFETCHCONTENT_QUIET=OFF -DTFDML_TELEMETRY=OFF

- Build: "C:/Program Files (x86)/Microsoft Visual Studio/2019/Enterprise/Common7/Tools/VsDevCmd.bat" -arch=x64 -no_logo && cmake --build S:\tensorflow-directml-plugin\build --config Debug --target tfdml_plugin_wheel
```

# Development Details

## Developer Mode

This repository may periodically reference in-development versions of DirectML for testing new features. For example, experimental APIs are added to `DirectMLPreview.h`, which may have breaking changes; once an API appears in `DirectML.h` it is immutable. A preview build of DirectML requires *developer mode* to be enabled or it will fail to load. This restriction is intended to avoid a long-term dependency on the preview library. **Packages on PyPI will only be released when the repository depends on a stable version of DirectML.**

You can determine if the current state of the repository references an in-development version of DirectML by inspecting `cmake/dependencies.cmake`. 

- If you see a reference to `Microsoft.AI.DirectML` then developer mode is NOT REQUIRED.
- If you see a reference to `Microsoft.AI.DirectML.Preview` then developer mode is REQUIRED.

Developer mode is, as the name indicates, only intended to be used for development! It should not be used for any other purpose. To enable developer mode:

- **Windows**
  - [Toggle "Developer Mode" to "On"](https://docs.microsoft.com/en-us/windows/uwp/get-started/enable-your-device-for-development) in the "For developers" tab of the "Update & Security" section of the Settings app.

- **WSL**
  - Create a plain-text file in your Linux home directory, `~/.directml.conf`, with the contents `devmode = 1`.

## DirectX Development Files

The development headers and libraries used to build DirectX-based applications are included in the Windows SDK; however, this SDK is not available when building for Linux, and some of required APIs may not yet exist in public versions of the SDK. For these reasons, the DirectX development files are integrated a little differently in this project.

The header files for Direct3D, DXCore, and DirectML are downloaded automatically. This project does not use any DirectX headers included in the Windows 10 SDK *except* for dxgi1_6.h, which is part of Windows 10 SDK 10.0.17763.0 or newer.

The use of the redistributable DirectML library is governed by a separate license that is found as part of the package (found in `tensorflow-plugins/directml/DirectML_LICENSE.txt` when extracted).

## Linux Wheels (Manylinux) and DirectX Libraries

Building *portable* Linux binaries is tricky in comparison to Windows or MacOS. The [Manylinux](https://github.com/pypa/manylinux) project attempts to wrangle some of this complexity by providing standards for build environments to produce binaries that are widely usable across GLIBC-based Linux distros. Manylinux also provides container images that implement the various manylinux standards. 

The official TensorFlow CI builds use Ubuntu 16.04 as the base OS for its container-based builds, and TFDML CI builds take a similar approach by leveraging Google's [custom-op-ubuntu16](https://hub.docker.com/layers/tensorflow/tensorflow/custom-op-ubuntu16/images/sha256-f0676af86cb61665ae20c7935340b4073e325ccbad1f2bc7b904577dd6d511c0?context=explore) image.

The DirectX libraries bundled into the TFDML wheel are built using a different cross-compiling technique intended to support both glibc and musl, so they may not be recognized as manylinux2010 compliant. The DX libraries only run in WSL2 distros, so this should not have any real-world consequence, but the auditwheel tool may report issues with these binaries when scanning the TFDML Linux wheels.

You will need to build this repository from source if the manylinux wheels on PyPI do not work for your target WSL2 distro (e.g. Alpine Linux uses musl instead of glibc). PyPI does not support uploading Linux binary wheels unless they conform to the manylinux standards.