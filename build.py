#!/usr/bin/env python
# Copyright (c) Microsoft Corporation. All Rights Reserved.
#
# Use of this source code is governed by an MIT-style
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Helper script to build tensorflow packages and tests."""

import os
import sys
import argparse
import subprocess
from pathlib import Path
from distutils.sysconfig import get_python_lib


def clean(args):
  subprocess.run("{} --output_user_root={} clean {}".format(
      args.bazel, args.build_output, "--expunge"),
                 shell=True,
                 check=True)


def build(args):
  """Runs bazel to build tfdml's build_pip_package target."""

  python_bin_path = sys.executable
  python_lib_path = get_python_lib()

  if sys.platform == "win32":
    python_bin_path = python_bin_path.replace('\\', '/')
    python_lib_path = python_lib_path.replace('\\', '/')

  cl = [args.bazel]
  cl.append("--output_user_root={}".format(args.build_output))
  cl.append("build")
  if args.subcommands:
    cl.append("--subcommands")
  cl.append("--config={}".format(args.config))
  if args.telemetry:
    cl.append("--config=telemetry")
  if sys.platform == "win32":
    cl.append("--config=windows")
  else:
    cl.append("--config=linux")
  cl.append("//tfdml:external")
  # cl.append("//tfdml/tools/pip_package:build_pip_package")
  cl.append("--action_env PYTHON_BIN_PATH={}".format(python_bin_path))
  cl.append("--action_env PYTHON_LIB_PATH={}".format(python_lib_path))

  if sys.platform == "win32":
    # Sometimes, Bazel needs help finding the compiler when installing only the
    # build tools without the visual studio IDE
    vswhere_path = os.path.join(os.environ["ProgramFiles(x86)"],
                                "Microsoft Visual Studio", "Installer",
                                "vswhere.exe")

    vswhere_params = [
        vswhere_path, "-products", "*", "-requires",
        "Microsoft.Component.MSBuild", "-property", "installationPath",
        "-latest"
    ]

    build_tools_path = subprocess.run(
        vswhere_params, check=True,
        capture_output=True).stdout.decode().splitlines()[0]

    vc_path = os.path.join(build_tools_path, "VC").replace("\\", "/")
    cl.append('--action_env BAZEL_VC="{}"'.format(vc_path))

  subprocess.run(" ".join(cl), shell=True, check=True)


def create_package(args):
  """
  Creates an installable Python package from tfdml's build_pip_package target.
  """

  build_pip_package_path = os.path.join(
      os.path.dirname(os.path.realpath(__file__)), "bazel-bin", "tfdml",
      "tools", "pip_package", "build_pip_package")

  if sys.platform == "win32":
    build_pip_package_path += ".exe"

  src_path = os.path.join(args.build_output, "python_package_src")
  dst_path = os.path.join(args.build_output, "python_package")

  cl = [build_pip_package_path, "--src", src_path, "--dst", dst_path]

  subprocess.run(" ".join(cl), shell=True, check=True)


def install_package(args):
  """Installs the generated TensorFlow Python package."""

  package_dir = os.path.join(args.build_output, "python_package")

  # Find the most recently created package
  package_path = sorted(Path(package_dir).iterdir(),
                        key=os.path.getmtime)[-1].as_posix()

  # Only force the reinstallation of tfdml
  subprocess.run(
      ["pip", "install", "--force-reinstall", "--no-deps", package_path],
      check=True)

  # Reinstall the dependencies that may have been updated
  subprocess.run(["pip", "install", package_path], check=True)


def main():
  parser = argparse.ArgumentParser()

  parser.add_argument("--config",
                      "-c",
                      choices=("debug", "release"),
                      default="debug",
                      help="Build configuration.")

  parser.add_argument("--clean",
                      "-x",
                      action="store_true",
                      help="Configure and build from scratch.")

  parser.add_argument("--install",
                      "-i",
                      action="store_true",
                      help="Install Python package using pip.")

  parser.add_argument(
      "--subcommands",
      "-s",
      action="store_true",
      help="Display the subcommands executed during a build (e.g. compiler "
      "command line invocations).")

  parser.add_argument(
      "--telemetry",
      action="store_true",
      help=
      "Allow builds to emit telemetry associated with the DMLTF client hint.")

  # Default to storing build output under <repo_root>/../tfdml_plugin_build/.
  default_build_output = os.path.join(
      os.path.dirname(os.path.realpath(__file__)), "..", "tfdml_plugin_build")

  parser.add_argument(
      "--build_output",
      "-o",
      default=default_build_output,
      help="Build output path. Defaults to {}.".format(default_build_output))

  parser.add_argument(
      "--bazel",
      "-b",
      default="bazel",
      help="Path to the bazel executable. Defaults to the first 'bazel' found "
      "in the path.")

  args = parser.parse_args()

  # Clean
  if args.clean:
    clean(args)

  # Build
  build(args)

  # Create Python package
  create_package(args)
  if args.install:
    install_package(args)


if __name__ == "__main__":
  main()
