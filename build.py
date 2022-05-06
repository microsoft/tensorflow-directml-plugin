#!/usr/bin/env python
# Copyright (c) Microsoft Corporation. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
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
import shutil
from pathlib import Path

def run_or_show(step_name, cl, show):
  if show:
    print(f"- {step_name}: {cl}")
  else:
    subprocess.run(cl, shell=True, check=True)


def set_tool_environment(args, cl):
  """Adds a build tool environment setup script to the command line, if necessary."""

  # On Windows this script uses MSVC regardless of the generator. The VS generators
  # know how to find the MSVC tools, but the Ninja generators assume the user
  # is running within a VS developer command prompt (i.e. has executed vcvarsall.bat).
  if os.name == "nt" and args.generator.startswith("Ninja"):
    import vswhere
    vs_path = vswhere.get_latest_path(products='*')
    bat_path = Path(vs_path) / "Common7" / "Tools" / "VsDevCmd.bat"
    cl.append(f"\"{bat_path}\" -arch=x64 -no_logo &&")


def configure_required(args, source_dir):
  """Checks if CMake configuration is required."""

  cmake_cache_path = os.path.join(args.build_output, "CMakeCache.txt")
  if not os.path.exists(cmake_cache_path):
    return True # Project hasn't been configured yet.
  
  with open(cmake_cache_path, "r") as file:
    cmake_cache_content = file.read()

  if f"CMAKE_GENERATOR:INTERNAL={args.generator}" not in cmake_cache_content:
    return True # Generator has changed.

  if f"TFDML_TELEMETRY:BOOL={'ON' if args.telemetry else 'OFF'}" not in cmake_cache_content:
    return True # Telemetry option has changed.

  if f"DTFDML_TELEMETRY_PROVIDER_GROUP_GUID:STRING={args.telemetry_provider_group_guid}" not in cmake_cache_content:
    return True # Telemetry Provider Group GUID has changed.
  
  if f"TFDML_WHEEL_VERSION_SUFFIX:STRING={args.wheel_version_suffix}" not in cmake_cache_content:
    return True # Package wheel version suffix has changed.

  return False


def configure(args, source_dir):
  """Configures the CMake project."""

  if not configure_required(args, source_dir):
    return

  cl = []
  set_tool_environment(args, cl)
  cl.append(args.cmake)
  cl.append(f"-S {source_dir}")
  cl.append(f"-B {args.build_output}")
  cl.append(f"-G \"{args.generator}\"")
  cl.append("-DFETCHCONTENT_QUIET=OFF") # Print details when fetching dependencies
  cl.append(f"-DTFDML_TELEMETRY={'ON' if args.telemetry else 'OFF'}")
  cl.append(f"-DTFDML_TELEMETRY_PROVIDER_GROUP_GUID={args.telemetry_provider_group_guid}")
  cl.append(f"-DTFDML_WHEEL_VERSION_SUFFIX=\"{args.wheel_version_suffix}\"")
  run_or_show("Configure", " ".join(cl), args.show)


def build(args):
  """Builds the CMake project."""

  cl = []
  set_tool_environment(args, cl)
  cl.append(args.cmake)
  cl.append(f"--build {args.build_output}")
  cl.append(f"--config {args.config.title()}")
  cl.append(f"--target {args.target}")
  if args.clean:
    cl.append("--clean-first")
  
  if args.no_parallel:
    # By default, Ninja builds in parallel.
    if args.generator.startswith("Ninja"):
      cl.append(f"-j 1") 
  else:
    # By default, MSBuild doesn't build in parallel.
    if args.generator.startswith("Visual Studio"):
      cl.append("-- /m")

  run_or_show("Build", " ".join(cl), args.show)


def install_wheel(args):
  """Installs the built plugin wheel into the current Python environment."""

  # Find the most recently created package
  build_files = sorted(Path(args.build_output).iterdir(),
                       key=os.path.getmtime,
                       reverse=True)
  wheel_files = filter(lambda f: f.parts[-1].endswith(".whl"), build_files)
  package_path = next(wheel_files).as_posix()

  # Only force the reinstallation of tfdml
  subprocess.run(
      ["pip", "install", "--force-reinstall", "--no-deps", package_path],
      check=True)

  # Reinstall the dependencies that may have been updated
  subprocess.run(["pip", "install", package_path], check=True)


def main():
  # Default to storing build output under <source_dir>/build/.
  source_dir = os.path.dirname(os.path.realpath(__file__))
  default_build_output = os.path.join(source_dir, "build")

  parser = argparse.ArgumentParser()

  parser.add_argument(
    "--cmake",
     default="cmake",
     help="Path to cmake executable. Defaults to cmake on PATH. If cmake isn't found it will be downloaded to the build folder.")

  # This script only supports a few of the CMake generators, but users can
  # manually invoke CMake if they prefer another.
  parser.add_argument(
    "--generator",
    "-g",
    choices=(
      "Visual Studio 17 2022", 
      "Visual Studio 16 2019", 
      "Visual Studio 15 2017", 
      "Ninja Multi-Config"),
    default="Ninja Multi-Config",
    help="CMake generator.")

  parser.add_argument(
    "--config",
    "-c",
    choices=("debug", "release"),
    default="debug",
    help="Build configuration.")

  parser.add_argument(
    "--target",
    "-t",
    default="tfdml_plugin_wheel",
    help="CMake target to build.")

  parser.add_argument(
    "--show",
    "-s",
    action="store_true",
    help="Show CMake commands, but don't execute them.")

  parser.add_argument(
    "--clean",
    "-x",
    action="store_true",
    help="Builds the 'clean' target before .")

  parser.add_argument(
    "--configure-only",
    action="store_true",
    help="Configure and generate the CMake project, but don't build it.")

  parser.add_argument(
    "--install",
    "-i",
    action="store_true",
    help="Install Python package using pip.")

  parser.add_argument(
    "--telemetry",
    action="store_true",
    help="Allow builds to emit telemetry associated with the DMLTF client hint.")

  parser.add_argument(
    "--telemetry_provider_group_guid",
    default="",
    help="The GUID of the telemetry provider group to use in the format '00000000-0000-0000-0000-000000000000'.")

  parser.add_argument(
      "--build_output",
      "-o",
      default=default_build_output,
      help="Build output path. Defaults to {}.".format(default_build_output))

  parser.add_argument(
      "--no_parallel",
      action="store_true",
      help="Limits build parallelism.")

  parser.add_argument(
      "--wheel_version_suffix",
      default="",
      help="Append some text to the Python wheel version.")

  args = parser.parse_args()

  configure(args, source_dir)
  if not args.configure_only:
    build(args)
    if args.install:
      install_wheel(args)

if __name__ == "__main__":
  main()
