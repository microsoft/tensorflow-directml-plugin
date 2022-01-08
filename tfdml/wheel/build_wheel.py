import os
import glob
import shutil
import sys
import re
import argparse
import subprocess
from contextlib import contextmanager


def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('--plugin_path', help='path to TFDML plugin library')
  parser.add_argument('--build_dir', help='cmake build directory')
  return parser.parse_args()


@contextmanager
def pushd(new_dir):
  previous_dir = os.getcwd()
  os.chdir(new_dir)
  try:
    yield
  finally:
    os.chdir(previous_dir)


def is_windows():
  return sys.platform == 'win32'


def copy_dml_redist_files(dst, dml_redist_dir, pix_dir):
  dml_config_path = os.path.join(dml_redist_dir, "include/DirectMLConfig.h")

  # Copy library and licenses
  with open(dml_config_path, 'r') as dml_config:
    dml_version = re.search(
        r'^#define DIRECTML_SOURCE_VERSION "([abcdef0-9]+)"',
        dml_config.read(),
        flags=re.MULTILINE).group(1)
    print(f'DML Version = {dml_version}')

    if is_windows():
      lib_src = f'{dml_redist_dir}/bin/x64-win/DirectML.dll'
      lib_dst = f'{dst}/DirectML.{dml_version}.dll'
    else:
      lib_src = f'{dml_redist_dir}/bin/x64-linux/libdirectml.so'
      lib_dst = f'{dst}/libdirectml.{dml_version}.so'

    shutil.copy(lib_src, lib_dst)
    shutil.copy(f'{dml_redist_dir}/LICENSE.txt',
                f'{dst}/DirectML_LICENSE.txt')
    shutil.copy(f'{dml_redist_dir}/ThirdPartyNotices.txt',
                f'{dst}/DirectML_ThirdPartyNotices.txt')

    # Copy PIX event runtime
    if is_windows():
        shutil.copy(f'{pix_dir}/bin/x64/WinPixEventRuntime.dll', dst)
        shutil.copy(f'{pix_dir}/license.txt',
                    f'{dst}/WinPixEventRuntime_LICENSE.txt')
        shutil.copy(
            f'{pix_dir}/ThirdPartyNotices.txt',
            f'{dst}/WinPixEventRuntime_ThirdPartyNotices.txt')


def prepare_src(src_dir, tfdml_plugin_path, cmake_build_dir):
  os.makedirs(src_dir)
  print(f'=== Preparing sources in dir {src_dir}')

  os.makedirs(f'{src_dir}/tensorflow-plugins')
  os.makedirs(f'{src_dir}/tensorflow-directml-plugin')
  os.chmod(tfdml_plugin_path, 0o777)
  shutil.copy(tfdml_plugin_path, f'{src_dir}/tensorflow-plugins')
  shutil.copy('tfdml/wheel/MANIFEST.in', src_dir)
  shutil.copy('tfdml/wheel/README', src_dir)
  shutil.copy('tfdml/wheel/setup.py', src_dir)
  shutil.copy(f'{cmake_build_dir}/TFDML_WHEEL_NAME', src_dir)
  shutil.copy(f'{cmake_build_dir}/TFDML_WHEEL_VERSION', src_dir)
  shutil.copy('tfdml/wheel/template_init.py', f'{src_dir}/tensorflow-directml-plugin/__init__.py')
  os.makedirs(f'{src_dir}/tensorflow-plugins/directml')
  copy_dml_redist_files(
    f'{src_dir}/tensorflow-plugins/directml',
    f'{cmake_build_dir}/_deps/directml_redist-src',
    f'{cmake_build_dir}/_deps/pix_event_runtime-src')


def build_wheel(staging_dir, cmake_build_dir):
  tf_path = os.path.join(cmake_build_dir, '_deps/tensorflow_whl-src')

  if not os.path.exists(tf_path):
    raise FileNotFoundError(f'{tf_path} could not be found')

  with pushd(staging_dir):
    try:
      os.remove('MANIFEST')
    except FileNotFoundError:
      pass

    print('=== Building wheel')
    env_copy = os.environ.copy()
    env_copy['PYTHONPATH'] = tf_path
    subprocess.run(['python', 'setup.py', 'bdist_wheel'],
                   env=env_copy,
                   check=True)
    try:
      os.makedirs(cmake_build_dir)
    except FileExistsError:
      pass

    for file_name in os.listdir('dist'):
      shutil.copy(os.path.join('dist', file_name), cmake_build_dir)

  print(f'=== Output wheel file is in: {cmake_build_dir}')


def main():
  args = parse_args()
  staging_dir = os.path.join(args.build_dir, "build_wheel_staging")
  try:
    prepare_src(staging_dir, args.plugin_path, args.build_dir)
    build_wheel(staging_dir, args.build_dir)
  finally:
    try:
      shutil.rmtree(staging_dir)
    except FileNotFoundError:
      pass


if __name__ == "__main__":
  main()
