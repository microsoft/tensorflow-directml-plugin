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
  parser.add_argument('--src', help='prepare sources in srcdir')
  parser.add_argument('--dst', help='build wheel in dstdir')
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


def copy_dml_redist_files(dml_redist_dir):
  if is_windows():
    runfiles_manifest_path = 'bazel-bin/tfdml/tools/pip_package/build_pip_package.exe.runfiles_manifest'
  else:
    runfiles_manifest_path = 'bazel-bin/tfdml/tools/pip_package/build_pip_package.runfiles_manifest'

  with open(runfiles_manifest_path, 'r') as manifest:
    dml_config_path = re.search(
        r'^dml_redist/.*?/include/DirectMLConfig\.h (.*)',
        manifest.read(),
        flags=re.MULTILINE).group(1)

  # Locate path to root of DirectML redist files
  dml_redist_root = os.path.split(os.path.split(dml_config_path)[0])[0]

  # Copy library and licenses
  with open(dml_config_path, 'r') as dml_config:
    dml_version = re.search(
        r'^#define DIRECTML_SOURCE_VERSION "([abcdef0-9]+)"',
        dml_config.read(),
        flags=re.MULTILINE).group(1)
    print(f'DML Version = {dml_version}')

    if is_windows():
      lib_src = f'{dml_redist_root}/bin/x64-win/DirectML.dll'
      lib_dst = f'{dml_redist_dir}/DirectML.{dml_version}.dll'
    else:
      lib_src = f'{dml_redist_root}/bin/x64-linux/libdirectml.so'
      lib_dst = f'{dml_redist_dir}/libdirectml.{dml_version}.so'

    shutil.copy(lib_src, lib_dst)
    shutil.copy(f'{dml_redist_root}/LICENSE.txt',
                f'{dml_redist_dir}/DirectML_LICENSE.txt')
    shutil.copy(f'{dml_redist_root}/ThirdPartyNotices.txt',
                f'{dml_redist_dir}/DirectML_ThirdPartyNotices.txt')

    # Copy PIX event runtime
    if is_windows():
      with open(runfiles_manifest_path, 'r') as manifest:
        pix_event_runtime_dll_path = re.search(
            r'^pix/.*?/WinPixEventRuntime\.dll (.*)',
            manifest.read(),
            flags=re.MULTILINE).group(1)
        pix_event_runtime_root = os.path.split(
            os.path.split(os.path.split(pix_event_runtime_dll_path)[0])[0])[0]
        shutil.copy(pix_event_runtime_dll_path, dml_redist_dir)
        shutil.copy(f'{pix_event_runtime_root}/license.txt',
                    f'{dml_redist_dir}/WinPixEventRuntime_LICENSE.txt')
        shutil.copy(
            f'{pix_event_runtime_root}/ThirdPartyNotices.txt',
            f'{dml_redist_dir}/WinPixEventRuntime_ThirdPartyNotices.txt')


def prepare_src(src_dir):
  os.makedirs(src_dir)
  print(f'=== Preparing sources in dir {src_dir}')

  if not os.path.exists('bazel-bin/tfdml'):
    raise Exception(
        'Could not find bazel-bin. Did you run from the root of the build tree?'
    )

  if is_windows():
    runfiles_manifest_path = 'bazel-bin/tfdml/tools/pip_package/build_pip_package.exe.runfiles_manifest'
    tfdml_plugin_path_regex = r'^.*tfdml_plugin\.dll (.*tfdml_plugin\.dll)$'
  else:
    runfiles_manifest_path = 'bazel-bin/tfdml/tools/pip_package/build_pip_package.runfiles_manifest'
    tfdml_plugin_path_regex = r'^.*libtfdml_plugin\.so (.*tfdml_plugin\.so)$'

  # Locate path to tfdml_plugin.dll or libtfdml_plugin.so in the manifest
  with open(runfiles_manifest_path, 'r') as manifest:
    tfdml_plugin_path = re.search(
        tfdml_plugin_path_regex,
        manifest.read(),
        flags=re.MULTILINE).group(1)

  os.makedirs(f'{src_dir}/tensorflow-plugins')
  os.chmod(tfdml_plugin_path, 0o777)
  shutil.copy(tfdml_plugin_path, f'{src_dir}/tensorflow-plugins')
  shutil.copy('tfdml/tools/pip_package/MANIFEST.in', src_dir)
  shutil.copy('tfdml/tools/pip_package/README', src_dir)
  shutil.copy('tfdml/tools/pip_package/setup.py', src_dir)
  shutil.copytree('tfdml/python', f'{src_dir}/tensorflow-directml-plugin')
  open(f'{src_dir}/tensorflow-directml-plugin/__init__.py', 'a').close()
  os.makedirs(f'{src_dir}/tensorflow-plugins/directml')
  copy_dml_redist_files(f'{src_dir}/tensorflow-plugins/directml')


def build_wheel(src_dir, dst_dir):
  tf_src_dir = os.path.split(os.path.realpath('.'))[-1]
  tf_path = os.path.realpath(f'bazel-{tf_src_dir}/external/tensorflow')

  if not os.path.exists(tf_path):
    raise FileNotFoundError(f'{tf_path} could not be found')

  with pushd(src_dir):
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
      os.makedirs(dst_dir)
    except FileExistsError:
      pass

    for file_name in os.listdir('dist'):
      shutil.copy(os.path.join('dist', file_name), dst_dir)

  print(f'=== Output wheel file is in: {dst_dir}')


def main():
  args = parse_args()
  src_path = os.path.realpath(args.src)
  try:
    prepare_src(src_path)
    build_wheel(args.src, args.dst)
  finally:
    try:
      shutil.rmtree(src_path)
    except FileNotFoundError:
      pass


if __name__ == "__main__":
  main()
