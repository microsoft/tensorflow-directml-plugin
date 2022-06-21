'''TensorFlow-DirectML-Plugin is an open-source DirectML backend for TensorFlow.
Accelerate machine learning training with TensorFlow on Windows and Windows
Subsystem for Linux (WSL).

TensorFlow-DirectML-Plugin leverages TensorFlow's Pluggable Device API to create
a DirectML backend for hardware-accelerated machine learning workflows on any
DirectX 12 compatible GPU.

Install the base tensorflow and tensorflow-directml-plugin packages to get started.
Check out the GitHub repository to try the samples or build from source.

TensorFlow, the TensorFlow logo, and any related marks are trademarks of Google Inc.
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import fnmatch
import os
import re
import sys

from setuptools import Command
from setuptools import find_packages
from setuptools import setup
from setuptools.command.install import install as InstallCommandBase
from setuptools.dist import Distribution
import pkg_resources

DOCLINES = __doc__.split('\n')

# This version string is semver compatible, but incompatible with pip.
# For pip, we will remove all '-' characters from this string, and use the
# result for pip.
# Also update tfdml/tfdml.bzl
_VERSION = '0.0.0'
with open('TFDML_WHEEL_VERSION', 'r') as f:
  _VERSION = f.read()

# this path can't be modified.
_PLUGIN_LIB_PATH = 'tensorflow-plugins'
_MY_PLUGIN_PATH = 'tensorflow-directml-plugin'

# The plugin should be compatible with any version of TF >= 2.9.1;
# however, this cannot be expressed as a dependency since there are different
# package names: tensorflow, tensorflow-cpu, tf-nightly, etc.
REQUIRED_PACKAGES = []

if sys.byteorder == 'little':
  # grpcio does not build correctly on big-endian machines due to lack of
  # BoringSSL support.
  # See https://github.com/tensorflow/tensorflow/issues/17882.
  REQUIRED_PACKAGES.append('grpcio >= 1.8.6')

# The wheel package name, change it as your requirements
project_name = ''
with open('TFDML_WHEEL_NAME', 'r') as f:
  project_name = f.read()

# python3 requires wheel 0.26
if sys.version_info.major == 3:
  REQUIRED_PACKAGES.append('wheel >= 0.26')
else:
  REQUIRED_PACKAGES.append('wheel')
  # mock comes with unittest.mock for python3, need to install for python2
  REQUIRED_PACKAGES.append('mock >= 2.0.0')

# weakref.finalize and enum were introduced in Python 3.4
if sys.version_info < (3, 4):
  REQUIRED_PACKAGES.append('backports.weakref >= 1.0rc1')
  REQUIRED_PACKAGES.append('enum34 >= 1.1.6')

# pylint: disable=line-too-long
CONSOLE_SCRIPTS = [
    #    'freeze_graph = tensorflow.python.tools.freeze_graph:run_main',
    #    'toco_from_protos = tensorflow.lite.toco.python.toco_from_protos:main',
    #    'tflite_convert = tensorflow.lite.python.tflite_convert:main',
    #    'toco = tensorflow.lite.python.tflite_convert:main',
    #    'saved_model_cli = tensorflow.python.tools.saved_model_cli:main',
    #    # We need to keep the TensorBoard command, even though the console script
    #    # is now declared by the tensorboard pip package. If we remove the
    #    # TensorBoard command, pip will inappropriately remove it during install,
    #    # even though the command is not removed, just moved to a different wheel.
    #    'tensorboard = tensorboard.main:run_main',
    #    'tf_upgrade_v2 = tensorflow.tools.compatibility.tf_upgrade_v2_main:main',
]
# pylint: enable=line-too-long

TEST_PACKAGES = [
    'scipy >= 0.15.1',
]


class BinaryDistribution(Distribution):

  def has_ext_modules(self):
    return True


class InstallCommand(InstallCommandBase):
  """Override the dir where the headers go."""

  def finalize_options(self):
    InstallCommandBase.finalize_options(self)
    self.install_headers = os.path.join(self.install_purelib,
                                        'tensorflow-plugins', 'include')


def find_files(pattern, root):
  """Return all the files matching pattern below root dir."""
  for dirpath, _, files in os.walk(root):
    for filename in fnmatch.filter(files, pattern):
      yield os.path.join(dirpath, filename)


so_lib_paths = [
    i for i in os.listdir('.')
    if os.path.isdir(i) and fnmatch.fnmatch(i, '_solib_*')
]

print(os.listdir('.'))

setup(
    name=project_name,
    version=_VERSION.replace('-', ''),
    description=DOCLINES[0],
    long_description='\n'.join(DOCLINES[2:]),
    url='https://github.com/microsoft/tensorflow-directml-plugin',
    download_url='https://github.com/microsoft/tensorflow-directml-plugin',
    author='Microsoft',
    author_email='askdirectml@microsoft.com',
    # Contained modules and scripts.
    packages=[_PLUGIN_LIB_PATH, _MY_PLUGIN_PATH],
    entry_points={
        'console_scripts': CONSOLE_SCRIPTS,
    },
    install_requires=REQUIRED_PACKAGES,
    tests_require=REQUIRED_PACKAGES + TEST_PACKAGES,
    package_data={
        _PLUGIN_LIB_PATH: ['*.so', '*.dll', '*/*.dll', '*/*.so', '*/*.txt'],
        _MY_PLUGIN_PATH: ['*', '*/*']
    },
    zip_safe=False,
    distclass=BinaryDistribution,
    cmdclass={
        'install': InstallCommand,
    },
    # PyPI package information.
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Software Development',
        'Topic :: Software Development :: Libraries',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],
    license='MIT',
    keywords='tensorflow tensor machine learning plugin',
)
