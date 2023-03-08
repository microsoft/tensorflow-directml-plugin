"""
Helpers used by the tensorflow-directml-plugin python tests
"""
import subprocess
import sys
import pathlib
import re


def should_skip_test(pattern, gpu_name):
    """
    Returns whether the test should be skipped based on a regex that should match the
    adapter name
    """
    return re.match(pattern, gpu_name) is not None
