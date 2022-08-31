"""
Helpers used by the tensorflow-directml-plugin python tests
"""
import subprocess
import sys
import pathlib
import re


def _get_adapter_name():
    """Returns the name of the first adapter created by tensorflow-directml-plugin"""
    file_path = str(
        pathlib.Path(__file__).parent.absolute().joinpath("dml_get_adapter_name.py")
    )
    log_output = subprocess.check_output(
        [sys.executable, file_path], stderr=subprocess.STDOUT
    )
    matches = re.match(
        r".*DirectML: creating device on adapter 0 \((.+?)\)\\n", str(log_output)
    )
    return matches.group(1)


def should_skip_test(pattern):
    """
    Returns whether the test should be skipped based on a regex that should match the
    adapter name
    """
    return re.match(pattern, _get_adapter_name()) is not None
