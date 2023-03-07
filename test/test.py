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
"""Helper script to drive testing tensorflow-directml-plugin."""

import subprocess
import json
import argparse
from pathlib import Path
import xml.etree.ElementTree as ET
import os
import fnmatch
import shutil
import time
import tempfile
import re
import sys
from multiprocessing import Pool


def _test_runner(test, timeout_seconds, start_time, redirect_output):
    try:
        elapsed_time = time.time() - start_time
        test.run(timeout_seconds - elapsed_time, redirect_output)
    except KeyboardInterrupt:
        return False
    return True


class _TestGroup:
    # pylint:disable=too-many-arguments
    def __init__(self, name, tests, is_python_test, timeout_seconds, results_dir):
        self.name = name
        self.is_python_test = is_python_test
        self.timeout_seconds = timeout_seconds
        self.tests = tests
        self.run_results_file_path = Path(results_dir) / f"run.{name}.json"
        self.summary_file_path = Path(results_dir) / f"summary.{name}.json"

    def show(self):
        """Prints the command lines that would be executed without executing anything"""
        if self.tests:
            print("-" * 80)
            print(f"Test Group '{self.name}'")
            print("-" * 80)
            for test in self.tests:
                test.show()

    def run(self, parallel, redirect_output):
        """Runs the tests"""
        if self.tests:
            start_time = time.time()
            results = []

            if parallel:
                with Pool(processes=6) as pool:
                    for test in self.tests:
                        result = pool.apply_async(
                            _test_runner,
                            (test, self.timeout_seconds, start_time, redirect_output),
                        )
                        results.append(result)

                    for result in results:
                        if not result.get():
                            raise KeyboardInterrupt
            else:
                for test in self.tests:
                    if not _test_runner(
                        test, self.timeout_seconds, start_time, redirect_output
                    ):
                        raise KeyboardInterrupt
            end_time = time.time()

            with open(self.run_results_file_path, "w", encoding="utf-8") as file:
                summary = {}
                summary["name"] = self.name
                summary["start_timestamp_seconds"] = start_time
                summary["end_timestamp_seconds"] = end_time
                summary["duration_seconds"] = end_time - start_time
                json.dump(summary, file)

    def summarize(self):
        """Outputs the results of the TestGroup run"""
        # Fetch last run metadata.
        start_timestamp_seconds = 0
        end_timestamp_seconds = 0
        duration_seconds = 0
        if Path(self.run_results_file_path).exists():
            with open(self.run_results_file_path, "r", encoding="utf-8") as json_file:
                json_data = json.load(json_file)
                start_timestamp_seconds = json_data["start_timestamp_seconds"]
                end_timestamp_seconds = json_data["end_timestamp_seconds"]
                duration_seconds = json_data["duration_seconds"]

        summary = {}
        summary["name"] = self.name
        summary["start_timestamp_seconds"] = start_timestamp_seconds
        summary["end_timestamp_seconds"] = end_timestamp_seconds
        summary["duration_seconds"] = duration_seconds
        summary["tests_total_count"] = 0
        summary["tests_passed_count"] = 0
        summary["tests_failed_count"] = 0
        summary["tests_skipped_count"] = 0
        summary["tests_timed_out_count"] = 0
        summary["cases_total_count"] = 0
        summary["cases_passed_count"] = 0
        summary["cases_failed_count"] = 0
        summary["cases_skipped_count"] = 0
        summary["tests"] = []

        for test in self.tests:
            test_summary = test.summarize()

            summary["tests_total_count"] += 1
            if test_summary["result"] == "passed":
                summary["tests_passed_count"] += 1
            elif test_summary["result"] == "failed":
                summary["tests_failed_count"] += 1
            elif test_summary["result"] == "skipped":
                summary["tests_skipped_count"] += 1
            elif test_summary["result"] == "timed_out":
                summary["tests_timed_out_count"] += 1

            summary["cases_total_count"] += test_summary["cases_total_count"]
            summary["cases_passed_count"] += test_summary["cases_passed_count"]
            summary["cases_failed_count"] += test_summary["cases_failed_count"]
            summary["cases_skipped_count"] += test_summary["cases_skipped_count"]
            summary["tests"].append(test_summary)

        with open(self.summary_file_path, "w", encoding="utf-8") as json_file:
            json.dump(summary, json_file)

        return summary

    def print_summary(self, summary):
        """Prints a summary of the results of the TestGroup run"""
        if not self.tests:
            return

        print()
        print("=" * 80)
        print(f"Test Group      : {summary['name']}")
        print(f"Test Duration   : {summary['duration_seconds']} seconds")
        print(
            f"Tests Total     : {summary['tests_total_count']} "
            f"({summary['cases_total_count']} cases)"
        )
        print(
            f"Tests Passed    : {summary['tests_passed_count']} "
            f"({summary['cases_passed_count']} cases)"
        )
        print(
            f"Tests Skipped   : {summary['tests_skipped_count']} "
            f"({summary['cases_skipped_count']} cases)"
        )
        print(
            f"Tests Failed    : {summary['tests_failed_count']} "
            f"({summary['cases_failed_count']} cases)"
        )
        print(f"Tests Timed Out : {summary['tests_timed_out_count']}")
        if summary["tests_failed_count"] > 0:
            print()
            print("Failed Tests:")
            fail_count = 0
            for test in summary["tests"]:
                if test["result"] == "failed":
                    print(f"{fail_count}: {test['name']}")
                    if len(test["cases_failed"]) > 0:
                        for failure, failure_index in enumerate(test["cases_failed"]):
                            print(f"{fail_count}.{failure_index}: {failure}")
                    fail_count += 1

        print("=" * 80)
        print()


# pylint:disable=too-many-instance-attributes
class _Test:
    """Represents a single test file that is part of a test group"""

    # pylint:disable=too-many-arguments
    def __init__(
        self,
        is_python_test,
        cwd,
        test_file_path,
        name,
        args,
        results_dir,
        gpu_name,
    ):
        self.is_python_test = is_python_test
        self.cwd = cwd
        self.name = name
        self.args = args
        self.results_dir = results_dir
        self.run_results_file_path = Path(results_dir) / f"run.{name}.json"
        self.results_file_path = Path(results_dir) / f"test.{name}.xml"

        if is_python_test:
            self.args.append(f"--xml_output_file={self.results_file_path}")
        else:
            self.args.append(f"--gtest_output=xml:{self.results_file_path}")

        if str(name) == "ops.reduction_ops_test":
            self.args = [gpu_name.replace(" ", "")] + self.args

        if is_python_test:
            self.command_line = f"python {test_file_path} {' '.join(self.args)}"
        else:
            self.command_line = f"{test_file_path} {' '.join(self.args)}"

    def show(self):
        """Prints the command lines that would be executed without executing anything"""
        print(f"{self.name}: {self.command_line}")

    def run(self, timeout_seconds, redirect_output):
        """Runs the test"""
        run_state = "not_run"

        # Run the test.
        start_time = time.time()
        if timeout_seconds <= 0:
            run_state = "timed_out"
        else:
            if redirect_output:
                log_file_path = Path(self.results_dir) / f"log.{self.name}.txt"

            print(f"Running '{self.name}' with a timeout of {timeout_seconds} seconds")
            sys.stdout.flush()
            environ = os.environ.copy()
            environ["PYTHONIOENCODING"] = "utf-8"
            process_result = None
            try:
                process_result = subprocess.run(
                    self.command_line,
                    cwd=self.cwd,
                    timeout=timeout_seconds,
                    stdin=subprocess.DEVNULL if redirect_output else None,
                    stdout=subprocess.PIPE if redirect_output else None,
                    stderr=subprocess.STDOUT if redirect_output else None,
                    universal_newlines=True,
                    encoding="utf-8",
                    env=environ,
                    shell=True,
                    check=True,
                )
                run_state = "completed"
            except subprocess.TimeoutExpired:
                run_state = "timed_out"
            except subprocess.CalledProcessError:
                run_state = "exited_abnormally"

            if process_result and redirect_output:
                results_subdir = Path(log_file_path).parent
                if not results_subdir.exists():
                    results_subdir.mkdir(exist_ok=True, parents=True)
                with open(log_file_path, "w", encoding="utf-8") as log_file:
                    log_file.write(str(process_result.stdout))
        end_time = time.time()

        # Write run results JSON.
        with open(self.run_results_file_path, "w", encoding="utf-8") as file:
            summary = {}
            summary["name"] = self.name
            summary["start_timestamp_seconds"] = start_time
            summary["end_timestamp_seconds"] = end_time
            summary["duration_seconds"] = end_time - start_time
            summary["run_state"] = run_state
            json.dump(summary, file)

    def summarize(self):
        """Outputs the results of the Test run"""
        summary = {}

        # Fetch last run metadata.
        if Path(self.run_results_file_path).exists():
            with open(self.run_results_file_path, "r", encoding="utf-8") as json_file:
                json_data = json.load(json_file)
            run_state = json_data["run_state"]
            summary["start_timestamp_seconds"] = json_data["start_timestamp_seconds"]
            summary["end_timestamp_seconds"] = json_data["end_timestamp_seconds"]
            summary["duration_seconds"] = json_data["duration_seconds"]
        else:
            run_state = "unknown"

        summary["name"] = self.name
        summary["result"] = "unknown"
        summary["cases_total_count"] = 0
        summary["cases_passed_count"] = 0
        summary["cases_failed_count"] = 0
        summary["cases_skipped_count"] = 0
        summary["cases_failed"] = []

        if Path(self.results_file_path).exists() and run_state != "timed_out":
            try:
                root = ET.parse(self.results_file_path).getroot()
            except ET.ParseError:
                print(f"Error while parsing '{self.results_file_path}'")
                summary["result"] = "failed"
                return summary

            for test_suite in root.findall("testsuite"):
                test_suite_name = test_suite.attrib["name"]
                for test_case in test_suite.findall("testcase"):
                    case_name = (
                        f"{self.name}::{test_suite_name}.{test_case.attrib['name']}"
                    )

                    summary["cases_total_count"] += 1

                    # Failures are saved as children nodes instead of attributes
                    failures = test_case.findall("failure") + test_case.findall("error")
                    if failures:
                        summary["cases_failed_count"] += 1
                        summary["cases_failed"].append(case_name)
                    else:
                        status = test_case.attrib.get("status", "")
                        result = test_case.attrib.get("result", "")

                        if status == "run" or result == "completed":
                            summary["cases_passed_count"] += 1
                        elif status == "skipped" or result == "suppressed":
                            summary["cases_skipped_count"] += 1

        _set_summary_result(summary, run_state)
        return summary


def _set_summary_result(summary, run_state):
    if run_state == "timed_out":
        summary["result"] = "timed_out"
    elif run_state == "exited_abnormally" or summary["cases_failed_count"] > 0:
        summary["result"] = "failed"
    elif summary["cases_passed_count"] > 0:
        summary["result"] = "passed"
    elif summary["cases_skipped_count"] > 0:
        summary["result"] = "skipped"
    else:
        summary["result"] = "unknown"


def _get_optional_json_property(json_object, property_name, default_value):
    if property_name in json_object:
        return json_object[property_name]
    return default_value


# Parses tests.json to build a list of test groups to execute.
def _parse_test_groups(
    tests_json_path,
    test_filter,
    allowed_test_groups,
    results_dir,
    run_disabled,
    gpu_name,
):
    test_groups = []
    test_names = set()

    with open(tests_json_path, "r", encoding="utf-8") as json_file:
        json_data = json.load(json_file)

    for json_test_group in json_data["groups"]:
        test_group_name = json_test_group["name"]

        if allowed_test_groups and test_group_name not in allowed_test_groups:
            continue

        test_group_tests = []

        for json_test in json_test_group["tests"]:
            test_disabled = _get_optional_json_property(json_test, "disabled", False)
            test = _parse_test(
                json_test_group["is_python_test"],
                tests_json_path,
                json_test,
                test_names,
                test_group_name,
                results_dir,
                gpu_name,
            )

            if (
                (not test_disabled or run_disabled)
                and os.name not in json_test.get("disabled_platforms", [])
                and fnmatch.fnmatch(test.name, test_filter)
            ):
                test_group_tests.append(test)

        test_groups.append(
            _TestGroup(
                test_group_name,
                test_group_tests,
                json_test_group["is_python_test"],
                _get_optional_json_property(json_test_group, "timeout_seconds", 300),
                results_dir,
            )
        )

    return test_groups


# pylint:disable=too-many-arguments
def _parse_test(
    is_python_test,
    tests_json_path,
    json_test,
    test_names,
    test_group_name,
    results_dir,
    gpu_name,
):
    test_file = Path(tests_json_path).parent / json_test["file"]
    test_base_name = _get_optional_json_property(
        json_test, "name", Path(test_file).stem
    )
    if not re.match(r"^\w+$", test_base_name):
        raise Exception(
            f"'{test_base_name}' is an invalid test name. Test names "
            f"must only contain word characters: a-z, A-Z, 0-9, and '_'."
        )

    test_full_name = f"{test_group_name}.{test_base_name}"
    test_args = _get_optional_json_property(json_test, "args", [])

    # Ensure test names are unique across all test groups.
    if test_full_name in test_names:
        raise Exception(
            f"{tests_json_path} contains a duplicate test: {test_full_name}."
        )
    test_names.add(test_full_name)

    return _Test(
        is_python_test,
        json_test.get("cwd"),
        test_file,
        test_full_name,
        test_args,
        results_dir,
        gpu_name,
    )


def _main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tests_json",
        type=str,
        default=Path(__file__).resolve().parent / "tests.json",
        help="Path to tests.json file.",
    )
    parser.add_argument(
        "--run",
        "-r",
        action="store_true",
        help="Shows test commands instead of executing them.",
    )
    parser.add_argument(
        "--show",
        "-w",
        action="store_true",
        help="Shows test commands instead of executing them.",
    )
    parser.add_argument(
        "--summarize", "-s", action="store_true", help="Summarizes test results."
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default=Path(tempfile.gettempdir()) / "tfdml_plugin_tests",
        help="Directory to save test results.",
    )
    parser.add_argument(
        "--tests",
        "-t",
        type=str,
        default="*",
        help="Filters tests. This is an fnmatch filter.",
    )
    parser.add_argument(
        "--groups",
        "-g",
        type=str,
        nargs="+",
        help="Filters test groups. This is a comma-separated list of group names.",
    )
    parser.add_argument(
        "--run_disabled",
        action="store_true",
        help="Runs tests even if they are disabled.",
    )
    parser.add_argument(
        "--redirect_output",
        action="store_true",
        help="Redirects test console output to log files in the results directory.",
    )
    parser.add_argument(
        "--parallel",
        "-p",
        action="store_true",
        help="Runs multiple tests in parallel using a multiprocessing pool.",
    )
    parser.add_argument(
        "--gpu_name",
        type=str,
        default="",
        help="Name of GPU, used for disabling tests for specific cards.",
    )
    args = parser.parse_args()

    if not (args.run or args.show or args.summarize):
        print("No mode specified. Did you intend to use --run, --summarize, or --show?")
        return

    print("args.gpu_name: ", args.gpu_name)

    # Parse tests from tests.json.
    test_groups = _parse_test_groups(
        args.tests_json,
        args.tests,
        args.groups,
        args.results_dir,
        args.run_disabled,
        args.gpu_name,
    )

    # Show test commands.
    if args.show:
        for test_group in test_groups:
            test_group.show()

    # Execute tests.
    if args.run:
        # Delete previous results, if any.
        if Path(args.results_dir).exists():
            shutil.rmtree(args.results_dir)
            os.makedirs(args.results_dir)

        for test_group in test_groups:
            test_group.run(args.parallel, args.redirect_output)

    failures = False

    # Summarize test results.
    for test_group in test_groups:
        summary = test_group.summarize()

        if args.summarize:
            test_group.print_summary(summary)

        if summary["tests_failed_count"] > 0 or summary["tests_timed_out_count"] > 0:
            failures = True

    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    _main()
