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
"""Helper script to drive testing tensorflow-directml-plugin."""

from genericpath import exists
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
import os
import re
import sys

class TestGroup:
    def __init__(self, name, tests, timeout_seconds, results_dir):
        self.name = name
        self.timeout_seconds = timeout_seconds
        self.tests = tests
        self.results_dir = results_dir
        self.results_file_path = None
        self.results_file_path = Path(results_dir) / f"run.{name}.json"
        self.summary_file_path = Path(results_dir) / f"summary.{name}.json"

    def show(self):
        if self.tests:
            print('-'*80)
            print(f"Test Group '{self.name}'")
            print('-'*80)
            for test in self.tests:
                test.show()

    def run(self):
        if self.tests:
            tests_timed_out = []
            tests_aborted = []
            start_time = time.time()
            for test in self.tests:
                elapsed_time = time.time() - start_time
                remaining_time = self.timeout_seconds - elapsed_time
                test.run(remaining_time)
                if test.timed_out:
                    tests_timed_out.append(test.name)
                if test.aborted:
                    tests_aborted.append(test.name)
            end_time = time.time()
            if self.results_file_path:
                with open(self.results_file_path, "w") as file:
                    summary = {}
                    summary["time_seconds"] = end_time - start_time
                    summary["tests_timed_out"] = tests_timed_out
                    summary["tests_aborted"] = tests_aborted
                    json.dump(summary, file)

    def summarize(self):
        time_seconds = 0
        tests_timed_out = []
        tests_aborted = []
        if Path(self.results_file_path).exists():
            with open(self.results_file_path, "r") as json_file:
                json_data = json.load(json_file)
                time_seconds = json_data["time_seconds"]
                tests_timed_out = json_data["tests_timed_out"]
                tests_aborted = json_data["tests_aborted"]

        summary = {}
        summary["group"] = self.name
        summary["time_seconds"] = time_seconds
        summary["cases_passed"] = []
        summary["cases_failed"] = []
        summary["cases_skipped"] = []
        summary["tests_timed_out"] = tests_timed_out
        summary["tests_aborted"] = tests_aborted

        for test in self.tests:
            test_summary = test.summarize()
            for test_case in test_summary["cases"]:
                if test_case["result"] == "passed":
                    summary["cases_passed"].append(test_case["name"])
                elif test_case["result"] == "failed":
                    summary["cases_failed"].append(test_case["name"])
                elif test_case["result"] == "skipped":
                    summary["cases_skipped"].append(test_case["name"])
        
        with open(self.summary_file_path, "w") as json_file:
            json.dump(summary, json_file)

        return summary

    def print_summary(self):
        if not self.tests:
            return
        summary = self.summarize()

        cases_passed_count = len(summary["cases_passed"])
        cases_failed_count = len(summary["cases_failed"])
        cases_skipped_count = len(summary["cases_skipped"])
        tests_timed_out = summary["tests_timed_out"]
        tests_aborted = summary["tests_aborted"]
        cases_ran_count = cases_passed_count + cases_failed_count + cases_skipped_count
        
        print()
        print('=' * 80)
        print(f"Test Group         : {summary['group']}")
        print(f"Test Duration      : {summary['time_seconds']} seconds")
        print(f"Test Cases Ran     : {cases_ran_count}")
        print(f"Test Cases Passed  : {cases_passed_count}")
        print(f"Test Cases Skipped : {cases_skipped_count}")
        print(f"Test Cases Failed  : {cases_failed_count}")
        print(f"Tests Timed Out    : {len(tests_timed_out)}")
        print(f"Tests Aborted      : {len(tests_aborted)}")
        if cases_failed_count > 0:
            print()
            print("Failing Test Cases: ")
            for i in range(0, cases_failed_count):
                failed_case = summary["cases_failed"][i]
                print(f"{i}: {failed_case}")
        if len(tests_timed_out) > 0:
            print()
            print("Timed-Out Tests: ")
            for i in range(0, len(tests_timed_out)):
                test = tests_timed_out[i]
                print(f"{i}: {test}")
        if len(tests_aborted) > 0:
            print()
            print("Aborted Tests: ")
            for i in range(0, len(tests_aborted)):
                test = tests_aborted[i]
                print(f"{i}: {test}")
        print('=' * 80)
        print()


class Test:
    def __init__(self, type, test_file_path, name, args, timeout_seconds, results_dir, redirect_output):
        self.type = type
        self.test_file_path = test_file_path
        self.name = name
        self.args = args
        self.timeout_seconds = timeout_seconds
        self.timed_out = False
        self.aborted = False

        self.log_file_path = None
        if redirect_output:
            self.log_file_path = Path(results_dir) / f"log.{name}.txt"

        self.results_file_path = None
        if type == "py_abseil":
            self.results_file_path = Path(results_dir) / f"test.{name}.xml"
            self.args.append(f"--xml_output_file {self.results_file_path}")
            self.command_line = f"python {test_file_path} {' '.join(self.args)}"
        else:
            raise Exception(f"Unknown test type: {type}")

    def show(self):
        print(f"{self.name}: {self.command_line}")

    def run(self, remaining_time_in_test_group):
        self.timed_out = False
        self.aborted = False
        timeout_seconds = remaining_time_in_test_group
        if self.timeout_seconds:
            timeout_seconds = min(self.timeout_seconds, timeout_seconds)
        if timeout_seconds <= 0:
            self.timed_out = True
            return

        print(f"Running '{self.name}' with a timeout of {timeout_seconds} seconds")
        sys.stdout.flush()
        environ = os.environ.copy()
        environ['PYTHONIOENCODING'] = 'utf-8'
        p = None
        try:
            p = subprocess.run(
                self.command_line, 
                timeout=timeout_seconds,
                stdin=subprocess.DEVNULL if self.log_file_path else None,
                stdout=subprocess.PIPE if self.log_file_path else None,
                stderr=subprocess.STDOUT if self.log_file_path else None,
                universal_newlines=True,
                encoding='utf-8',
                env=environ,
                shell=True
            )
        except subprocess.TimeoutExpired:
            self.timed_out = True
        
        if p.returncode != 0:
            self.aborted = True

        if p and self.log_file_path:
            results_subdir = Path(self.log_file_path).parent
            if not results_subdir.exists():
                results_subdir.mkdir(exist_ok=True, parents=True)
            with open(self.log_file_path, "w", encoding='utf-8') as log_file:
                log_file.write(str(p.stdout))

    def summarize(self):
        summary = {}
        summary["name"] = self.name
        summary["cases"] = []

        if not Path(self.results_file_path).exists():
            return summary
        
        try:
            root = ET.parse(self.results_file_path).getroot()
        except:
            return summary

        for test_suite in root.findall("testsuite"):
            test_suite_name = test_suite.attrib["name"]
            for test_case in test_suite.findall("testcase"):
                test_case_summary = {}
                test_case_summary["name"] = f"{self.name}::{test_suite_name}.{test_case.attrib['name']}"
                test_case_summary["module"] = Path(self.results_file_path).stem
                test_case_summary["time"] = test_case.attrib["time"]

                # Failures are saved as children nodes instead of attributes
                failures = test_case.findall("failure") + test_case.findall("error")
                if failures:
                    test_case_summary["result"] = "failed"
                else:
                    status = test_case.attrib.get("status", "")
                    result = test_case.attrib.get("result", "")

                    if status == "run" or result == "completed":
                        test_case_summary["result"] = "passed"
                    elif status == "skipped" or result == "suppressed":
                        test_case_summary["result"] = "skipped"
                summary["cases"].append(test_case_summary)
        
        return summary


def get_optional_json_property(json_object, property_name, default_value):
    if property_name in json_object:
        return json_object[property_name]
    return default_value


# Parses tests.json to build a list of test groups to execute.
def parse_test_groups(tests_json_path, test_filter, allowed_test_groups, results_dir, run_disabled, redirect_output):
    test_groups = []
    test_names = set()

    with open(tests_json_path, "r") as json_file:
        json_data = json.load(json_file)

    for json_test_group in json_data["groups"]:
        test_group_name = json_test_group["name"]
        
        if allowed_test_groups and test_group_name not in allowed_test_groups:
            continue

        test_group_tests = []
        test_group_timeout_seconds = get_optional_json_property(json_test_group, "timeout_seconds", 300)

        for json_test in json_test_group["tests"]:
            test_type = "py_abseil"
            test_file = Path(tests_json_path).parent / json_test["file"]
            test_base_name = get_optional_json_property(json_test, "name", Path(test_file).stem)
            if not re.match("^\w+$", test_base_name):
                raise Exception(f"'{test_base_name}' is an invalid test name. Test names must only contain word characters: a-z, A-Z, 0-9, and '_'.")

            test_full_name = f"{test_group_name}.{test_base_name}"
            test_args = get_optional_json_property(json_test, "args", [])
            test_disabled = get_optional_json_property(json_test, "disabled", False)
            test_timeout_seconds = get_optional_json_property(json_test, "timeout_seconds", None)

            # Ensure test names are unique across all test groups.
            if test_full_name in test_names:
                raise Exception(f"{tests_json_path} contains a duplicate test: {test_full_name}.")
            test_names.add(test_full_name)

            if (not test_disabled or run_disabled) and fnmatch.fnmatch(test_full_name, test_filter):
                test_group_tests.append(Test(
                    test_type,
                    test_file,
                    test_full_name,
                    test_args,
                    test_timeout_seconds,
                    results_dir,
                    redirect_output
                ))
        
        test_groups.append(TestGroup(
            test_group_name, 
            test_group_tests, 
            test_group_timeout_seconds,
            results_dir
        ))
    
    return test_groups


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tests_json", 
        type=str, 
        default=Path(__file__).resolve().parent / "tests.json",
        help="Path to tests.json file."
    )
    parser.add_argument(
        "--run", "-r",
        action="store_true", 
        help="Shows test commands instead of executing them."
    )
    parser.add_argument(
        "--show", "-w",
        action="store_true", 
        help="Shows test commands instead of executing them."
    )
    parser.add_argument(
        "--summarize", "-s",
        action="store_true",
        help="Summarizes test results."
    )
    parser.add_argument(
        "--results_dir", 
        type=str, 
        default=Path(tempfile.gettempdir()) / "tfdml_plugin_tests", 
        help="Directory to save test results."
    )
    parser.add_argument(
        "--tests", "-t",
        type=str, 
        default="*",
        help="Filters tests. This is an fnmatch filter."
    )
    parser.add_argument(
        "--groups", "-g",
        type=str, 
        nargs='+',
        help="Filters test groups. This is a comma-separated list of group names."
    )
    parser.add_argument(
        "--run_disabled",
        action="store_true",
        help="Runs tests even if they are disabled."
    )
    parser.add_argument(
        "--redirect_output",
        action="store_true",
        help="Redirects test console output to log files in the results directory."
    )
    args = parser.parse_args()

    # Parse tests from tests.json.
    test_groups = parse_test_groups(
        args.tests_json, 
        args.tests, 
        args.groups,
        args.results_dir, 
        args.run_disabled, 
        args.redirect_output
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
            test_group.run()

    # Summarize test results.
    if args.summarize:
        for test_group in test_groups:
            test_group.print_summary()


if __name__ == "__main__":
    main()