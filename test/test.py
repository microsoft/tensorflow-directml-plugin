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
import re
import fnmatch
import shutil
import time

class TestGroup:
    def __init__(self, name, tests, timeout_seconds, results_dir):
        self.name = name
        self.timeout_seconds = timeout_seconds
        self.tests = tests
        self.results_dir = results_dir
        self.results_file_path = None
        self.tests_timed_out = []
        if results_dir:
            self.results_file_path = Path(results_dir) / f"{name}.json"

    def show(self):
        if self.tests:
            print('-'*80)
            print(f"Test Group '{self.name}'")
            print('-'*80)
            for test in self.tests:
                test.show()

    def run(self):
        if self.tests:
            self.tests_timed_out = []
            start_time = time.time()
            for test in self.tests:
                test.run()
                if test.timed_out:
                    self.tests_timed_out.append(test.name)
            end_time = time.time()
            if self.results_file_path:
                with open(self.results_file_path, "w") as file:
                    summary = {}
                    summary["time_seconds"] = end_time - start_time
                    summary["tests_timed_out"] = self.tests_timed_out
                    json.dump(summary, file)

    def summarize(self):
        time_seconds = 0
        tests_timed_out = []
        if Path(self.results_file_path).exists():
            with open(self.results_file_path, "r") as json_file:
                json_data = json.load(json_file)
                time_seconds = json_data["time_seconds"]
                tests_timed_out = json_data["tests_timed_out"]

        summary = {}
        summary["group"] = self.name
        summary["time_seconds"] = time_seconds
        summary["cases_ran"] = 0
        summary["cases_passed"] = 0
        summary["cases_failed"] = 0
        summary["cases_skipped"] = 0
        summary["tests_timed_out"] = tests_timed_out

        for test in self.tests:
            test_summary = test.summarize()
            if test_summary:
                for test_case in test_summary:
                    summary["cases_ran"] += 1
                    if test_case["Result"] == "Pass":
                        summary["cases_passed"] += 1
                    elif test_case["Result"] == "Fail":
                        summary["cases_failed"] += 1
                    elif test_case["Result"] == "Skipped":
                        summary["cases_skipped"] += 1
        
        return summary


class Test:
    def __init__(self, type, test_file_path, name, args, timeout_seconds, results_dir):
        self.type = type
        self.test_file_path = test_file_path
        self.name = name
        self.args = args
        self.timeout_seconds = timeout_seconds
        self.results_file_path = None
        self.log_file_path = None
        self.timed_out = False

        if type == "py_abseil":
            if results_dir:
                self.results_file_path = Path(results_dir) / f"{name}.xml"
                self.log_file_path = Path(results_dir) / f"{name}.txt"
                self.args.append(f"--xml_output_file {self.results_file_path}")
            self.command_line = f"python {test_file_path} {' '.join(self.args)}"
        else:
            raise Exception(f"Unknown test type: {type}")

    def show(self):
        print(f"{self.name}: {self.command_line}")

    def run(self):
        print(f"Running '{self.name}'")
        environ = os.environ.copy()
        environ['PYTHONIOENCODING'] = 'utf-8'
        p = None
        try:
            p = subprocess.run(
                self.command_line, 
                timeout=self.timeout_seconds,
                stdin=subprocess.DEVNULL if self.log_file_path else None,
                stdout=subprocess.PIPE if self.log_file_path else None,
                stderr=subprocess.STDOUT if self.log_file_path else None,
                universal_newlines=True,
                encoding='utf-8',
                env=environ,
            )
        except subprocess.TimeoutExpired:
            self.timed_out = True

        if p and self.log_file_path:
            results_subdir = Path(self.log_file_path).parent
            if not results_subdir.exists():
                results_subdir.mkdir(exist_ok=True, parents=True)
            with open(self.log_file_path, "w", encoding='utf-8') as log_file:
                log_file.write(str(p.stdout))

    def summarize(self):
        if not Path(self.results_file_path).exists():
            return None
        
        try:
            summary = []
            root = ET.parse(self.results_file_path).getroot()
            for test_suite in root.findall("testsuite"):
                test_suite_name = test_suite.attrib["name"]
                for test_case in test_suite.findall("testcase"):
                    test_case_name = test_case.attrib['name']
                    json_test_case = {}
                    json_test_case["Name"] = f"{test_suite_name}.{test_case_name}"
                    json_test_case["Module"] = Path(self.results_file_path).stem
                    json_test_case["Time"] = test_case.attrib["time"]

                    # Failures are saved as children nodes instead of attributes
                    failures = test_case.findall("failure") + test_case.findall("error")

                    if failures:
                        json_test_case["Result"] = "Fail"
                        error_strings = []

                        for failure in failures:
                            failure_message = failure.attrib["message"]
                            if re.match(r".+\.(cpp|h):\d+", failure_message) is None:
                                error_strings.append(failure_message)
                            else:
                                file_path = re.sub(r"(.+):\d+", lambda match: match.group(1), failure_message)
                                line_number = re.sub(r".+:(\d+)", lambda match: match.group(1), failure_message)
                                message = re.sub(r"&#xA(.+)", lambda match: match.group(1), failure_message)
                                error_strings.append(f"{message} [{file_path}:{line_number}]")

                        json_test_case["Errors"] = "".join(error_strings).replace("&#xA", "     ")
                    else:
                        status = test_case.attrib.get("status", "")
                        result = test_case.attrib.get("result", "")

                        if status == "run" or result == "completed":
                            json_test_case["Result"] = "Pass"
                        elif status == "skipped" or result == "suppressed":
                            json_test_case["Result"] = "Skipped"
                        else:
                            json_test_case["Result"] = "Blocked"
                    summary.append(json_test_case)

            return summary
        except ET.ParseError:
            return None


def get_optional_json_property(json_object, property_name, default_value):
    if property_name in json_object:
        return json_object[property_name]
    return default_value


# Parses tests.json to build a list of test groups to execute.
def parse_test_groups(tests_json_path, test_filter, results_dir, run_disabled):
    test_groups = []
    test_names = set()

    with open(tests_json_path, "r") as json_file:
        json_data = json.load(json_file)

    for json_test_group in json_data["groups"]:
        test_group_name = json_test_group["name"]
        test_group_tests = []
        test_group_timeout_seconds = get_optional_json_property(json_test_group, "timeout_seconds", 300)

        for json_test in json_test_group["tests"]:
            test_type = "py_abseil"
            test_file = Path(tests_json_path).parent / json_test["file"]
            test_base_name = get_optional_json_property(json_test, "name", Path(test_file).stem)
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
                    results_dir
                ))
        
        test_groups.append(TestGroup(
            test_group_name, 
            test_group_tests, 
            test_group_timeout_seconds,
            results_dir
        ))
    
    return test_groups


def summarize_results(test_groups, results_dir):
    if not results_dir:
        return

    for test_group in test_groups:
        summary = test_group.summarize()
        print(summary)


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
        default=None, 
        help="Directory to save test results. If empty no result files are written."
    )
    parser.add_argument(
        "--filter", "-f",
        type=str, 
        default="*",
        help="Filters test names to select a subset of the tests."
    )
    parser.add_argument(
        "--run_disabled",
        action="store_true",
        help="Runs tests even if they are disabled."
    )
    parser.add_argument(
        "--clean_results", "-x",
        action="store_true",
        help="Deletes the results_dir if it already exists."
    )
    args = parser.parse_args()

    # Parse tests from tests.json.
    test_groups = parse_test_groups(args.tests_json, args.filter, args.results_dir, args.run_disabled)

    # Show test commands.
    if args.show:
        for test_group in test_groups:
            test_group.show()
    
    # Delete previous results, if any.
    if args.clean_results and Path(args.results_dir).exists():
        shutil.rmtree(args.results_dir)

    # Execute tests.
    if args.run:
        for test_group in test_groups:
            test_group.run()

    # Summarize test results.
    if args.summarize:
        if not args.results_dir:
            raise Exception("You must specify a --results_dir when using the --summarize option.")
        summarize_results(test_groups, args.results_dir)

if __name__ == "__main__":
    main()