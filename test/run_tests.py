from genericpath import exists
import subprocess
import json
import argparse
from pathlib import Path
import xml.etree.ElementTree as ET
import os
import re

class TestGroup:
    def __init__(self, name, tests, timeout_seconds):
        self.name = name
        self.timeout_seconds = timeout_seconds
        self.tests = tests

    def show(self):
        if self.tests:
            print(f"Test Group '{self.name}':")
            for test in self.tests:
                test.show()

    def run(self):
        if self.tests:
            print(f"Test Group '{self.name}':")
            for test in self.tests:
                test.run()

    def summarize(self):
        summary = {}
        summary["Name"] = self.name
        summary["Time"] = 0.0
        # summary["Tests"] = []
        summary["Counts"] = {}
        summary["Counts"]["Total"] = 0
        summary["Counts"]["Passed"] = 0
        summary["Counts"]["Failed"] = 0
        summary["Counts"]["Skipped"] = 0
        summary["Counts"]["Blocked"] = 0

        # if test was expected to output a file but it doesn't exist (or is empty) then it's a test errro
        for test in self.tests:
            test_summary = test.summarize()
            if test_summary:
                for test_case in test_summary:
                    summary["Counts"]["Total"] += 1
                    if test_case["Result"] == "Pass":
                        summary["Counts"]["Passed"] += 1
                    elif test_case["Result"] == "Fail":
                        summary["Counts"]["Failed"] += 1
                    elif test_case["Result"] == "Skipped":
                        summary["Counts"]["Skipped"] += 1
                    else:
                        summary["Counts"]["Blocked"] += 1
        
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

        if type == "py_abseil":
            if results_dir:
                self.results_file_path = Path(results_dir) / f"{name}.xml"
                self.log_file_path = Path(results_dir) / f"{name}.txt"
                self.args.append(f"--xml_output_file {self.results_file_path}")
            self.command_line = f"python {test_file_path} {' '.join(self.args)}"
        else:
            raise f"Unknown test type: {type}"

    def show(self):
        print(f"    {self.command_line}")

    def run(self):
        print(f"    Test: {self.command_line}")
        environ = os.environ.copy()
        environ['PYTHONIOENCODING'] = 'utf-8'
        p = None
        try:
            p = subprocess.run(
                self.command_line, 
                timeout=self.timeout_seconds,
                stdin=subprocess.DEVNULL if self.log_path else None,
                stdout=subprocess.PIPE if self.log_path else None,
                stderr=subprocess.STDOUT if self.log_path else None,
                universal_newlines=True,
                encoding='utf-8',
                env=environ,
            )
        except subprocess.TimeoutExpired:
            print("Test timed out!")

        if p and self.log_path:
            results_subdir = Path(self.log_path).parent
            if not results_subdir.exists():
                results_subdir.mkdir(exist_ok=True, parents=True)
            with open(self.log_path, "w", encoding='utf-8') as log_file:
                log_file.write(str(p.stdout))

    def summarize(self):
        if not Path(self.results_path).exists():
            return None
        
        try:
            summary = []
            root = ET.parse(self.results_path).getroot()
            for test_suite in root.findall("testsuite"):
                test_suite_name = test_suite.attrib["name"]
                for test_case in test_suite.findall("testcase"):
                    test_case_name = test_case.attrib['name']
                    json_test_case = {}
                    json_test_case["Name"] = f"{test_suite_name}.{test_case_name}"
                    json_test_case["Module"] = Path(self.results_path).stem
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
def parse_test_groups(tests_json_path, test_filter = "", results_dir = None):
    test_groups = []
    test_names = set()

    with open(tests_json_path) as json_file:
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
                raise f"{tests_json_path} contains a duplicate test: {test_full_name}."
            test_names.add(test_full_name)

            if not test_disabled and re.match(test_filter, test_full_name):
                test_group_tests.append(Test(
                    test_type,
                    test_file,
                    test_full_name,
                    test_args,
                    test_timeout_seconds,
                    results_dir
                ))
        
        test_groups.append(TestGroup(test_group_name, test_group_tests, test_group_timeout_seconds))
    
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
        "--show", 
        action="store_true", 
        help="Shows test commands instead of executing them."
    )
    parser.add_argument(
        "--results_dir", 
        type=str, 
        default=None, 
        help="Directory to save test results. If empty no result files are written."
    )
    parser.add_argument(
        "--test_filter", 
        type=str, 
        default="",
        help="Filters test names to select a subset of the tests."
    )
    args = parser.parse_args()

    # Parse tests from tests.json.
    test_groups = parse_test_groups(args.tests_json, args.test_filter, args.results_dir)

    # Run or show all tests.
    for test_group in test_groups:
        if args.show:
            test_group.show()
        else:
            test_group.run()

    # Merge and summarize test results.
    summarize_results(test_groups, args.results_dir)

if __name__ == "__main__":
    main()