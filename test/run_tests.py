from genericpath import exists
import subprocess
import json
import argparse
from pathlib import Path
import xml.etree.ElementTree as ET
import os

class Test:
    def __init__(self, test_group, command_line, timeout_seconds, results_path):
        self.test_group = test_group
        self.command_line = command_line
        self.timeout_seconds = timeout_seconds
        self.results_path = results_path
        if results_path:
            self.log_path = Path(results_path).parent / f"{Path(results_path).stem}.txt"
    
    def show(self):
        print(f"[{self.test_group}] {self.command_line} {self.log_path}")

    def run(self):
        print(f"Running Test: [{self.test_group}] {self.command_line}")
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


# Parses tests.json to build a list of command lines to execute.
def parse_tests(tests_root_dir, results_dir = None):
    test_processes = []

    with open(Path(tests_root_dir) / "tests.json") as json_file:
        tests_metadata = json.load(json_file)

    for test_group_name in tests_metadata["groups"]:
        test_group = tests_metadata["groups"][test_group_name]

        # Test groups of type 'python_abseil' point at a directory with .py test scripts.
        if test_group["type"] == "python_abseil":
            test_scripts_dir = tests_root_dir / Path(test_group["test_script_dir"])
            for test_script_path in test_scripts_dir.glob("*.py"):
                test_command_line = f"python {test_script_path}"

                if "disabled_tests" in test_group and test_script_path.name in test_group["disabled_tests"]:
                    continue

                results_path = None
                if results_dir:
                    results_path = Path(results_dir) / test_group_name / f"{test_script_path.stem}.xml"
                    test_command_line += f" --xml_output_file {results_path}"

                test_processes.append(Test(
                    test_group_name,
                    test_command_line, 
                    test_group["timeout_minutes"] * 60, 
                    results_path
                ))
        else:
            print("test_group test not supported")
    
    return test_processes

def summarize_results(tests, results_dir):
    if not results_dir:
        return

    for test in tests:
        test_results_path = Path(test.results_path)
        print(test_results_path)
        print(test_results_path.exists())

    # if test was expected to output a file but it doesn't exist (or is empty) then it's a test errro

    pass

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tests_dir", 
        type=str, 
        default=Path(__file__).resolve().parent,
        help="Directory containing tests."
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
    args = parser.parse_args()

    # Parse tests from tests.json.
    tests = parse_tests(args.tests_dir, args.results_dir)

    # Run or show all tests.
    for test in tests:
        if args.show:
            test.show()
        else:
            test.run()

    # Merge and summarize test results.
    summarize_results(tests, args.results_dir)

if __name__ == "__main__":
    main()