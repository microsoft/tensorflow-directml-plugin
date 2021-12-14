import subprocess
import json
import os
import sys
import argparse
from pathlib import Path

# Parses tests.json to build a list of command lines to execute.
def get_test_command_lines(tests_root_dir, results_dir = None):
    test_command_lines = []

    with open(Path(tests_root_dir) / "tests.json") as json_file:
        tests_metadata = json.load(json_file)

    for test_group_name in tests_metadata["groups"]:
        test_group = tests_metadata["groups"][test_group_name]

        # Test groups of type 'python_abseil' point at a directory with .py test scripts.
        if test_group["type"] == "python_abseil":
            test_scripts_dir = tests_root_dir / Path(test_group["test_script_dir"])
            for test_script_path in test_scripts_dir.glob("*.py"):
                test_command_line = f"python {test_script_path}"
                if results_dir:
                    xml_path = Path(results_dir) / f"test_results_{len(test_command_lines)}.xml"
                    test_command_line += f" --xml_output_file {xml_path}"
                test_command_lines.append(test_command_line)
        else:
            print("test_group test not supported")
    
    return test_command_lines

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, default=None, help="Directory to save test results. If empty no result files are written.")
    args = parser.parse_args()

    test_root_dir = Path(__file__).resolve().parent

    # Execute tests
    test_command_lines = get_test_command_lines(test_root_dir, args.results_dir)
    for test_command_line in test_command_lines:
        print(test_command_line)

    # Merge results and optionally convert to ap format.

if __name__ == "__main__":
    main()