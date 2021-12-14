import subprocess
import json
import os
import sys
import argparse
from pathlib import Path

# Parses tests.json to build a list of command lines to execute.
def get_test_command_lines(tests_root_dir, results_dir):
    test_command_lines = []

    with open(os.path.join(sys.path[0], "tests.json")) as json_file:
        data = json.load(json_file)

    for test_group_name in data["groups"]:
        test_group = data["groups"][test_group_name]

        # Test groups of type 'python_abseil' point at a directory with .py test scripts.
        if test_group["type"] == "python_abseil":
            test_scripts_dir = tests_root_dir / Path(test_group["test_script_dir"])
            for test_script_path in test_scripts_dir.glob("*.py"):
                xml_path = results_dir / f"test_results_{len(test_command_lines)}.xml"
                test_command_lines.append(f"python {test_script_path} --xml_output_file {xml_path}")
        else:
            print("test_group test not supported")
    
    return test_command_lines

def main():
    test_root_dir = Path(__file__).resolve().parent
    results_dir = test_root_dir / "results"

    # Execute tests
    test_command_lines = get_test_command_lines(test_root_dir, results_dir)
    for test_command_line in test_command_lines:
        print(test_command_line)

    # Merge results and optionally convert to ap format.

if __name__ == "__main__":
    main()