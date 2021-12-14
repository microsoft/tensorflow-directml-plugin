# load json
# go through test groups
# launch each script in its own process -> own xml file
# gather xml files into single summary
# (optional) convert xml summary to AP format

import subprocess
import json
import os
import sys
import argparse
from pathlib import Path

def main():
    script_parent_dir = Path(__file__).resolve().parent

    with open(os.path.join(sys.path[0], "tests.json")) as json_file:
        data = json.load(json_file)
    
    for test_group_name in data["groups"]:
        print(f"run {test_group_name}")
        test_group = data["groups"][test_group_name]

        if test_group["type"] == "python_abseil":
            test_scripts_dir = script_parent_dir / Path(test_group["test_script_dir"])
            for p in test_scripts_dir.glob("*.py"):
                print(p)
            print(test_scripts_dir)
            # TODO: find py files

            pass
        else:
            print("test_group test not supported")

if __name__ == "__main__":
    main()