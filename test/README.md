# Testing TensorFlow-DirectML-Plugin

This directory contains tests for validating the DirectML plugin. Testing is done with a combination of python scripts and native test executables. The python tests should be run in a python environment with the tensorflow-directml-plugin package installed. All testing is driven by the following two files:

- [run_tests.py](run_tests.py) : main script that will execute all test content (python and native tests) and summarize the results. Uses tests.json to drive testing.
- [tests.json](tests.json) : describes the test content and how it should be executed.

Tests are organized into *test groups*, which contain related tests written with the same testing framework (e.g. python + abseil). The groups are defined in tests.json. Here is a summary of the current test groups:

| Test Group    | Type          | Purpose                                                                 |
| ------------- | ------------- | ----------------------------------------------------------------------- |
| ops           | Python/Abseil | Validates DirectML implementations of TF operators                      |
| models        | Python/Abseil | Validates training/inference on specific models                         |
| plugin_python | Python/Abseil | Validates specific features of the DML plugin (e.g. pluggable profiler) |
| plugin_cpp    | Native/GTest  | Validates internal pieces of the DML plugin (e.g. kernel cache)         |

# Running Tests

To fully validate the plugin you should run all of the tests. Make sure to set the `--results_dir` parameter to a location to store test results; if you don't set this then you see test output to the console, but you won't get a summary of results across all test groups.

```
> python .\run_tests.py --results_dir results
[ops] python S:\tensorflow-directml-plugin\test\python\ops\concat_op_test.py
[ops] python S:\tensorflow-directml-plugin\test\python\ops\gather_nd_op_test.py
[ops] python S:\tensorflow-directml-plugin\test\python\ops\gather_op_test.py
[ops] python S:\tensorflow-directml-plugin\test\python\ops\matmul_op_test.py
[plugin_python] python S:\tensorflow-directml-plugin\test\python\plugin\profiler_tests.py
[plugin_python] python S:\tensorflow-directml-plugin\test\python\plugin\simple.py
...
```

## Running Tests in Isolation

You may find it useful to run a single test in isolation. To see the available tests you can use the `--show` switch as shown below. Each line indicates the test group (in brackets) and the command line to run the test:

```
> python .\run_tests.py --show
[ops] python S:\tensorflow-directml-plugin\test\python\ops\concat_op_test.py
[ops] python S:\tensorflow-directml-plugin\test\python\ops\gather_nd_op_test.py
[ops] python S:\tensorflow-directml-plugin\test\python\ops\gather_op_test.py
[ops] python S:\tensorflow-directml-plugin\test\python\ops\matmul_op_test.py
[plugin_python] python S:\tensorflow-directml-plugin\test\python\plugin\profiler_tests.py
[plugin_python] python S:\tensorflow-directml-plugin\test\python\plugin\simple.py
...
```

Python tests may be further scoped down to individual classes or methods. The following example shows how to execute the `testBasic` method of the `ConcatOffsetTest` test:

```
python S:\tensorflow-directml-plugin\test\python\ops\concat_op_test.py -- ConcatOffsetTest.testBasic
```

# Test JSON Metadata

## python_abseil 

Test groups of this type reference a directory of python scripts. 

- Each python top-level python file is expected to contain abseil test classes. 
- Only the top-level .py files will be considered as tests; the test group **does not recurse** into subdirectories. 
- You may have helper files in subdirectories that are imported/used by the top-level .py files.

**Fields**

| Field                         | Required | Type                | Default | Description                                                                  |
| ----------------------------- | -------- | ------------------- | ------- | ---------------------------------------------------------------------------- |
| type                          | Yes      | string              |         | Must be "python_abseil".                                                     |
| test_script_dir               | Yes      | string              |         | Directory (relative to tests.json) containing the test scripts.              |
| group_timeout_seconds         | No       | number              | 900     | Max number of seconds to wait for all tests in the group to complete.        |
| default_test_timeout_seconds  | No       | number              | 30      | Max number of seconds each test script can run before being terminated.      |
| override_test_timeout_seconds | No       | map(string, number) | {}      | Max number of seconds specific test scripts can run before being terminated. |
| disabled_tests                | No       | array(string)       | []      | Names of scripts in test_script_dir to skip executing.                       |

**Example**

Consider a directory structure with the following files:
```
tests/
│   tests.json
│
├───basic_tests/
│   │   bar.py
│   │   broken_test.py
│   │   foo.py
│   │   slow_test.py
│   │
│   └───helpers/
│           common.py
│
└───native/
        ...
```

The `tests.json` may look like the following:

```json
{
    "groups": {
        "basic_tests": {
            "type": "python_abseil",
            "test_script_dir": "basic_tests",
            "group_timeout_seconds": 100,
            "default_test_timeout_seconds": 15,
            "override_test_timeout_seconds": {
                "slow_test.py": 120
            },
            "disabled_tests": [
                "broken_test.py"
            ]
        },
        "native": {
            ...
        }
    }
}
```

Running the `basic_tests` test group, as defined above, will execute three tests: 
- `bar.py` (given 15 seconds to complete)
- `foo.py` (given 15 seconds to complete)
- `slow_test.py` (given 120 seconds to complete)

All three tests combined have a max duration of 150 seconds, but the test group is only given 100 seconds to complete. Any tests that have not started within the `group_test_timeout_seconds` duration will not be executed. Any tests that have not completed within the `group_test_timeout_seconds` duration will be terminated.