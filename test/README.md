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

## Test Type: python_abseil 

Test groups of this type reference a directory of python scripts. Each python script is expected to contain abseil test classes. Test scripts should **not** be organized into subdirectories; only the top-level .py files will be considered for 

Example:
```json
{
"ops": {
    "type": "python_abseil",
    "test_timeout_minutes": 30,
    "test_script_dir": "python/ops",
    "disabled_tests": [
        "batch_matmul_op_test.py"
    ]
}
```

| Field                | Required | Type          | Description                                                                      |
| -------------------- | -------- | ------------- | -------------------------------------------------------------------------------- |
| type                 | Yes      | string        | Must be "python_abseil".                                                         |
| test_timeout_minutes | Yes      | number        | Max number of minutes to wait for each test.                                     |
| test_script_dir      | Yes      | string        | Directory (relative to root test content directory) containing the test scripts. |
| disabled_tests       | No       | array(string) | Names of scripts in test_script_dir to skip executing.                           |
