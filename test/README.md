# Testing TensorFlow-DirectML-Plugin

This directory contains tests for validating the DirectML plugin. Testing is done with a combination of python scripts and native test executables. The python tests should be run in a python environment with the tensorflow-directml-plugin package installed. All testing is driven by the following two files:

- [test.py](run_tests.py) : main script that will execute all test content (python and native tests) and summarize the results. Uses tests.json to drive testing.
- [tests.json](tests.json) : describes the test content and how it should be executed.

Tests are organized into *test groups*, which contain related tests written with the same testing framework (e.g. python + abseil). The groups are defined in tests.json. Here is a summary of the current test groups:

| Test Group | Purpose                                                                 |
| ---------- | ----------------------------------------------------------------------- |
| ops        | Validates DirectML implementations of TF operators                      |
| models     | Validates training/inference on specific models                         |
| plugin     | Validates specific features of the DML plugin (e.g. pluggable profiler) |

# Running Tests

To fully validate the plugin you should run all of the tests and print a summary at the end:

```
> python .\test.py --run --summarize
```

The output from each test will, by default, appear in the console. This is useful to see progress, but you may also find it easier to redirect each test's output to its own log file:

```
> python .\test.py --run --summarize --redirect_output
```

Finally, you can run a subset of the tests with the `--filter` option. The following example shows only running tests in the `plugin` test group:

```
> python .\test.py --run --summarize --filter plugin.*
```

# Viewing Test Results

Whenever `test.py` is launched with the `--run` option it will generate result files: one file for each test group, and one file for each test. By default these files are output to `%TEMP%/tfdml_plugin_tests` (Windows) or `/tmp/tfdml_plugin_tests` (Linux). You can control the output location directory with the `--results_dir` parameter. Be aware that this directory is deleted between runs with the `--run` argument. Below is an example of what this directory may contain:

```
ops.concat_op_test.xml
ops.gather_nd_op_test.xml
ops.gather_op_test.xml
ops.json
plugin.dml_visible_devices_empty1.xml
plugin.dml_visible_devices_empty2.xml
plugin.dml_visible_devices_single.xml
plugin.dml_visible_devices_swapped.xml
plugin.json
plugin.profiler_test.xml
```

There is a JSON file for each test group (`ops.json` and `plugin.json` above). This JSON file contains data on the execution of the group as a whole (duration, timed-out tests, etc.). Additionally, there is a log file each test within each test group; the name of each file is `<test_group>.<test_name>.xml` (for Python/Abseil tests). 

You can inspect the result files manually to see the detailed errors and results. However, the `--summarize` option can be used to parse the result files and give you a high-level summary. You may see output like the following:

```
--------------------------------------------------------------------------------
Test Group         : plugin
Test Cases Ran     : 7
Test Cases Passed  : 5
Test Cases Skipped : 0
Test Cases Failed  : 2
Failing Test Cases :
0: plugin.profiler_test::ProfilerTest.testTraceKernelEvents
1: plugin.profiler_test::ProfilerTest.testXPlaneKernelEvents
--------------------------------------------------------------------------------
```

The above output indicates the `plugin` test group encountered two failing test cases:
- plugin.profiler_test::ProfilerTest.testTraceKernelEvents
- plugin.profiler_test::ProfilerTest.testXPlaneKernelEvents

The full name of each test case has the format `<test_group>.<test_name>::<test_class>.<test_method>`. In this example you would want to open `plugin.profiler_test.xml` to see the full error messages.

# Debugging Tests

When debugging you will often want to run a single failing test in isolation under a debugger. Since `test.py` launches subprocesses for each test, you may find it easier to launch the actual test executable. You can find the full command line used to test with the `--show` option. In combination with the `--filter` option you can limit this to show only the command line for tests you're interested in:

```
> python .\test.py --show --filter plugin.profiler_test
plugin.profiler_test: python S:\tensorflow-directml-plugin\test\plugin\profiler_test.py --xml_output_file C:\Users\justoeck\AppData\Local\Temp\tfdml_plugin_tests\plugin.profiler_test.xml
```

Take note of the `--xml_output_file ...` portion of the command line; you don't need to include this when debugging as it is used to write the XML result file for Abseil tests. 

If you want to scope down the testing even further you can limit the Abseil test to run a single method. For example, you can execute the following command line to debug the `testTraceKernelEvents` failure in the previous example:

```
python S:\tensorflow-directml-plugin\test\plugin\profiler_test.py -- ProfilerTest.testTraceKernelEvents
```

# Test JSON Metadata

Test groups of this type reference a directory of python scripts. 

- The full name of a test is "<group_name>.<test_name>". Tests without an explicit name use the stem of their filename.
- If the test file ends with .py it is assumed to be a python/abseil test.
- If the test file ends with .exe or has no extension it is assumed to be a google test executable.
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