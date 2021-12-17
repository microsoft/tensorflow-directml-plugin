# Testing TensorFlow-DirectML-Plugin

This directory contains tests for validating the DirectML plugin. Testing is done with a combination of python scripts and native test executables. The python tests should be run in a python environment with the `tensorflow-directml-plugin` and `tensorflow >= 2.7.0` packages installed. All testing is driven by the following two files:

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

Finally, you can run a subset of the tests with the `--groups` and `--tests` options. 

The `--groups` option is a coarse-grained filter mainly used for nightly testing. The following example shows how to restrict testing to the `plugin` and `ops` test groups:

```
> python .\test.py --run --summarize --groups ops, plugin
```

The `--tests` option is a fine-grained filter mainly used for debugging. The following example shows only running tests containing "devices" in the name, regardless of group:

```
> python .\test.py --run --summarize --tests *devices*
```

Many of the options above have short-hand aliases. Run `python test.py --help` for more details.

# Viewing Test Results

Whenever `test.py` is launched with the `--run` option it will generate result files, which are output to `%TEMP%/tfdml_plugin_tests` (Windows) or `/tmp/tfdml_plugin_tests` (Linux). You can control the output location directory with the `--results_dir` parameter. Be aware that this directory is deleted between runs with the `--run` argument. Below is an example of what this directory may contain:

```
log.ops.concat_op_test.txt
log.ops.gather_nd_op_test.txt
log.ops.gather_op_test.txt
log.ops.matmul_op_test.txt
log.plugin.dml_visible_devices_empty1.txt
log.plugin.dml_visible_devices_empty2.txt
log.plugin.dml_visible_devices_single.txt
log.plugin.dml_visible_devices_swapped.txt
log.plugin.profiler_test.txt
run.ops.json
run.plugin.json
summary.ops.json
summary.plugin.json
test.ops.concat_op_test.xml
test.ops.gather_nd_op_test.xml
test.ops.gather_op_test.xml
test.ops.matmul_op_test.xml
test.plugin.dml_visible_devices_empty1.xml
test.plugin.dml_visible_devices_empty2.xml
test.plugin.dml_visible_devices_single.xml
test.plugin.dml_visible_devices_swapped.xml
test.plugin.profiler_test.xml
```

You may see the following types of results:

- `test.<group>.<test>.xml` : Abseil Testing results for a single test.
- `log.<group>.<test>.txt` : console output for a single test (if output is redirected).
- `run.<group>.json` : group-level runtime stats, like execution duration and timed-out tests.
- `summary.<group>.json` : overall results for the entire group (if `--summarize` used).

You can inspect the result files manually to see the detailed errors and results. However, the `--summarize` option can be used to parse the result files and give you a high-level summary. You may see output like the following:

```
================================================================================
Test Group      : plugin
Test Duration   : 17.735952138900757 seconds
Tests Total     : 5 (6 cases)
Tests Passed    : 4 (4 cases)
Tests Skipped   : 0 (3 cases)
Tests Failed    : 1 (2 cases)
Tests Timed Out : 0

Failed Tests:
0: plugin.profiler_test

Failed Test Cases :
0: plugin.profiler_test::ProfilerTest.testTraceKernelEvents
1: plugin.profiler_test::ProfilerTest.testXPlaneKernelEvents
================================================================================
```

Results are reported both in terms of *tests* (e.g. `plugin.profiler_test`) and *test cases* (e.g. `plugin.profiler_test::ProfilerTest.testTraceKernelEvents`). Tests comprise one or more cases; however, it's possible for a test to fail to launch or timeout and thus report no case results. Similarly, a test is only considered *skipped* if it runs to completion without a single passing or failing test case.

The above output indicates the `plugin` test group encountered two failing test cases from the same test:
- `plugin.profiler_test::ProfilerTest.testTraceKernelEvents`
- `plugin.profiler_test::ProfilerTest.testXPlaneKernelEvents`

The full name of each test case has the format `<group>.<test>::<test_class>.<test_method>`. In this example you would want to open `test.plugin.profiler_test.xml` to see the full error messages.

# Debugging Tests

When debugging you will often want to run a single failing test in isolation under a debugger. Since `test.py` launches subprocesses for each test, you may find it easier to launch the actual test executable. You can find the full command line used to test with the `--show` option. In combination with the `--tests` option you can limit this to show only the command line for tests you're interested in:

```
> python .\test.py --show --tests plugin.profiler_test
plugin.profiler_test: python S:\tensorflow-directml-plugin\test\plugin\profiler_test.py --xml_output_file C:\Users\justoeck\AppData\Local\Temp\tfdml_plugin_tests\plugin.profiler_test.xml
```

Take note of the `--xml_output_file ...` portion of the command line; you don't need to include this when debugging as it is used to write the XML result file for Abseil tests. 

If you want to scope down the testing even further you can limit the Abseil test to run a single method. For example, you can execute the following command line to debug the `testTraceKernelEvents` failure in the previous example:

```
python S:\tensorflow-directml-plugin\test\plugin\profiler_test.py -- ProfilerTest.testTraceKernelEvents
```

# Test Metadata (tests.json)

The [tests.json](tests.json) file describes all of the test content available to run using `test.py`. The schema for this file is [tests_schema.json](tests_schema.json). 

Below is a summary of the properties associated with each test group:

| Field           | Required | Type        | Default | Description                                                           |
| --------------- | -------- | ----------- | ------- | --------------------------------------------------------------------- |
| name            | Yes      | string      |         | Unique name for the group of tests.                                   |
| tests           | Yes      | array(test) |         | List of tests (may be python, gtest, etc.).                           |
| timeout_seconds | No       | number      | 300     | Max number of seconds to wait for all tests in the group to complete. |

Below is a summary of the properties associated with each test:

| Field           | Required | Type          | Default     | Description                                                                                                                            |
| --------------- | -------- | ------------- | ----------- | -------------------------------------------------------------------------------------------------------------------------------------- |
| name            | No       | string        | *Generated* | Unique name for the test. If omitted, the test's base filename will be used.                                                           |
| file            | Yes      | string        |             | Path to the test script/executable relative to tests.json.                                                                             |
| type            | No       | string        | *Inferred*  | Type of test referenced by the file property. May be `py_abseil` or `gtest`. If omitted, the type is inferred from the file extension. |
| args            | No       | array(string) | []          | Command-line arguments to append when running the test.                                                                                |
| disabled        | No       | boolean       | false       | Skip executing the test file.                                                                                                          |
| timeout_seconds | No       | number        | null        | Max number of seconds to wait for the tests to complete. If omitted, the test will run up to the test group's timeout.                 |

A few tips:
- Every test must have a unique name. The `name` property is useful if you want to run the same test file with different arguments.
- You should not provide any arguments for logging result files in tests.json itself; this is handled automatically by `test.py`.