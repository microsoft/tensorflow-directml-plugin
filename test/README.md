# Testing TensorFlow-DirectML-Plugin

This directory contains tests for validating the DirectML plugin. Testing is done with a combination of python scripts and native test executables. The python tests should be run in a python environment with the `tensorflow-directml-plugin` and `tensorflow-cpu >= 2.11.0` packages installed. All testing is driven by the following two files:

- [test.py](test.py) : main script that will execute all test content (python and native tests) and summarize the results. Uses tests.json to drive testing.
- [tests.json](tests.json) : describes the test content and how it should be executed.

## Test Groups, Tests, and Test Cases

Tests are organized into three levels: *Test Group* > *Test* > *Test Case*:
- A **test group** is a logically related set of tests. Each group comprises one or more tests.
- A **test** is a single test script or executable file. Each test comprises one or more test cases. The name of a test is usually the basename of the script/executable file, but this is only a default; the same file may be reused by multiple tests so long as those tests have unique names.
- A **test case** is the smallest unit of testing with a result (failed, passed, etc.).

For example, `plugin.profiler_test::ProfilerTest.testTensorFlowStats` is the fully qualified name of a test case:
- Test Group = `plugin`
- Test = `profiler_test` (the test file also happens to be `profiler_test.py`)
- Test Case = `ProfilerTest.testTensorFlowStats`

The *test groups* and their associated *tests* are defined in `tests.json`. Test cases are determined at runtime when a test is launched.

## Test Results

Tests report a single status based on the outcome of their cases:

| Result    | Meaning                                                                                              |
| --------- | ---------------------------------------------------------------------------------------------------- |
| passed    | At least one case passed. Some cases may have skipped. Zero cases failed.                            |
| skipped   | All cases skipped.                                                                                   |
| failed    | At least one case failed, the test crashed (i.e. encountered *errors*), or the test failed to start. |
| timed_out | The test was terminated because it ran longer than max allowed duration.                             |

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
> python .\test.py --run --summarize --groups ops plugin
```

The `--tests` option is a fine-grained filter mainly used for debugging. The following example shows only running tests containing "devices" in the name, regardless of group:

```
> python .\test.py --run --summarize --tests *devices*
```

Many of the options above have short-hand aliases. Run `python test.py --help` for more details.

# Viewing Test Results

Whenever `test.py` is launched with the `--run` option it will generate result files, which are output to `%TEMP%/tfdml_plugin_tests` (Windows) or `/tmp/tfdml_plugin_tests` (Linux). You can control the output location directory with the `--results_dir` parameter. Be aware that this directory is deleted between runs with the `--run` argument. Below is an example of what this directory may contain:

```
log.examples.buggy.txt
log.examples.good_test.txt
log.examples.not_good_test.txt
log.plugin.dml_visible_devices_empty1.txt
log.plugin.dml_visible_devices_empty2.txt
log.plugin.dml_visible_devices_single.txt
log.plugin.dml_visible_devices_swapped.txt
log.plugin.profiler_test.txt
run.examples.buggy.json
run.examples.good_test.json
run.examples.json
run.examples.not_good_test.json
run.examples.slow_test.json
run.plugin.dml_visible_devices_empty1.json
run.plugin.dml_visible_devices_empty2.json
run.plugin.dml_visible_devices_single.json
run.plugin.dml_visible_devices_swapped.json
run.plugin.json
run.plugin.profiler_test.json
summary.examples.json
summary.plugin.json
test.examples.buggy.xml
test.examples.good_test.xml
test.examples.not_good_test.xml
test.examples.slow_test.xml
test.plugin.dml_visible_devices_empty1.xml
test.plugin.dml_visible_devices_empty2.xml
test.plugin.dml_visible_devices_single.xml
test.plugin.dml_visible_devices_swapped.xml
```

You may see the following types of results:

- `log.<group>.<test>.txt` : console output for a single test (if output is redirected).
- `run.<group>.json` : group-level runtime stats, like execution duration.
- `run.<group>.<test>.json` : test-level runtime stats, like execution duration and exit code.
- `test.<group>.<test>.xml` : Abseil Testing results for a single test.
- `summary.<group>.json` : overall results for the entire group (if `--summarize` used).

You can inspect the result files manually to see the detailed errors and results. However, the `--summarize` option can be used to parse the result files and give you a high-level summary. You will see output like the following:

```
================================================================================
Test Group      : plugin
Test Duration   : 18.850157976150513 seconds
Tests Total     : 5 (4 cases)
Tests Passed    : 4 (4 cases)
Tests Skipped   : 0 (0 cases)
Tests Failed    : 1 (0 cases)
Tests Timed Out : 0

Failed Tests:
0: plugin.profiler_test
================================================================================

================================================================================
Test Group      : examples
Test Duration   : 11.100826263427734 seconds
Tests Total     : 4 (10 cases)
Tests Passed    : 1 (5 cases)
Tests Skipped   : 0 (1 cases)
Tests Failed    : 2 (4 cases)
Tests Timed Out : 1

Failed Tests:
0: examples.buggy
0.0: examples.buggy::BuggyTest1.testCase1
0.1: examples.buggy::BuggyTest2.testBravo
0.2: examples.buggy::BuggyTest2.testCharlie
1: examples.not_good_test
1.0: examples.not_good_test::ThisTestFails.testCaseA
================================================================================
```

Each failing test is listed (lines starting with a single digit, like `0:`, `1:`, etc.) along with the failing cases (lines starting with `0.0:`, `0.1:`, etc.). If a test fails to launch or record case results you will only get a test-level result for that test; this is why `plugin.profiler_test` shows as failed but has no failing cases. You will want to view the log files for tests that crash or terminate abnormally to see if there's any useful diagnostic information.

# Debugging Tests

When debugging you will often want to run a single failing test in isolation under a debugger. Since `test.py` launches subprocesses for each test, you may find it easier to launch the actual test executable. You can find the full command line used to test with the `--show` option. In combination with the `--tests` option you can limit this to show only the command line for tests you're interested in:

```
> python .\test.py --show --tests plugin.profiler_test
plugin.profiler_test: python S:\tensorflow-directml-plugin\test\plugin\profiler_test.py --xml_output_file C:\Users\justoeck\AppData\Local\Temp\tfdml_plugin_tests\test.plugin.profiler_test.xml
```

Take note of the `--xml_output_file ...` portion of the command line; you don't need to include this when debugging as it is used to write the XML result file for Abseil tests. 

If you want to scope down the testing even further you can limit the Abseil test to run a single method. For example, you can execute the following command line to debug the `ProfilerTest.testTraceKernelEvents` case:

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