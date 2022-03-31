#!/usr/bin/env python
# Copyright (c) Microsoft Corporation. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import contextlib
from tensorflow.python.framework import ops
from tensorflow.python.client import device_lib
from tensorflow.python.framework import test_util
from tensorflow.python.eager import context
from tensorflow.python.util import compat
from tensorflow.python.keras import keras_parameterized

@contextlib.contextmanager
def device(use_gpu):
    """Uses a DML gpu when requested."""
    if use_gpu:
        dev = "/device:DML:0"
    else:
        dev = "/device:CPU:0"
    with ops.device(dev):
        yield

@contextlib.contextmanager
def use_gpu():
    """Uses a DML gpu when requested."""
    with device(use_gpu=True):
        yield

@contextlib.contextmanager
def force_gpu():
    """Force the DML gpu to be used."""
    with ops.device("/device:DML:0"):
        yield

def is_gpu_available(cuda_only=False, min_cuda_compute_capability=None):
    """Returns whether TensorFlow can access a GPU."""
    # This was needed earlier when we had support for SYCL in TensorFlow.
    del cuda_only
    del min_cuda_compute_capability

    for local_device in device_lib.list_local_devices():
        if local_device.device_type == "DML":
            return True
    return False

def gpu_device_name():
    """Returns the name of a DML device if available or a empty string."""
    for x in device_lib.list_local_devices():
        if x.device_type == "DML":
            return compat.as_str(x.name)
    return ""

class TestCase(test_util.TensorFlowTestCase):
    """Base class for tests that need to test DML devices."""
    @contextlib.contextmanager
    def cached_session(self,
                        graph=None,
                        config=None,
                        use_gpu=True,
                        force_gpu=False):
        """Returns a DML TensorFlow Session for use in executing tests."""
        if context.executing_eagerly():
            yield test_util.FakeEagerSession(self)
        else:
            sess = self._get_cached_session(graph,
                                            config,
                                            force_gpu,
                                            crash_if_inconsistent_args=True)
            with self._constrain_devices_and_set_default_dml(sess, use_gpu,
                                                         force_gpu) as cached:
                yield cached


    @contextlib.contextmanager
    def session(self, graph=None, config=None, use_gpu=True, force_gpu=False):
        """A context manager for a DML TensorFlow Session for use in executing tests."""
        if context.executing_eagerly():
            yield test_util.EagerSessionWarner()
        else:
            with self._create_session(graph, config, force_gpu) as sess:
                with self._constrain_devices_and_set_default_dml(sess, use_gpu, force_gpu):
                    yield sess

    @contextlib.contextmanager
    def _constrain_devices_and_set_default_dml(self, sess, use_gpu, force_gpu):
        """Set the session and its graph to global default and constrain devices."""
        if context.executing_eagerly():
            yield None
        else:
            with sess.graph.as_default(), sess.as_default():
                if force_gpu:
                    # Use the name of an actual device if one is detected, or
                    # '/device:DML:0' otherwise
                    gpu_name = gpu_device_name()
                    if not gpu_name:
                        gpu_name = "/device:DML:0"
                    with sess.graph.device(gpu_name):
                        yield sess
                elif use_gpu:
                    yield sess
                else:
                    with sess.graph.device("/device:CPU:0"):
                        yield sess

class KerasParameterizedTestCase(TestCase, keras_parameterized.TestCase):
    pass
