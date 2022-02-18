#!/usr/bin/env python
# Copyright (c) Microsoft Corporation. All Rights Reserved.
#
# Use of this source code is governed by an MIT-style
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import contextlib
from tensorflow.python.framework import ops

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
