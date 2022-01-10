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
"""Tests for tensorflow.ops.tf.fill."""

import tensorflow as tf
tf.debugging.set_log_device_placement(True)

import numpy as np

from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.platform import test

class GatherNdTest(test.TestCase):

  @test_util.run_in_graph_and_eager_modes
  def testFill(self):
    with self.cached_session():
      fill = array_ops.fill([2, 3], 1.0)
      self.assertAllEqual([[1, 1, 1], [1, 1, 1]], fill)

  def testFillBadShape(self):
    with self.cached_session():
      with self.assertRaisesOpError(
          r"dims must be a vector, got shape \[2,2\]"):
        array_ops.fill([[2, 3], [2, 3]], 1.0)

  def testFillBadValue(self):
    with self.cached_session():
      with self.assertRaisesOpError(
          r"value must be a scalar, got shape \[2\]"):
        array_ops.fill([2, 3], [1.0, 2.0])


if __name__ == "__main__":
  test.main()