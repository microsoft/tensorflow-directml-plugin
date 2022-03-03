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
"""Tests for tensorflow.ops.tf.deepcopy."""

from tensorflow.python.framework import test_util
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.framework import ops
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import nn
from tensorflow.python.platform import test
from tensorflow.python.eager import context
import numpy as np
import dml_test_util

class DeepcopyTest(dml_test_util.TestCase):

#   @test_util.run_in_graph_and_eager_modes
#   def testDeepcopy(self):
#     with self.cached_session():
#         x1 = np.array([[0, -1, 2, -3, 4], [-5, 6, -7, 8, -9]])
#         x1 = ops.convert_to_tensor(x1)
#         x1 = math_ops.abs(x1)
#         y2 = x1
#         y = gen_array_ops.DeepCopy(x=x1)
#         x1 = math_ops.square(x1)
#         print("x: ", x1)
#         print("y: ", y)
#         print("y2: ", y2)
#         self.assertNotAllEqual(x1, y)
#         self.assertAllEqual(x1, y2)


  @test_util.run_deprecated_v1
  def testDeepcopy(self):
    with self.cached_session():
        x1 = np.array([[0, -1, 2, -3, 4], [-5, 6, -7, 8, -9]])
        x1 = ops.convert_to_tensor(x1)
        x1 = math_ops.abs(x1)
        y2 = x1
        y = gen_array_ops.DeepCopy(x=x1)
        x1 = math_ops.square(x1)
        print("x: ", x1)
        print("y: ", y)
        print("y2: ", y2)
        self.assertNotAllEqual(x1, y)
        self.assertAllEqual(x1, y2)


if __name__ == "__main__":
  test.main()