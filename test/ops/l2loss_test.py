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
"""Tests for tensorflow.nn.l2_loss."""

import tensorflow as tf
tf.debugging.set_log_device_placement(True)
import numpy as np

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import test_util
from tensorflow.python.platform import test
import dml_test_util

class L2LossTest(dml_test_util.TestCase):

  @test_util.run_in_graph_and_eager_modes
  def testL2Loss(self):
    with self.cached_session():
      data = np.array([-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5])
      expected_result = np.sum(np.square(data)) / 2
      for dtype in (dtypes.float16, dtypes.float32,):
        result = tf.nn.l2_loss(constant_op.constant(data, dtype=dtype))
        self.assertAllCloseAccordingToType(result, expected_result)


if __name__ == "__main__":
  test.main()