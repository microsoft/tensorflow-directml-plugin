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
"""Tests for tensorflow.ops.tf.ones_like."""

import tensorflow as tf
from tensorflow.python.framework import test_util
from tensorflow.python.platform import test
import dml_test_util
import tensorflow as tf

class OnesLikeTest(dml_test_util.TestCase):

  @test_util.run_in_graph_and_eager_modes
  def testOnesLike(self):
    with self.cached_session():
      for dtype in [tf.float16, tf.float32, tf.int64]:
        const_input = tf.constant([[1, 1, 3], [4, 5, 6]], dtype=dtype)
        result = tf.raw_ops.OnesLike(x=const_input)
        self.assertAllEqual([[1, 1, 1], [1, 1, 1]], result)


if __name__ == "__main__":
  test.main()