# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Tests for cudnn recurrent layers."""

import numpy as np

from tensorflow.python import keras
from tensorflow.python.framework import test_util
from tensorflow.python.keras import keras_parameterized
from absl.testing import parameterized
from tensorflow.python.platform import test
from tensorflow.python.framework import test_util
from tensorflow.python.keras import layers
from tensorflow.python.keras.utils import tf_inspect
import tensorflow as tf


@keras_parameterized.run_all_keras_modes
class CuDNNTest(keras_parameterized.TestCase):

  @parameterized.named_parameters(
      ('cudnngru', keras.layers.CuDNNGRU),
      ('cudnnlstm', keras.layers.CuDNNLSTM),
  )
  @test_util.run_gpu_only
  def testCudnnParamsConversion(self, layer_class):
    num_layers = 1
    input_size = 10
    timesteps = 6
    units = 2
    num_samples = 1
    return_sequences = False
    input_shape=(num_samples, timesteps, input_size)
    input_dtype='float32'
    input_data=None
    expected_output_dtype=None

    input_data_shape = list(input_shape)
    for i, e in enumerate(input_data_shape):
        if e is None:
            input_data_shape[i] = np.random.randint(1, 4)

    input_data = 10 * np.random.random(input_data_shape)
    input_data -= 0.5
    input_data = input_data.astype(input_dtype)

    if expected_output_dtype is None:
        expected_output_dtype = input_dtype

    # instantiation
    kwargs={'units': units, 'return_sequences': return_sequences}
    layer = layer_class(**kwargs)

    weights = layer.get_weights()
    layer.set_weights(weights)

    # test and instantiation from weights
    if 'weights' in tf_inspect.getargspec(layer_class.__init__):
      kwargs['weights'] = weights
      layer = layer_class(**kwargs)

    # test in functional API
    x = layers.Input(shape=input_shape[1:], dtype=input_dtype)
    y = layer(x)

    test_weights = []
    weights_input = layer.get_weights()[0]
    weights_input = np.transpose(weights_input)
    for x in weights_input:
        test_weights.append(x)

    weights_hidden = layer.get_weights()[1]
    weights_hidden = np.transpose(weights_hidden)
    for x in weights_hidden:
        test_weights.append(x)

    num_params = weights_input.shape[0]
    biases = layer.get_weights()[2]

    kwargs2={'num_layers': num_layers, 'num_units': units, 'input_size': input_size, 'weights': test_weights, 'biases': biases.tolist()}
    params = tf.raw_ops.CudnnRNNCanonicalToParams(**kwargs2)

    kwargs3 = {'num_layers': num_layers, 'num_units': units, 'input_size': input_size, 'params': params, 'num_params': num_params}
    weights2, biases2 = tf.raw_ops.CudnnRNNParamsToCanonical(**kwargs3)

    weights2 = [np.array(x) for x in weights2]
    num_input_weights = len(weights2)//2

    weights2_input, weights2_hidden = weights2[:num_input_weights], weights2[num_input_weights:]

    weights2_input = np.concatenate(weights2_input, axis=0)
    weights2_hidden = np.concatenate(weights2_hidden, axis=0)

    biases2 = [np.array(x) for x in biases2]
    biases2 = np.array(biases2)
    biases2 = np.ndarray.flatten(biases2)

    self.assertEqual(weights_input.shape, weights2_input.shape)
    self.assertEqual(weights_hidden.shape, weights2_hidden.shape)
    self.assertEqual(biases.shape, biases2.shape)

    self.assertAllEqual(weights_input, weights2_input)
    self.assertAllEqual(weights_hidden, weights2_hidden)
    self.assertAllEqual(biases.tolist(), biases2)

if __name__ == '__main__':
  test.main()