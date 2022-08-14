/* Copyright (c) Microsoft Corporation.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "absl/cleanup/cleanup.h"
#include "tensorflow/c/c_api.h"
#include "tensorflow/c/c_api_experimental.h"
#include <fstream>
#include <gtest/gtest.h>
#include <numeric>
#include <random>

static TF_Buffer* ReadBufferFromFile(const char* file_path)
{
    std::ifstream buffer_file(file_path, std::ifstream::binary);
    if (buffer_file.fail())
    {
        return nullptr;
    }

    if (!buffer_file.is_open())
    {
        return nullptr;
    }

    if (buffer_file.seekg(0, std::ifstream::end).fail())
    {
        return nullptr;
    }

    int num_bytes = buffer_file.tellg();
    if (num_bytes <= 0)
    {
        return nullptr;
    }

    if (buffer_file.seekg(0, std::ifstream::beg).fail())
    {
        return nullptr;
    }

    char* bytes = new char[num_bytes];
    if (buffer_file.read(bytes, num_bytes).fail())
    {
        return nullptr;
    }

    auto buffer = TF_NewBuffer();
    buffer->data = bytes;
    buffer->length = num_bytes;
    buffer->data_deallocator = [](void* bytes, size_t num_bytes)
    { delete[] static_cast<char*>(bytes); };

    return buffer;
}

// Demonstrate some basic assertions.
TEST(CApiTests, TestSqueezeNet)
{
    TF_Status* status = TF_NewStatus();
    auto status_cleanup =
        absl::MakeCleanup([status] { TF_DeleteStatus(status); });

    // Load the TFDML binary, which will automatically register all supported
    // kernel implementations and create a "GPU" device
    const TF_Library* tfdml_lib =
        TF_LoadPluggableDeviceLibrary("libtfdml_plugin_framework.so", status);
    ASSERT_EQ(TF_GetCode(status), TF_OK);

    // Read the frozen graph from the file
    TF_Buffer* buffer = ReadBufferFromFile("squeezenet_model/squeezenet.pb");
    ASSERT_NE(buffer, nullptr);
    auto buffer_cleanup =
        absl::MakeCleanup([buffer] { TF_DeleteBuffer(buffer); });

    auto graph = TF_NewGraph();
    auto graph_cleanup = absl::MakeCleanup([graph] { TF_DeleteGraph(graph); });

    auto options = TF_NewImportGraphDefOptions();
    auto options_cleanup = absl::MakeCleanup(
        [options] { TF_DeleteImportGraphDefOptions(options); });

    TF_ImportGraphDefOptionsSetDefaultDevice(options, "/device:GPU:0");
    TF_GraphImportGraphDef(graph, buffer, options, status);
    ASSERT_EQ(TF_GetCode(status), TF_OK);

    auto input_op = TF_Output{TF_GraphOperationByName(graph, "input_1"), 0};
    ASSERT_NE(input_op.oper, nullptr);

    int num_input_dims = TF_GraphGetTensorNumDims(graph, input_op, status);
    ASSERT_EQ(TF_GetCode(status), TF_OK);

    std::vector<int64_t> input_dims(num_input_dims);
    TF_GraphGetTensorShape(
        graph,
        input_op,
        input_dims.data(),
        num_input_dims,
        status);
    ASSERT_EQ(TF_GetCode(status), TF_OK);

    // Set a batch dimension of 1
    ASSERT_EQ(input_dims[0], -1);
    input_dims[0] = 1;

    int num_input_elements = std::accumulate(
        input_dims.begin(),
        input_dims.end(),
        1,
        std::multiplies<int>());

    TF_DataType input_dtype = TF_OperationOutputType(input_op);
    ASSERT_EQ(input_dtype, TF_FLOAT);

    // Generate random input values with a fixed seed
    std::random_device random_device;
    std::mt19937 generator(random_device());
    generator.seed(42);
    std::uniform_real_distribution<> distribution(0.0f, 1.0f);

    float* input_vals = new float[num_input_elements];
    for (int i = 0; i < num_input_elements; ++i)
    {
        input_vals[i] = distribution(generator);
    }

    // Create a tensor with the generated values
    auto input_tensor = TF_NewTensor(
        TF_FLOAT,
        input_dims.data(),
        num_input_dims,
        input_vals,
        num_input_elements * sizeof(float),
        [](void* data, size_t len, void* arg)
        { delete[] static_cast<float*>(data); },
        nullptr);
    ASSERT_NE(input_tensor, nullptr);

    auto input_tensor_cleanup =
        absl::MakeCleanup([input_tensor] { TF_DeleteTensor(input_tensor); });

    TF_Tensor* output_tensor = nullptr;
    auto output_tensor_cleanup =
        absl::MakeCleanup([output_tensor] { TF_DeleteTensor(output_tensor); });

    auto output_op =
        TF_Output{TF_GraphOperationByName(graph, "loss/Softmax"), 0};

    // Build the session
    auto session_options = TF_NewSessionOptions();
    auto session_options_cleanup = absl::MakeCleanup(
        [session_options] { TF_DeleteSessionOptions(session_options); });

    auto session = TF_NewSession(graph, session_options, status);
    ASSERT_EQ(TF_GetCode(status), TF_OK);

    // Run the session
    TF_SessionRun(
        session,
        nullptr,
        &input_op,
        &input_tensor,
        1,
        &output_op,
        &output_tensor,
        1,
        nullptr,
        0,
        nullptr,
        status);
    ASSERT_EQ(TF_GetCode(status), TF_OK);

    TF_CloseSession(session, status);
    ASSERT_EQ(TF_GetCode(status), TF_OK);

    TF_DeleteSession(session, status);
    ASSERT_EQ(TF_GetCode(status), TF_OK);

    // This SqueezeNet frozen model uses NHWC. Since the CPU doesn't support
    // this format, we won't be comparing the results here. Operator a model
    // accuracy tests are done in the python tests instead.
    ASSERT_NE(output_tensor, nullptr);

    int64_t num_output_elements = TF_TensorElementCount(output_tensor);
    ASSERT_GT(num_output_elements, 0);

    TF_DataType output_dtype = TF_TensorType(output_tensor);
    ASSERT_EQ(output_dtype, TF_FLOAT);
}