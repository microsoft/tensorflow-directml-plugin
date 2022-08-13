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
#include <dlfcn.h>
#include <fstream>
#include <gtest/gtest.h>

static TF_Buffer* ReadBufferFromFile(const char* file_path)
{
    std::ifstream buffer_file(file_path, std::ifstream::binary);
    EXPECT_FALSE(buffer_file.fail());
    EXPECT_TRUE(buffer_file.is_open());

    EXPECT_FALSE(buffer_file.seekg(0, std::ifstream::end).fail());
    int num_bytes = buffer_file.tellg();
    EXPECT_GT(num_bytes, 0);
    EXPECT_FALSE(buffer_file.seekg(0, std::ifstream::beg).fail());

    char* bytes = new char[num_bytes];
    EXPECT_FALSE(buffer_file.read(bytes, num_bytes).fail());

    auto buffer = TF_NewBuffer();
    buffer->data = bytes;
    buffer->length = num_bytes;
    buffer->data_deallocator = [](void* bytes, size_t num_bytes)
    { delete[] static_cast<char*>(bytes); };

    return buffer;
}

// Demonstrate some basic assertions.
TEST(CApiTest, TestCApi)
{
    TF_Status* status = TF_NewStatus();
    auto status_cleanup =
        absl::MakeCleanup([status] { TF_DeleteStatus(status); });

    const TF_Library* tfdml_lib =
        TF_LoadPluggableDeviceLibrary("libtfdml_plugin_framework.so", status);
    EXPECT_EQ(TF_GetCode(status), TF_OK);
    printf("LILILILI\n");

    TF_Buffer* buffer =
        ReadBufferFromFile("squeezenet_model/squeezenet.pb");
    EXPECT_NE(buffer, nullptr);
    auto buffer_cleanup =
        absl::MakeCleanup([buffer] { TF_DeleteBuffer(buffer); });

    auto graph = TF_NewGraph();
    auto graph_cleanup = absl::MakeCleanup([graph] { TF_DeleteGraph(graph); });

    auto options = TF_NewImportGraphDefOptions();
    auto options_cleanup = absl::MakeCleanup(
        [options] { TF_DeleteImportGraphDefOptions(options); });

    TF_GraphImportGraphDef(graph, buffer, options, status);

    printf("****************\n");
    printf("****************\n");
    printf("****************\n");
    printf("Error: %s\n", TF_Message(status));

    EXPECT_EQ(TF_GetCode(status), TF_OK);

    size_t op_index = 0;

    while (TF_Operation* op = TF_GraphNextOperation(graph, &op_index))
    {
        auto op_name = TF_OperationName(op);
        auto op_type = TF_OperationOpType(op);
        auto device_name = TF_OperationDevice(op);

        printf(
            "op_name: %s, op_type: %s, device_name: %s\n",
            op_name,
            op_type,
            device_name);
    }

    // https://github.com/Neargye/hello_tf_c_api
}