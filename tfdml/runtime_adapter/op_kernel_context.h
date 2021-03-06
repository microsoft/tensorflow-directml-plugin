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

#pragma once

#include "absl/types/span.h"
#include "tfdml/runtime_adapter/status.h"
#include "tfdml/runtime_adapter/statusor.h"
#include "tfdml/runtime_adapter/tensor.h"
#include "tfdml/runtime_adapter/types.h"

struct TF_OpKernelContext;

namespace tfdml
{
class Device;
class OpKernel;
class ResourceMgr;

class OpKernelContext
{
  public:
    OpKernelContext(TF_OpKernelContext* context, OpKernel* op_kernel);
    Tensor input(int input_index);
    int num_inputs() const;
    int num_outputs() const;
    void CtxFailure(const char* file, int line, const Status& s);
    void CtxFailureWithWarning(const char* file, int line, const Status& s);
    void forward_ref_input_to_ref_output(int input_index, int output_index);
    TF_DataType input_dtype(int index);
    TF_DataType expected_output_dtype(int index);
    StatusOr<Tensor> allocate_output(int index, const TensorShape& shape);
    StatusOr<Tensor> forward_input_or_allocate_output(
        absl::Span<const int> candidate_input_indices,
        int output_index,
        const TensorShape& output_shape,
        int* forwarded_input = nullptr);
    const Status& status() const;
    Device* device() const;
    Status allocate_temp(
        TF_DataType dtype,
        const TensorShape& shape,
        Tensor* tensor,
        bool on_host = false);
    MemoryType input_memory_type(int index) const;
    MemoryType output_memory_type(int index) const;
    Status set_output(int index, const Tensor& tensor);
    const OpKernel& op_kernel() const;
    Status AssignVariable(int var_index, int value_index, bool validate_shape);

    Status AssignUpdateVariable(
        int var_index,
        int value_index,
        void (*updateFunc)(
            TF_OpKernelContext* ctx,
            TF_Tensor* tensor,
            TF_Tensor* value,
            int Op));

    Status AssignRefVariable(
        int input_ref_index,
        int output_ref_index,
        int value_index,
        bool use_locking,
        bool validate_shape);

    Status AddNVariant(void (*binary_add_func)(
        TF_OpKernelContext* ctx,
        TF_Tensor* a,
        TF_Tensor* b,
        TF_Tensor* out));

    Status ZerosLikeVariant(void (*zeros_like_func)(
        TF_OpKernelContext* ctx,
        TF_Tensor* input,
        TF_Tensor* out));

    Status GetInputTensorFromVariable(
        int index,
        bool lock_held,
        bool is_variant,
        Tensor* tensor);
    TF_OpKernelContext* raw() const;

  private:
    TF_OpKernelContext* const context_;
    Status status_;
    Device* device_;
    OpKernel* const op_kernel_;
};
} // namespace tfdml
