/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.
Portions Copyright (c) Microsoft Corporation.

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

#include "tfdml/kernels/pch.h"

namespace tfdml
{

class DmlBitcastKernel : public OpKernel
{
  public:
    explicit DmlBitcastKernel(
        OpKernelConstruction* c,
        std::shared_ptr<const NodeDef> node_def)
        : OpKernel(std::move(node_def))
    {
    }

  private:
    void ComputeImpl(OpKernelContext* ctx) final
    {
        const Tensor& input = ctx->input(0);
        int dim_count = input.dims();
        TF_DataType input_dtype = input.dtype();
        TF_DataType output_dtype = ctx->expected_output_dtype(0);

        int input_dtype_size = DataTypeSize(input_dtype);
        int output_dtype_size = DataTypeSize(output_dtype);

        OP_REQUIRES(
            ctx,
            input_dtype_size >= output_dtype_size ||
                (dim_count > 0 && input.dim_size(dim_count - 1) ==
                                      output_dtype_size / input_dtype_size),
            errors::InvalidArgument(
                "Cannot bitcast from ",
                input_dtype,
                " to ",
                output_dtype));

        TensorShape output_shape = input.shape();

        if (output_dtype_size < input_dtype_size)
        {
            output_shape.AddDim(input_dtype_size / output_dtype_size);
        }
        else if (output_dtype_size > input_dtype_size)
        {
            output_shape.RemoveLastDims(1);
        }

        TF_Tensor* output = TF_AllocateTensor(
            output_dtype,
            output_shape.data(),
            0,
            output_dtype_size);

        Status status;
        TF_TensorBitcastFrom(
            input.raw(),
            output_dtype,
            output,
            output_shape.data(),
            output_shape.dims(),
            status.raw());
        OP_REQUIRES_OK(ctx, status);

        if (status.ok())
        {
            TF_SetOutput(ctx->raw(), 0, output, status.raw());
        }
        TF_DeleteTensor(output);
    }
};

void RegisterKernels_Bitcast()
{
    KernelDefinition<ops::Bitcast, DmlBitcastKernel>::Register();
}

} // namespace tfdml
