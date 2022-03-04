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

namespace tfdml {

class DmlEmptyKernel : public OpKernel {
 public:
  explicit DmlEmptyKernel(OpKernelConstruction* ctx,
        std::shared_ptr<const NodeDef> node_def)
        : OpKernel(std::move(node_def)){
    OP_REQUIRES_OK(ctx, ctx->GetAttr("init", &init_));
  }

  void Compute(OpKernelContext* ctx) {
    const Tensor& shape = ctx->input(0);
    OP_REQUIRES(
        ctx, TensorShapeUtils::IsVector(shape.shape()),
        errors::InvalidArgument("shape must be a vector of int32, got shape ",
                                shape.shape().DebugString()));
    // auto dims = shape.base<int32_t>()[0];
    TensorShape out_shape;
    // OP_REQUIRES_OK(ctx, TensorShapeUtils::MakeShape(
    //                         reinterpret_cast<const int32_t*>(dims.data()),
    //                         dims.size(), &out_shape));
    OP_REQUIRES_OK(ctx, TensorShapeUtils::MakeShape(shape, &out_shape));

    // Tensor output_tensor = nullptr;
    // OP_REQUIRES_OK(ctx, ctx->allocate_output(0, out_shape, &output_tensor));
    StatusOr<Tensor> status_or_output_tensor =
            ctx->allocate_output(0, out_shape);
    OP_REQUIRES_OK(ctx, status_or_output_tensor.status());

    if (init_ && out_shape.num_elements() > 0) {
      // auto device_context =
      //     static_cast<DMLDeviceContext*>(ctx->op_device_context());
      DmlDevice* device = static_cast<DmlDevice*>(ctx->device());
      auto* device_context = device->GetDeviceContext();

      // D3D12BufferRegion output_buffer =
      //     device_context->GetBufferForTensor(*output_tensor);
      D3D12BufferRegion output_buffer = device_context->GetBufferForTensor(
            status_or_output_tensor.ValueOrDie());

      device_context->ZeroBuffer(output_buffer);
    }
  }

 private:
  bool init_;
};

void RegisterKernels_Empty()
{
    using K = KernelDefinition<
        ops::Empty,
        DmlEmptyKernel>::WithHostMemoryArguments<ops::Empty::Argument::shape>;
    
    K::template WithTypeConstraint<ops::Empty::Attribute::dtype, TF_FLOAT>::Register();
    K::template WithTypeConstraint<ops::Empty::Attribute::dtype, TF_HALF>::Register();
    K::template WithTypeConstraint<ops::Empty::Attribute::dtype, TF_INT64>::Register();
    K::template WithTypeConstraint<ops::Empty::Attribute::dtype, TF_INT32>::Register();
}

}  // namespace tfdml
