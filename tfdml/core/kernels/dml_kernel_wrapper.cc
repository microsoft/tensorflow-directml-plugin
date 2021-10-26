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

#include "tfdml/core/kernels/dml_kernel_wrapper.h"

#include "tfdml/core/common_runtime/dml/dml_execution_context.h"
#include "tfdml/core/common_runtime/dml/dml_operator_helper.h"
#include "tfdml/core/common_runtime/dml/dml_tracing.h"
#include "tfdml/core/common_runtime/dml/dml_util.h"
#include "tfdml/core/util/types.h"

namespace tfdml
{

DmlKernelWrapperBase::DmlKernelWrapperBase(
    DmlKernelCachePolicy cache_policy,
    const char* op_type_string,
    const char* op_name)
    : OpKernel(op_type_string, op_name),
      cache_policy_(cache_policy)
{
}

void DmlKernelWrapperBase::Compute(OpKernelContext* ctx)
{
    DmlTracing::Instance().LogKernelCompute(
        ctx->op_kernel().type_string(),
        ctx->op_kernel().name());

    DmlDevice* dml_device = static_cast<DmlDevice*>(ctx->device());
    const DmlKernelManager& kernel_manager = *dml_device->GetKernelManager();

    // Compute the output shapes
    const ShapeHelper* shape_helper = GetShapeHelper();

    std::shared_ptr<DmlKernel> kernel;
    std::vector<TensorShape> output_shapes;
    const InitializationHelper* init_helper = nullptr;
    DmlKernelKey key;

    if (cache_policy_ != DmlKernelCachePolicy::Never)
    {
        // Construct a kernel key which uniquely identifies the kernel instance
        // we need
        key = CreateKernelKey(ctx);

        // Retrieve an appropriate DmlKernel from the cache. If the kernel
        // hasn't been cached yet, it will be null
        kernel = TryGetCachedKernel(kernel_manager, key);
    }

    // If we found a cached kernel, simply retrieve its initialization helper
    if (kernel)
    {
        init_helper = kernel->GetInitializationHelper();
        output_shapes = shape_helper->GetOutputShapes(ctx, init_helper);
    }
    else
    {
        auto shared_helper = CreateInitializationHelper(ctx);
        init_helper = shared_helper.get();

        if (!ctx->status().ok())
        {
            return;
        }

        output_shapes = shape_helper->GetOutputShapes(ctx, init_helper);

        // Check that the number of output shapes matches the number of outputs
        OP_REQUIRES(
            ctx,
            ctx->num_outputs() == output_shapes.size(),
            errors::InvalidArgument(
                "The shape helper supplied an incorrect number of output "
                "shapes. ",
                ctx->num_outputs(),
                " were expected, but ",
                output_shapes.size(),
                " were provided."));

        if (shared_helper->IsNoOpKernel(ctx, output_shapes))
        {
            // Don't bother constructing/executing no-op'd kernels. Instead,
            // just construct empty output tensors and return immediately.
            for (int i = 0; i < ctx->num_outputs(); ++i)
            {
                StatusOr<Tensor> status_or_output_tensor =
                    ctx->allocate_output(i, output_shapes[i]);
                OP_REQUIRES_OK(ctx, status_or_output_tensor.status());

                Tensor output_tensor =
                    status_or_output_tensor.ConsumeValueOrDie();

                // If the tensor is nonempty, fill it with zero's
                if (output_tensor.NumElements() != 0)
                {
                    DMLDeviceContext* device_context =
                        dml_device->GetDeviceContext();

                    D3D12BufferRegion buffer =
                        device_context->GetBufferForTensor(output_tensor);
                    device_context->ZeroBuffer(buffer);
                }
            }
            return;
        }

        std::vector<std::pair<int, int>> forwardIndices;
        forwardIndices.reserve(ctx->num_outputs());

        for (int i = 0; i < ctx->num_outputs(); ++i)
        {
            absl::optional<int> inputIndexToForward =
                shared_helper->GetForwardableInputIndex(ctx, output_shapes, i);

            bool cancelForward = true;

            // If the input is considered forwardable for the op
            if (inputIndexToForward)
            {
                int inputIndex = inputIndexToForward.value();
                const Tensor& input = ctx->input(inputIndex);

                // Element counts must also match
                if (input.NumElements() == output_shapes[i].num_elements())
                {
                    forwardIndices.emplace_back(
                        std::make_pair(inputIndexToForward.value(), i));
                    cancelForward = false;
                }
            }

            // If not all outputs are forwardable, we will forward nothing and
            // proceed with the kernel normally.
            if (cancelForward)
            {
                forwardIndices.clear();
                break;
            }
        }

        // If we are forwarding (if this vector is not empty)
        if (!forwardIndices.empty())
        {
            for (size_t i = 0U; i < forwardIndices.size(); ++i)
            {
                int inputIndex = forwardIndices[i].first;
                int outputIndex = forwardIndices[i].second;
                const Tensor& input = ctx->input(inputIndex);

                Tensor output;
                // Copies underlying data pointer, but uses the output shape
                // provided.
                output.CopyFrom(input, output_shapes[outputIndex]);

                ctx->set_output(outputIndex, output);
            }

            return;
        }

        DmlKernelConstruction dml_construction(
            dml_device,
            ctx,
            output_shapes,
            shared_helper);

        if (cache_policy_ == DmlKernelCachePolicy::Never)
        {
            // This kernel has requested to never be cached; create a new one
            // directly
            kernel = CreateKernel(&dml_construction, init_helper);
        }
        else
        {
            kernel = CreateCachedKernel(
                &dml_construction,
                kernel_manager,
                key,
                init_helper);
        }

        // Check for validation done during kernel construction
        if (!ctx->status().ok())
        {
            return;
        }
    }

    assert(kernel != nullptr);

    // Execute the kernel
    DmlKernelContext dml_ctx(
        dml_device,
        ctx,
        init_helper,
        output_shapes,
        kernel->GetOutputRefsForwarding());

    // Check for errors triggered during the kernel context's constructor (e.g.
    // OOM when allocating the output buffers)
    if (!ctx->status().ok())
    {
        return;
    }

    auto status_or_event = ComputeKernel(kernel.get(), &dml_ctx);
    OP_REQUIRES_OK(ctx, status_or_event.status());

    // Keep this kernel alive at least until it's completed execution on the GPU
    kernel_manager.QueueReference(kernel, status_or_event.ConsumeValueOrDie());
}

DmlKernelKey DmlKernelWrapperBase::CreateKernelKey(OpKernelContext* ctx) const
{
    DmlKernelKey key = {};
    key.op_type_name = this->type_string();
    key.attributes = this->GetAttributes();

    for (int i = 0; i < ctx->num_inputs(); ++i)
    {
        // Resource types cannot be hashed or copied, so they cannot form part
        // of a kernel key. Therefore, resource tensors cannot be used as
        // constant CPU inputs. This is okay because it's unlikely a kernel
        // would ever want to take a dependency on the value of a *resource
        // handle*, rather than the contents of the tensor the handle refers to.
        const bool is_resource_type = ctx->input_dtype(i) == TF_RESOURCE;
        Tensor tensor = ctx->input(i);

        DmlInputTensorKey tensor_key = {};
        tensor_key.is_constant_cpu_input =
            (ctx->input_memory_type(i) == HOST_MEMORY && !is_resource_type);

        if (tensor_key.is_constant_cpu_input)
        {
            tensor_key.tensor = std::move(tensor);
        }
        else
        {
            tensor_key.tensor =
                TensorShapeAndType{tensor.shape(), tensor.dtype()};
        }

        key.input_tensors.push_back(std::move(tensor_key));
    }

    return key;
}

} // namespace tfdml