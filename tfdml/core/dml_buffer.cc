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

#include "dml_buffer.h"

#include "dml_bfc_allocator.h"
#include "dml_device.h"
#include "tensorflow/c/kernels.h"
#include "tfdml/runtime_adapter/op_kernel_context.h"

namespace tfdml
{

/*explicit*/ DmlBuffer::DmlBuffer(
    TF_OpKernelContext* op_kernel_context,
    DmlAllocator* allocator,
    uint64_t size_in_bytes)
    : allocator_(allocator)
{
    // Allocate a dummy tensor to leverage the BFCAllocator that wraps our
    // DmlAllocator. Calling allocator->Alloc() would not use its BFC
    // capabilities and would unconditionally allocate new memory instead of
    // reusing existing memory.
    TF_AllocatorAttributes alloc_attributes;
    alloc_attributes.struct_size = TF_ALLOCATOR_ATTRIBUTES_STRUCT_SIZE;
    alloc_attributes.on_host = false;

    TensorShape shape({static_cast<int64_t>(size_in_bytes)});

    Status status;
    TF_Tensor* raw_tensor = TF_AllocateTemp(
        op_kernel_context,
        TF_UINT8,
        shape.data(),
        shape.dims(),
        &alloc_attributes,
        status.raw());

    if (!status.ok())
    {
        return;
    }

    tensor_ = Tensor(raw_tensor);

    buffer_region_ =
        allocator_->CreateBufferRegion(tensor_.raw_data(), size_in_bytes);
}

ID3D12Resource* DmlBuffer::ResourceInUavState() const
{
    return buffer_region_.ResourceInUavState();
}

ID3D12Resource* DmlBuffer::ResourceInCopySrcState() const
{
    return buffer_region_.ResourceInCopySrcState();
}

ID3D12Resource* DmlBuffer::ResourceInCopyDstState() const
{
    return buffer_region_.ResourceInCopyDstState();
}

uint64_t DmlBuffer::Offset() const
{
    return buffer_region_ ? buffer_region_.Offset() : 0;
}

uint64_t DmlBuffer::SizeInBytes() const
{
    return buffer_region_ ? buffer_region_.SizeInBytes() : 0;
}

DML_BUFFER_BINDING DmlBuffer::GetBufferBinding() const
{
    return buffer_region_ ? buffer_region_.GetBufferBinding()
                          : DML_BUFFER_BINDING{};
}

} // namespace tfdml
