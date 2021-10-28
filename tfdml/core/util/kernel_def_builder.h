/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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

#pragma once

#include "absl/container/inlined_vector.h"
#include "absl/types/optional.h"
#include "absl/types/span.h"
#include "tensorflow/c/kernels.h"
#include "tensorflow/c/tf_datatype.h"
#include "tfdml/core/util/macros.h"
#include "tfdml/core/util/op_defs.h"
#include "tfdml/core/util/op_kernel_construction.h"
#include "tfdml/core/util/op_kernel_context.h"
#include "tfdml/core/util/status.h"

struct TF_OpKernelConstruction;
struct TF_OpKernelContext;

namespace tfdml
{

template <typename OpDef, typename Kernel> class KernelBuilder
{
    using Argument = typename OpDef::Argument;
    using Attribute = typename OpDef::Attribute;

  public:
    KernelBuilder() = default;

    KernelBuilder<OpDef, Kernel>& TypeConstraint(
        Attribute attr,
        TF_DataType data_type)
    {
        type_constraints_.emplace_back(attr, data_type);
        return *this;
    }

    template <typename T>
    KernelBuilder<OpDef, Kernel>& TypeConstraint(Attribute attr)
    {
        return TypeConstraint(attr, DataTypeToEnum<T>());
    }

    KernelBuilder<OpDef, Kernel>& HostMemory(Argument arg)
    {
        host_memory_args_.push_back(arg);
        return *this;
    }

    KernelBuilder<OpDef, Kernel>& Priority(int32_t priority)
    {
        priority_ = priority;
        return *this;
    }

    void Register()
    {
        auto* builder = TF_NewKernelBuilder(
            OpDef::name,
            DEVICE_DML,
            &CreateKernel,
            &ComputeKernel,
            &DeleteKernel);
        CHECK(builder != nullptr);

        Status status;
        for (const auto& type_constraint : type_constraints_)
        {
            const auto& attr_desc =
                GetAttributeDesc<OpDef>(type_constraint.first);
            TF_KernelBuilder_TypeConstraint(
                builder,
                attr_desc.name,
                type_constraint.second,
                status.raw());
            CHECK(status.ok());
        }

        for (const auto& arg : host_memory_args_)
        {
            const auto& arg_desc = GetArgumentDesc<OpDef>(arg);
            TF_KernelBuilder_HostMemory(builder, arg_desc.name);
        }

        if (priority_)
        {
            TF_KernelBuilder_Priority(builder, *priority_);
        }

        TF_RegisterKernelBuilder(OpDef::name, builder, status.raw());
        CHECK(status.ok());
    }

  private:
    KernelBuilder(const KernelBuilder&) = delete;
    void operator=(const KernelBuilder&) = delete;
    absl::InlinedVector<Argument, 4> host_memory_args_;
    absl::InlinedVector<std::pair<Attribute, TF_DataType>, 4> type_constraints_;
    absl::optional<int32_t> priority_;

    static void* CreateKernel(TF_OpKernelConstruction* raw_ctx)
    {
        TF_StringView name_string_view =
            TF_OpKernelConstruction_GetName(raw_ctx);
        OpKernelConstruction ctx(raw_ctx);
        return new Kernel(&ctx, OpDef::name, name_string_view.data);
    }

    static void ComputeKernel(void* kernel, TF_OpKernelContext* raw_ctx)
    {
        Kernel* concrete_kernel = static_cast<Kernel*>(kernel);
        OpKernelContext ctx(raw_ctx, concrete_kernel);
        concrete_kernel->Compute(&ctx);
    }

    static void DeleteKernel(void* kernel)
    {
        Kernel* concrete_kernel = static_cast<Kernel*>(kernel);
        delete concrete_kernel;
    }
};

} // namespace tfdml
