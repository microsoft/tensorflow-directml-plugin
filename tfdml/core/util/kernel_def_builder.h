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
#include "tfdml/core/util/op_kernel_construction.h"
#include "tfdml/core/util/op_kernel_context.h"
#include "tfdml/core/util/status.h"

struct TF_OpKernelConstruction;
struct TF_OpKernelContext;

namespace tfdml
{

class KernelDefBuilder;

struct InitOnStartupMarker
{
    InitOnStartupMarker(KernelDefBuilder& kernel_def_builder) {}
};

#define REGISTER_KERNEL_BUILDER_UNIQ(ctr, op_name, kernel_builder, ...)        \
    constexpr char const op_type_string##ctr[] = op_name;                      \
    static ::tfdml::InitOnStartupMarker const                                  \
        registrar__body__##ctr##__object = ::tfdml::InitOnStartupMarker(       \
            kernel_builder.Build<__VA_ARGS__, op_type_string##ctr>());

#define REGISTER_KERNEL_BUILDER_UNIQ_HELPER(ctr, op_name, kernel_builder, ...) \
    REGISTER_KERNEL_BUILDER_UNIQ(ctr, op_name, kernel_builder, __VA_ARGS__)

#define REGISTER_KERNEL_BUILDER_IMPL(op_name, kernel_builder, ...)             \
    REGISTER_KERNEL_BUILDER_UNIQ_HELPER(                                       \
        __COUNTER__,                                                           \
        op_name,                                                               \
        kernel_builder,                                                        \
        __VA_ARGS__)

#define TF_EXTRACT_KERNEL_NAME_Name(name_str) name_str, Name(name_str)
#define TF_EXTRACT_KERNEL_NAME_IMPL(m, ...) m(__VA_ARGS__)
#define TF_EXTRACT_KERNEL_NAME(m, kernel_builder, ...)                         \
    TF_EXTRACT_KERNEL_NAME_IMPL(                                               \
        m,                                                                     \
        TF_EXTRACT_KERNEL_NAME_##kernel_builder,                               \
        __VA_ARGS__)

#define REGISTER_KERNEL_BUILDER(kernel_builder, ...)                           \
    TF_EXTRACT_KERNEL_NAME(                                                    \
        REGISTER_KERNEL_BUILDER_IMPL,                                          \
        kernel_builder,                                                        \
        __VA_ARGS__)

// Forward declare proto so that kernels don't need to depend on it
class KernelDef;

// Builder class passed to the REGISTER_KERNEL_BUILDER() macro.
class KernelDefBuilder
{
  public:
    KernelDefBuilder() = default;

    // Required: specify the type of device this kernel supports.
    // Returns *this.
    KernelDefBuilder& Device(const char* device_type);
    //  KernelDefBuilder& Device(DeviceType device_type);

    template <typename T>
    KernelDefBuilder& TypeConstraint(const char* attr_name)
    {
        type_constraints_.emplace_back(attr_name, DataTypeToEnum<T>());
        return *this;
    }

    KernelDefBuilder& HostMemory(const char* arg_name);

    // Specify that this kernel requires a particular value for the
    // "_kernel" attr.  May only be specified once.  Returns *this.
    KernelDefBuilder& Label(const char* label);

    // Specify a priority number for this kernel.
    KernelDefBuilder& Priority(int32_t priority);

    template <typename TKernel, auto& op_type_string> KernelDefBuilder& Build()
    {
        auto* builder = TF_NewKernelBuilder(
            op_type_string,
            device_type_,
            &CreateKernel<TKernel, op_type_string>,
            &ComputeKernel<TKernel>,
            &DeleteKernel<TKernel>);
        CHECK(builder != nullptr);

        Status status;
        for (const auto& type_constraint : type_constraints_)
        {
            TF_KernelBuilder_TypeConstraint(
                builder,
                type_constraint.first.c_str(),
                type_constraint.second,
                status.raw());
            CHECK(status.ok());
        }

        for (const std::string& host_memory_attr : host_memory_attrs_)
        {
            TF_KernelBuilder_HostMemory(builder, host_memory_attr.c_str());
        }

        if (priority_)
        {
            TF_KernelBuilder_Priority(builder, *priority_);
        }

        TF_RegisterKernelBuilder(op_type_string, builder, status.raw());
        CHECK(status.ok());

        return *this;
    }

  private:
    KernelDefBuilder(const KernelDefBuilder&) = delete;
    void operator=(const KernelDefBuilder&) = delete;
    const char* device_type_;
    absl::InlinedVector<std::string, 4> host_memory_attrs_;
    absl::InlinedVector<std::pair<std::string, TF_DataType>, 4>
        type_constraints_;
    absl::optional<int32_t> priority_;

    template <typename TKernel, auto& op_type_string>
    static void* CreateKernel(TF_OpKernelConstruction* raw_ctx)
    {
        TF_StringView name_string_view =
            TF_OpKernelConstruction_GetName(raw_ctx);
        OpKernelConstruction ctx(raw_ctx);
        return new TKernel(&ctx, op_type_string, name_string_view.data);
    }

    template <typename TKernel>
    static void ComputeKernel(void* kernel, TF_OpKernelContext* raw_ctx)
    {
        TKernel* concrete_kernel = static_cast<TKernel*>(kernel);
        OpKernelContext ctx(raw_ctx, concrete_kernel);
        concrete_kernel->Compute(&ctx);
    }

    template <typename TKernel> static void DeleteKernel(void* kernel)
    {
        TKernel* concrete_kernel = static_cast<TKernel*>(kernel);
        delete concrete_kernel;
    }
};

class Name : public KernelDefBuilder
{
  public:
    explicit Name(const char* op) {}
};

} // namespace tfdml
