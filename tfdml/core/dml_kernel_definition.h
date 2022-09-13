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
#include "tfdml/core/dml_ops_common.h"
#include "tfdml/core/dml_tracing.h"
#include "tfdml/runtime_adapter/macros.h"
#include "tfdml/runtime_adapter/node_def.h"
#include "tfdml/runtime_adapter/op_defs.h"
#include "tfdml/runtime_adapter/op_kernel_construction.h"
#include "tfdml/runtime_adapter/op_kernel_context.h"
#include "tfdml/runtime_adapter/status.h"

#ifdef _DEBUG
// #define DBG_NEW new ( _NORMAL_BLOCK , __FILE__ , __LINE__ )
    #define DBG_NEW new
    // Replace _NORMAL_BLOCK with _CLIENT_BLOCK if you want the
    // allocations to be of _CLIENT_BLOCK type
#else
    #define DBG_NEW new
#endif

struct TF_OpKernelConstruction;
struct TF_OpKernelContext;

extern std::atomic<size_t> memory_usage;

namespace tfdml
{

// Type that contains zero or more op arguments.
template <typename Op, typename Op::Argument...>
struct OpArgumentList;

// Type that describes a data-type constraint imposed on an op attribute.
template <typename Op, typename Op::Attribute Attribute, TF_DataType DataType>
struct OpTypeConstraint
{
    static constexpr typename Op::Attribute AttributeValue = Attribute;
    static constexpr TF_DataType DataTypeValue = DataType;

    static_assert(
        Op::attribute_descs[ConvertOpDefEnumToIndex(Attribute)].type ==
            AttributeType::Type,
        "Type constraints are only valid on attributes with type "
        "'AttributeType::Type'. Check that you are registering a "
        "constraint on the intended attribute!");
};

// Type that contains zero or more type constraints.
template <typename Op, typename... OpTypeConstraints>
struct OpTypeConstraintList;

// Template for declaring a kernel registration type that statically defines the
// following:
// - Op : declares the operator definition that the kernel implements
// - Kernel : declares the type of the kernel class that implements the operator
// - Priority : declares an integer value that influences the kernel's selection
// at runtime
// - TypeConstraints : declares zero or more data-type constraints imposed on
// operator attributes
// - HostArguments : declares zero or more operator arguments that must reside
// in host memory
//
// At a minimum you must specify the Op and Kernel traits to form a valid kernel
// registration. Setting all of the traits together is possible but difficult to
// read; you are encouraged to use the 'With*' helpers for extending the type.
// For example:
//
// KernelDefinition<ops::AssignVariableOp, DmlAssignVariableOp>
//    ::WithTypeConstraint<ops::AssignVariableOp::Attribute::dtype, TF_FLOAT>
//    ::WithHostMemoryArguments<ops::AssignVariableOp::Argument::resource>
//    ::Register();
template <
    typename Op,
    typename Kernel,
    uint32_t Priority = 0,
    typename TypeConstraints = OpTypeConstraintList<Op>,
    typename HostArguments = OpArgumentList<Op>>
class KernelDefinition;

// Specialized template necessary for having two parameter packs
// (TypeConstraints and HostArguments).
template <
    typename Op,
    typename Kernel,
    uint32_t PriorityValue,
    typename... TypeConstraints,
    typename Op::Argument... HostArguments>
class KernelDefinition<
    Op,
    Kernel,
    PriorityValue,
    OpTypeConstraintList<Op, TypeConstraints...>,
    OpArgumentList<Op, HostArguments...>>
{
  public:
    using OpType = Op;

    // Sets the priority value.
    template <uint32_t Value>
    using WithPriority = KernelDefinition<
        Op,
        Kernel,
        Value,
        OpTypeConstraintList<Op, TypeConstraints...>,
        OpArgumentList<Op, HostArguments...>>;

    // Extend the kernel registration type with an additional type constraint.
    template <typename Op::Attribute A, TF_DataType Type>
    using WithTypeConstraint = KernelDefinition<
        Op,
        Kernel,
        PriorityValue,
        OpTypeConstraintList<
            Op,
            TypeConstraints...,
            OpTypeConstraint<Op, A, Type>>,
        OpArgumentList<Op, HostArguments...>>;

    // Extend the kernel registration type with additional host-memory
    // arguments.
    template <typename Op::Argument... NewHostArgs>
    using WithHostMemoryArguments = KernelDefinition<
        Op,
        Kernel,
        PriorityValue,
        OpTypeConstraintList<Op, TypeConstraints...>,
        OpArgumentList<Op, HostArguments..., NewHostArgs...>>;

    static void Register()
    {
        auto* builder = TF_NewKernelBuilder(
            Op::name,
            DEVICE_DML,
            &CreateKernel,
            &ComputeKernel,
            &DeleteKernel);
        CHECK(builder != nullptr);

        SetTypeConstraints<TypeConstraints...>(builder);

        for (auto arg :
             std::initializer_list<typename Op::Argument>{HostArguments...})
        {
            const auto& arg_desc = GetArgumentDesc<Op>(arg);
            TF_KernelBuilder_HostMemory(builder, arg_desc.name);
        }

        if (PriorityValue)
        {
            TF_KernelBuilder_Priority(builder, PriorityValue);
        }

        Status status;
        TF_RegisterKernelBuilder(Op::name, builder, status.raw());
        CHECK(status.ok());
    }

  private:
    template <typename C = void, typename... Cs>
    static void SetTypeConstraints(TF_KernelBuilder* builder)
    {
        if constexpr (!std::is_same<C, void>::value)
        {
            Status status;
            const auto& attr_desc = GetAttributeDesc<Op>(C::AttributeValue);
            TF_KernelBuilder_TypeConstraint(
                builder,
                attr_desc.name,
                C::DataTypeValue,
                status.raw());
            CHECK(status.ok());
        }
        if constexpr (sizeof...(Cs) > 0)
        {
            SetTypeConstraints<Cs...>(builder);
        }
    }

    static void* CreateKernel(TF_OpKernelConstruction* raw_ctx)
    {
        memory_usage += sizeof(Kernel);
        printf("***********CreateKernel memory_usage: %llu\n", memory_usage);

        OpKernelConstruction ctx(raw_ctx);
        NodeDef node_def = NodeDef::Create<Op, HostArguments...>(ctx);
        auto abc = DBG_NEW Kernel(
            &ctx,
            std::make_shared<const NodeDef>(std::move(node_def)));
        return abc;
    }

    static void ComputeKernel(void* kernel, TF_OpKernelContext* raw_ctx)
    {
        Kernel* concrete_kernel = static_cast<Kernel*>(kernel);
        OpKernelContext ctx(raw_ctx, concrete_kernel);
        concrete_kernel->Compute(&ctx);

#ifdef DIRECTML_ENABLE_TELEMETRY
        DmlTracing::Instance().LogKernelComputeTelemetry(Op::name);
#endif
    }

    static void DeleteKernel(void* kernel)
    {
        memory_usage -= sizeof(Kernel);
        printf("***********DeleteKernel memory_usage: %llu\n", memory_usage);
        Kernel* concrete_kernel = static_cast<Kernel*>(kernel);
        delete concrete_kernel;
    }
};

// Helper for registering a kernel definition K once for each data type
// constraint. This simple helper is intended for kernels that share the
// same definition aside from a single data-type attribute.
template <
    typename K,
    typename K::OpType::Attribute Attr,
    TF_DataType T,
    TF_DataType... Ts>
void RegisterWithTypes()
{
    K::template WithTypeConstraint<Attr, T>::Register();
    if constexpr (sizeof...(Ts) > 0)
    {
        RegisterWithTypes<K, Attr, Ts...>();
    }
}

} // namespace tfdml
