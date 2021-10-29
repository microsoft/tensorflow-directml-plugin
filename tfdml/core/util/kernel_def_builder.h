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

// Type that contains zero or more op arguments.
template <typename Op, typename Op::Argument...> struct OpArgumentList;

// Type that describes a data-type constraint imposed on an op attribute.
template <typename Op, typename Op::Attribute Attribute, TF_DataType DataType>
struct OpTypeConstraint
{
    static constexpr typename Op::Attribute Attribute = Attribute;
    static constexpr TF_DataType DataType = DataType;
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
//    ::WithHostMemoryArgument<ops::AssignVariableOp::Argument::resource>
//    ::Register();
template <
    typename Op,
    typename Kernel,
    uint32_t Priority = 0,
    typename TypeConstraints = OpTypeConstraintList<Op>,
    typename HostArguments = OpArgumentList<Op>>
struct KernelDefinition;

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

    // Extend the kernel registration type with an additional host-memory
    // argument.
    template <typename Op::Argument HostArg>
    using WithHostMemoryArgument = KernelDefinition<
        Op,
        Kernel,
        PriorityValue,
        OpTypeConstraintList<Op, TypeConstraints...>,
        OpArgumentList<Op, HostArguments..., HostArg>>;

    static void Register()
    {
        // auto* builder = TF_NewKernelBuilder(
        //     OpDef::name,
        //     DEVICE_DML,
        //     &CreateKernel,
        //     &ComputeKernel,
        //     &DeleteKernel);
        // CHECK(builder != nullptr);

        // Status status;
        // for (const auto& type_constraint : type_constraints_)
        // {
        //     const auto& attr_desc =
        //         GetAttributeDesc<OpDef>(type_constraint.first);
        //     TF_KernelBuilder_TypeConstraint(
        //         builder,
        //         attr_desc.name,
        //         type_constraint.second,
        //         status.raw());
        //     CHECK(status.ok());
        // }

        // for (const auto& arg : host_memory_args_)
        // {
        //     const auto& arg_desc = GetArgumentDesc<OpDef>(arg);
        //     TF_KernelBuilder_HostMemory(builder, arg_desc.name);
        // }

        // if (priority_)
        // {
        //     TF_KernelBuilder_Priority(builder, *priority_);
        // }

        // TF_RegisterKernelBuilder(OpDef::name, builder, status.raw());
        // CHECK(status.ok());


        // SetTypeConstraints<TypeConstraints...>();

        // for (auto arg :
        // std::initializer_list<Op::Argument>{HostArguments...})
        // {
        // }
    }

  private:
    template <typename T = void, typename... Ts>
    static void SetTypeConstraints()
    {
        // if constexpr (!std::is_same_v<T, void>)
        // {
        //     constexpr auto attr = T::Attribute;
        //     constexpr auto dtype = T::DataType;
        // }
        // if constexpr (sizeof...(Ts) > 0)
        // {
        //     SetTypeConstraints<Ts...>();
        // }
    }

    // static void* CreateKernel(TF_OpKernelConstruction* raw_ctx)
    // {
    //     TF_StringView name_string_view =
    //         TF_OpKernelConstruction_GetName(raw_ctx);
    //     OpKernelConstruction ctx(raw_ctx);
    //     return new Kernel(&ctx, OpDef::name, name_string_view.data);
    // }

    // static void ComputeKernel(void* kernel, TF_OpKernelContext* raw_ctx)
    // {
    //     Kernel* concrete_kernel = static_cast<Kernel*>(kernel);
    //     OpKernelContext ctx(raw_ctx, concrete_kernel);
    //     concrete_kernel->Compute(&ctx);
    // }

    // static void DeleteKernel(void* kernel)
    // {
    //     Kernel* concrete_kernel = static_cast<Kernel*>(kernel);
    //     delete concrete_kernel;
    // }
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
    K::WithTypeConstraint<Attr, T>::Register();
    if constexpr (sizeof...(Ts) > 0)
    {
        RegisterWithTypes<K, Attr, Ts...>();
    }
}

} // namespace tfdml
