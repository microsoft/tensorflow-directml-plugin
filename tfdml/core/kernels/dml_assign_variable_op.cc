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

#include "tensorflow/c/tf_datatype.h"
#include "tfdml/core/common_runtime/dml/dml_common.h"
#include "tfdml/core/util/kernel_def_builder.h"
#include "tfdml/core/util/macros.h"
#include "tfdml/core/util/op_kernel.h"
#include "tfdml/core/util/op_kernel_construction.h"
#include "tfdml/core/util/op_kernel_context.h"
#include "tfdml/core/util/refcount.h"
#include "tfdml/core/util/resource_var.h"
#include "tfdml/core/util/tensor.h"

namespace tfdml
{
class DmlAssignVariableOp : public OpKernel
{
  public:
    explicit DmlAssignVariableOp(
        OpKernelConstruction* c,
        const char* op_type_string,
        const char* op_name)
        : OpKernel(op_type_string, op_name)
    {
        OP_REQUIRES_OK(c, c->GetAttr("dtype", &dtype_));
        if (!c->GetAttr(
                  "_grappler_relax_allocator_constraints",
                  &relax_constraints_)
                 .ok())
        {
            relax_constraints_ = false;
        }
    }

    void Compute(OpKernelContext* context)
    {
        OP_REQUIRES(
            context,
            dtype_ == context->input(1).dtype(),
            errors::InvalidArgument(
                "Variable and value dtypes don't match; respectively, ",
                DataTypeString(dtype_),
                " and ",
                DataTypeString(context->input(1).dtype())));
        RefCountPtr<Var> variable;
        const Tensor& value = context->input(1);
        // Note: every resource-variable-manipulating op assumes copy-on-write
        // semantics, and creates a copy of the variable's Tensor if its
        // refcount is bigger than 1 when we try to modify it. This means we
        // never need to copy the original tensor for AssignVariableOp; even if
        // there are other live users of it we know none can modify it so this
        // is always safe (even in esoteric cases where the same tensor is used
        // to initialize multiple variables or the tensor is a constant this is
        // safe, as future writes will trigger copies).

        const Tensor handle_input = context->input(0);

        OP_REQUIRES_OK(
            context,
            LookupOrCreateResource<Var>(
                context,
                handle_input.base<tensorflow::ResourceHandleProto>()[0],
                &variable,
                [this, &value](Var** ptr)
                {
                    *ptr = new Var(dtype_);
                    *(*ptr)->tensor() = value;
                    (*ptr)->is_initialized = true;
                    return Status::OK();
                }));
        std::unique_lock<std::shared_mutex> ml(*variable->mu());
        OP_REQUIRES(
            context,
            variable->tensor()->dtype() == dtype_,
            errors::InvalidArgument(
                "Trying to assign variable with wrong dtype. Expected ",
                DataTypeString(variable->tensor()->dtype()),
                " got ",
                DataTypeString(dtype_)));

        *variable->tensor() = value;
        variable->is_initialized = true;
    }

  private:
    TF_DataType dtype_;
    bool relax_constraints_;
};

void RegisterKernels_AssignVariableOp()
{
    // We deliberately register the same types here that CUDA does.
    for (auto& type :
         {TF_BOOL,
          TF_COMPLEX64,
          TF_COMPLEX128,
          TF_HALF,
          TF_FLOAT,
          TF_DOUBLE,
          TF_INT64})
    {
        using Op = ops::AssignVariableOp;
        KernelBuilder<Op, DmlAssignVariableOp>()
            .TypeConstraint(Op::Attribute::dtype, type)
            .HostMemory(Op::Argument::resource)
            .Register();
    }
}

} // namespace tfdml
