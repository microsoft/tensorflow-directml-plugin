# Kernel Cookbook

## Implement the kernel

To add a new kernel implementation for `tensorflow-directml-plugin`, follow the following steps:

1. Add a file to [tfdml/core/kernels](tfdml/core/kernels) that follows the existing `dml_<kernel_name>_op.cc` nomenclature (e.g. `dml_demo_op.cc`)
2. Add `tfdml/kernels/dml_demo_op.cc` to the `kernels` library in [CMakeLists.txt](CMakeLists.txt)
3. In `dml_demo_op.cc`, use the following template as a starting point to implement the kernel:

    ```cpp
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
    class DemoShapeHelper : public ShapeHelper
    {
        std::vector<TensorShape> GetOutputShapes(
            OpKernelContext* ctx,
            const InitializationHelper* initialization_helper) const final
        {
            // TODO: Implement me
        }
    };

    class DemoInitHelper : public InitializationHelper
    {
    public:
        struct Attributes
        {
            explicit Attributes(OpKernelConstruction* ctx)
            {
                // TODO: Implement me
            }
        };

        DemoInitHelper(
            OpKernelContext* ctx,
            std::shared_ptr<const Attributes> attr)
            : attr_(std::move(attr))
        {
            // TODO: Implement me
        }

        bool IsNoOpKernel(
            OpKernelContext* ctx,
            absl::Span<const TensorShape> output_shapes) const override
        {
            // TODO: Implement me
        }
    };

    class DmlDemoKernel : public DmlKernel
    {
    public:
        using InitHelper = DemoInitHelper;

        explicit DmlDemoKernel(
            DmlKernelConstruction* ctx,
            const InitHelper* init_helper)
        {
            // TODO: Implement me
        }
    };

    void RegisterKernels_Demo()
    {
        // TODO: Implement me
    }

    } // namespace tfdml
    ```

4. Implement `DemoShapeHelper::ShapeHelper`. `DemoShapeHelper::ShapeHelper` is generally pretty simple: all it does is compute the shape of the outputs for the kernel. For example, for a simple elementwise operation, you would do the following:

    ```cpp
    std::vector<TensorShape> GetOutputShapes(
        OpKernelContext* ctx,
        const InitializationHelper* initialization_helper) const final
    {
        return {ctx->input(0).shape()};
    }
    ```

    Most of the time, simply copying the implementation from the [tensorflow](https://github.com/tensorflow/tensorflow) repository is enough to implement this function. Also, since a lot of operators share the same shape logic, you can take a look at the common shape helpers available in [dml_operator_helper.cc](tfdml/core/dml_operator_helper.cc) and see if one already works for your kernel.

5. Implement the `Attributes` constructor. The `Attributes` constructor is where you will validate and fetch all attributes that you need from the kernel. For example, to fetch an integer named `Foo` from the `OpKernelConstruction` and validate that it exists, you would do the following:

    ```cpp
    struct Attributes
    {
        int foo;
        OP_REQUIRES_OK(ctx, ctx->GetAttr("Foo", &foo));
    };
    ```

    Note that not all kernels have attribute. In that case, instead of implementing an empty struct, you can simply use the `EmptyAttributes` helper:

    ```cpp
    using Attributes = EmptyAttributes;
    ```

6. Implement the `DemoInitHelper` constructor. The `DemoInitHelper` constructor is where you wil validate and fetch all inputs that you need from the kernel. For example, to fetch the first input and validate that it has 4 dimensions, you would the following:

    ```cpp
    DemoInitHelper(
        OpKernelContext* ctx,
        std::shared_ptr<const Attributes> attr)
        : attr_(std::move(attr))
    {
        const Tensor input = ctx->input(0);
        OP_REQUIRES(
            ctx,
            input.dims() == 4,
            errors::InvalidArgument(
                "input must be 4-dimensional",
                input.shape().DebugString()));
    }
    ```

    It is **very rare** that you won't need an initialization helper since most operators require some kind of input validation, but if you don't need it, you can use the `NoOpInitializationHelper` instead inside the `DmlDemoKernel` class:

    ```cpp
    using InitHelper = NoOpInitializationHelper;
    ```

7. **[Optional]** Implement the `IsNoOpKernel` function. When `IsNoOpKernel` is ommitted, the default behavior is to treat operators that have empty inputs or inputs (i.e. `input.NumElements() == 0`) as no-op kernels. Although this is the right behavior for most operators, some operators need a custom behavior. For example, if we look at [dml_concat_op.cc](tfdml/kernels/dml_concat_op.cc), we see that the default behavior has been overriden to allow empty inputs, **as long as** at least one input is not empty. The rules for whether a kernel should be treated as a no-op are not are very kernel-dependent, so you should dive into corresponding kernel implementations in the [tensorflow](https://github.com/tensorflow/tensorflow) codebase to make sure that all edge cases are handled correctly.

8. Implement the `DmlDemoKernel` constructor. The `DmlDemoKernel` constructor is where all the heavy lifting is done and where we compile the kernel. Most kernels follow the same recipe:

    1. Get the inputs
    2. If the operator has attributes, get them from the initialization helper
    3. Create the DML tensor descs
    4. Initialize the DML graph
    5. Compile the DML operator

    A basic implementation will look like this:

    ```cpp
    explicit DmlDemoKernel(
        DmlKernelConstruction* ctx,
        const InitHelper* init_helper)
    {
        auto input_shapes = {
            ctx->input(0).shape(),
            ctx->input(1).shape(),
        };
        const TensorShape& output_shape =
            init_helper->GetCollapsedOutputShape();

        DmlKernelTensors tensors = CreateKernelTensors(
            ctx,
            input_shapes,
            output_shape);
        auto inputs = GetDmlTensorDescs(tensors.inputs);

        auto scope = dml::Graph(ctx->GetDmlDevice());
        auto x = dml::InputTensor(scope, 0, inputs[0]);
        auto y = dml::InputTensor(scope, 1, inputs[1]);
        auto result = dml::DemoOperator(x, y);

        ComPtr<IDMLCompiledOperator> compiled_op =
            scope.Compile(DML_EXECUTION_FLAG_NONE, {result});

        Initialize(ctx, std::move(tensors), compiled_op.Get());
    }
    ```

9. Implement `RegisterKernels_Demo`. `RegisterKernels_Demo` is where we register the kernel that we just created. If we want to register our `Demo` kernel for the `float` and `half` types, it would look something like this:

    ```cpp
    void RegisterKernels_Demo()
    {
        using K = KernelDefinition<
            ops::Demo,
            DmlKernelWrapper<DmlDemoKernel, DemoShapeHelper>>;

        constexpr auto T = ops::Demo::Attribute::T;
        K::WithTypeConstraint<T, TF_FLOAT>::Register();
        K::WithTypeConstraint<T, TF_HALF>::Register();
    }
    ```

10. Finally, in [plugin_kernel.cc](tfdml/plugin/plugin_kernel.cc), add the function definition and call it inside `TF_InitKernel`:

    ```cpp
    //...
    namespace tfdml
    {
    // ...
    void RegisterKernels_Demo();
    // ...
    TFDML_EXPORT void TF_InitKernel()
    {
    // ...
    tfdml::RegisterKernels_Demo();
    // ...
    }
    }
    ```

## Test the kernel

Of course, each kernel that we add should have some kind of test collateral. Testing machine learning operators for all edge cases and handling precision issues can get relatively complex, but fortunately for us, we can reuse most of the TensorFlow tests that have already proven themselves. In the TensorFlow codebase, just look for a file that loosely follow the pattern `<operator_name>_test.py` and copy the file over to `test/ops`. Once the file is in there, simply add it to the [list of files that the CI tests](test/tests.json).