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

#include "tfdml/kernels/dml_lstm_helpers.h"
#include "tfdml/kernels/pch.h"

namespace tfdml
{

// Specifies the types of a RNN model.
enum class RnnMode
{
    kRnnRelu = 0,
    kRnnTanh = 1,
    kRnnLstm = 2,
    kRnnGru = 3,
};

// Specifies the number of directions used in a RNN model. When bidirection
// is used, the input states and output sequence contain data for both
// directions.
enum class RnnDirectionMode
{
    kRnnUnidirectional = 0,
    kRnnBidirectional = 1,
};

enum class TFRNNInputMode
{
    kRNNLinearInput = 0,
    kRNNSkipInput = 1,
    kAutoSelect = 9999999
};

Status ParseRNNMode(const std::string& str, RnnMode* rnn_mode)
{
    if (str == "rnn_relu")
    {
        *rnn_mode = RnnMode::kRnnRelu;
        return Status::OK();
    }
    else if (str == "rnn_tanh")
    {
        *rnn_mode = RnnMode::kRnnTanh;
        return Status::OK();
    }
    else if (str == "lstm")
    {
        *rnn_mode = RnnMode::kRnnLstm;
        return Status::OK();
    }
    else if (str == "gru")
    {
        *rnn_mode = RnnMode::kRnnGru;
        return Status::OK();
    }
    return errors::InvalidArgument("Invalid RNN mode: ", str);
}

Status ParseRNNDirectionMode(
    const std::string& str,
    RnnDirectionMode* rnn_dir_mode)
{
    if (str == "unidirectional")
    {
        *rnn_dir_mode = RnnDirectionMode::kRnnUnidirectional;
        return Status::OK();
    }
    else if (str == "bidirectional")
    {
        *rnn_dir_mode = RnnDirectionMode::kRnnBidirectional;
        return Status::OK();
    }
    return errors::InvalidArgument("Invalid RNN direction mode: ", str);
}

struct CudnnModelTypes
{
    RnnMode rnn_mode;
    TFRNNInputMode rnn_input_mode;
    RnnDirectionMode rnn_direction_mode;
    bool HasInputC() const
    {
        // Only LSTM has input-c. All other models use only
        // input-h.
        return rnn_mode == RnnMode::kRnnLstm;
    }
};

// Convert weight and bias params from a platform-specific layout to the
// canonical form.
template <typename T>
class CudnnRNNParamsToCanonicalKernel : public OpKernel
{
  public:
    explicit CudnnRNNParamsToCanonicalKernel(
        OpKernelConstruction* ctx,
        std::shared_ptr<const NodeDef> node_def)
        : OpKernel(std::move(node_def))
    {
        if (ctx->HasAttr("num_proj"))
        {
            OP_REQUIRES_OK(ctx, ctx->GetAttr("num_proj", &num_proj_));
        }
        else
        {
            num_proj_ = 0;
        }
        if (ctx->HasAttr("num_params"))
        {
            OP_REQUIRES_OK(ctx, ctx->GetAttr("num_params", &num_params_));
        }
        else
        {
            num_params_ = 0;
        }
        if (ctx->HasAttr("num_params_biases"))
        {
            OP_REQUIRES_OK(
                ctx,
                ctx->GetAttr("num_params_biases", &num_params_biases_));
        }
        else
        {
            num_params_biases_ = 0;
        }
        if (ctx->HasAttr("num_params_weights"))
        {
            OP_REQUIRES_OK(
                ctx,
                ctx->GetAttr("num_params_weights", &num_params_weights_));
        }
        else
        {
            num_params_weights_ = 0;
        }
        if (num_proj_ == 0)
        {
            num_params_weights_ = num_params_;
            num_params_biases_ = num_params_;
        }

        std::string str;
        OP_REQUIRES_OK(ctx, ctx->GetAttr("rnn_mode", &str));
        OP_REQUIRES_OK(ctx, ParseRNNMode(str, &model_types_.rnn_mode));
        OP_REQUIRES_OK(ctx, ctx->GetAttr("direction", &str));
        OP_REQUIRES_OK(
            ctx,
            ParseRNNDirectionMode(str, &model_types_.rnn_direction_mode));

        num_dirs = model_types_.rnn_direction_mode == RnnDirectionMode::kRnnBidirectional ? 2 : 1;
    }

  private:
    void ComputeImpl(OpKernelContext* ctx) final
    {
        const Tensor& num_layers_tensor = ctx->input(0);
        const Tensor& num_units_tensor = ctx->input(1);
        const Tensor& input_size_tensor = ctx->input(2);

        OP_REQUIRES(
            ctx,
            TensorShapeUtils::IsScalar(num_layers_tensor.shape()),
            errors::InvalidArgument(
                "num_layers must be a scalar, received shape: ",
                num_layers_tensor.shape().DebugString()));

        OP_REQUIRES(
            ctx,
            TensorShapeUtils::IsScalar(num_units_tensor.shape()),
            errors::InvalidArgument(
                "num_units must be a scalar, received shape: ",
                num_units_tensor.shape().DebugString()));

        OP_REQUIRES(
            ctx,
            TensorShapeUtils::IsScalar(input_size_tensor.shape()),
            errors::InvalidArgument(
                "input_size must be a scalar, received shape: ",
                input_size_tensor.shape().DebugString()));

        num_layers_ = num_layers_tensor.base<int>()[0];
        num_units_ = num_units_tensor.base<int>()[0];
        input_size_ = input_size_tensor.base<int>()[0];

        num_params_weights_per_layer_ =
            num_params_weights_ / num_layers_ / num_dirs_;
        num_params_input_state_ = num_params_weights_per_layer_ / 2;

        OP_REQUIRES(
            ctx,
            num_params_weights_ % (num_layers_ * num_dirs_) == 0,
            errors::InvalidArgument(
                "Number of params (weights) is not a multiple"
                "of num_layers * num_dirs."));

        OP_REQUIRES(
            ctx,
            num_params_biases_ % (num_layers_ * num_dirs_) == 0,
            errors::InvalidArgument(
                "Number of params (biases) is not a multiple"
                "of num_layers * num_dirs."));
        if (num_proj_ == 0)
        {
            OP_REQUIRES(
                ctx,
                num_params_weights_per_layer_ % 2 == 0,
                errors::InvalidArgument(
                    "Number of params (weights) per layer is not"
                    "an even number with no projection."));
        }
        else
        {
            OP_REQUIRES(
                ctx,
                num_params_weights_per_layer_ % 2 != 0,
                errors::InvalidArgument(
                    "Number of params (weights) per layer is not"
                    "an odd number with projection."));
        }

        h_num_units_ = (num_proj_ == 0 ? num_units_ : num_proj_);
        c_num_units_ = (num_proj_ == 0 ? 0 : num_units_);

        DmlDevice* device = static_cast<DmlDevice*>(ctx->device());
        auto* device_context = device->GetDeviceContext();

        uint64_t src_offset = 0;
        for (int i = 0; i < num_params_weights_; i++)
        {
            const int layer_idx = i / num_params_weights_per_layer_;
            const int index_within_layer = i % num_params_weights_per_layer_;
            int width = 0,
                height = (num_proj_ == 0 ? h_num_units_ : c_num_units_);
            bool apply_on_input_state =
                index_within_layer < num_params_input_state_;

            if (model_types_.rnn_direction_mode ==
                RnnDirectionMode::kRnnUnidirectional)
            {
                if (layer_idx == 0 && apply_on_input_state)
                {
                    width = input_size_;
                }
                else
                {
                    width = h_num_units_;
                }
            }
            else
            {
                if (apply_on_input_state)
                {
                    if (layer_idx <= 1)
                    {
                        // First fwd or bak layer.
                        width = input_size_;
                    }
                    else
                    {
                        // Following layers, cell inputs are concatenated
                        // outputs of its prior layer.
                        width = 2 * h_num_units_;
                    }
                }
                else
                {
                    width = h_num_units_;
                }
            }

            int id_in_layer = i % num_params_weights_per_layer_;
            if (num_proj_ != 0 &&
                id_in_layer == num_params_weights_per_layer_ - 1)
            {
                std::swap(height, width);
            }

            int64_t size = height * width;
            const uint64_t data_type_size = sizeof(T);
            int64_t size_in_bytes = size * data_type_size;

            StatusOr<Tensor> status_or_output_tensor =
                ctx->allocate_output(i, TensorShape({height, width}));
            OP_REQUIRES_OK(ctx, status_or_output_tensor.status());

            D3D12BufferRegion input_buffer =
                device_context->GetBufferForTensor(ctx->input(3));

            D3D12BufferRegion output_buffer =
                device_context->GetBufferForTensor(
                    status_or_output_tensor.ValueOrDie());

            device_context->CopyBufferToBuffer(
                output_buffer,
                input_buffer.Subregion(src_offset, size_in_bytes));

            src_offset += size_in_bytes;
        }

        for (int i = 0; i < num_params_biases_; i++)
        {
            int64_t size = num_units_;
            const uint64_t data_type_size = sizeof(T);
            int64_t size_in_bytes = size * data_type_size;

            StatusOr<Tensor> status_or_output_tensor = ctx->allocate_output(
                num_params_weights_ + i,
                TensorShape({size}));
            OP_REQUIRES_OK(ctx, status_or_output_tensor.status());

            D3D12BufferRegion input_buffer =
                device_context->GetBufferForTensor(ctx->input(3));

            D3D12BufferRegion output_buffer =
                device_context->GetBufferForTensor(
                    status_or_output_tensor.ValueOrDie());

            device_context->CopyBufferToBuffer(
                output_buffer,
                input_buffer.Subregion(src_offset, size_in_bytes));

            src_offset += size_in_bytes;
        }
    }

  private:
    int num_layers_;
    int num_units_;
    int input_size_;
    int h_num_units_;
    int c_num_units_;
    int num_dirs_;
    int num_params_weights_per_layer_;
    int num_params_input_state_;
    int num_proj_;
    int num_params_;
    int num_params_biases_;
    int num_params_weights_;
    CudnnModelTypes model_types_;
};

class CudnnRNNCanonicalToParamsInitHelper : public InitializationHelper
{
  public:
    struct Attributes
    {
        explicit Attributes(OpKernelConstruction* ctx)
        {
            if (ctx->HasAttr("num_proj"))
            {
                OP_REQUIRES_OK(ctx, ctx->GetAttr("num_proj", &num_proj));
            }
            else
            {
                num_proj = 0;
            }
            if (ctx->HasAttr("num_params"))
            {
                OP_REQUIRES_OK(ctx, ctx->GetAttr("num_params", &num_params));
            }
            else
            {
                num_params = 0;
            }
            if (ctx->HasAttr("num_params_biases"))
            {
                OP_REQUIRES_OK(
                    ctx,
                    ctx->GetAttr("num_params_biases", &num_params_biases));
            }
            else
            {
                num_params_biases = 0;
            }
            if (ctx->HasAttr("num_params_weights"))
            {
                OP_REQUIRES_OK(
                    ctx,
                    ctx->GetAttr("num_params_weights", &num_params_weights));
            }
            else
            {
                num_params_weights = 0;
            }
            if (num_proj == 0)
            {
                num_params_weights = num_params;
                num_params_biases = num_params;
            }

            std::string str;
            OP_REQUIRES_OK(ctx, ctx->GetAttr("rnn_mode", &str));
            OP_REQUIRES_OK(ctx, ParseRNNMode(str, &model_types.rnn_mode));
            OP_REQUIRES_OK(ctx, ctx->GetAttr("direction", &str));
            OP_REQUIRES_OK(
                ctx,
                ParseRNNDirectionMode(str, &model_types.rnn_direction_mode));

            num_dirs = model_types.rnn_direction_mode == RnnDirectionMode::kRnnBidirectional ? 2 : 1;
        }
        int num_proj;
        int num_layers;
        int num_dirs;
        int num_params;
        int num_params_biases;
        int num_params_weights;
        CudnnModelTypes model_types;
    };

    CudnnRNNCanonicalToParamsInitHelper(
        OpKernelContext* ctx,
        std::shared_ptr<const Attributes> attr)
        : attr_(std::move(attr))
    {
        const Tensor& num_layers_tensor = ctx->input(0);
        const Tensor& num_units_tensor = ctx->input(1);
        const Tensor& input_size_tensor = ctx->input(2);

        OP_REQUIRES(
            ctx,
            TensorShapeUtils::IsScalar(num_layers_tensor.shape()),
            errors::InvalidArgument(
                "num_layers must be a scalar, received shape: ",
                num_layers_tensor.shape().DebugString()));

        OP_REQUIRES(
            ctx,
            TensorShapeUtils::IsScalar(num_units_tensor.shape()),
            errors::InvalidArgument(
                "num_units must be a scalar, received shape: ",
                num_units_tensor.shape().DebugString()));

        OP_REQUIRES(
            ctx,
            TensorShapeUtils::IsScalar(input_size_tensor.shape()),
            errors::InvalidArgument(
                "input_size must be a scalar, received shape: ",
                input_size_tensor.shape().DebugString()));

        num_layers_ = num_layers_tensor.base<int>()[0];
        num_units_ = num_units_tensor.base<int>()[0];
        input_size_ = input_size_tensor.base<int>()[0];

        num_params_weights_per_layer_ =
            attr_->num_params_weights / num_layers_ / attr_->num_dirs;
        num_params_input_state_ = num_params_weights_per_layer_ / 2;

        output_size_ = 0;
        int last_index =
            3 + attr_->num_params_weights + attr_->num_params_biases;
        for (int i = 3; i < last_index; ++i)
        {
            auto input_size = ctx->input(i).shape();
            output_size_ += input_size.num_elements();
        }
    }

    int GetNumProj() const { return attr_->num_proj; }
    int GetNumParamsWeights() const { return attr_->num_params_weights; }
    int GetNumParamsBiases() const { return attr_->num_params_biases; }
    int GetNumLayers() const { return num_layers_; }
    int GetNumUnits() const { return num_units_; }
    int GetInputSize() const { return input_size_; }
    int64_t GetOutputSize() const { return output_size_; }
    RnnMode GetRNNMode() const { return attr_->model_types.rnn_mode; }
    RnnDirectionMode GetRNNDirectionMode() const
    {
        return attr_->model_types.rnn_direction_mode;
    }

  private:
    std::shared_ptr<const Attributes> attr_;
    int num_params_weights_per_layer_;
    int num_params_input_state_;
    int num_layers_;
    int num_units_;
    int input_size_;
    int64_t output_size_;
};

class CudnnRNNCanonicalToParamsShapeHelper : public ShapeHelper
{
  public:
    using InitHelper = CudnnRNNCanonicalToParamsInitHelper;
    std::vector<TensorShape> GetOutputShapes(
        OpKernelContext* ctx,
        const InitializationHelper* initialization_helper) const override
    {
        auto init_helper =
            static_cast<const InitHelper*>(initialization_helper);

        int64_t output_size = init_helper->GetOutputSize();

        TensorShape output_shape({output_size});

        return {std::move(output_shape)};
    }
};

class CudnnRNNCanonicalToParamsKernel : public DmlKernel
{
  public:
    using InitHelper = CudnnRNNCanonicalToParamsInitHelper;

    explicit CudnnRNNCanonicalToParamsKernel(
        DmlKernelConstruction* ctx,
        const InitHelper* init_helper)
    {
        uint32_t u_output_size =
            static_cast<uint32_t>(init_helper->GetOutputSize());
        TensorShape output_shape({ctx->GetOutputTensorShape(0).num_elements()});

        DmlKernelTensors tensors;

        DmlTensorInfo output;
        output.kernel_index = 0;
        output.desc = DmlTensorDesc::Create(
            ctx->GetOutputDataType(0),
            output_shape,
            output_shape);
        tensors.outputs = {output};

        uint32_t first_weight_index = 3;
        uint32_t last_weight_index =
            first_weight_index + init_helper->GetNumParamsWeights();
        uint32_t first_bias_index = last_weight_index;
        uint32_t last_bias_index =
            first_bias_index + init_helper->GetNumParamsBiases();

        for (uint32_t i = first_weight_index; i < last_weight_index; i++)
        {
            TensorShape input_shape(
                {1, 1, 1, ctx->GetInputTensorShape(i).num_elements()});
            DmlTensorInfo input;
            input.kernel_index = i;
            input.desc = DmlTensorDesc::Create(
                ctx->GetInputDataType(i),
                input_shape,
                input_shape);

            tensors.inputs.push_back(std::move(input));
        }

        for (uint32_t i = first_bias_index; i < last_bias_index; i++)
        {
            TensorShape input_shape(
                {1, 1, 1, ctx->GetInputTensorShape(i).num_elements()});
            DmlTensorInfo input;
            input.kernel_index = i;
            input.desc = DmlTensorDesc::Create(
                ctx->GetInputDataType(i),
                input_shape,
                input_shape);

            tensors.inputs.push_back(std::move(input));
        }

        auto inputs = GetDmlTensorDescs(tensors.inputs);
        auto scope = dml::Graph(ctx->GetDmlDevice());

        absl::InlinedVector<dml::Expression, 4> input_tensors;
        input_tensors.reserve(last_bias_index - first_weight_index);

        for (int i = 0; i < last_bias_index - first_weight_index; ++i)
        {
            input_tensors.push_back(dml::InputTensor(scope, i, inputs[i]));
        }

        auto result = dml::Join(input_tensors, 3);
        result = dml::Reinterpret(
            result,
            dml::TensorDimensions({u_output_size}),
            {});

        Microsoft::WRL::ComPtr<IDMLCompiledOperator> compiled_op =
            scope.Compile(DML_EXECUTION_FLAG_NONE, {result});

        Initialize(ctx, std::move(tensors), compiled_op.Get());
    }
};

template <typename T>
void RegisterCudnnRNNParamsToCanonical()
{
    using Op = ops::CudnnRNNParamsToCanonical;

    KernelDefinition<Op, CudnnRNNParamsToCanonicalKernel<T>>::
        template WithHostMemoryArguments<Op::Argument::num_layers>::
            template WithHostMemoryArguments<Op::Argument::num_units>::
                template WithHostMemoryArguments<Op::Argument::input_size>::
                    template WithTypeConstraint<
                        Op::Attribute::T,
                        DataTypeToEnum<T>()>::Register();
}

template <
    typename T,
    typename... Ts,
    std::enable_if_t<sizeof...(Ts) >= 1>* = nullptr>
static void RegisterCudnnRNNParamsToCanonical()
{
    RegisterCudnnRNNParamsToCanonical<T>();
    RegisterCudnnRNNParamsToCanonical<Ts...>();
}

template <typename T>
void RegisterCudnnRNNParamsToCanonicalV2()
{
    using Op = ops::CudnnRNNParamsToCanonicalV2;

    KernelDefinition<Op, CudnnRNNParamsToCanonicalKernel<T>>::
        template WithHostMemoryArguments<Op::Argument::num_layers>::
            template WithHostMemoryArguments<Op::Argument::num_units>::
                template WithHostMemoryArguments<Op::Argument::input_size>::
                    template WithTypeConstraint<
                        Op::Attribute::T,
                        DataTypeToEnum<T>()>::Register();
}

template <
    typename T,
    typename... Ts,
    std::enable_if_t<sizeof...(Ts) >= 1>* = nullptr>
static void RegisterCudnnRNNParamsToCanonicalV2()
{
    RegisterCudnnRNNParamsToCanonicalV2<T>();
    RegisterCudnnRNNParamsToCanonicalV2<Ts...>();
}

void RegisterCudnnRNNCanonicalToParams()
{
    using Op = ops::CudnnRNNCanonicalToParams;
    using K = KernelDefinition<
        Op,
        DmlKernelWrapper<
            CudnnRNNCanonicalToParamsKernel,
            CudnnRNNCanonicalToParamsShapeHelper>>::
        template WithHostMemoryArguments<Op::Argument::num_layers>::
            template WithHostMemoryArguments<Op::Argument::num_units>::
                template WithHostMemoryArguments<Op::Argument::input_size>;

    RegisterWithTypes<K, Op::Attribute::T, TF_FLOAT, TF_HALF, TF_DOUBLE>();
}

void RegisterCudnnRNNCanonicalToParamsV2()
{
    using Op = ops::CudnnRNNCanonicalToParamsV2;
    using K = KernelDefinition<
        Op,
        DmlKernelWrapper<
            CudnnRNNCanonicalToParamsKernel,
            CudnnRNNCanonicalToParamsShapeHelper>>::
        template WithHostMemoryArguments<Op::Argument::num_layers>::
            template WithHostMemoryArguments<Op::Argument::num_units>::
                template WithHostMemoryArguments<Op::Argument::input_size>;

    RegisterWithTypes<K, Op::Attribute::T, TF_FLOAT, TF_HALF, TF_DOUBLE>();
}

void RegisterKernels_CudnnRNN()
{
    RegisterCudnnRNNParamsToCanonical<float, Eigen::half, double>();
    RegisterCudnnRNNParamsToCanonicalV2<float, Eigen::half, double>();
    RegisterCudnnRNNCanonicalToParams();
    RegisterCudnnRNNCanonicalToParamsV2();
}

} // namespace tfdml