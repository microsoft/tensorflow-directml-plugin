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
// Specifies the types of a RNN model.
enum class RnnMode {
  kRnnRelu = 0,
  kRnnTanh = 1,
  kRnnLstm = 2,
  kRnnGru = 3,
};

// Specifies the number of directions used in a RNN model. When bidirection
// is used, the input states and output sequence contain data for both
// directions.
enum class RnnDirectionMode {
  kRnnUnidirectional = 0,
  kRnnBidirectional = 1,
};

enum class TFRNNInputMode {
  kRNNLinearInput = 0,
  kRNNSkipInput = 1,
  kAutoSelect = 9999999
};

Status ParseRNNMode(const std::string& str, RnnMode* rnn_mode) {
  if (str == "rnn_relu") {
    *rnn_mode = RnnMode::kRnnRelu;
    return Status::OK();
  } else if (str == "rnn_tanh") {
    *rnn_mode = RnnMode::kRnnTanh;
    return Status::OK();
  } else if (str == "lstm") {
    *rnn_mode = RnnMode::kRnnLstm;
    return Status::OK();
  } else if (str == "gru") {
    *rnn_mode = RnnMode::kRnnGru;
    return Status::OK();
  }
  return errors::InvalidArgument("Invalid RNN mode: ", str);
}

Status ParseTFRNNInputMode(const std::string& str, TFRNNInputMode* rnn_input_mode) {
  if (str == "linear_input") {
    *rnn_input_mode = TFRNNInputMode::kRNNLinearInput;
    return Status::OK();
  } else if (str == "skip_input") {
    *rnn_input_mode = TFRNNInputMode::kRNNSkipInput;
    return Status::OK();
  } else if (str == "auto_select") {
    *rnn_input_mode = TFRNNInputMode::kAutoSelect;
    return Status::OK();
  }
  return errors::InvalidArgument("Invalid RNN input mode: ", str);
}

Status ParseRNNDirectionMode(const std::string& str,
                             RnnDirectionMode* rnn_dir_mode) {
  if (str == "unidirectional") {
    *rnn_dir_mode = RnnDirectionMode::kRnnUnidirectional;
    return Status::OK();
  } else if (str == "bidirectional") {
    *rnn_dir_mode = RnnDirectionMode::kRnnBidirectional;
    return Status::OK();
  }
  return errors::InvalidArgument("Invalid RNN direction mode: ", str);
}

struct CudnnModelTypes {
  RnnMode rnn_mode;
  TFRNNInputMode rnn_input_mode;
  RnnDirectionMode rnn_direction_mode;
  bool HasInputC() const {
    // For Cudnn 5.0, only LSTM has input-c. All other models use only
    // input-h.
    return rnn_mode == RnnMode::kRnnLstm;
  }

//   std::string DebugString() const {
//     return strings::Printf(
//         "[rnn_mode, rnn_input_mode, rnn_direction_mode]: %d, %d, %d ",
//         static_cast<int>(rnn_mode), static_cast<int>(rnn_input_mode),
//         static_cast<int>(rnn_direction_mode));
//   }
};

class CudnnRNNCommonInitHelper : public InitializationHelper
{
  protected:
    struct Attributes
    {
        explicit Attributes(OpKernelConstruction* ctx)
        {
            OP_REQUIRES_OK(ctx, ctx->GetAttr("dropout", &dropout));
            OP_REQUIRES_OK(ctx, ctx->GetAttr("seed", &seed));
            OP_REQUIRES_OK(ctx, ctx->GetAttr("seed2", &seed2));
            std::string str;
            OP_REQUIRES_OK(ctx, ctx->GetAttr("rnn_mode", &str));
            OP_REQUIRES_OK(ctx, ParseRNNMode(str, &model_types.rnn_mode));
            OP_REQUIRES_OK(ctx, ctx->GetAttr("input_mode", &str));
            OP_REQUIRES_OK(ctx, ParseTFRNNInputMode(str, &model_types.rnn_input_mode));
            OP_REQUIRES_OK(ctx, ctx->GetAttr("direction", &str));
            OP_REQUIRES_OK(ctx, ParseRNNDirectionMode(str, &model_types.rnn_direction_mode));
        }
        int seed;
        int seed2;
        float dropout;
        bool reset_rnd_gen_state;
        CudnnModelTypes model_types;
    };

    CudnnRNNCommonInitHelper(OpKernelContext* ctx, std::shared_ptr<const Attributes> attr)
        : attr_(std::move(attr))
    {

        // const Tensor& input_tensor = ctx->input(0);
        // const Tensor& init_h_tensor = ctx->input(1);
        // const Tensor& init_c_tensor = ctx->input(2);
        // const Tensor& params_tensor = ctx->input(3);
    }

    bool HasInputC() const { return attr_->model_types.HasInputC(); }
    RnnMode rnn_mode() const { return attr_->model_types.rnn_mode; }
    TFRNNInputMode rnn_input_mode() const { return attr_->model_types.rnn_input_mode; }
    RnnDirectionMode rnn_direction_mode() const {
        return attr_->model_types.rnn_direction_mode;
    }
    const CudnnModelTypes& model_types() const { return attr_->model_types; }
    float dropout() const { return attr_->dropout; }
    uint64_t seed() { return (static_cast<uint64_t>(attr_->seed) << 32) | attr_->seed; }

  private:
    std::shared_ptr<const Attributes> attr_;
};

class CudnnRNNParamsSizeInitHelper : public InitializationHelper
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
            std::string str;
            OP_REQUIRES_OK(ctx, ctx->GetAttr("rnn_mode", &str));
            OP_REQUIRES_OK(ctx, ParseRNNMode(str, &model_types.rnn_mode));
            OP_REQUIRES_OK(ctx, ctx->GetAttr("direction", &str));
            OP_REQUIRES_OK(ctx, ParseRNNDirectionMode(str, &model_types.rnn_direction_mode));
        }
        int num_proj;
        CudnnModelTypes model_types;
    };

    CudnnRNNParamsSizeInitHelper(OpKernelContext* ctx, std::shared_ptr<const Attributes> attr)
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

        int h_num_units = (attr_->num_proj == 0 ? num_units_ : attr_->num_proj);
        int c_num_units = (attr_->num_proj == 0 ? 0 : num_units_);
    }

    int GetNumProj() const { return attr_->num_proj; }
    int GetNumLayers() const { return num_layers_; }
    int GetNumUnits() const { return num_units_; }
    int GetInputSize() const { return input_size_; }
    RnnMode GetRNNMode() const { return attr_->model_types.rnn_mode; }
    RnnDirectionMode GetRNNDirectionMode() const {
        return attr_->model_types.rnn_direction_mode;
    }

  private:
    std::shared_ptr<const Attributes> attr_;
    int num_layers_;
    int num_units_;
    int input_size_;
};

// A class that returns the size of the opaque parameter buffer. The user should
// use that to create the actual parameter buffer for training. However, it
// should not be used for saving and restoring.
class CudnnRNNParamsSizeOp : public DmlKernel {
 public:
    using InitHelper = CudnnRNNParamsSizeInitHelper;
    explicit CudnnRNNParamsSizeOp(DmlKernelConstruction* ctx,
            const InitHelper* init_helper)
    {
        DmlKernelTensors tensors = GetTensorInfos(ctx, DmlKernelParams{});
        auto scope = dml::Graph(ctx->GetDmlDevice());
        auto input_descs = GetDmlTensorDescs(tensors.inputs);
        auto num_layers = dml::InputTensor(scope, 0, input_descs[0]);
        auto num_units = dml::InputTensor(scope, 1, input_descs[1]);
        auto input_size = dml::InputTensor(scope, 2, input_descs[2]);


        int params_per_layer = 0;

        switch (init_helper->GetRNNMode()) {
            case RnnMode::kRnnRelu:
                params_per_layer = 2;
                break;
            case RnnMode::kRnnTanh:
                params_per_layer = 2;
                break;
            case RnnMode::kRnnLstm:
                params_per_layer = 8;
                break;
            case RnnMode::kRnnGru:
                params_per_layer = 6;
                break;
        }

        int num_layers_int = init_helper->GetNumLayers();
        // int input_size = init_helper->GetInputSize();
        // int num_units = init_helper->GetNumUnits();
        int num_gates = params_per_layer/2;
        bool is_bidirectional = init_helper->GetRNNDirectionMode() == RnnDirectionMode::kRnnBidirectional ? true : false;

        dml::Expression total_size = 0;
        for (int i = 0; i < num_layers_int; i++) {
            dml::Expression layer_size = 0;
            if (i == 0) {
                layer_size = num_units * input_size * num_gates;
            } else {
                if (is_bidirectional) {
                    layer_size = num_units * 2 * num_units * num_gates;
                } else {
                    layer_size = num_units * num_units * num_gates;
                }
            } 
            // size of hidden states
            layer_size += num_units * num_units * num_gates;
            if (is_bidirectional) {
                layer_size *= 2;
            }
            total_size += layer_size;
        }



        Microsoft::WRL::ComPtr<IDMLCompiledOperator> compiled_op =
            scope.Compile(DML_EXECUTION_FLAG_NONE, {total_size});

        Initialize(ctx, std::move(tensors), compiled_op.Get());
    }
};

// Convert weight and bias params from a platform-specific layout to the
// canonical form.
template <typename T>
class CudnnRNNParamsToCanonicalKernel: public OpKernel {
 public:
    // using InitHelper = CudnnRNNParamsToCanonicalInitHelper;
    explicit CudnnRNNParamsToCanonicalKernel(OpKernelConstruction* ctx,
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
            OP_REQUIRES_OK(ctx, ctx->GetAttr("num_params_biases", &num_params_biases_));
        }
        else
        {
            num_params_biases_ = 0;
        }
        if (ctx->HasAttr("num_params_weights"))
        {
            OP_REQUIRES_OK(ctx, ctx->GetAttr("num_params_weights", &num_params_weights_));
        }
        else
        {
            num_params_weights_ = 0;
        }
        if (num_proj_ == 0) {
            num_params_weights_ = num_params_;
            num_params_biases_ = num_params_;
        }
                    
        std::string str;
        OP_REQUIRES_OK(ctx, ctx->GetAttr("rnn_mode", &str));
        OP_REQUIRES_OK(ctx, ParseRNNMode(str, &model_types_.rnn_mode));
        OP_REQUIRES_OK(ctx, ctx->GetAttr("direction", &str));
        OP_REQUIRES_OK(ctx, ParseRNNDirectionMode(str, &model_types_.rnn_direction_mode));

        num_dirs_ = 1;
        if (model_types_.rnn_direction_mode == RnnDirectionMode::kRnnBidirectional) {
            num_dirs_ = 2;
        }
        num_params_weights_per_layer_ = num_params_weights_ / num_layers_ / num_dirs_;
        num_params_input_state_ = num_params_weights_per_layer_ / 2;

        OP_REQUIRES(
            ctx, num_params_weights_ % (num_layers_ * num_dirs_) == 0,
            errors::InvalidArgument("Number of params (weights) is not a multiple"
                                    "of num_layers * num_dirs."));

        OP_REQUIRES(
            ctx, num_params_biases_ % (num_layers_ * num_dirs_) == 0,
            errors::InvalidArgument("Number of params (biases) is not a multiple"
                                    "of num_layers * num_dirs."));
        if (num_proj_ == 0) {
            OP_REQUIRES(
                ctx, num_params_weights_per_layer_ % 2 == 0,
                errors::InvalidArgument("Number of params (weights) per layer is not"
                                        "an even number with no projection."));
        } else {
            OP_REQUIRES(
                ctx, num_params_weights_per_layer_ % 2 != 0,
                errors::InvalidArgument("Number of params (weights) per layer is not"
                                        "an odl number with projection."));
        }

        h_num_units_ = (num_proj_ == 0 ? num_units_ : num_proj_);
        c_num_units_ = (num_proj_ == 0 ? 0 : num_units_);
    }
    void Compute(OpKernelContext* ctx)
    {
        // auto num_units = ctx->input(1);
        auto input_size = ctx->input(2);

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

        DmlDevice* device = static_cast<DmlDevice*>(ctx->device());
        auto* device_context = device->GetDeviceContext();

        // int num_params_input_state = init_helper->GetNumParamsInputState();
        // int h_num_units = init_helper->GetHNumUnits();
        // int c_num_units = init_helper->GetCNumUnits();

        uint64_t src_offset = 0;
        for (int i = 0; i < num_params_weights_; i++) {
            // TODO: get size
            const int layer_idx = i / num_params_weights_per_layer_;
            const int index_within_layer = i % num_params_weights_per_layer_;
            int width = 0, height = (num_proj_ == 0 ? h_num_units_ : c_num_units_);
            bool apply_on_input_state = index_within_layer < num_params_input_state_;

            if (model_types_.rnn_direction_mode == RnnDirectionMode::kRnnUnidirectional) {
                if (layer_idx == 0 && apply_on_input_state) {
                    width = input_size_;
                } else {
                    width = h_num_units_;
                }
            } else {
                if (apply_on_input_state) {
                    if (layer_idx <= 1) {
                        // First fwd or bak layer.
                        width = input_size_;
                    } else {
                        // Following layers, cell inputs are concatenated outputs of
                        // its prior layer.
                        width = 2 * h_num_units_;
                    }
                } else {
                    width = h_num_units_;
                }
            }

            // TODO: check size

            int id_in_layer = i % num_params_weights_per_layer_;
            if (num_proj_ != 0 && id_in_layer == num_params_weights_per_layer_ - 1) {
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

            D3D12BufferRegion output_buffer = device_context->GetBufferForTensor(
                status_or_output_tensor.ValueOrDie());

            device_context->CopyBufferToBuffer(
                    output_buffer,
                    input_buffer.Subregion(src_offset, size_in_bytes));

            src_offset += size_in_bytes;
        }

        for (int i = 0; i < num_params_biases_; i++) {
            int64_t size = num_units_;
            const uint64_t data_type_size = sizeof(T);
            int64_t size_in_bytes = size * data_type_size;

            StatusOr<Tensor> status_or_output_tensor =
                ctx->allocate_output(num_params_weights_+i, TensorShape({size}));
            OP_REQUIRES_OK(ctx, status_or_output_tensor.status());

            D3D12BufferRegion input_buffer =
                device_context->GetBufferForTensor(ctx->input(3));

            D3D12BufferRegion output_buffer = device_context->GetBufferForTensor(
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
                OP_REQUIRES_OK(ctx, ctx->GetAttr("num_params_biases", &num_params_biases));
            }
            else
            {
                num_params_biases = 0;
            }
            if (ctx->HasAttr("num_params_weights"))
            {
                OP_REQUIRES_OK(ctx, ctx->GetAttr("num_params_weights", &num_params_weights));
            }
            else
            {
                num_params_weights = 0;
            }
            if (num_proj == 0) {
                num_params_weights = num_params;
                num_params_biases = num_params;
            }
                        
            std::string str;
            OP_REQUIRES_OK(ctx, ctx->GetAttr("rnn_mode", &str));
            OP_REQUIRES_OK(ctx, ParseRNNMode(str, &model_types.rnn_mode));
            OP_REQUIRES_OK(ctx, ctx->GetAttr("direction", &str));
            OP_REQUIRES_OK(ctx, ParseRNNDirectionMode(str, &model_types.rnn_direction_mode));

            num_dirs = 1;
            if (model_types.rnn_direction_mode == RnnDirectionMode::kRnnBidirectional) {
                num_dirs = 2;
            }
            num_params_weights_per_layer = num_params_weights / num_layers / num_dirs;
            num_params_input_state = num_params_weights_per_layer / 2;

            OP_REQUIRES(
                ctx, num_params_weights % (num_layers * num_dirs) == 0,
                errors::InvalidArgument("Number of params (weights) is not a multiple"
                                        "of num_layers * num_dirs."));

            OP_REQUIRES(
                ctx, num_params_biases % (num_layers * num_dirs) == 0,
                errors::InvalidArgument("Number of params (biases) is not a multiple"
                                        "of num_layers * num_dirs."));
            if (num_proj == 0) {
                OP_REQUIRES(
                    ctx, num_params_weights_per_layer % 2 == 0,
                    errors::InvalidArgument("Number of params (weights) per layer is not"
                                            "an even number with no projection."));
            } else {
                OP_REQUIRES(
                    ctx, num_params_weights_per_layer % 2 != 0,
                    errors::InvalidArgument("Number of params (weights) per layer is not"
                                            "an odl number with projection."));
            }
        }
        int num_proj;
        int num_layers;
        int num_dirs;
        int num_params_weights_per_layer;
        int num_params_input_state;
        int num_params;
        int num_params_biases;
        int num_params_weights;
        CudnnModelTypes model_types;
    };

    CudnnRNNCanonicalToParamsInitHelper(OpKernelContext* ctx, std::shared_ptr<const Attributes> attr)
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

        output_size_ = 0;
        int last_index = 3 + attr_->num_params_weights + attr_->num_params_biases;
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
    RnnDirectionMode GetRNNDirectionMode() const {
        return attr_->model_types.rnn_direction_mode;
    }

  private:
    std::shared_ptr<const Attributes> attr_;
    int num_layers_;
    int num_units_;
    int input_size_;
    int64_t output_size_;
};

using InitHelper = CudnnRNNCanonicalToParamsInitHelper;

class CudnnRNNCanonicalToParamsShapeHelper : public ShapeHelper
{
  public:
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
        uint32_t last_weight_index = first_weight_index + init_helper->GetNumParamsWeights();
        uint32_t first_bias_index = 3;
        uint32_t last_bias_index = first_weight_index + init_helper->GetNumParamsBiases();

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

        CHECK(!tensors.inputs.empty());

        auto inputs = GetDmlTensorDescs(tensors.inputs);
        auto scope = dml::Graph(ctx->GetDmlDevice());

        absl::InlinedVector<dml::Expression, 4> input_tensors;
        input_tensors.reserve(inputs.size());

        for (int i = 0; i < inputs.size(); ++i)
        {
            input_tensors.push_back(dml::InputTensor(scope, i, inputs[i]));
        }

        auto result = dml::Join(input_tensors, 0);

        Microsoft::WRL::ComPtr<IDMLCompiledOperator> compiled_op =
            scope.Compile(DML_EXECUTION_FLAG_NONE, {result});

        Initialize(ctx, std::move(tensors), compiled_op.Get());
    }
};

class CudnnRNNForwardInitHelper : public InitializationHelper
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
            if (ctx->HasAttr("time_major"))
            {
                OP_REQUIRES_OK(ctx, ctx->GetAttr("time_major", &time_major));
            }
            else
            {
                time_major = false; // TODO: double check
            }
            OP_REQUIRES_OK(ctx, ctx->GetAttr("dropout", &dropout));
            OP_REQUIRES_OK(ctx, ctx->GetAttr("seed", &seed));
            OP_REQUIRES_OK(ctx, ctx->GetAttr("seed2", &seed2));
            OP_REQUIRES_OK(ctx, ctx->GetAttr("is_training", &is_training));
            std::string str;
            OP_REQUIRES_OK(ctx, ctx->GetAttr("input_mode", &str));
            OP_REQUIRES_OK(ctx, ParseTFRNNInputMode(str, &model_types.rnn_input_mode));
            OP_REQUIRES_OK(ctx, ctx->GetAttr("rnn_mode", &str));
            OP_REQUIRES_OK(ctx, ParseRNNMode(str, &model_types.rnn_mode));
            OP_REQUIRES_OK(ctx, ctx->GetAttr("direction", &str));
            OP_REQUIRES_OK(ctx, ParseRNNDirectionMode(str, &model_types.rnn_direction_mode));

        }
        int num_proj;
        bool time_major;
        float dropout;
        int seed;
        int seed2;
        bool is_training;
        CudnnModelTypes model_types;
    };

    CudnnRNNForwardInitHelper(OpKernelContext* ctx, std::shared_ptr<const Attributes> attr)
        : attr_(std::move(attr))
    {
        // TODO: account for extra input for V3
        const Tensor& input_tensor = ctx->input(0);
        const Tensor& input_h_tensor = ctx->input(1);
        const Tensor& input_c_tensor = ctx->input(2);
        const Tensor& params_tensor = ctx->input(3);

        input_h_shape_ = input_h_tensor.shape();
        input_c_shape_ = input_c_tensor.shape();

        OP_REQUIRES(
            ctx,
            input_tensor.dims()==3,
            errors::InvalidArgument(
                "RNN input must be a 3-D vector."));

        if (attr_->time_major) {
            max_seq_length_ = input_tensor.dim_size(0);
            batch_size_ = input_tensor.dim_size(1);
        } else {
            batch_size_ = input_tensor.dim_size(0);
            max_seq_length_ = input_tensor.dim_size(1); 
        }

        input_size_ = input_tensor.dim_size(2);
        input_shape_ = input_tensor.shape();
        dir_count_ = (attr_->model_types.rnn_direction_mode ==  RnnDirectionMode::kRnnBidirectional)
            ? 2
            : 1;

        OP_REQUIRES(
            ctx,
            input_h_tensor.dims()==3,
            errors::InvalidArgument(
                "RNN input_h must be a 3-D vector."));

        if (attr_->time_major) {
            num_layers_ = input_h_tensor.dim_size(0) / dir_count_;
        } else {
            num_layers_ = input_h_tensor.dim_size(1) / dir_count_;
        }

        num_units_ = input_h_tensor.dim_size(2);

        if (attr_->time_major) {
            hidden_state_shape_ = TensorShape({dir_count_ * num_layers_,
                            batch_size_, num_units_});
        } else {
            hidden_state_shape_ = TensorShape({batch_size_,
                            dir_count_ * num_layers_,
                            num_units_});
        }

        OP_REQUIRES(
            ctx,
            input_h_tensor.shape() == hidden_state_shape_,
            errors::InvalidArgument(
                "Invalid input_h shape: ",
                input_h_tensor.shape().DebugString()));

        if (attr_->model_types.HasInputC()) {
            cell_num_units_ = input_c_tensor.dim_size(2);
            if (attr_->time_major) {
                cell_state_shape_ =
                TensorShape({dir_count_ * num_layers_,
                            batch_size_, cell_num_units_});
            } else {
                cell_state_shape_ =
                TensorShape({batch_size_,
                            dir_count_ * num_layers_,
                            cell_num_units_});
            }
            if (attr_->num_proj == 0) {
                OP_REQUIRES(
                    ctx,
                    input_h_tensor.shape() == input_c_tensor.shape(),
                    errors::InvalidArgument(
                        "input_h and input_c must have the same shape w/o projection: ",
                        input_h_tensor.shape().DebugString(), " ",
                        input_c_tensor.shape().DebugString()));

                OP_REQUIRES(
                    ctx,
                    input_h_tensor.dim_size(2) <= input_c_tensor.dim_size(2) &&
                    attr_->num_proj == input_h_tensor.dim_size(2) &&
                    input_h_tensor.dim_size(0) == input_c_tensor.dim_size(0) &&
                    input_h_tensor.dim_size(1) == input_c_tensor.dim_size(1),
                    errors::InvalidArgument(
                        "Invalid input_h and input_c w/ projection size: ", attr_->num_proj, " ",
                        input_h_tensor.shape().DebugString(), " ",
                        input_c_tensor.shape().DebugString()));
            }
        } else {
            if (attr_->time_major) {
                cell_state_shape_ =
                TensorShape({dir_count_ * num_layers_,
                            batch_size_, num_units_});
            } else {
                cell_state_shape_ =
                TensorShape({batch_size_,
                            dir_count_ * num_layers_,
                            num_units_});
            }
            cell_num_units_ = 0;
        }

        if (attr_->time_major) {
            output_shape_ = 
                TensorShape({max_seq_length_, batch_size_,
                            dir_count_ * num_units_});
        } else {
            output_shape_ = 
                TensorShape({batch_size_, max_seq_length_,
                            dir_count_ * num_units_});
        }
    }

    int GetNumProj() const { return attr_->num_proj; }
    int GetNumLayers() const { return num_layers_; }
    int GetNumUnits() const { return num_units_; }
    int GetInputSize() const { return input_size_; }
    int GetBatchSize() const { return batch_size_; }
    int GetMaxSeqLen() const { return max_seq_length_; }
    RnnMode GetRNNMode() const { return attr_->model_types.rnn_mode; }
    RnnDirectionMode GetRNNDirectionMode() const {
        return attr_->model_types.rnn_direction_mode;
    }
    TensorShape GetOutputShape() const { return output_shape_; }
    TensorShape GetInputHShape() const { return input_h_shape_; }
    TensorShape GetInputCShape() const { return input_c_shape_; }

    // bool time_major;
    // float dropout;
    // int seed;
    // int seed2;
    // bool is_training;
    // CudnnModelTypes model_types;

  private:
    std::shared_ptr<const Attributes> attr_;
    int max_seq_length_;
    int batch_size_;
    int input_size_;
    int dir_count_;
    int num_layers_;
    int num_units_;
    int cell_num_units_ = 0;
    TensorShape input_shape_;
    TensorShape hidden_state_shape_;
    TensorShape cell_state_shape_;
    TensorShape output_shape_;
    TensorShape input_h_shape_;
    TensorShape input_c_shape_;
};

using InitHelper = CudnnRNNForwardInitHelper;

class CudnnRNNForwardShapeHelper : public ShapeHelper
{
  public:
    std::vector<TensorShape> GetOutputShapes(
        OpKernelContext* ctx,
        const InitializationHelper* initialization_helper) const override
    {
        auto init_helper =
            static_cast<const InitHelper*>(initialization_helper);
        std::vector<TensorShape> outputShapes;
        // TODO: account for additional outputs for V2/V3
        outputShapes.reserve(4);

        // output,
        // output_h,
        // output_c,
        // reserve_space

        // output tensor shape
        outputShapes.push_back(init_helper->GetOutputShape());

        // output h tensor shape
        outputShapes.push_back(init_helper->GetInputHShape());

        // output c tensor shape
        outputShapes.push_back(init_helper->GetInputCShape());

        // reserve tensor shape
        // TODO: figure out shape
        outputShapes.push_back(TensorShape({}));

        return outputShapes;
    }
};

//   The requirements to use the cuDNN implementation are:

//   1. `activation` == `tanh`
//   2. `recurrent_activation` == `sigmoid`
//   3. `recurrent_dropout` == 0
//   4. `unroll` is `False`
//   5. `use_bias` is `True`
//   6. Inputs, if use masking, are strictly right-padded.
//   7. Eager execution is enabled in the outermost context.

class CudnnRNNForwardOp : public DmlKernel
{
  public:
    using InitHelper = CudnnRNNForwardInitHelper;

    explicit CudnnRNNForwardOp(
        DmlKernelConstruction* ctx,
        const InitHelper* init_helper)
    {
        DmlKernelTensors tensors;
        auto input_descs = GetDmlTensorDescs(tensors.inputs);
        auto scope = dml::Graph(ctx->GetDmlDevice());
        

        auto inputs = dml::InputTensor(scope, 0, input_descs[0]);
        auto input_h = dml::InputTensor(scope, 1, input_descs[1]);
        auto input_c = dml::InputTensor(scope, 2, input_descs[2]);
        auto params = dml::InputTensor(scope, 3, input_descs[3]);

        auto timesteps = ctx->GetInputTensorShape(0).dim_size(0);
        uint32_t batch_size = static_cast<uint32_t>(init_helper->GetBatchSize());
        uint32_t input_size = static_cast<uint32_t>(init_helper->GetInputSize());
        int max_seq_length = init_helper->GetMaxSeqLen();
        int32_t slice_stride[] = {1, 1, 1, 1};

        dml::TensorDesc::Dimensions x_extent{1, 1, batch_size, input_size};

        std::vector<dml::Expression> c_tensors;
        c_tensors.reserve(max_seq_length);
        std::vector<dml::Expression> h_tensors;
        h_tensors.reserve(max_seq_length);

        for (uint32_t t = 0; t < timesteps; ++t) {
            dml::TensorDesc::Dimensions tensor_offset{0, t, 0, 0};
            dml::TensorDesc::Dimensions prev_offset{0, t - 1, 0, 0};

            auto input_tensor =
                dml::Slice(inputs, tensor_offset, x_extent, slice_stride);

            // auto cs_prev_tensor = t == 0 ? cs_prev : cs_tensors.at(t - 1);
            // auto h_prev_tensor = t == 0 ? h_prev : h_tensors.at(t - 1);

            if (init_helper->GetRNNMode() == RnnMode::kRnnLstm) {
                auto c_tm1 = t == 0 ? input_c : c_tensors.at(t - 1);
                auto h_tm1 = t == 0 ? input_h : h_tensors.at(t - 1);

                // // Concat xh = [x, h].
                // auto xh = dml::Join({inputs, h_tm1}, 3);

                // // TODO: get w and b from params
                // auto w = params;
                // auto b = params;
                // // states1 = xh * w + b
                // auto gates_gemm = dml::Gemm(xh, w);
                // dml::Expression gates = gates_gemm;
                // gates += b;

                // // split z into 4
                // // i = sigmoid(z[0])
                // // Input gate.
                // auto i = dml::Slice(gates, i_offset, cell_extent, slice_stride);
                // i = dml::ActivationSigmoid(i);
                // // f = sigmoid(z[1])
                // // Forget gate (w/ bias).
                // auto f = dml::Slice(gates, f_offset, cell_extent, slice_stride);
                // f = dml::ActivationSigmoid(f);
                // // c = f* init_C + i * tanh(z[2])
                // // Cell input.
                // auto ci = dml::Slice(gates, c_offset, cell_extent, slice_stride);
                // ci = dml::ActivationTanh(ci);
                // // cs = ci .* i + f .* cs_prev
                // auto cs = i * ci + f * c_tm1;
                // // o = sigmoid(z[3])
                // // Output gate.
                // auto o = dml::Slice(gates, o_offset, cell_extent, slice_stride);
                // o = dml::ActivationSigmoid(o);
                // // co = tanh(cs)
                // auto co = dml::ActivationTanh(cs);
                // // h = o * tanh(c)
                // // h = o * co
                // auto h = o * co;
                // i_tensors.push_back(i);
                // cs_tensors.push_back(cs);
                // f_tensors.push_back(f);
                // o_tensors.push_back(o);
                // ci_tensors.push_back(ci);
                // co_tensors.push_back(co);
                // h_tensors.push_back(h);
                // // return h, [h, c]
            } else {
                auto h_tm1 = t == 0 ? input_h : h_tensors.at(t - 1);
                // // z = cell_inputs (dot) params[weights[t]]
                // // Concat x_h_prev = [x, h_prev].
                // auto x_h_prev = dml::Join({x, h_prev}, 3);
                // // z += bias[t]
                // // r_u_bar = x_h_prev * w_ru + b_ru
                // auto r_u_bar_gemm = dml::Gemm(x_h_prev, w_ru);
                // dml::Expression r_u_bar = r_u_bar_gemm;
                // r_u_bar += b_ru;
                // // split z into 3
                // // m_i = init_h * recurrent kernel??
                // // Slice r_u_bar into r, u and apply the sigmoid.
                // auto r = dml::Slice(r_u_bar, ru_r_offsets, cell_extents, slice_strides);
                // r = dml::ActivationSigmoid(r);

                // auto u = dml::Slice(r_u_bar, ru_u_offsets, cell_extents, slice_strides);
                // u = dml::ActivationSigmoid(u);
                // // m_i = m_i + bias[t]
                // // Concat x_h_prevr = [x,h_prev*r]
                // auto h_prevr = h_prev * r;
                // auto x_h_prevr = dml::Join({x, h_prevr}, 3);
                // // r_z, r_r, r_h = split m_i into 3
                // // z = sigmoid(x_z + r_z)
                // // r = sigmoid(x_r + r_r)
                // // hh = tanh (x_h + r * r_h)
                // // c = tanh(x_h_prevr*w_c+b_c), Note b_c is broadcasted before adding.
                // auto c_gemm = dml::Gemm(x_h_prevr, w_c);
                // dml::Expression c = c_gemm;
                // c += b_c;
                // c = dml::ActivationTanh(c);
                // // h = z * input_h + (1-z) * hh
                // // h= u*h_prev + (1-u)*c
                // auto h = u * (h_prev - c) + c;
                // // return h, [h]
            }
        }
    }
};

class CudnnRNNBackwardInitHelper : public InitializationHelper
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
            if (ctx->HasAttr("time_major"))
            {
                OP_REQUIRES_OK(ctx, ctx->GetAttr("time_major", &time_major));
            }
            else
            {
                time_major = false; // TODO: double check
            }
            OP_REQUIRES_OK(ctx, ctx->GetAttr("dropout", &dropout));
            OP_REQUIRES_OK(ctx, ctx->GetAttr("seed", &seed));
            OP_REQUIRES_OK(ctx, ctx->GetAttr("seed2", &seed2));
            std::string str;
            OP_REQUIRES_OK(ctx, ctx->GetAttr("input_mode", &str));
            OP_REQUIRES_OK(ctx, ParseTFRNNInputMode(str, &model_types.rnn_input_mode));
            OP_REQUIRES_OK(ctx, ctx->GetAttr("rnn_mode", &str));
            OP_REQUIRES_OK(ctx, ParseRNNMode(str, &model_types.rnn_mode));
            OP_REQUIRES_OK(ctx, ctx->GetAttr("direction", &str));
            OP_REQUIRES_OK(ctx, ParseRNNDirectionMode(str, &model_types.rnn_direction_mode));

        }
        int num_proj;
        bool time_major;
        float dropout;
        int seed;
        int seed2;
        CudnnModelTypes model_types;
    };

    CudnnRNNBackwardInitHelper(OpKernelContext* ctx, std::shared_ptr<const Attributes> attr)
        : attr_(std::move(attr))
    {
        const Tensor& input_tensor = ctx->input(0);
        const Tensor& input_h_tensor = ctx->input(1);
        const Tensor& input_c_tensor = ctx->input(2);
        const Tensor& params_tensor = ctx->input(3);

        input_h_shape_ = input_h_tensor.shape();
        input_c_shape_ = input_c_tensor.shape();
        params_shape_ = params_tensor.shape();

        OP_REQUIRES(
            ctx,
            input_tensor.dims()==3,
            errors::InvalidArgument(
                "RNN input must be a 3-D vector."));

        if (attr_->time_major) {
            max_seq_length_ = input_tensor.dim_size(0);
            batch_size_ = input_tensor.dim_size(1);
        } else {
            batch_size_ = input_tensor.dim_size(0);
            max_seq_length_ = input_tensor.dim_size(1); 
        }

        input_size_ = input_tensor.dim_size(2);
        input_shape_ = input_tensor.shape();
        dir_count_ = (attr_->model_types.rnn_direction_mode ==  RnnDirectionMode::kRnnBidirectional)
            ? 2
            : 1;

        OP_REQUIRES(
            ctx,
            input_h_tensor.dims()==3,
            errors::InvalidArgument(
                "RNN input_h must be a 3-D vector."));

        if (attr_->time_major) {
            num_layers_ = input_h_tensor.dim_size(0) / dir_count_;
        } else {
            num_layers_ = input_h_tensor.dim_size(1) / dir_count_;
        }

        num_units_ = input_h_tensor.dim_size(2);

        if (attr_->time_major) {
            hidden_state_shape_ = TensorShape({dir_count_ * num_layers_,
                            batch_size_, num_units_});
        } else {
            hidden_state_shape_ = TensorShape({batch_size_,
                            dir_count_ * num_layers_,
                            num_units_});
        }

        OP_REQUIRES(
            ctx,
            input_h_tensor.shape() == hidden_state_shape_,
            errors::InvalidArgument(
                "Invalid input_h shape: ",
                input_h_tensor.shape().DebugString()));

        if (attr_->model_types.HasInputC()) {
            cell_num_units_ = input_c_tensor.dim_size(2);
            if (attr_->time_major) {
                cell_state_shape_ =
                TensorShape({dir_count_ * num_layers_,
                            batch_size_, cell_num_units_});
            } else {
                cell_state_shape_ =
                TensorShape({batch_size_,
                            dir_count_ * num_layers_,
                            cell_num_units_});
            }
            if (attr_->num_proj == 0) {
                OP_REQUIRES(
                    ctx,
                    input_h_tensor.shape() == input_c_tensor.shape(),
                    errors::InvalidArgument(
                        "input_h and input_c must have the same shape w/o projection: ",
                        input_h_tensor.shape().DebugString(), " ",
                        input_c_tensor.shape().DebugString()));

                OP_REQUIRES(
                    ctx,
                    input_h_tensor.dim_size(2) <= input_c_tensor.dim_size(2) &&
                    attr_->num_proj == input_h_tensor.dim_size(2) &&
                    input_h_tensor.dim_size(0) == input_c_tensor.dim_size(0) &&
                    input_h_tensor.dim_size(1) == input_c_tensor.dim_size(1),
                    errors::InvalidArgument(
                        "Invalid input_h and input_c w/ projection size: ", attr_->num_proj, " ",
                        input_h_tensor.shape().DebugString(), " ",
                        input_c_tensor.shape().DebugString()));
            }
        } else {
            if (attr_->time_major) {
                cell_state_shape_ =
                TensorShape({dir_count_ * num_layers_,
                            batch_size_, num_units_});
            } else {
                cell_state_shape_ =
                TensorShape({batch_size_,
                            dir_count_ * num_layers_,
                            num_units_});
            }
            cell_num_units_ = 0;
        }

        if (attr_->time_major) {
            output_shape_ = 
                TensorShape({max_seq_length_, batch_size_,
                            dir_count_ * num_units_});
        } else {
            output_shape_ = 
                TensorShape({batch_size_, max_seq_length_,
                            dir_count_ * num_units_});
        }

        //////////

        const Tensor& output_tensor = ctx->input(4);
        const Tensor& output_backprop_tensor = ctx->input(7);
        const Tensor& output_h_tensor = ctx->input(5);
        const Tensor& output_h_backprop_tensor = ctx->input(8);

        // if (attr_->model_types.HasInputC()) {
        const Tensor& output_c_tensor = ctx->input(6);
        const Tensor& output_c_backprop_tensor = ctx->input(9);
        // }

        const Tensor& reserve_space_tensor = ctx->input(10);

        OP_REQUIRES(
            ctx,
            output_shape_ == output_tensor.shape(),
            errors::InvalidArgument(
                "Invalid output shape: ",
                output_tensor.shape().DebugString(), " ",
                output_shape_.DebugString()));

        OP_REQUIRES(
            ctx,
            hidden_state_shape_ == output_h_tensor.shape(),
            errors::InvalidArgument(
                "Invalid output_h shape: ",
                output_h_tensor.shape().DebugString(), " ",
                hidden_state_shape_.DebugString()));

        OP_REQUIRES(
            ctx,
            output_shape_ == output_backprop_tensor.shape(),
            errors::InvalidArgument(
                "Invalid output_backprop shape: ",
                output_backprop_tensor.shape().DebugString(), " ",
                output_shape_.DebugString()));

        OP_REQUIRES(
            ctx,
            hidden_state_shape_ == output_h_backprop_tensor.shape(),
            errors::InvalidArgument(
                "Invalid output_h_backprop shape: ",
                output_h_backprop_tensor.shape().DebugString(), " ",
                hidden_state_shape_.DebugString()));

        if (attr_->model_types.HasInputC()) {
            OP_REQUIRES(
                ctx,
                cell_state_shape_ == output_c_tensor.shape(),
                errors::InvalidArgument(
                    "Invalid output_c shape: ",
                    output_c_tensor.shape().DebugString(), " ",
                    cell_state_shape_.DebugString()));

            OP_REQUIRES(
                ctx,
                cell_state_shape_ == output_c_backprop_tensor.shape(),
                errors::InvalidArgument(
                    "Invalid output_c_backprop shape: ",
                    output_c_backprop_tensor.shape().DebugString(), " ",
                    cell_state_shape_.DebugString()));
        }
    }

    int GetNumProj() const { return attr_->num_proj; }
    int GetNumLayers() const { return num_layers_; }
    int GetNumUnits() const { return num_units_; }
    int GetInputSize() const { return input_size_; }
    int GetBatchSize() const { return batch_size_; }
    int GetMaxSeqLen() const { return max_seq_length_; }
    RnnMode GetRNNMode() const { return attr_->model_types.rnn_mode; }
    RnnDirectionMode GetRNNDirectionMode() const {
        return attr_->model_types.rnn_direction_mode;
    }
    TensorShape GetOutputShape() const { return output_shape_; }
    TensorShape GetInputShape() const { return input_shape_; }
    TensorShape GetHiddenStateShape() const { return hidden_state_shape_; }
    TensorShape GetCellStateShape() const { return cell_state_shape_; }
    TensorShape GetInputHShape() const { return input_h_shape_; }
    TensorShape GetInputCShape() const { return input_c_shape_; }
    TensorShape GetParamsShape() const { return params_shape_; }


  private:
    std::shared_ptr<const Attributes> attr_;
    int max_seq_length_;
    int batch_size_;
    int input_size_;
    int dir_count_;
    int num_layers_;
    int num_units_;
    int cell_num_units_ = 0;
    TensorShape input_shape_;
    TensorShape hidden_state_shape_;
    TensorShape cell_state_shape_;
    TensorShape output_shape_;
    TensorShape input_h_shape_;
    TensorShape input_c_shape_;
    TensorShape params_shape_;
};

using InitHelper = CudnnRNNBackwardInitHelper;

class CudnnRNNBackwardShapeHelper : public ShapeHelper
{
  public:
    std::vector<TensorShape> GetOutputShapes(
        OpKernelContext* ctx,
        const InitializationHelper* initialization_helper) const override
    {
        auto init_helper =
            static_cast<const InitHelper*>(initialization_helper);
        std::vector<TensorShape> outputShapes;
        outputShapes.reserve(4);

        const TensorShape& input_shape = init_helper->GetInputShape();
        const TensorShape& hidden_state_shape = init_helper->GetHiddenStateShape();
        const TensorShape& cell_state_shape = init_helper->GetCellStateShape();
        const TensorShape& params_shape = init_helper->GetParamsShape();

        // input_backprop tensor shape
        outputShapes.push_back(input_shape);

        // input_h_backprop tensor shape
        outputShapes.push_back(hidden_state_shape);

        // input_c_backprop tensor shape
        // TODO: or dummy
        outputShapes.push_back(cell_state_shape);

        // params_backprop tensor shape
        outputShapes.push_back(params_shape);

        return outputShapes;
    }
};

class CudnnRNNBackwardOp : public DmlKernel
{
  public:
    using InitHelper = CudnnRNNBackwardInitHelper;

    explicit CudnnRNNBackwardOp(
        DmlKernelConstruction* ctx,
        const InitHelper* init_helper)
    {
        DmlKernelTensors tensors;
        auto input_descs = GetDmlTensorDescs(tensors.inputs);
        auto scope = dml::Graph(ctx->GetDmlDevice());
        

        auto inputs = dml::InputTensor(scope, 0, input_descs[0]);
        auto input_h = dml::InputTensor(scope, 1, input_descs[1]);
        auto input_c = dml::InputTensor(scope, 2, input_descs[2]);
        auto params = dml::InputTensor(scope, 3, input_descs[3]);

        auto timesteps = ctx->GetInputTensorShape(0).dim_size(0);
        uint32_t batch_size = static_cast<uint32_t>(init_helper->GetBatchSize());
        uint32_t input_size = static_cast<uint32_t>(init_helper->GetInputSize());
        int max_seq_length = init_helper->GetMaxSeqLen();
        int32_t slice_stride[] = {1, 1, 1, 1};

        dml::TensorDesc::Dimensions x_extent{1, 1, batch_size, input_size};

        std::vector<dml::Expression> c_tensors;
        c_tensors.reserve(max_seq_length);
        std::vector<dml::Expression> h_tensors;
        h_tensors.reserve(max_seq_length);

        for (uint32_t t = 0; t < timesteps; ++t) {
            dml::TensorDesc::Dimensions tensor_offset{0, t, 0, 0};
            dml::TensorDesc::Dimensions prev_offset{0, t - 1, 0, 0};

            auto input_tensor =
                dml::Slice(inputs, tensor_offset, x_extent, slice_stride);

            // auto cs_prev_tensor = t == 0 ? cs_prev : cs_tensors.at(t - 1);
            // auto h_prev_tensor = t == 0 ? h_prev : h_tensors.at(t - 1);

            if (init_helper->GetRNNMode() == RnnMode::kRnnLstm) {
                auto c_tm1 = t == 0 ? input_c : c_tensors.at(t - 1);
                auto h_tm1 = t == 0 ? input_h : h_tensors.at(t - 1);

                // TODO: LSTM gradients
            } else {
                auto h_tm1 = t == 0 ? input_h : h_tensors.at(t - 1);
                // TODO: GRU gradients
            }
        }
    }
};

void RegisterCudnnRNNParamsSize()
{
    using Op = ops::CudnnRNNParamsSize;
    using K = KernelDefinition<
        Op,
        DmlKernelWrapper<CudnnRNNParamsSizeOp, ScalarOutputShapeHelper>>::
        template WithHostMemoryArguments<Op::Argument::num_layers>::
        template WithHostMemoryArguments<Op::Argument::num_units>::
        template WithHostMemoryArguments<Op::Argument::input_size>::
        template WithHostMemoryArguments<Op::Argument::params_size>::
        template WithTypeConstraint<
                        Op::Attribute::S,
                        TF_INT32>;

    RegisterWithTypes<
        K,
        Op::Attribute::T,
        TF_FLOAT,
        TF_HALF,
        TF_DOUBLE>();
}

template <typename T>
void RegisterCudnnRNNParamsToCanonical()
{
    using Op = ops::CudnnRNNParamsToCanonical;
 
    KernelDefinition<
        Op,
        CudnnRNNParamsToCanonicalKernel<T>>::
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
 
    KernelDefinition<
        Op,
        CudnnRNNParamsToCanonicalKernel<T>>::
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
        DmlKernelWrapper<CudnnRNNCanonicalToParamsKernel, CudnnRNNCanonicalToParamsShapeHelper>>::
        template WithHostMemoryArguments<Op::Argument::num_layers>::
        template WithHostMemoryArguments<Op::Argument::num_units>::
        template WithHostMemoryArguments<Op::Argument::input_size>;

    RegisterWithTypes<
        K,
        Op::Attribute::T,
        TF_FLOAT,
        TF_HALF,
        TF_DOUBLE>();
}

void RegisterCudnnRNNCanonicalToParamsV2()
{
    using Op = ops::CudnnRNNCanonicalToParamsV2;
    using K = KernelDefinition<
        Op,
        DmlKernelWrapper<CudnnRNNCanonicalToParamsKernel, CudnnRNNCanonicalToParamsShapeHelper>>::
        template WithHostMemoryArguments<Op::Argument::num_layers>::
        template WithHostMemoryArguments<Op::Argument::num_units>::
        template WithHostMemoryArguments<Op::Argument::input_size>;

    RegisterWithTypes<
        K,
        Op::Attribute::T,
        TF_FLOAT,
        TF_HALF,
        TF_DOUBLE>();
}

void RegisterCudnnRNN()
{
    using Op = ops::CudnnRNN;
    using K = KernelDefinition<
        Op,
        DmlKernelWrapper<CudnnRNNForwardOp, CudnnRNNForwardShapeHelper>>;

    RegisterWithTypes<
        K,
        Op::Attribute::T,
        TF_FLOAT,
        TF_HALF,
        TF_DOUBLE>();
}

void RegisterCudnnRNNV2()
{
    using Op = ops::CudnnRNNV2;
    using K = KernelDefinition<
        Op,
        DmlKernelWrapper<CudnnRNNForwardOp, CudnnRNNForwardShapeHelper>>::
        template WithHostMemoryArguments<Op::Argument::host_reserved>;

    RegisterWithTypes<
        K,
        Op::Attribute::T,
        TF_FLOAT,
        TF_HALF,
        TF_DOUBLE>();
}

void RegisterCudnnRNNV3()
{
    using Op = ops::CudnnRNNV3;
    using K = KernelDefinition<
        Op,
        DmlKernelWrapper<CudnnRNNForwardOp, CudnnRNNForwardShapeHelper>>::
        template WithHostMemoryArguments<Op::Argument::sequence_lengths>::
        template WithHostMemoryArguments<Op::Argument::host_reserved>;

    RegisterWithTypes<
        K,
        Op::Attribute::T,
        TF_FLOAT,
        TF_HALF,
        TF_DOUBLE>();
}

void RegisterCudnnRNNBackprop()
{
    using Op = ops::CudnnRNNBackprop;
    using K = KernelDefinition<
        Op,
        DmlKernelWrapper<CudnnRNNBackwardOp, CudnnRNNBackwardShapeHelper>>;

    RegisterWithTypes<
        K,
        Op::Attribute::T,
        TF_FLOAT,
        TF_HALF,
        TF_DOUBLE>();
}

void RegisterCudnnRNNBackpropV2()
{
    using Op = ops::CudnnRNNBackpropV2;
    using K = KernelDefinition<
        Op,
        DmlKernelWrapper<CudnnRNNBackwardOp, CudnnRNNBackwardShapeHelper>>::
        template WithHostMemoryArguments<Op::Argument::host_reserved>;

    RegisterWithTypes<
        K,
        Op::Attribute::T,
        TF_FLOAT,
        TF_HALF,
        TF_DOUBLE>();
}

void RegisterCudnnRNNBackpropV3()
{
    using Op = ops::CudnnRNNBackpropV3;
    using K = KernelDefinition<
        Op,
        DmlKernelWrapper<CudnnRNNBackwardOp, CudnnRNNBackwardShapeHelper>>::
        template WithHostMemoryArguments<Op::Argument::sequence_lengths>::
        template WithHostMemoryArguments<Op::Argument::host_reserved>;

    RegisterWithTypes<
        K,
        Op::Attribute::T,
        TF_FLOAT,
        TF_HALF,
        TF_DOUBLE>();
}

void RegisterKernels_CudnnRNN()
{
    RegisterCudnnRNNParamsSize();
    RegisterCudnnRNNParamsToCanonical<float, Eigen::half, double>();
    RegisterCudnnRNNParamsToCanonicalV2<float, Eigen::half, double>();
    RegisterCudnnRNNCanonicalToParams();
    RegisterCudnnRNNCanonicalToParamsV2();
    RegisterCudnnRNN();
    RegisterCudnnRNNV2();
    RegisterCudnnRNNV3();
    RegisterCudnnRNNBackprop();
    RegisterCudnnRNNBackpropV2();
    RegisterCudnnRNNBackpropV3();
}

} // namespace tfdml