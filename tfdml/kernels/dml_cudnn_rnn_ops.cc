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
#include "tfdml/kernels/dml_lstm_helpers.h"

namespace tfdml
{
// using TensorDimensions = SmallVector<uint32_t, 4>;

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

inline dml::TensorDimensions DimensionFromShape(
    const TensorShape& shape)
{
    if (shape.dims() == 2) {
        return dml::TensorDimensions{
            1,
            1,
            static_cast<uint32_t>(shape.dim_size(0)),
            static_cast<uint32_t>(shape.dim_size(1))};
    } else {
        return dml::TensorDimensions{
            1,
            static_cast<uint32_t>(shape.dim_size(0)),
            static_cast<uint32_t>(shape.dim_size(1)),
            static_cast<uint32_t>(shape.dim_size(2))};
    }
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
                                        "an odd number with projection."));
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

// using InitHelper = CudnnRNNCanonicalToParamsInitHelper;

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
        uint32_t first_bias_index = last_weight_index;
        uint32_t last_bias_index = first_bias_index + init_helper->GetNumParamsBiases();

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
                time_major = true; // TODO: double check
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
        const Tensor& input_tensor = ctx->input(0);
        const Tensor& input_h_tensor = ctx->input(1);
        const Tensor& input_c_tensor = ctx->input(2);
        const Tensor& params_tensor = ctx->input(3);

        Tensor sequence_lengths_tensor;

        // Extra input for V3
        if (ctx->num_inputs() == 5) {
            sequence_lengths_tensor = ctx->input(4);
        }

        // Extra output for V2/V3
        if (ctx->num_outputs() == 5) {
            host_reserved_output_ = true;
        }

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
    int GetDirCount() const { return dir_count_; }
    int GetInputSize() const { return input_size_; }
    int GetBatchSize() const { return batch_size_; }
    int GetMaxSeqLen() const { return max_seq_length_; }
    int GetCellSize() const { return dir_count_ * num_units_; }
    bool HasHostReservedOutput() const { return host_reserved_output_; }
    bool IsTraining() const { return attr_->is_training; }
    RnnMode GetRNNMode() const { return attr_->model_types.rnn_mode; }
    RnnDirectionMode GetRNNDirectionMode() const {
        return attr_->model_types.rnn_direction_mode;
    }
    TensorShape GetOutputShape() const { return output_shape_; }
    TensorShape GetInputHShape() const { return input_h_shape_; }
    TensorShape GetInputCShape() const { return input_c_shape_; }

  private:
    std::shared_ptr<const Attributes> attr_;
    bool host_reserved_output_ = false;
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

class CudnnRNNForwardShapeHelper : public ShapeHelper
{
  public:
  using InitHelper = CudnnRNNForwardInitHelper;
    std::vector<TensorShape> GetOutputShapes(
        OpKernelContext* ctx,
        const InitializationHelper* initialization_helper) const override
    {
        auto init_helper =
            static_cast<const InitHelper*>(initialization_helper);
        std::vector<TensorShape> outputShapes;
        outputShapes.reserve(5);

        // output tensor shape
        outputShapes.push_back(init_helper->GetOutputShape());

        // output h tensor shape
        outputShapes.push_back(init_helper->GetInputHShape());

        // output c tensor shape
        outputShapes.push_back(init_helper->GetInputCShape());

        // reserve tensor shape
        outputShapes.push_back(TensorShape({}));

        if (init_helper->HasHostReservedOutput()) {
            outputShapes.push_back(TensorShape({2}));
        }

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
        CHECK(ctx->GetInputCount() == 4 || ctx->GetInputCount() == 5);
        CHECK(ctx->GetOutputCount() == 4 || ctx->GetOutputCount() == 5);

        DmlKernelParams tparams;
        if (ctx->GetInputCount() == 4) {
            tparams.kernel_input_indices = {0, 1, 2, 3};
        } else {
            tparams.kernel_input_indices = {0, 1, 2, 3, 4};
        }
        tparams.kernel_output_indices = {0, 1, 2};
        auto tensors = GetTensorInfos(ctx, tparams);

        auto input_descs = GetDmlTensorDescs(tensors.inputs);
        DML_TENSOR_DATA_TYPE dtype =
            GetDmlDataTypeFromTfDataType(ctx->GetInputDataType(1));
        auto scope = dml::Graph(ctx->GetDmlDevice());

        auto inputs = dml::InputTensor(scope, 0, input_descs[0]);
        auto input_h = dml::InputTensor(scope, 1, input_descs[1]);
        auto input_c = dml::InputTensor(scope, 2, input_descs[2]);
        auto params = dml::InputTensor(scope, 3, input_descs[3]);

        dml::Expression sequence_lengths;
        if (ctx->GetInputCount() == 5) {
            sequence_lengths = dml::InputTensor(scope, 4, input_descs[4]);
        }

        auto timesteps = ctx->GetInputTensorShape(0).dim_size(0);
        uint32_t batch_size = static_cast<uint32_t>(init_helper->GetBatchSize());
        uint32_t input_size = static_cast<uint32_t>(init_helper->GetInputSize());
        int max_seq_length = init_helper->GetMaxSeqLen();
        int32_t slice_stride[] = {1, 1, 1, 1};

        dml::TensorDesc::Dimensions x_extent{1, 1, batch_size, input_size};

        int num_dirs = init_helper->GetDirCount();
        int num_layers = init_helper->GetNumLayers();
        int num_units = init_helper->GetNumUnits();

        // TensorShape w_shape{num_layers * num_dirs, 3 * num_units, input_size};
        // TensorShape r_shape{num_layers * num_dirs, 3 * num_units, num_units};
        // TensorShape b_shape{num_layers * num_dirs, 6 * num_units};

        TensorShape w_shape{num_dirs, 3 * num_units, input_size};
        TensorShape r_shape{num_dirs, 3 * num_units, num_units};
        TensorShape b_shape{num_dirs, 6 * num_units};

        dml::TensorDimensions w_dims = DimensionFromShape(w_shape);
        dml::TensorDimensions r_dims = DimensionFromShape(r_shape);
        dml::TensorDimensions b_dims = DimensionFromShape(b_shape);

        dml::TensorDimensions dummy_dims = DimensionFromShape(TensorShape{1,1,1,1});

        TensorShape output_c_shape{num_layers * num_dirs, batch_size, num_units};
        dml::TensorDimensions output_c_dims = DimensionFromShape(output_c_shape);

        uint32_t w_end = w_shape.num_elements();
        uint32_t r_end = w_end + r_shape.num_elements();
        uint32_t b_end = r_end + b_shape.num_elements();

        std::vector<dml::Expression> c_tensors;
        c_tensors.reserve(max_seq_length);
        std::vector<dml::Expression> h_tensors;
        h_tensors.reserve(max_seq_length);

        dml::TensorDesc::Dimensions w_extent =
                    DimensionFromExtent({1, w_end});
        auto w = dml::Slice(params, dml::TensorDesc::Dimensions{0, 0, 0, 0}, w_extent, slice_stride);
        dml::TensorDesc::Dimensions r_extent =
                    DimensionFromExtent({1, r_shape.num_elements()});
        auto r = dml::Slice(params, dml::TensorDesc::Dimensions{0, 0, 0, w_end}, r_extent, slice_stride);
        dml::TensorDesc::Dimensions b_extent =
                    DimensionFromExtent({1, b_shape.num_elements()});
        auto b = dml::Slice(params, dml::TensorDesc::Dimensions{0, 0, 0, r_end}, b_extent, slice_stride);

        w = dml::Reinterpret(w, w_dims, {});
        r = dml::Reinterpret(r, r_dims, {});
        b = dml::Reinterpret(b, b_dims, {});

        for (uint32_t t = 0; t < timesteps; ++t) {
            dml::TensorDesc::Dimensions tensor_offset{0, t, 0, 0};
            dml::TensorDesc::Dimensions prev_offset{0, t - 1, 0, 0};

            auto input_tensor =
                dml::Slice(inputs, tensor_offset, x_extent, slice_stride);

            if (init_helper->GetRNNMode() == RnnMode::kRnnLstm) {
                auto c_tm1 = t == 0 ? input_c : c_tensors.at(t - 1);
                auto h_tm1 = t == 0 ? input_h : h_tensors.at(t - 1);

                // Concat xh = [x, h].
                auto xh = dml::Join({input_tensor, h_tm1}, 3);

                // states1 = xh * w + b
                auto gates_gemm = dml::Gemm(xh, w);
                dml::Expression gates = gates_gemm;
                gates += dml::Gemm(gates, r);
                gates += b;

                auto cell_size = init_helper->GetCellSize();
                functor::LSTMBlockCell cell(batch_size, input_size, cell_size);

                dml::TensorDesc::Dimensions i_offset =
                    DimensionFromOffset(cell.gates_i_offsets());
                dml::TensorDesc::Dimensions c_offset =
                    DimensionFromOffset(cell.gates_c_offsets(ICFO));
                dml::TensorDesc::Dimensions f_offset =
                    DimensionFromOffset(cell.gates_f_offsets(ICFO));
                dml::TensorDesc::Dimensions o_offset =
                    DimensionFromOffset(cell.gates_o_offsets());
                dml::TensorDesc::Dimensions cell_extent =
                    DimensionFromExtent(cell.cell_extents());
                

                // i = sigmoid(z[0])
                // Input gate.
                auto i = dml::Slice(gates, i_offset, cell_extent, slice_stride);
                i = dml::ActivationSigmoid(i);
                // f = sigmoid(z[1])
                // Forget gate (w/ bias).
                auto f = dml::Slice(gates, f_offset, cell_extent, slice_stride);
                f = dml::ActivationSigmoid(f);
                // c = f* init_C + i * tanh(z[2])
                // Cell input.
                auto ci = dml::Slice(gates, c_offset, cell_extent, slice_stride);
                ci = dml::ActivationTanh(ci);
                // cs = ci .* i + f .* cs_prev
                auto cs = i * ci + f * c_tm1;
                // o = sigmoid(z[3])
                // Output gate.
                auto o = dml::Slice(gates, o_offset, cell_extent, slice_stride);
                o = dml::ActivationSigmoid(o);
                // co = tanh(cs)
                auto co = dml::ActivationTanh(cs);
                // h = o * co
                auto h = o * co;
                c_tensors.push_back(cs);
                h_tensors.push_back(h);
                // return h, [h, c]
            } else {
                auto h_tm1 = t == 0 ? input_h : h_tensors.at(t - 1);

                uint32_t axis_size = 3 * num_units;
                auto split_b = dml::Split(b, 3, {axis_size, axis_size});
                auto& b_i = split_b[0];
                auto& b_r = split_b[1];
                auto b_i_tile = dml::Tile(b_i, {1, 1, batch_size, 1});
                auto b_r_tile = dml::Tile(b_r, {1, 1, batch_size, 1});

                auto matrix_x_gemm = dml::Gemm(input_tensor, w, dml::NullOpt, DML_MATRIX_TRANSFORM_NONE, DML_MATRIX_TRANSFORM_TRANSPOSE);
                dml::Expression matrix_x = matrix_x_gemm;
                matrix_x += b_i_tile;

                auto cell_size = static_cast<uint32_t>(init_helper->GetCellSize());
                dml::TensorDesc::Dimensions x_z_offsets = {0, 0, 0, 0};
                dml::TensorDesc::Dimensions x_r_offsets = {0, 0, 0, cell_size};
                dml::TensorDesc::Dimensions cell_extents = {1, 1, batch_size, cell_size};
                int32_t slice_strides[] = {1, 1, 1, 1};

                // split z into 3
                uint32_t matrix_axis_size = num_units;
                auto split_matrix_x = dml::Split(matrix_x, 3, {matrix_axis_size, matrix_axis_size, matrix_axis_size});
                auto x_z = split_matrix_x[0];
                auto x_r = split_matrix_x[1];
                auto x_h = split_matrix_x[2];

                auto matrix_inner_gemm = dml::Gemm(h_tm1, r, dml::NullOpt, DML_MATRIX_TRANSFORM_NONE, DML_MATRIX_TRANSFORM_TRANSPOSE);
                dml::Expression matrix_inner = matrix_inner_gemm;
                matrix_inner += b_r_tile;

                auto split_matrix_inner = dml::Split(matrix_inner, 3, {matrix_axis_size, matrix_axis_size, matrix_axis_size});
                auto recurrent_z = split_matrix_inner[0];
                auto recurrent_r = split_matrix_inner[1];
                auto recurrent_h = split_matrix_inner[2];

                auto z = dml::ActivationSigmoid(x_z + recurrent_z);
                auto r1 = dml::ActivationSigmoid(x_r + recurrent_r);
                auto hh = dml::ActivationTanh(x_h + r1 * recurrent_h);

                auto h = z * h_tm1 + (1 - z) * hh;
                // return h, [h]
                h_tensors.push_back(h);
            }
        }
        std::vector<dml::Expression> outputs = {h_tensors.back(), h_tensors.back()};
        if (init_helper->GetRNNMode() == RnnMode::kRnnLstm) {
            outputs.push_back(c_tensors.back());
        }
        auto output_c_dummy = dml::ScalarTensor(scope, 1, output_c_dims);

        outputs.push_back(output_c_dummy);

        Microsoft::WRL::ComPtr<IDMLCompiledOperator> compiled_op =
            scope.Compile(DML_EXECUTION_FLAG_NONE, outputs);

        Initialize(ctx, std::move(tensors), compiled_op.Get());
    }

    StatusOr<DmlGpuEvent> Compute(DmlKernelContext* ctx) const override {
        for (int i = 0; i < 3; ++i) {
            ctx->GetDmlDeviceContext()->ZeroBuffer(
                ctx->GetDmlDeviceContext()->GetBufferForTensor(ctx->GetOutputTensor(i)));
        }

        ctx->GetDmlDeviceContext()->FillBufferWithPattern(
            ctx->GetDmlDeviceContext()->GetBufferForTensor(ctx->GetOutputTensor(4)),
            {1, 0} //TODO: dnn:kDefaultAlgorithm has value -1 but FillBufferWithPattern takes uint8
        );
        return DmlKernel::Compute(ctx);
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
                time_major = true; // TODO: double check
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

        // TODO: account for extra inputs for V2/V3

        const Tensor& output_tensor = ctx->input(4);
        const Tensor& output_h_tensor = ctx->input(5);
        
        
        Tensor output_backprop_tensor = nullptr;
        Tensor output_h_backprop_tensor = nullptr;
        Tensor output_c_tensor = nullptr;
        Tensor output_c_backprop_tensor = nullptr;
        Tensor reserve_space_tensor = nullptr;

        if (attr_->model_types.HasInputC()) {
            output_c_tensor = ctx->input(6);
            output_backprop_tensor = ctx->input(7);
            output_h_backprop_tensor = ctx->input(8);
            output_c_backprop_tensor = ctx->input(9);
            reserve_space_tensor = ctx->input(10);
        } else {
            output_backprop_tensor = ctx->input(6);
            output_h_backprop_tensor = ctx->input(7);
            reserve_space_tensor = ctx->input(8);
        }


        ////////

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

        ////////

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
    int GetDirCount() const { return dir_count_; }
    int GetInputSize() const { return input_size_; }
    int GetBatchSize() const { return batch_size_; }
    int GetMaxSeqLen() const { return max_seq_length_; }
    int GetCellSize() const { return dir_count_ * num_units_; }
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
        
        // input: [seq_length, batch_size, input_size]
        auto inputs = dml::InputTensor(scope, 0, input_descs[0]);
        // input_h: [num_layer * dir, batch_size, num_units]
        auto input_h = dml::InputTensor(scope, 1, input_descs[1]);
        // input_c (only LSTM): [num_layer * dir, batch_size, num_units]
        auto input_c = dml::InputTensor(scope, 2, input_descs[2]);
        // opaque- same as forward??
        auto params = dml::InputTensor(scope, 3, input_descs[3]);
        // output: [seq_length, batch_size, dir * num_units]
        auto output = dml::InputTensor(scope, 4, input_descs[4]);
        // output_h: [same as input_h]
        auto output_h = dml::InputTensor(scope, 5, input_descs[5]);
        // output_c: [same as input_c]
        auto output_c = dml::InputTensor(scope, 6, input_descs[6]);
        // output_backprop: [same as output in forward pass]
        auto output_backprop = dml::InputTensor(scope, 7, input_descs[7]);
        // output_h_backprop: [same as output_h in forward pass]
        auto output_h_backprop = dml::InputTensor(scope, 8, input_descs[8]);
        // output_c_backprop: [same as output_c in forward pass]
        auto output_c_backprop = dml::InputTensor(scope, 9, input_descs[9]);
        // reserve_space: same reserve_space produced in forward
        // auto reserve_space = dml::InputTensor(scope, 10, input_descs[10]);
        // host_reserved
        // auto host_reserved = dml::InputTensor(scope, 11, input_descs[11]);

        auto timesteps = ctx->GetInputTensorShape(0).dim_size(0);
        uint32_t batch_size = static_cast<uint32_t>(init_helper->GetBatchSize());
        uint32_t input_size = static_cast<uint32_t>(init_helper->GetInputSize());
        int max_seq_length = init_helper->GetMaxSeqLen();
        int32_t slice_stride[] = {1, 1, 1, 1};

        dml::TensorDesc::Dimensions x_extent{1, 1, batch_size, input_size};

        int num_dirs = init_helper->GetDirCount();
        int num_layers = init_helper->GetNumLayers(); 
        int num_units = init_helper->GetNumUnits();
        uint32_t u_num_units = static_cast<uint32_t>(init_helper->GetNumUnits());

        // TensorShape w_shape{num_layers * num_dirs, 3 * num_units, input_size};
        // TensorShape r_shape{num_layers * num_dirs, 3 * num_units, num_units};
        // TensorShape b_shape{num_layers * num_dirs, 6 * num_units};

        TensorShape w_shape{num_dirs, 3 * num_units, input_size};
        TensorShape r_shape{num_dirs, 3 * num_units, num_units};
        TensorShape b_shape{num_dirs, 6 * num_units};

        dml::TensorDimensions w_dims = DimensionFromShape(w_shape);
        dml::TensorDimensions r_dims = DimensionFromShape(r_shape);
        dml::TensorDimensions b_dims = DimensionFromShape(b_shape);

        uint32_t w_end = w_shape.num_elements();
        uint32_t r_end = w_end + r_shape.num_elements();
        uint32_t b_end = r_end + b_shape.num_elements();

        std::vector<dml::Expression> c_tensors;
        c_tensors.reserve(max_seq_length);
        std::vector<dml::Expression> h_tensors;
        h_tensors.reserve(max_seq_length);

        dml::TensorDesc::Dimensions w_extent =
                    DimensionFromExtent({0, w_end});
        auto w = dml::Slice(params, dml::TensorDesc::Dimensions{0, 0, 0, 0}, w_extent, slice_stride);
        dml::TensorDesc::Dimensions r_extent =
                    DimensionFromExtent({0, r_shape.num_elements()});
        auto r = dml::Slice(params, dml::TensorDesc::Dimensions{0, 0, w_end, 0}, r_extent, slice_stride);
        dml::TensorDesc::Dimensions b_extent =
                    DimensionFromExtent({0, b_shape.num_elements()});
        auto b = dml::Slice(params, dml::TensorDesc::Dimensions{0, 0, r_end, 0}, b_extent, slice_stride);

        w = dml::Reinterpret(w, w_dims, {});
        r = dml::Reinterpret(r, r_dims, {});
        b = dml::Reinterpret(b, b_dims, {});

        std::vector<dml::Expression> x_grad_tensors;

        DML_TENSOR_DATA_TYPE dtype =
            GetDmlDataTypeFromTfDataType(ctx->GetInputDataType(1));
        
        dml::TensorDesc::Dimensions output_extent{1, 1, batch_size, u_num_units};
        dml::TensorDesc::Dimensions xh_x_offset{0, 0, 0, 0};
        dml::TensorDesc::Dimensions xh_h_offset{0, 0, 0, input_size};

        auto b_grad = dml::ZeroTensor(scope, dtype, b_dims);
        auto cs_prev_grad = dml::ZeroTensor(scope, dtype, output_extent);
        auto h_prev_grad = dml::ZeroTensor(scope, dtype, output_extent);
        auto w_grad = dml::ZeroTensor(scope, dtype, w_dims);

        dml::Expression input_backprop;
        dml::Expression input_h_backprop;
        dml::Expression input_c_backprop;
        dml::Expression params_backprop;

        for (uint32_t t = timesteps - 1; t >= 0; --t) {
            dml::TensorDesc::Dimensions tensor_offset{0, t, 0, 0};
            dml::TensorDesc::Dimensions prev_offset{0, t - 1, 0, 0};

            auto input_tensor =
                dml::Slice(inputs, tensor_offset, x_extent, slice_stride);
            auto output_tensor =
                dml::Slice(output, tensor_offset, x_extent, slice_stride);

            if (init_helper->GetRNNMode() == RnnMode::kRnnLstm) {
                auto c_tm1 = t == 0 ? input_c : c_tensors.at(t - 1);
                auto h_tm1 = t == 0 ? input_h : h_tensors.at(t - 1);

                auto xh = dml::Join({input_tensor, h_tm1}, 3);
                auto gates_gemm = dml::Gemm(xh, w);
                dml::Expression gates = gates_gemm;
                gates += dml::Gemm(gates, r);
                gates += b;

                auto cell_size = init_helper->GetCellSize();
                functor::LSTMBlockCell cell(batch_size, input_size, cell_size);

                dml::TensorDesc::Dimensions i_offset =
                    DimensionFromOffset(cell.gates_i_offsets());
                dml::TensorDesc::Dimensions c_offset =
                    DimensionFromOffset(cell.gates_c_offsets(ICFO));
                dml::TensorDesc::Dimensions f_offset =
                    DimensionFromOffset(cell.gates_f_offsets(ICFO));
                dml::TensorDesc::Dimensions o_offset =
                    DimensionFromOffset(cell.gates_o_offsets());
                dml::TensorDesc::Dimensions cell_extent =
                    DimensionFromExtent(cell.cell_extents());

                auto i = dml::Slice(gates, i_offset, cell_extent, slice_stride);
                i = dml::ActivationSigmoid(i);
                // f = sigmoid(z[1])
                // Forget gate (w/ bias).
                auto f = dml::Slice(gates, f_offset, cell_extent, slice_stride);
                f = dml::ActivationSigmoid(f);
                // c = f* init_C + i * tanh(z[2])
                // Cell input.
                auto ci = dml::Slice(gates, c_offset, cell_extent, slice_stride);
                ci = dml::ActivationTanh(ci);
                // cs = ci .* i + f .* cs_prev
                auto cs = i * ci + f * c_tm1;
                // o = sigmoid(z[3])
                // Output gate.
                auto o = dml::Slice(gates, o_offset, cell_extent, slice_stride);
                o = dml::ActivationSigmoid(o);
                // co = tanh(cs)
                auto co = dml::ActivationTanh(cs);
                // h = o * co
                auto h = o * co;
                c_tensors.push_back(cs);
                h_tensors.push_back(h);

                // LSTM gradients
                uint32_t t_ind = static_cast<uint32_t>(t);
                dml::TensorDesc::Dimensions tensor_offset{0, t_ind, 0, 0};
                dml::TensorDesc::Dimensions prev_offset{0, t_ind - 1, 0, 0};

                auto x_tensor =
                    dml::Slice(input_tensor, tensor_offset, x_extent, slice_stride);

                auto cs_prev_tensor = c_tm1;
                auto h_prev_tensor = h_tm1;

                dml::Expression cs_tensor;

                auto cs_grad_tensor = output_c_backprop;
                    // dml::Slice(output_backprop, tensor_offset, output_extent, slice_stride);

                cs_grad_tensor += cs_prev_grad;

                auto h_grad_tensor = output_h_backprop;
                    // dml::Slice(h_grad, tensor_offset, output_extent, slice_stride);

                h_grad_tensor += h_prev_grad;

                // do[t] = sigm'(o[t]) .* dh[t] .* co[t]
                auto do_tensor =
                    o * (1 - o) * h_grad_tensor * co;

                // dcs[t] += tanh'(cs[t]) .* dh[t] .* o[t] + dcs[t + 1] .* f[t + 1]
                auto dcs =
                    (1 - dml::Pow(co, 2.0f)) * h_grad_tensor * o +
                    cs_grad_tensor;

                // dci[t] = tanh'(ci[t]) dcs[t] i[t]
                auto dci = (1 - dml::Pow(ci, 2.0f)) * dcs * i;

                // df[t] = sigm'(f[t]) dcs[t] cs[t - 1]
                auto df = f * (1 - f) * dcs * cs_prev_tensor;

                // di[t] = sigm'(i[t]) dcs[t] ci[t]
                auto di = i * (1 - i) * dcs * ci;

                auto dgates = dml::Join({di, dci, df, do_tensor}, 3);

                cs_prev_grad = dcs * f;

                auto xh_grad_gemm = dml::Gemm(
                    dgates,
                    w,
                    dml::NullOpt,
                    DML_MATRIX_TRANSFORM_NONE,
                    DML_MATRIX_TRANSFORM_TRANSPOSE);
                dml::Expression xh_grad = xh_grad_gemm;

                // auto xh = dml::Join({x_tensor, h_prev_tensor}, 3);

                auto x_grad_tensor =
                    dml::Slice(xh_grad, xh_x_offset, x_extent, slice_stride);
                h_prev_grad =
                    dml::Slice(xh_grad, xh_h_offset, output_extent, slice_stride);

                auto w_grad_gemm = dml::Gemm(
                    xh,
                    dgates,
                    dml::NullOpt,
                    DML_MATRIX_TRANSFORM_TRANSPOSE,
                    DML_MATRIX_TRANSFORM_NONE);
                w_grad += w_grad_gemm;

                b_grad += dml::Reduce(dgates, DML_REDUCE_FUNCTION_SUM, {2});

                // add to vector of x_grad tensors
                x_grad_tensors.insert(x_grad_tensors.begin(), x_grad_tensor);

            } else {
                absl::InlinedVector<dml::Expression, 2> param_grad_tensors;
                auto h_tm1 = t == 0 ? input_h : h_tensors.at(t - 1);

                uint32_t axis_size = 3 * num_units;
                auto split_b = dml::Split(b, 1, {axis_size, axis_size});
                auto& b_i = split_b[0];
                auto& b_r = split_b[1];

                auto matrix_x_gemm = dml::Gemm(input_tensor, w);
                dml::Expression matrix_x = matrix_x_gemm;
                matrix_x += b_i;

                auto cell_size = static_cast<uint32_t>(init_helper->GetCellSize());
                dml::TensorDesc::Dimensions x_z_offsets = {0, 0, 0, 0};
                dml::TensorDesc::Dimensions x_r_offsets = {0, 0, 0, cell_size};
                dml::TensorDesc::Dimensions cell_extents = {1, 1, batch_size, cell_size};
                int32_t slice_strides[] = {1, 1, 1, 1};

                // split z into 3
                uint32_t matrix_axis_size = num_units;
                auto split_matrix_x = dml::Split(matrix_x, 1, {matrix_axis_size, matrix_axis_size, matrix_axis_size});

                auto x_z = split_matrix_x[0];
                auto x_r = split_matrix_x[1];
                auto x_h = split_matrix_x[2];

                auto matrix_inner_gemm = dml::Gemm(h_tm1, r);
                dml::Expression matrix_inner = matrix_inner_gemm;
                matrix_inner += b_r;

                auto split_matrix_inner = dml::Split(matrix_inner, 1, {matrix_axis_size, matrix_axis_size, matrix_axis_size});
                auto recurrent_z = split_matrix_inner[0];
                auto recurrent_r = split_matrix_inner[1];
                auto recurrent_h = split_matrix_inner[2];

                auto z = dml::ActivationSigmoid(x_z + recurrent_z);
                auto r1 = dml::ActivationSigmoid(x_r + recurrent_r);
                auto hh = dml::ActivationTanh(x_h + r1 * recurrent_h);

                auto h = z * h_tm1 + (1 - z) * hh;
                h_tensors.push_back(h);

                // GRU gradients
                auto d0 = output_tensor;
                auto d1 = z * d0;
                auto d2 = h_tm1 * d0;
                auto d3 = hh * d0;
                auto d4 = -1 * d3;
                auto d5 = d2 + d4;
                auto d6 = (1 - z) * d0;
                auto d7 = d5 * (z * (1 - z));
                auto d8 = d6 * (1 - dml::Pow(hh, 2.0));
                auto d9 = d8 * x_h;
                auto d10 = d8 * recurrent_h;
                auto d11 = d7 * x_z;
                auto d12 = d7 * recurrent_z;
                auto d14 = d10 * r1;
                auto d15 = d10 * h_tm1;
                auto d16 = d15 * (r1 * (1 - r1));
                auto d13 = d16 * x_r;
                auto d17 = d16 * recurrent_r;

                auto dx = d9 + d11 + d13;
                auto d_h_prev = d12 + d14 + d1 + d17;
                // u_z + u_r + u_h
                auto d_r = (input_tensor * d7) + (input_tensor * d16) + (input_tensor * d8);
                param_grad_tensors.push_back(d_r);
                // w_z + w_r + w_h
                auto d_w = (h_tm1 * d7) + (h_tm1 * d16) + ((h_tm1 * r1) * d8);
                param_grad_tensors.push_back(d_w);
                
                auto d_params = dml::Join(param_grad_tensors, 3);

                input_backprop = dx;
                input_h_backprop = d_h_prev;
                params_backprop = d_params;
            }
        }
        std::vector<dml::Expression> outputs = {input_backprop, input_h_backprop, params_backprop};
        // if (init_helper->GetRNNMode() == RnnMode::kRnnLstm) {
        //     outputs.push_back(c_tensors.back());
        // }
        Microsoft::WRL::ComPtr<IDMLCompiledOperator> compiled_op =
            scope.Compile(DML_EXECUTION_FLAG_NONE, outputs);

        Initialize(ctx, std::move(tensors), compiled_op.Get());
    }

    StatusOr<DmlGpuEvent> Compute(DmlKernelContext* ctx) const override {
        for (int i = 0; i < ctx->GetOutputCount(); ++i) {
            ctx->GetDmlDeviceContext()->ZeroBuffer(
                ctx->GetDmlDeviceContext()->GetBufferForTensor(ctx->GetOutputTensor(i)));
        }
        return DmlKernel::Compute(ctx);
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