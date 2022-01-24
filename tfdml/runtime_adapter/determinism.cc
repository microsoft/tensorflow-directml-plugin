/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#include "absl/strings/string_view.h"
#include "tfdml/runtime_adapter/env_var.h"
#include "tfdml/runtime_adapter/macros.h"

namespace tfdml
{

namespace
{

class DeterminismState
{
  public:
    explicit DeterminismState(absl::string_view env_var) : env_var_(env_var) {}
    bool Required()
    {
        if (state_ == Value::NOT_SET)
        {
            bool env_var_set = false;
            TF_CHECK_OK(ReadBoolFromEnvVar(env_var_, false, &env_var_set));
            state_ = env_var_set ? Value::ENABLED : Value::DISABLED;
        }

        return state_ == Value::ENABLED;
    }

  private:
    absl::string_view env_var_;
    enum class Value
    {
        DISABLED,
        ENABLED,
        NOT_SET
    };
    Value state_ = Value::NOT_SET;
};

} // namespace

DeterminismState OpDeterminismState = DeterminismState("TF_DETERMINISTIC_OPS");

bool OpDeterminismRequired() { return OpDeterminismState.Required(); }

} // namespace tfdml
