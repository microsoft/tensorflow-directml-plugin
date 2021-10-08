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

#include "env_var.h"

#include <stdlib.h>

#include "absl/strings/ascii.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"

namespace tfdml {

Status ReadBoolFromEnvVar(absl::string_view env_var_name, bool default_val,
                          bool* value) {
  *value = default_val;
  const char* tf_env_var_val = getenv(std::string(env_var_name).c_str());
  if (tf_env_var_val == nullptr) {
    return Status::OK();
  }
  std::string str_value = absl::AsciiStrToLower(tf_env_var_val);
  if (str_value == "0" || str_value == "false") {
    *value = false;
    return Status::OK();
  } else if (str_value == "1" || str_value == "true") {
    *value = true;
    return Status::OK();
  }
  return errors::InvalidArgument(absl::StrCat(
      "Failed to parse the env-var ${", env_var_name, "} into bool: ",
      tf_env_var_val, ". Use the default value: ", default_val));
}

Status ReadInt64FromEnvVar(absl::string_view env_var_name, int64_t default_val,
                           int64_t* value) {
  *value = default_val;
  const char* tf_env_var_val = getenv(std::string(env_var_name).c_str());
  if (tf_env_var_val == nullptr) {
    return Status::OK();
  }
  if (absl::SimpleAtoi<int64_t>(tf_env_var_val, value)) {
    return Status::OK();
  }
  return errors::InvalidArgument(absl::StrCat(
      "Failed to parse the env-var ${", env_var_name, "} into int64: ",
      tf_env_var_val, ". Use the default value: ", default_val));
}

Status ReadStringFromEnvVar(absl::string_view env_var_name,
                            absl::string_view default_val, std::string* value) {
  const char* tf_env_var_val = getenv(std::string(env_var_name).c_str());
  if (tf_env_var_val != nullptr) {
    *value = tf_env_var_val;
  } else {
    *value = std::string(default_val);
  }
  return Status::OK();
}

}  // namespace tfdml