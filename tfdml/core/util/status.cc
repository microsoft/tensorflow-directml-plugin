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

#include "status.h"

#include "tensorflow/c/tf_status.h"

namespace tfdml {

void delete_status(TF_Status* s) {
  if (s != nullptr) {
    TF_DeleteStatus(s);
  }
}

Status::Status() : Status(TF_OK, "") {}

Status::Status(TF_Code code, const char* message)
    : safe_status_(TF_NewStatus(), delete_status) {
  TF_SetStatus(safe_status_.get(), code, message);
}

Status::Status(TF_Code code, const std::string& message)
    : Status(code, message.c_str()) {}

Status::Status(TF_Code code, std::string&& message)
    : Status(code, message.c_str()) {}

TF_Code Status::code() const { return TF_GetCode(safe_status_.get()); }
bool Status::ok() const { return TF_GetCode(safe_status_.get()) == TF_OK; }

TF_Status* Status::raw() const { return safe_status_.get(); }

const char* Status::error_message() const {
  return TF_Message(safe_status_.get());
}

Status Status::OK() { return Status(); }

void Status::Update(const Status& new_status) {
  if (ok()) {
    *this = new_status;
  }
}

}  // namespace tfdml
