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

#pragma once

#include <memory>
#include <string>

#include "absl/strings/str_cat.h"
#include "tensorflow/c/tf_status.h"

namespace tfdml {

class Status {
 public:
  explicit Status();
  explicit Status(TF_Code code, const char* message);
  explicit Status(TF_Code code, const std::string& message);
  explicit Status(TF_Code code, std::string&& message);
  TF_Code code() const;
  bool ok() const;
  const char* error_message() const;
  TF_Status* raw() const;
  static Status OK();
  void Update(const Status& new_status);

 private:
  std::shared_ptr<TF_Status> safe_status_;
};

namespace errors {
// Convenience functions for generating and using error
// status.
// Example usage:
//   status.Update(errors::InvalidArgument("The ", foo, " isn't right."));
//   if (errors::IsInvalidArgument(status)) { ... }
//   switch (status.code()) { case error::INVALID_ARGUMENT: ... }

#define DECLARE_ERROR(FUNC, CODE)                        \
  template <typename... Args>                            \
  ::tfdml::Status FUNC(Args... args) {                   \
    return ::tfdml::Status(CODE, absl::StrCat(args...)); \
  }                                                      \
  inline bool Is##FUNC(const ::tfdml::Status& status) {  \
    return status.code() == CODE;                        \
  }

DECLARE_ERROR(Cancelled, TF_CANCELLED)
DECLARE_ERROR(InvalidArgument, TF_INVALID_ARGUMENT)
DECLARE_ERROR(NotFound, TF_NOT_FOUND)
DECLARE_ERROR(AlreadyExists, TF_ALREADY_EXISTS)
DECLARE_ERROR(ResourceExhausted, TF_RESOURCE_EXHAUSTED)
DECLARE_ERROR(Unavailable, TF_UNAVAILABLE)
DECLARE_ERROR(FailedPrecondition, TF_FAILED_PRECONDITION)
DECLARE_ERROR(OutOfRange, TF_OUT_OF_RANGE)
DECLARE_ERROR(Unimplemented, TF_UNIMPLEMENTED)
DECLARE_ERROR(Internal, TF_INTERNAL)
DECLARE_ERROR(Aborted, TF_ABORTED)
DECLARE_ERROR(DeadlineExceeded, TF_DEADLINE_EXCEEDED)
DECLARE_ERROR(DataLoss, TF_DATA_LOSS)
DECLARE_ERROR(Unknown, TF_UNKNOWN)
DECLARE_ERROR(PermissionDenied, TF_PERMISSION_DENIED)
DECLARE_ERROR(Unauthenticated, TF_UNAUTHENTICATED)

#undef DECLARE_ERROR
}  // namespace errors

}  // namespace tfdml
