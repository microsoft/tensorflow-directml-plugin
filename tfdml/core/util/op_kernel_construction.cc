/* Copyright (c) Microsoft Corporation.

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

#include "op_kernel_construction.h"

#include "tensorflow/c/kernels.h"

namespace tfdml {
OpKernelConstruction::OpKernelConstruction(TF_OpKernelConstruction* context)
    : context_(context) {}

void OpKernelConstruction::CtxFailure(const char* file, int line,
                                      const Status& s) {
  TF_VLog(1, "OP_REQUIRES failed at %s:%d : %s", file, line, s.error_message());
  status_.Update(s);
  TF_OpKernelConstruction_Failure(context_, status_.raw());
}

void OpKernelConstruction::CtxFailureWithWarning(const char* file, int line,
                                                 const Status& s) {
  TF_Log(TF_WARNING, "OP_REQUIRES failed at %s:%d : %s", file, line,
         s.error_message());
  status_.Update(s);
  TF_OpKernelConstruction_Failure(context_, status_.raw());
}

}  // namespace tfdml
