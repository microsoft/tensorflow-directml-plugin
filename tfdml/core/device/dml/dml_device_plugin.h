#pragma once

#include "tensorflow/c/experimental/stream_executor/stream_executor.h"
#include "tensorflow/c/tf_status.h"

void SE_InitPluginFns(SE_PlatformRegistrationParams* const params,
                      TF_Status* const status);
