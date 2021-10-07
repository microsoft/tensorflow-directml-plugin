#include "plugin_device.h"

#include <iostream>

#include "tensorflow/c/experimental/stream_executor/stream_executor.h"
#include "tfdml/core/device/dml/dml_device_plugin.h"

extern "C" {
void SE_InitPlugin(SE_PlatformRegistrationParams* params, TF_Status* status) {
  params->platform->struct_size = SP_PLATFORM_STRUCT_SIZE;
  params->platform->name = DEVICE_NAME;
  params->platform->type = DEVICE_TYPE;
  params->platform->supports_unified_memory = false;

  // TODO: Set force_memory_growth to true once the PR is merged
  // https://github.com/tensorflow/tensorflow/pull/51705
  params->platform->use_bfc_allocator = true;
  params->major_version = 0;
  params->minor_version = 0;
  params->patch_version = 1;
  SE_InitPluginFns(params, status);
}
}
