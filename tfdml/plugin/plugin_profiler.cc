/* Copyright (c) Microsoft Corporation.
Use of this source code is governed by an MIT-style
license that can be found in the LICENSE file or at
https://opensource.org/licenses/MIT.
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "tensorflow/c/experimental/pluggable_profiler/pluggable_profiler.h"
#include "tfdml/core/dml_device_cache.h"
#include "tfdml/core/dml_tracing.h"

// {D113B493-BBA2-4993-8608-D706A73B91CE}
static constexpr GUID PIX_EVAL_CAPTURABLE_WORK_GUID = {
    0xd113b493,
    0xbba2,
    0x4993,
    {0x86, 0x08, 0xd7, 0x06, 0xa7, 0x3b, 0x91, 0xce}};

void profiler_start(const TP_Profiler* profiler, TF_Status* status) {
  DmlTracing::Instance().LogSessionRunStart();
  auto& device_cache = tfdml::DmlDeviceCache::Instance();

  for (uint32_t i = 0; i < device_cache.GetAdapterCount(); ++i) {
    const auto* state = device_cache.GetOrCreateDeviceState(i);

    if (state->sharing_contract) {
      state->sharing_contract->BeginCapturableWork(
          PIX_EVAL_CAPTURABLE_WORK_GUID);
    }
  }
}

void profiler_stop(const TP_Profiler* profiler, TF_Status* status) {
  DmlTracing::Instance().LogSessionRunEnd();
  auto& device_cache = tfdml::DmlDeviceCache::Instance();

  for (uint32_t i = 0; i < device_cache.GetAdapterCount(); ++i) {
    const auto* state = device_cache.GetOrCreateDeviceState(i);

    if (state->sharing_contract) {
      state->sharing_contract->EndCapturableWork(PIX_EVAL_CAPTURABLE_WORK_GUID);
    }
  }
}

void profiler_collect_data_xspace(const TP_Profiler* profiler, uint8_t* buffer,
                                  size_t* size_in_bytes, TF_Status* status) {
  // We don't need xspace data for PIX
  *size_in_bytes = 0;
  TF_SetStatus(status, TF_OK, "");
}

void profiler_destroy_profiler(TP_Profiler* profiler) {
  // Nothing to do here
}

void profiler_destroy_profiler_fns(TP_ProfilerFns* profiler_fns) {
  // Nothing to do here
}

void TF_InitProfiler(TF_ProfilerRegistrationParams* params, TF_Status* status) {
  params->profiler->struct_size = TP_PROFILER_STRUCT_SIZE;
  params->profiler_fns->struct_size = TP_PROFILER_FNS_STRUCT_SIZE;

  params->profiler->device_type = "DML";

  params->profiler_fns->start = profiler_start;
  params->profiler_fns->stop = profiler_stop;
  params->profiler_fns->collect_data_xspace = profiler_collect_data_xspace;
  params->destroy_profiler = profiler_destroy_profiler;
  params->destroy_profiler_fns = profiler_destroy_profiler_fns;
} 