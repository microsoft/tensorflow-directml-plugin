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

#include "tensorflow/c/kernels.h"

namespace tfdml
{
void RegisterKernels_AddN();
void RegisterKernels_AssignVariableOps();
void RegisterKernels_BatchNorm();
void RegisterKernels_BatchToSpace();
void RegisterKernels_BiasAdd();
void RegisterKernels_Bitcast();
void RegisterKernels_BroadcastTo();
void RegisterKernels_Cast();
void RegisterKernels_CheckNumerics();
void RegisterKernels_Concat();
void RegisterKernels_Conv();
void RegisterKernels_CropAndResizeGradBoxes();
void RegisterKernels_CropAndResizeGradImage();
void RegisterKernels_CropAndResize();
void RegisterKernels_Cross();
void RegisterKernels_Cwise();
void RegisterKernels_DataFormatDimMap();
void RegisterKernels_DataFormatVecPermute();
void RegisterKernels_DeepCopy();
void RegisterKernels_Diag();
void RegisterKernels_DiagPart();
void RegisterKernels_DynamicStitch();
void RegisterKernels_Empty();
void RegisterKernels_ExtractImagePatches();
void RegisterKernels_ExtractVolumePatches();
void RegisterKernels_Fill();
void RegisterKernels_Gather();
void RegisterKernels_GatherNd();
void RegisterKernels_GRU();
void RegisterKernels_Image();
void RegisterKernels_Inplace();
void RegisterKernels_L2Loss();
void RegisterKernels_LRN();
void RegisterKernels_LSTM();
void RegisterKernels_MatMul();
void RegisterKernels_MatrixBandPart();
void RegisterKernels_MatrixDiag();
void RegisterKernels_MatrixDiagPart();
void RegisterKernels_MatrixSetDiag();
void RegisterKernels_MirrorPadGrad();
void RegisterKernels_OneHot();
void RegisterKernels_OnesLike();
void RegisterKernels_Pack();
void RegisterKernels_Pad();
void RegisterKernels_ParallelConcat();
void RegisterKernels_Pooling();
void RegisterKernels_Random();
void RegisterKernels_Range();
void RegisterKernels_Reduce();
void RegisterKernels_Relu();
void RegisterKernels_Resize();
void RegisterKernels_ResizeGrad();
void RegisterKernels_Reverse();
void RegisterKernels_ReverseSequence();
void RegisterKernels_Roll();
void RegisterKernels_Scan();
void RegisterKernels_Scatter();
void RegisterKernels_ScatterNd();
void RegisterKernels_Select();
void RegisterKernels_Slice();
void RegisterKernels_Snapshot();
void RegisterKernels_SpaceDepth();
void RegisterKernels_SpaceToBatch();
void RegisterKernels_SparseXent();
void RegisterKernels_Split();
void RegisterKernels_StridedSlice();
void RegisterKernels_StridedSliceCpu();
void RegisterKernels_Swapping();
void RegisterKernels_Tile();
void RegisterKernels_TopK();
void RegisterKernels_Training();
void RegisterKernels_Transpose();
void RegisterKernels_Unpack();
void RegisterKernels_Where();
void RegisterKernels_Xent();
void RegisterKernels_ZerosLike();
} // namespace tfdml

void TF_InitKernel()
{
    // NOTE: we could add logic here to conditionally register kernels based on
    // D3D12 adapter capabilities (for example).
    tfdml::RegisterKernels_AddN();
    tfdml::RegisterKernels_AssignVariableOps();
    tfdml::RegisterKernels_BatchNorm();
    tfdml::RegisterKernels_BatchToSpace();
    tfdml::RegisterKernels_BiasAdd();
    tfdml::RegisterKernels_Bitcast();
    tfdml::RegisterKernels_BroadcastTo();
    tfdml::RegisterKernels_Cast();
    tfdml::RegisterKernels_CheckNumerics();
    tfdml::RegisterKernels_Concat();
    tfdml::RegisterKernels_Conv();
    tfdml::RegisterKernels_CropAndResizeGradBoxes();
    tfdml::RegisterKernels_CropAndResizeGradImage();
    tfdml::RegisterKernels_CropAndResize();
    tfdml::RegisterKernels_Cross();
    tfdml::RegisterKernels_Cwise();
    tfdml::RegisterKernels_DataFormatDimMap();
    tfdml::RegisterKernels_DataFormatVecPermute();
    tfdml::RegisterKernels_DeepCopy();
    tfdml::RegisterKernels_Diag();
    tfdml::RegisterKernels_DiagPart();
    tfdml::RegisterKernels_DynamicStitch();
    tfdml::RegisterKernels_Empty();
    tfdml::RegisterKernels_ExtractImagePatches();
    tfdml::RegisterKernels_ExtractVolumePatches();
    tfdml::RegisterKernels_Fill();
    tfdml::RegisterKernels_Gather();
    tfdml::RegisterKernels_GatherNd();
    tfdml::RegisterKernels_GRU();
    tfdml::RegisterKernels_Image();
    tfdml::RegisterKernels_Inplace();
    tfdml::RegisterKernels_L2Loss();
    tfdml::RegisterKernels_LRN();
    tfdml::RegisterKernels_LSTM();
    tfdml::RegisterKernels_MatMul();
    tfdml::RegisterKernels_MatrixBandPart();
    tfdml::RegisterKernels_MatrixDiag();
    tfdml::RegisterKernels_MatrixDiagPart();
    tfdml::RegisterKernels_MatrixSetDiag();
    tfdml::RegisterKernels_MirrorPadGrad();
    tfdml::RegisterKernels_OneHot();
    tfdml::RegisterKernels_OnesLike();
    tfdml::RegisterKernels_Pack();
    tfdml::RegisterKernels_Pad();
    tfdml::RegisterKernels_ParallelConcat();
    tfdml::RegisterKernels_Pooling();
    tfdml::RegisterKernels_Random();
    tfdml::RegisterKernels_Range();
    tfdml::RegisterKernels_Reduce();
    tfdml::RegisterKernels_Relu();
    tfdml::RegisterKernels_Resize();
    tfdml::RegisterKernels_ResizeGrad();
    tfdml::RegisterKernels_Reverse();
    tfdml::RegisterKernels_ReverseSequence();
    tfdml::RegisterKernels_Roll();
    tfdml::RegisterKernels_Scan();
    tfdml::RegisterKernels_Scatter();
    tfdml::RegisterKernels_ScatterNd();
    tfdml::RegisterKernels_Select();
    tfdml::RegisterKernels_Slice();
    tfdml::RegisterKernels_Snapshot();
    tfdml::RegisterKernels_SpaceDepth();
    tfdml::RegisterKernels_SpaceToBatch();
    tfdml::RegisterKernels_SparseXent();
    tfdml::RegisterKernels_Split();
    tfdml::RegisterKernels_StridedSlice();
    tfdml::RegisterKernels_StridedSliceCpu();
    tfdml::RegisterKernels_Swapping();
    tfdml::RegisterKernels_Tile();
    tfdml::RegisterKernels_TopK();
    tfdml::RegisterKernels_Training();
    tfdml::RegisterKernels_Transpose();
    tfdml::RegisterKernels_Unpack();
    tfdml::RegisterKernels_Where();
    tfdml::RegisterKernels_Xent();
    tfdml::RegisterKernels_ZerosLike();
}
