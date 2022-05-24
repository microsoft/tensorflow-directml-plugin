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

// clang-format off

// This file is generated. Do not edit it directly. See generate_op_defs_core.py.

#include "tfdml/runtime_adapter/op_defs.h"

namespace tfdml
{
namespace ops
{

constexpr std::array<ArgumentDesc, Mod::input_arg_count + Mod::output_arg_count> Mod::argument_descs;
constexpr std::array<AttributeDesc, 1> Mod::attribute_descs;

constexpr std::array<ArgumentDesc, RemoteCall::input_arg_count + RemoteCall::output_arg_count> RemoteCall::argument_descs;
constexpr std::array<AttributeDesc, 3> RemoteCall::attribute_descs;

constexpr std::array<ArgumentDesc, RefOutput::input_arg_count + RefOutput::output_arg_count> RefOutput::argument_descs;
constexpr std::array<AttributeDesc, 0> RefOutput::attribute_descs;

constexpr std::array<ArgumentDesc, TensorMapInsert::input_arg_count + TensorMapInsert::output_arg_count> TensorMapInsert::argument_descs;
constexpr std::array<AttributeDesc, 2> TensorMapInsert::attribute_descs;

constexpr std::array<ArgumentDesc, DTensorAllReduce::input_arg_count + DTensorAllReduce::output_arg_count> DTensorAllReduce::argument_descs;
constexpr std::array<AttributeDesc, 3> DTensorAllReduce::attribute_descs;

constexpr std::array<ArgumentDesc, RpcClient::input_arg_count + RpcClient::output_arg_count> RpcClient::argument_descs;
constexpr std::array<AttributeDesc, 2> RpcClient::attribute_descs;

constexpr std::array<ArgumentDesc, DTensorShardedPrefix::input_arg_count + DTensorShardedPrefix::output_arg_count> DTensorShardedPrefix::argument_descs;
constexpr std::array<AttributeDesc, 1> DTensorShardedPrefix::attribute_descs;

constexpr std::array<ArgumentDesc, StatsAggregatorSummary::input_arg_count + StatsAggregatorSummary::output_arg_count> StatsAggregatorSummary::argument_descs;
constexpr std::array<AttributeDesc, 0> StatsAggregatorSummary::attribute_descs;

constexpr std::array<ArgumentDesc, EncodeProto::input_arg_count + EncodeProto::output_arg_count> EncodeProto::argument_descs;
constexpr std::array<AttributeDesc, 4> EncodeProto::attribute_descs;

constexpr std::array<ArgumentDesc, Old::input_arg_count + Old::output_arg_count> Old::argument_descs;
constexpr std::array<AttributeDesc, 0> Old::attribute_descs;

constexpr std::array<ArgumentDesc, Relayout::input_arg_count + Relayout::output_arg_count> Relayout::argument_descs;
constexpr std::array<AttributeDesc, 2> Relayout::attribute_descs;

constexpr std::array<ArgumentDesc, AudioSpectrogram::input_arg_count + AudioSpectrogram::output_arg_count> AudioSpectrogram::argument_descs;
constexpr std::array<AttributeDesc, 3> AudioSpectrogram::attribute_descs;

constexpr std::array<ArgumentDesc, DTensorReduceScatter::input_arg_count + DTensorReduceScatter::output_arg_count> DTensorReduceScatter::argument_descs;
constexpr std::array<AttributeDesc, 3> DTensorReduceScatter::attribute_descs;

constexpr std::array<ArgumentDesc, _ArrayToList::input_arg_count + _ArrayToList::output_arg_count> _ArrayToList::argument_descs;
constexpr std::array<AttributeDesc, 3> _ArrayToList::attribute_descs;

constexpr std::array<ArgumentDesc, DTensorAllGather::input_arg_count + DTensorAllGather::output_arg_count> DTensorAllGather::argument_descs;
constexpr std::array<AttributeDesc, 3> DTensorAllGather::attribute_descs;

constexpr std::array<ArgumentDesc, UniqueDataset::input_arg_count + UniqueDataset::output_arg_count> UniqueDataset::argument_descs;
constexpr std::array<AttributeDesc, 3> UniqueDataset::attribute_descs;

constexpr std::array<ArgumentDesc, ApplyRMSProp::input_arg_count + ApplyRMSProp::output_arg_count> ApplyRMSProp::argument_descs;
constexpr std::array<AttributeDesc, 2> ApplyRMSProp::attribute_descs;

constexpr std::array<ArgumentDesc, BesselY0::input_arg_count + BesselY0::output_arg_count> BesselY0::argument_descs;
constexpr std::array<AttributeDesc, 1> BesselY0::attribute_descs;

constexpr std::array<ArgumentDesc, AttrBool::input_arg_count + AttrBool::output_arg_count> AttrBool::argument_descs;
constexpr std::array<AttributeDesc, 1> AttrBool::attribute_descs;

constexpr std::array<ArgumentDesc, DatasetToTFRecord::input_arg_count + DatasetToTFRecord::output_arg_count> DatasetToTFRecord::argument_descs;
constexpr std::array<AttributeDesc, 0> DatasetToTFRecord::attribute_descs;

constexpr std::array<ArgumentDesc, RefIn::input_arg_count + RefIn::output_arg_count> RefIn::argument_descs;
constexpr std::array<AttributeDesc, 1> RefIn::attribute_descs;

constexpr std::array<ArgumentDesc, DTensorAllScatter::input_arg_count + DTensorAllScatter::output_arg_count> DTensorAllScatter::argument_descs;
constexpr std::array<AttributeDesc, 3> DTensorAllScatter::attribute_descs;

constexpr std::array<ArgumentDesc, TwoFloatInputsIntOutput::input_arg_count + TwoFloatInputsIntOutput::output_arg_count> TwoFloatInputsIntOutput::argument_descs;
constexpr std::array<AttributeDesc, 0> TwoFloatInputsIntOutput::attribute_descs;

constexpr std::array<ArgumentDesc, TensorDataset::input_arg_count + TensorDataset::output_arg_count> TensorDataset::argument_descs;
constexpr std::array<AttributeDesc, 3> TensorDataset::attribute_descs;

constexpr std::array<ArgumentDesc, FloatOutputStringOutput::input_arg_count + FloatOutputStringOutput::output_arg_count> FloatOutputStringOutput::argument_descs;
constexpr std::array<AttributeDesc, 0> FloatOutputStringOutput::attribute_descs;

constexpr std::array<ArgumentDesc, PriorityQueue::input_arg_count + PriorityQueue::output_arg_count> PriorityQueue::argument_descs;
constexpr std::array<AttributeDesc, 5> PriorityQueue::attribute_descs;

constexpr std::array<ArgumentDesc, CopyToMesh::input_arg_count + CopyToMesh::output_arg_count> CopyToMesh::argument_descs;
constexpr std::array<AttributeDesc, 3> CopyToMesh::attribute_descs;

constexpr std::array<ArgumentDesc, ScanDataset::input_arg_count + ScanDataset::output_arg_count> ScanDataset::argument_descs;
constexpr std::array<AttributeDesc, 8> ScanDataset::attribute_descs;

constexpr std::array<ArgumentDesc, ExperimentalUnbatchDataset::input_arg_count + ExperimentalUnbatchDataset::output_arg_count> ExperimentalUnbatchDataset::argument_descs;
constexpr std::array<AttributeDesc, 2> ExperimentalUnbatchDataset::attribute_descs;

constexpr std::array<ArgumentDesc, _FusedConv3D::input_arg_count + _FusedConv3D::output_arg_count> _FusedConv3D::argument_descs;
constexpr std::array<AttributeDesc, 10> _FusedConv3D::attribute_descs;

constexpr std::array<ArgumentDesc, RecordInput::input_arg_count + RecordInput::output_arg_count> RecordInput::argument_descs;
constexpr std::array<AttributeDesc, 7> RecordInput::attribute_descs;

constexpr std::array<ArgumentDesc, TensorArrayWrite::input_arg_count + TensorArrayWrite::output_arg_count> TensorArrayWrite::argument_descs;
constexpr std::array<AttributeDesc, 1> TensorArrayWrite::attribute_descs;

constexpr std::array<ArgumentDesc, DTensorRestoreV2::input_arg_count + DTensorRestoreV2::output_arg_count> DTensorRestoreV2::argument_descs;
constexpr std::array<AttributeDesc, 3> DTensorRestoreV2::attribute_descs;

constexpr std::array<ArgumentDesc, ConfigureAndInitializeGlobalTPU::input_arg_count + ConfigureAndInitializeGlobalTPU::output_arg_count> ConfigureAndInitializeGlobalTPU::argument_descs;
constexpr std::array<AttributeDesc, 0> ConfigureAndInitializeGlobalTPU::attribute_descs;

constexpr std::array<ArgumentDesc, Foo3::input_arg_count + Foo3::output_arg_count> Foo3::argument_descs;
constexpr std::array<AttributeDesc, 0> Foo3::attribute_descs;

constexpr std::array<ArgumentDesc, LookupTableSizeV2::input_arg_count + LookupTableSizeV2::output_arg_count> LookupTableSizeV2::argument_descs;
constexpr std::array<AttributeDesc, 0> LookupTableSizeV2::attribute_descs;

constexpr std::array<ArgumentDesc, SampleDistortedBoundingBoxV2::input_arg_count + SampleDistortedBoundingBoxV2::output_arg_count> SampleDistortedBoundingBoxV2::argument_descs;
constexpr std::array<AttributeDesc, 7> SampleDistortedBoundingBoxV2::attribute_descs;

constexpr std::array<ArgumentDesc, A::input_arg_count + A::output_arg_count> A::argument_descs;
constexpr std::array<AttributeDesc, 0> A::attribute_descs;

constexpr std::array<ArgumentDesc, RiscBinaryArithmetic::input_arg_count + RiscBinaryArithmetic::output_arg_count> RiscBinaryArithmetic::argument_descs;
constexpr std::array<AttributeDesc, 2> RiscBinaryArithmetic::attribute_descs;

constexpr std::array<ArgumentDesc, RpcServer::input_arg_count + RpcServer::output_arg_count> RpcServer::argument_descs;
constexpr std::array<AttributeDesc, 0> RpcServer::attribute_descs;

constexpr std::array<ArgumentDesc, OnesLike::input_arg_count + OnesLike::output_arg_count> OnesLike::argument_descs;
constexpr std::array<AttributeDesc, 1> OnesLike::attribute_descs;

constexpr std::array<ArgumentDesc, ShutdownTPUSystem::input_arg_count + ShutdownTPUSystem::output_arg_count> ShutdownTPUSystem::argument_descs;
constexpr std::array<AttributeDesc, 0> ShutdownTPUSystem::attribute_descs;

constexpr std::array<ArgumentDesc, ResourceApplyKerasMomentum::input_arg_count + ResourceApplyKerasMomentum::output_arg_count> ResourceApplyKerasMomentum::argument_descs;
constexpr std::array<AttributeDesc, 3> ResourceApplyKerasMomentum::attribute_descs;

constexpr std::array<ArgumentDesc, IntAttr::input_arg_count + IntAttr::output_arg_count> IntAttr::argument_descs;
constexpr std::array<AttributeDesc, 1> IntAttr::attribute_descs;

constexpr std::array<ArgumentDesc, RpcServerStart::input_arg_count + RpcServerStart::output_arg_count> RpcServerStart::argument_descs;
constexpr std::array<AttributeDesc, 0> RpcServerStart::attribute_descs;

constexpr std::array<ArgumentDesc, XlaSplitND::input_arg_count + XlaSplitND::output_arg_count> XlaSplitND::argument_descs;
constexpr std::array<AttributeDesc, 4> XlaSplitND::attribute_descs;

constexpr std::array<ArgumentDesc, RefExit::input_arg_count + RefExit::output_arg_count> RefExit::argument_descs;
constexpr std::array<AttributeDesc, 1> RefExit::attribute_descs;

constexpr std::array<ArgumentDesc, RpcServerRegister::input_arg_count + RpcServerRegister::output_arg_count> RpcServerRegister::argument_descs;
constexpr std::array<AttributeDesc, 4> RpcServerRegister::attribute_descs;

constexpr std::array<ArgumentDesc, DeleteRpcFutureResource::input_arg_count + DeleteRpcFutureResource::output_arg_count> DeleteRpcFutureResource::argument_descs;
constexpr std::array<AttributeDesc, 0> DeleteRpcFutureResource::attribute_descs;

constexpr std::array<ArgumentDesc, EnqueueTPUEmbeddingArbitraryTensorBatch::input_arg_count + EnqueueTPUEmbeddingArbitraryTensorBatch::output_arg_count> EnqueueTPUEmbeddingArbitraryTensorBatch::argument_descs;
constexpr std::array<AttributeDesc, 6> EnqueueTPUEmbeddingArbitraryTensorBatch::attribute_descs;

constexpr std::array<ArgumentDesc, DestroyTemporaryVariable::input_arg_count + DestroyTemporaryVariable::output_arg_count> DestroyTemporaryVariable::argument_descs;
constexpr std::array<AttributeDesc, 2> DestroyTemporaryVariable::attribute_descs;

constexpr std::array<ArgumentDesc, RpcCall::input_arg_count + RpcCall::output_arg_count> RpcCall::argument_descs;
constexpr std::array<AttributeDesc, 1> RpcCall::attribute_descs;

constexpr std::array<ArgumentDesc, IntInputIntOutput::input_arg_count + IntInputIntOutput::output_arg_count> IntInputIntOutput::argument_descs;
constexpr std::array<AttributeDesc, 0> IntInputIntOutput::attribute_descs;

constexpr std::array<ArgumentDesc, _MklBatchMatMul::input_arg_count + _MklBatchMatMul::output_arg_count> _MklBatchMatMul::argument_descs;
constexpr std::array<AttributeDesc, 3> _MklBatchMatMul::attribute_descs;

constexpr std::array<ArgumentDesc, _MklFusedBatchNormGradV3::input_arg_count + _MklFusedBatchNormGradV3::output_arg_count> _MklFusedBatchNormGradV3::argument_descs;
constexpr std::array<AttributeDesc, 5> _MklFusedBatchNormGradV3::attribute_descs;

constexpr std::array<ArgumentDesc, SparseReduceSumSparse::input_arg_count + SparseReduceSumSparse::output_arg_count> SparseReduceSumSparse::argument_descs;
constexpr std::array<AttributeDesc, 2> SparseReduceSumSparse::attribute_descs;

constexpr std::array<ArgumentDesc, ControlTrigger::input_arg_count + ControlTrigger::output_arg_count> ControlTrigger::argument_descs;
constexpr std::array<AttributeDesc, 0> ControlTrigger::attribute_descs;

constexpr std::array<ArgumentDesc, RpcCheckStatus::input_arg_count + RpcCheckStatus::output_arg_count> RpcCheckStatus::argument_descs;
constexpr std::array<AttributeDesc, 0> RpcCheckStatus::attribute_descs;

constexpr std::array<ArgumentDesc, _ConfigureDistributedTPU::input_arg_count + _ConfigureDistributedTPU::output_arg_count> _ConfigureDistributedTPU::argument_descs;
constexpr std::array<AttributeDesc, 2> _ConfigureDistributedTPU::attribute_descs;

constexpr std::array<ArgumentDesc, GenerateVocabRemapping::input_arg_count + GenerateVocabRemapping::output_arg_count> GenerateVocabRemapping::argument_descs;
constexpr std::array<AttributeDesc, 3> GenerateVocabRemapping::attribute_descs;

constexpr std::array<ArgumentDesc, KernelLabel::input_arg_count + KernelLabel::output_arg_count> KernelLabel::argument_descs;
constexpr std::array<AttributeDesc, 0> KernelLabel::attribute_descs;

constexpr std::array<ArgumentDesc, QuantizedConcatV2::input_arg_count + QuantizedConcatV2::output_arg_count> QuantizedConcatV2::argument_descs;
constexpr std::array<AttributeDesc, 3> QuantizedConcatV2::attribute_descs;

constexpr std::array<ArgumentDesc, ParallelBatchDataset::input_arg_count + ParallelBatchDataset::output_arg_count> ParallelBatchDataset::argument_descs;
constexpr std::array<AttributeDesc, 5> ParallelBatchDataset::attribute_descs;

constexpr std::array<ArgumentDesc, ResourceApplyAdaMax::input_arg_count + ResourceApplyAdaMax::output_arg_count> ResourceApplyAdaMax::argument_descs;
constexpr std::array<AttributeDesc, 2> ResourceApplyAdaMax::attribute_descs;

constexpr std::array<ArgumentDesc, RpcGetValue::input_arg_count + RpcGetValue::output_arg_count> RpcGetValue::argument_descs;
constexpr std::array<AttributeDesc, 1> RpcGetValue::attribute_descs;

constexpr std::array<ArgumentDesc, MergeSummary::input_arg_count + MergeSummary::output_arg_count> MergeSummary::argument_descs;
constexpr std::array<AttributeDesc, 1> MergeSummary::attribute_descs;

constexpr std::array<ArgumentDesc, BlockLSTM::input_arg_count + BlockLSTM::output_arg_count> BlockLSTM::argument_descs;
constexpr std::array<AttributeDesc, 4> BlockLSTM::attribute_descs;

constexpr std::array<ArgumentDesc, Log1p::input_arg_count + Log1p::output_arg_count> Log1p::argument_descs;
constexpr std::array<AttributeDesc, 1> Log1p::attribute_descs;

constexpr std::array<ArgumentDesc, TfLiteSubgraphExecute::input_arg_count + TfLiteSubgraphExecute::output_arg_count> TfLiteSubgraphExecute::argument_descs;
constexpr std::array<AttributeDesc, 2> TfLiteSubgraphExecute::attribute_descs;

constexpr std::array<ArgumentDesc, ConstructionFails::input_arg_count + ConstructionFails::output_arg_count> ConstructionFails::argument_descs;
constexpr std::array<AttributeDesc, 0> ConstructionFails::attribute_descs;

constexpr std::array<ArgumentDesc, KernelLabelRequired::input_arg_count + KernelLabelRequired::output_arg_count> KernelLabelRequired::argument_descs;
constexpr std::array<AttributeDesc, 0> KernelLabelRequired::attribute_descs;

constexpr std::array<ArgumentDesc, BoostedTreesCreateQuantileStreamResource::input_arg_count + BoostedTreesCreateQuantileStreamResource::output_arg_count> BoostedTreesCreateQuantileStreamResource::argument_descs;
constexpr std::array<AttributeDesc, 1> BoostedTreesCreateQuantileStreamResource::attribute_descs;

constexpr std::array<ArgumentDesc, GraphDefVersion::input_arg_count + GraphDefVersion::output_arg_count> GraphDefVersion::argument_descs;
constexpr std::array<AttributeDesc, 0> GraphDefVersion::attribute_descs;

constexpr std::array<ArgumentDesc, RequiresOlderGraphVersion::input_arg_count + RequiresOlderGraphVersion::output_arg_count> RequiresOlderGraphVersion::argument_descs;
constexpr std::array<AttributeDesc, 0> RequiresOlderGraphVersion::attribute_descs;

constexpr std::array<ArgumentDesc, Imag::input_arg_count + Imag::output_arg_count> Imag::argument_descs;
constexpr std::array<AttributeDesc, 2> Imag::attribute_descs;

constexpr std::array<ArgumentDesc, StackV2::input_arg_count + StackV2::output_arg_count> StackV2::argument_descs;
constexpr std::array<AttributeDesc, 2> StackV2::attribute_descs;

constexpr std::array<ArgumentDesc, FakeQueue::input_arg_count + FakeQueue::output_arg_count> FakeQueue::argument_descs;
constexpr std::array<AttributeDesc, 0> FakeQueue::attribute_descs;

constexpr std::array<ArgumentDesc, Round::input_arg_count + Round::output_arg_count> Round::argument_descs;
constexpr std::array<AttributeDesc, 1> Round::attribute_descs;

constexpr std::array<ArgumentDesc, ScatterMin::input_arg_count + ScatterMin::output_arg_count> ScatterMin::argument_descs;
constexpr std::array<AttributeDesc, 3> ScatterMin::attribute_descs;

constexpr std::array<ArgumentDesc, GetDeadline::input_arg_count + GetDeadline::output_arg_count> GetDeadline::argument_descs;
constexpr std::array<AttributeDesc, 0> GetDeadline::attribute_descs;

constexpr std::array<ArgumentDesc, BesselK1::input_arg_count + BesselK1::output_arg_count> BesselK1::argument_descs;
constexpr std::array<AttributeDesc, 1> BesselK1::attribute_descs;

constexpr std::array<ArgumentDesc, SleepOp::input_arg_count + SleepOp::output_arg_count> SleepOp::argument_descs;
constexpr std::array<AttributeDesc, 0> SleepOp::attribute_descs;

constexpr std::array<ArgumentDesc, SleepIdentityOp::input_arg_count + SleepIdentityOp::output_arg_count> SleepIdentityOp::argument_descs;
constexpr std::array<AttributeDesc, 1> SleepIdentityOp::attribute_descs;

constexpr std::array<ArgumentDesc, Assign::input_arg_count + Assign::output_arg_count> Assign::argument_descs;
constexpr std::array<AttributeDesc, 3> Assign::attribute_descs;

constexpr std::array<ArgumentDesc, StubResourceHandleOp::input_arg_count + StubResourceHandleOp::output_arg_count> StubResourceHandleOp::argument_descs;
constexpr std::array<AttributeDesc, 2> StubResourceHandleOp::attribute_descs;

constexpr std::array<ArgumentDesc, TensorListReserve::input_arg_count + TensorListReserve::output_arg_count> TensorListReserve::argument_descs;
constexpr std::array<AttributeDesc, 2> TensorListReserve::attribute_descs;

constexpr std::array<ArgumentDesc, _MklReshape::input_arg_count + _MklReshape::output_arg_count> _MklReshape::argument_descs;
constexpr std::array<AttributeDesc, 2> _MklReshape::attribute_descs;

constexpr std::array<ArgumentDesc, ResourceCreateOp::input_arg_count + ResourceCreateOp::output_arg_count> ResourceCreateOp::argument_descs;
constexpr std::array<AttributeDesc, 0> ResourceCreateOp::attribute_descs;

constexpr std::array<ArgumentDesc, CollectiveGather::input_arg_count + CollectiveGather::output_arg_count> CollectiveGather::argument_descs;
constexpr std::array<AttributeDesc, 7> CollectiveGather::attribute_descs;

constexpr std::array<ArgumentDesc, _XlaHostComputeMlir::input_arg_count + _XlaHostComputeMlir::output_arg_count> _XlaHostComputeMlir::argument_descs;
constexpr std::array<AttributeDesc, 6> _XlaHostComputeMlir::attribute_descs;

constexpr std::array<ArgumentDesc, WriteAudioSummary::input_arg_count + WriteAudioSummary::output_arg_count> WriteAudioSummary::argument_descs;
constexpr std::array<AttributeDesc, 1> WriteAudioSummary::attribute_descs;

constexpr std::array<ArgumentDesc, SetSize::input_arg_count + SetSize::output_arg_count> SetSize::argument_descs;
constexpr std::array<AttributeDesc, 2> SetSize::attribute_descs;

constexpr std::array<ArgumentDesc, _Recv::input_arg_count + _Recv::output_arg_count> _Recv::argument_descs;
constexpr std::array<AttributeDesc, 6> _Recv::attribute_descs;

constexpr std::array<ArgumentDesc, RandomShuffleQueueV2::input_arg_count + RandomShuffleQueueV2::output_arg_count> RandomShuffleQueueV2::argument_descs;
constexpr std::array<AttributeDesc, 8> RandomShuffleQueueV2::attribute_descs;

constexpr std::array<ArgumentDesc, NInTwice::input_arg_count + NInTwice::output_arg_count> NInTwice::argument_descs;
constexpr std::array<AttributeDesc, 1> NInTwice::attribute_descs;

constexpr std::array<ArgumentDesc, ResourceInitializedOp::input_arg_count + ResourceInitializedOp::output_arg_count> ResourceInitializedOp::argument_descs;
constexpr std::array<AttributeDesc, 0> ResourceInitializedOp::attribute_descs;

constexpr std::array<ArgumentDesc, ExperimentalLatencyStatsDataset::input_arg_count + ExperimentalLatencyStatsDataset::output_arg_count> ExperimentalLatencyStatsDataset::argument_descs;
constexpr std::array<AttributeDesc, 2> ExperimentalLatencyStatsDataset::attribute_descs;

constexpr std::array<ArgumentDesc, RiscSlice::input_arg_count + RiscSlice::output_arg_count> RiscSlice::argument_descs;
constexpr std::array<AttributeDesc, 2> RiscSlice::attribute_descs;

constexpr std::array<ArgumentDesc, _SwitchN::input_arg_count + _SwitchN::output_arg_count> _SwitchN::argument_descs;
constexpr std::array<AttributeDesc, 2> _SwitchN::attribute_descs;

constexpr std::array<ArgumentDesc, WriteScalarSummary::input_arg_count + WriteScalarSummary::output_arg_count> WriteScalarSummary::argument_descs;
constexpr std::array<AttributeDesc, 1> WriteScalarSummary::attribute_descs;

constexpr std::array<ArgumentDesc, SparseDenseCwiseAdd::input_arg_count + SparseDenseCwiseAdd::output_arg_count> SparseDenseCwiseAdd::argument_descs;
constexpr std::array<AttributeDesc, 1> SparseDenseCwiseAdd::attribute_descs;

constexpr std::array<ArgumentDesc, TwoIntOutputs::input_arg_count + TwoIntOutputs::output_arg_count> TwoIntOutputs::argument_descs;
constexpr std::array<AttributeDesc, 0> TwoIntOutputs::attribute_descs;

constexpr std::array<ArgumentDesc, Div::input_arg_count + Div::output_arg_count> Div::argument_descs;
constexpr std::array<AttributeDesc, 1> Div::attribute_descs;

constexpr std::array<ArgumentDesc, Conv3D::input_arg_count + Conv3D::output_arg_count> Conv3D::argument_descs;
constexpr std::array<AttributeDesc, 5> Conv3D::attribute_descs;

constexpr std::array<ArgumentDesc, Fill::input_arg_count + Fill::output_arg_count> Fill::argument_descs;
constexpr std::array<AttributeDesc, 2> Fill::attribute_descs;

constexpr std::array<ArgumentDesc, ResourceUsingOp::input_arg_count + ResourceUsingOp::output_arg_count> ResourceUsingOp::argument_descs;
constexpr std::array<AttributeDesc, 0> ResourceUsingOp::attribute_descs;

constexpr std::array<ArgumentDesc, StatelessRandomGammaV2::input_arg_count + StatelessRandomGammaV2::output_arg_count> StatelessRandomGammaV2::argument_descs;
constexpr std::array<AttributeDesc, 3> StatelessRandomGammaV2::attribute_descs;

constexpr std::array<ArgumentDesc, RefInputIntInput::input_arg_count + RefInputIntInput::output_arg_count> RefInputIntInput::argument_descs;
constexpr std::array<AttributeDesc, 0> RefInputIntInput::attribute_descs;

constexpr std::array<ArgumentDesc, EncodeJpeg::input_arg_count + EncodeJpeg::output_arg_count> EncodeJpeg::argument_descs;
constexpr std::array<AttributeDesc, 9> EncodeJpeg::attribute_descs;

constexpr std::array<ArgumentDesc, NonMaxSuppressionWithOverlaps::input_arg_count + NonMaxSuppressionWithOverlaps::output_arg_count> NonMaxSuppressionWithOverlaps::argument_descs;
constexpr std::array<AttributeDesc, 0> NonMaxSuppressionWithOverlaps::attribute_descs;

constexpr std::array<ArgumentDesc, BesselJ0::input_arg_count + BesselJ0::output_arg_count> BesselJ0::argument_descs;
constexpr std::array<AttributeDesc, 1> BesselJ0::attribute_descs;

constexpr std::array<ArgumentDesc, IsResourceHandleRefCounting::input_arg_count + IsResourceHandleRefCounting::output_arg_count> IsResourceHandleRefCounting::argument_descs;
constexpr std::array<AttributeDesc, 0> IsResourceHandleRefCounting::attribute_descs;

constexpr std::array<ArgumentDesc, Int64Output::input_arg_count + Int64Output::output_arg_count> Int64Output::argument_descs;
constexpr std::array<AttributeDesc, 0> Int64Output::attribute_descs;

constexpr std::array<ArgumentDesc, MakeWeakResourceHandle::input_arg_count + MakeWeakResourceHandle::output_arg_count> MakeWeakResourceHandle::argument_descs;
constexpr std::array<AttributeDesc, 0> MakeWeakResourceHandle::attribute_descs;

constexpr std::array<ArgumentDesc, QueueSize::input_arg_count + QueueSize::output_arg_count> QueueSize::argument_descs;
constexpr std::array<AttributeDesc, 0> QueueSize::attribute_descs;

constexpr std::array<ArgumentDesc, TestStringOutput::input_arg_count + TestStringOutput::output_arg_count> TestStringOutput::argument_descs;
constexpr std::array<AttributeDesc, 0> TestStringOutput::attribute_descs;

constexpr std::array<ArgumentDesc, Reciprocal::input_arg_count + Reciprocal::output_arg_count> Reciprocal::argument_descs;
constexpr std::array<AttributeDesc, 1> Reciprocal::attribute_descs;

constexpr std::array<ArgumentDesc, EditDistance::input_arg_count + EditDistance::output_arg_count> EditDistance::argument_descs;
constexpr std::array<AttributeDesc, 2> EditDistance::attribute_descs;

constexpr std::array<ArgumentDesc, Namespace_TestStringOutput::input_arg_count + Namespace_TestStringOutput::output_arg_count> Namespace_TestStringOutput::argument_descs;
constexpr std::array<AttributeDesc, 0> Namespace_TestStringOutput::attribute_descs;

constexpr std::array<ArgumentDesc, Foo1::input_arg_count + Foo1::output_arg_count> Foo1::argument_descs;
constexpr std::array<AttributeDesc, 0> Foo1::attribute_descs;

constexpr std::array<ArgumentDesc, TestAttr::input_arg_count + TestAttr::output_arg_count> TestAttr::argument_descs;
constexpr std::array<AttributeDesc, 1> TestAttr::attribute_descs;

constexpr std::array<ArgumentDesc, EncodePng::input_arg_count + EncodePng::output_arg_count> EncodePng::argument_descs;
constexpr std::array<AttributeDesc, 2> EncodePng::attribute_descs;

constexpr std::array<ArgumentDesc, ThreadPoolDataset::input_arg_count + ThreadPoolDataset::output_arg_count> ThreadPoolDataset::argument_descs;
constexpr std::array<AttributeDesc, 2> ThreadPoolDataset::attribute_descs;

constexpr std::array<ArgumentDesc, Softplus::input_arg_count + Softplus::output_arg_count> Softplus::argument_descs;
constexpr std::array<AttributeDesc, 1> Softplus::attribute_descs;

constexpr std::array<ArgumentDesc, StopGradient::input_arg_count + StopGradient::output_arg_count> StopGradient::argument_descs;
constexpr std::array<AttributeDesc, 1> StopGradient::attribute_descs;

constexpr std::array<ArgumentDesc, FiveFloatOutputs::input_arg_count + FiveFloatOutputs::output_arg_count> FiveFloatOutputs::argument_descs;
constexpr std::array<AttributeDesc, 0> FiveFloatOutputs::attribute_descs;

constexpr std::array<ArgumentDesc, B::input_arg_count + B::output_arg_count> B::argument_descs;
constexpr std::array<AttributeDesc, 0> B::attribute_descs;

constexpr std::array<ArgumentDesc, ApplyProximalGradientDescent::input_arg_count + ApplyProximalGradientDescent::output_arg_count> ApplyProximalGradientDescent::argument_descs;
constexpr std::array<AttributeDesc, 2> ApplyProximalGradientDescent::attribute_descs;

constexpr std::array<ArgumentDesc, ParseTensor::input_arg_count + ParseTensor::output_arg_count> ParseTensor::argument_descs;
constexpr std::array<AttributeDesc, 1> ParseTensor::attribute_descs;

constexpr std::array<ArgumentDesc, PreventGradient::input_arg_count + PreventGradient::output_arg_count> PreventGradient::argument_descs;
constexpr std::array<AttributeDesc, 2> PreventGradient::attribute_descs;

constexpr std::array<ArgumentDesc, Bucketize::input_arg_count + Bucketize::output_arg_count> Bucketize::argument_descs;
constexpr std::array<AttributeDesc, 2> Bucketize::attribute_descs;

constexpr std::array<ArgumentDesc, DynamicStitch::input_arg_count + DynamicStitch::output_arg_count> DynamicStitch::argument_descs;
constexpr std::array<AttributeDesc, 2> DynamicStitch::attribute_descs;

constexpr std::array<ArgumentDesc, Foo2::input_arg_count + Foo2::output_arg_count> Foo2::argument_descs;
constexpr std::array<AttributeDesc, 0> Foo2::attribute_descs;

constexpr std::array<ArgumentDesc, CopyOp::input_arg_count + CopyOp::output_arg_count> CopyOp::argument_descs;
constexpr std::array<AttributeDesc, 1> CopyOp::attribute_descs;

constexpr std::array<ArgumentDesc, FloatInput::input_arg_count + FloatInput::output_arg_count> FloatInput::argument_descs;
constexpr std::array<AttributeDesc, 0> FloatInput::attribute_descs;

constexpr std::array<ArgumentDesc, ConditionalAccumulator::input_arg_count + ConditionalAccumulator::output_arg_count> ConditionalAccumulator::argument_descs;
constexpr std::array<AttributeDesc, 5> ConditionalAccumulator::attribute_descs;

constexpr std::array<ArgumentDesc, XlaVariadicReduceV2::input_arg_count + XlaVariadicReduceV2::output_arg_count> XlaVariadicReduceV2::argument_descs;
constexpr std::array<AttributeDesc, 3> XlaVariadicReduceV2::attribute_descs;

constexpr std::array<ArgumentDesc, Erfinv::input_arg_count + Erfinv::output_arg_count> Erfinv::argument_descs;
constexpr std::array<AttributeDesc, 1> Erfinv::attribute_descs;

constexpr std::array<ArgumentDesc, XlaCustomCall::input_arg_count + XlaCustomCall::output_arg_count> XlaCustomCall::argument_descs;
constexpr std::array<AttributeDesc, 5> XlaCustomCall::attribute_descs;

constexpr std::array<ArgumentDesc, QuantizedConv2DWithBias::input_arg_count + QuantizedConv2DWithBias::output_arg_count> QuantizedConv2DWithBias::argument_descs;
constexpr std::array<AttributeDesc, 7> QuantizedConv2DWithBias::attribute_descs;

constexpr std::array<ArgumentDesc, None::input_arg_count + None::output_arg_count> None::argument_descs;
constexpr std::array<AttributeDesc, 0> None::attribute_descs;

constexpr std::array<ArgumentDesc, TwoFloatInputsFloatOutput::input_arg_count + TwoFloatInputsFloatOutput::output_arg_count> TwoFloatInputsFloatOutput::argument_descs;
constexpr std::array<AttributeDesc, 0> TwoFloatInputsFloatOutput::attribute_descs;

constexpr std::array<ArgumentDesc, CompositeTensorVariantToComponents::input_arg_count + CompositeTensorVariantToComponents::output_arg_count> CompositeTensorVariantToComponents::argument_descs;
constexpr std::array<AttributeDesc, 2> CompositeTensorVariantToComponents::attribute_descs;

constexpr std::array<ArgumentDesc, RiscCeil::input_arg_count + RiscCeil::output_arg_count> RiscCeil::argument_descs;
constexpr std::array<AttributeDesc, 1> RiscCeil::attribute_descs;

constexpr std::array<ArgumentDesc, MatrixDiagV2::input_arg_count + MatrixDiagV2::output_arg_count> MatrixDiagV2::argument_descs;
constexpr std::array<AttributeDesc, 1> MatrixDiagV2::attribute_descs;

constexpr std::array<ArgumentDesc, RiscCondition::input_arg_count + RiscCondition::output_arg_count> RiscCondition::argument_descs;
constexpr std::array<AttributeDesc, 4> RiscCondition::attribute_descs;

constexpr std::array<ArgumentDesc, IntOutput::input_arg_count + IntOutput::output_arg_count> IntOutput::argument_descs;
constexpr std::array<AttributeDesc, 0> IntOutput::attribute_descs;

constexpr std::array<ArgumentDesc, CudnnRNNV3::input_arg_count + CudnnRNNV3::output_arg_count> CudnnRNNV3::argument_descs;
constexpr std::array<AttributeDesc, 10> CudnnRNNV3::attribute_descs;

constexpr std::array<ArgumentDesc, PolymorphicDefaultOut::input_arg_count + PolymorphicDefaultOut::output_arg_count> PolymorphicDefaultOut::argument_descs;
constexpr std::array<AttributeDesc, 1> PolymorphicDefaultOut::attribute_descs;

constexpr std::array<ArgumentDesc, All::input_arg_count + All::output_arg_count> All::argument_descs;
constexpr std::array<AttributeDesc, 2> All::attribute_descs;

constexpr std::array<ArgumentDesc, QuantizedBatchNormWithGlobalNormalization::input_arg_count + QuantizedBatchNormWithGlobalNormalization::output_arg_count> QuantizedBatchNormWithGlobalNormalization::argument_descs;
constexpr std::array<AttributeDesc, 4> QuantizedBatchNormWithGlobalNormalization::attribute_descs;

constexpr std::array<ArgumentDesc, CudnnRNNCanonicalToParamsV2::input_arg_count + CudnnRNNCanonicalToParamsV2::output_arg_count> CudnnRNNCanonicalToParamsV2::argument_descs;
constexpr std::array<AttributeDesc, 10> CudnnRNNCanonicalToParamsV2::attribute_descs;

constexpr std::array<ArgumentDesc, IntInputFloatInput::input_arg_count + IntInputFloatInput::output_arg_count> IntInputFloatInput::argument_descs;
constexpr std::array<AttributeDesc, 0> IntInputFloatInput::attribute_descs;

constexpr std::array<ArgumentDesc, FakeQuantWithMinMaxArgs::input_arg_count + FakeQuantWithMinMaxArgs::output_arg_count> FakeQuantWithMinMaxArgs::argument_descs;
constexpr std::array<AttributeDesc, 4> FakeQuantWithMinMaxArgs::attribute_descs;

constexpr std::array<ArgumentDesc, SampleDistortedBoundingBox::input_arg_count + SampleDistortedBoundingBox::output_arg_count> SampleDistortedBoundingBox::argument_descs;
constexpr std::array<AttributeDesc, 8> SampleDistortedBoundingBox::attribute_descs;

constexpr std::array<ArgumentDesc, FlatMapDataset::input_arg_count + FlatMapDataset::output_arg_count> FlatMapDataset::argument_descs;
constexpr std::array<AttributeDesc, 5> FlatMapDataset::attribute_descs;

constexpr std::array<ArgumentDesc, FloatOutput::input_arg_count + FloatOutput::output_arg_count> FloatOutput::argument_descs;
constexpr std::array<AttributeDesc, 0> FloatOutput::attribute_descs;

constexpr std::array<ArgumentDesc, TwoFloatOutputs::input_arg_count + TwoFloatOutputs::output_arg_count> TwoFloatOutputs::argument_descs;
constexpr std::array<AttributeDesc, 0> TwoFloatOutputs::attribute_descs;

constexpr std::array<ArgumentDesc, RefOutputFloatOutput::input_arg_count + RefOutputFloatOutput::output_arg_count> RefOutputFloatOutput::argument_descs;
constexpr std::array<AttributeDesc, 0> RefOutputFloatOutput::attribute_descs;

constexpr std::array<ArgumentDesc, Pad::input_arg_count + Pad::output_arg_count> Pad::argument_descs;
constexpr std::array<AttributeDesc, 2> Pad::attribute_descs;

constexpr std::array<ArgumentDesc, XlaRecvFromHost::input_arg_count + XlaRecvFromHost::output_arg_count> XlaRecvFromHost::argument_descs;
constexpr std::array<AttributeDesc, 3> XlaRecvFromHost::attribute_descs;

constexpr std::array<ArgumentDesc, DefaultAttrs::input_arg_count + DefaultAttrs::output_arg_count> DefaultAttrs::argument_descs;
constexpr std::array<AttributeDesc, 14> DefaultAttrs::attribute_descs;

constexpr std::array<ArgumentDesc, DummyIterationCounter::input_arg_count + DummyIterationCounter::output_arg_count> DummyIterationCounter::argument_descs;
constexpr std::array<AttributeDesc, 0> DummyIterationCounter::attribute_descs;

constexpr std::array<ArgumentDesc, RefInputFloatInput::input_arg_count + RefInputFloatInput::output_arg_count> RefInputFloatInput::argument_descs;
constexpr std::array<AttributeDesc, 0> RefInputFloatInput::attribute_descs;

constexpr std::array<ArgumentDesc, FloorDiv::input_arg_count + FloorDiv::output_arg_count> FloorDiv::argument_descs;
constexpr std::array<AttributeDesc, 1> FloorDiv::attribute_descs;

constexpr std::array<ArgumentDesc, Igammac::input_arg_count + Igammac::output_arg_count> Igammac::argument_descs;
constexpr std::array<AttributeDesc, 1> Igammac::attribute_descs;

constexpr std::array<ArgumentDesc, CreateSummaryDbWriter::input_arg_count + CreateSummaryDbWriter::output_arg_count> CreateSummaryDbWriter::argument_descs;
constexpr std::array<AttributeDesc, 0> CreateSummaryDbWriter::attribute_descs;

constexpr std::array<ArgumentDesc, SkipDataset::input_arg_count + SkipDataset::output_arg_count> SkipDataset::argument_descs;
constexpr std::array<AttributeDesc, 3> SkipDataset::attribute_descs;

constexpr std::array<ArgumentDesc, RiscReal::input_arg_count + RiscReal::output_arg_count> RiscReal::argument_descs;
constexpr std::array<AttributeDesc, 2> RiscReal::attribute_descs;

constexpr std::array<ArgumentDesc, IntInput::input_arg_count + IntInput::output_arg_count> IntInput::argument_descs;
constexpr std::array<AttributeDesc, 0> IntInput::attribute_descs;

constexpr std::array<ArgumentDesc, CudnnRNNBackpropV3::input_arg_count + CudnnRNNBackpropV3::output_arg_count> CudnnRNNBackpropV3::argument_descs;
constexpr std::array<AttributeDesc, 9> CudnnRNNBackpropV3::attribute_descs;

constexpr std::array<ArgumentDesc, OrderedMapStage::input_arg_count + OrderedMapStage::output_arg_count> OrderedMapStage::argument_descs;
constexpr std::array<AttributeDesc, 6> OrderedMapStage::attribute_descs;

constexpr std::array<ArgumentDesc, MatrixDiagPart::input_arg_count + MatrixDiagPart::output_arg_count> MatrixDiagPart::argument_descs;
constexpr std::array<AttributeDesc, 1> MatrixDiagPart::attribute_descs;

constexpr std::array<ArgumentDesc, IntOutputFloatOutput::input_arg_count + IntOutputFloatOutput::output_arg_count> IntOutputFloatOutput::argument_descs;
constexpr std::array<AttributeDesc, 0> IntOutputFloatOutput::attribute_descs;

constexpr std::array<ArgumentDesc, ChooseFastestDataset::input_arg_count + ChooseFastestDataset::output_arg_count> ChooseFastestDataset::argument_descs;
constexpr std::array<AttributeDesc, 4> ChooseFastestDataset::attribute_descs;

constexpr std::array<ArgumentDesc, TwoIntInputs::input_arg_count + TwoIntInputs::output_arg_count> TwoIntInputs::argument_descs;
constexpr std::array<AttributeDesc, 0> TwoIntInputs::attribute_descs;

constexpr std::array<ArgumentDesc, IteratorGetDevice::input_arg_count + IteratorGetDevice::output_arg_count> IteratorGetDevice::argument_descs;
constexpr std::array<AttributeDesc, 0> IteratorGetDevice::attribute_descs;

constexpr std::array<ArgumentDesc, CrossReplicaSum::input_arg_count + CrossReplicaSum::output_arg_count> CrossReplicaSum::argument_descs;
constexpr std::array<AttributeDesc, 1> CrossReplicaSum::attribute_descs;

constexpr std::array<ArgumentDesc, Size::input_arg_count + Size::output_arg_count> Size::argument_descs;
constexpr std::array<AttributeDesc, 2> Size::attribute_descs;

constexpr std::array<ArgumentDesc, TwoFloatInputs::input_arg_count + TwoFloatInputs::output_arg_count> TwoFloatInputs::argument_descs;
constexpr std::array<AttributeDesc, 0> TwoFloatInputs::attribute_descs;

constexpr std::array<ArgumentDesc, _MklAvgPoolGrad::input_arg_count + _MklAvgPoolGrad::output_arg_count> _MklAvgPoolGrad::argument_descs;
constexpr std::array<AttributeDesc, 5> _MklAvgPoolGrad::attribute_descs;

constexpr std::array<ArgumentDesc, Where::input_arg_count + Where::output_arg_count> Where::argument_descs;
constexpr std::array<AttributeDesc, 1> Where::attribute_descs;

constexpr std::array<ArgumentDesc, RefInputFloatInputIntOutput::input_arg_count + RefInputFloatInputIntOutput::output_arg_count> RefInputFloatInputIntOutput::argument_descs;
constexpr std::array<AttributeDesc, 0> RefInputFloatInputIntOutput::attribute_descs;

constexpr std::array<ArgumentDesc, ApplyFtrl::input_arg_count + ApplyFtrl::output_arg_count> ApplyFtrl::argument_descs;
constexpr std::array<AttributeDesc, 3> ApplyFtrl::attribute_descs;

constexpr std::array<ArgumentDesc, PaddingFIFOQueue::input_arg_count + PaddingFIFOQueue::output_arg_count> PaddingFIFOQueue::argument_descs;
constexpr std::array<AttributeDesc, 5> PaddingFIFOQueue::attribute_descs;

constexpr std::array<ArgumentDesc, ListInput::input_arg_count + ListInput::output_arg_count> ListInput::argument_descs;
constexpr std::array<AttributeDesc, 2> ListInput::attribute_descs;

constexpr std::array<ArgumentDesc, _MklQuantizedDepthwiseConv2D::input_arg_count + _MklQuantizedDepthwiseConv2D::output_arg_count> _MklQuantizedDepthwiseConv2D::argument_descs;
constexpr std::array<AttributeDesc, 8> _MklQuantizedDepthwiseConv2D::attribute_descs;

constexpr std::array<ArgumentDesc, ListOutput::input_arg_count + ListOutput::output_arg_count> ListOutput::argument_descs;
constexpr std::array<AttributeDesc, 1> ListOutput::attribute_descs;

constexpr std::array<ArgumentDesc, ExperimentalMaxIntraOpParallelismDataset::input_arg_count + ExperimentalMaxIntraOpParallelismDataset::output_arg_count> ExperimentalMaxIntraOpParallelismDataset::argument_descs;
constexpr std::array<AttributeDesc, 2> ExperimentalMaxIntraOpParallelismDataset::attribute_descs;

constexpr std::array<ArgumentDesc, Unary::input_arg_count + Unary::output_arg_count> Unary::argument_descs;
constexpr std::array<AttributeDesc, 1> Unary::attribute_descs;

constexpr std::array<ArgumentDesc, OpWithDefaultAttr::input_arg_count + OpWithDefaultAttr::output_arg_count> OpWithDefaultAttr::argument_descs;
constexpr std::array<AttributeDesc, 1> OpWithDefaultAttr::attribute_descs;

constexpr std::array<ArgumentDesc, OpWithFutureDefaultAttr::input_arg_count + OpWithFutureDefaultAttr::output_arg_count> OpWithFutureDefaultAttr::argument_descs;
constexpr std::array<AttributeDesc, 0> OpWithFutureDefaultAttr::attribute_descs;

constexpr std::array<ArgumentDesc, Split::input_arg_count + Split::output_arg_count> Split::argument_descs;
constexpr std::array<AttributeDesc, 2> Split::attribute_descs;

constexpr std::array<ArgumentDesc, StringListAttr::input_arg_count + StringListAttr::output_arg_count> StringListAttr::argument_descs;
constexpr std::array<AttributeDesc, 2> StringListAttr::attribute_descs;

constexpr std::array<ArgumentDesc, StatelessWhile::input_arg_count + StatelessWhile::output_arg_count> StatelessWhile::argument_descs;
constexpr std::array<AttributeDesc, 5> StatelessWhile::attribute_descs;

constexpr std::array<ArgumentDesc, FuncAttr::input_arg_count + FuncAttr::output_arg_count> FuncAttr::argument_descs;
constexpr std::array<AttributeDesc, 1> FuncAttr::attribute_descs;

constexpr std::array<ArgumentDesc, XlaSetBound::input_arg_count + XlaSetBound::output_arg_count> XlaSetBound::argument_descs;
constexpr std::array<AttributeDesc, 0> XlaSetBound::attribute_descs;

constexpr std::array<ArgumentDesc, FuncListAttr::input_arg_count + FuncListAttr::output_arg_count> FuncListAttr::argument_descs;
constexpr std::array<AttributeDesc, 1> FuncListAttr::attribute_descs;

constexpr std::array<ArgumentDesc, LMDBDataset::input_arg_count + LMDBDataset::output_arg_count> LMDBDataset::argument_descs;
constexpr std::array<AttributeDesc, 2> LMDBDataset::attribute_descs;

constexpr std::array<ArgumentDesc, NPolymorphicRestrictIn::input_arg_count + NPolymorphicRestrictIn::output_arg_count> NPolymorphicRestrictIn::argument_descs;
constexpr std::array<AttributeDesc, 2> NPolymorphicRestrictIn::attribute_descs;

constexpr std::array<ArgumentDesc, Simple::input_arg_count + Simple::output_arg_count> Simple::argument_descs;
constexpr std::array<AttributeDesc, 0> Simple::attribute_descs;

constexpr std::array<ArgumentDesc, MulNoNan::input_arg_count + MulNoNan::output_arg_count> MulNoNan::argument_descs;
constexpr std::array<AttributeDesc, 1> MulNoNan::attribute_descs;

constexpr std::array<ArgumentDesc, DecodeProtoV2::input_arg_count + DecodeProtoV2::output_arg_count> DecodeProtoV2::argument_descs;
constexpr std::array<AttributeDesc, 6> DecodeProtoV2::attribute_descs;

constexpr std::array<ArgumentDesc, OutT::input_arg_count + OutT::output_arg_count> OutT::argument_descs;
constexpr std::array<AttributeDesc, 1> OutT::attribute_descs;

constexpr std::array<ArgumentDesc, FilterByLastComponentDataset::input_arg_count + FilterByLastComponentDataset::output_arg_count> FilterByLastComponentDataset::argument_descs;
constexpr std::array<AttributeDesc, 2> FilterByLastComponentDataset::attribute_descs;

constexpr std::array<ArgumentDesc, XlaDotV2::input_arg_count + XlaDotV2::output_arg_count> XlaDotV2::argument_descs;
constexpr std::array<AttributeDesc, 5> XlaDotV2::attribute_descs;

constexpr std::array<ArgumentDesc, OutfeedDequeueV2::input_arg_count + OutfeedDequeueV2::output_arg_count> OutfeedDequeueV2::argument_descs;
constexpr std::array<AttributeDesc, 2> OutfeedDequeueV2::attribute_descs;

constexpr std::array<ArgumentDesc, ReservedInput::input_arg_count + ReservedInput::output_arg_count> ReservedInput::argument_descs;
constexpr std::array<AttributeDesc, 0> ReservedInput::attribute_descs;

constexpr std::array<ArgumentDesc, Polymorphic::input_arg_count + Polymorphic::output_arg_count> Polymorphic::argument_descs;
constexpr std::array<AttributeDesc, 1> Polymorphic::attribute_descs;

constexpr std::array<ArgumentDesc, _XlaSendFromHost::input_arg_count + _XlaSendFromHost::output_arg_count> _XlaSendFromHost::argument_descs;
constexpr std::array<AttributeDesc, 3> _XlaSendFromHost::attribute_descs;

constexpr std::array<ArgumentDesc, Tile::input_arg_count + Tile::output_arg_count> Tile::argument_descs;
constexpr std::array<AttributeDesc, 2> Tile::attribute_descs;

constexpr std::array<ArgumentDesc, AssignVariableXlaConcatND::input_arg_count + AssignVariableXlaConcatND::output_arg_count> AssignVariableXlaConcatND::argument_descs;
constexpr std::array<AttributeDesc, 4> AssignVariableXlaConcatND::attribute_descs;

constexpr std::array<ArgumentDesc, PolymorphicOut::input_arg_count + PolymorphicOut::output_arg_count> PolymorphicOut::argument_descs;
constexpr std::array<AttributeDesc, 1> PolymorphicOut::attribute_descs;

constexpr std::array<ArgumentDesc, CollectiveBcastSend::input_arg_count + CollectiveBcastSend::output_arg_count> CollectiveBcastSend::argument_descs;
constexpr std::array<AttributeDesc, 7> CollectiveBcastSend::attribute_descs;

constexpr std::array<ArgumentDesc, _MklSlice::input_arg_count + _MklSlice::output_arg_count> _MklSlice::argument_descs;
constexpr std::array<AttributeDesc, 2> _MklSlice::attribute_descs;

constexpr std::array<ArgumentDesc, TPUReshardVariables::input_arg_count + TPUReshardVariables::output_arg_count> TPUReshardVariables::argument_descs;
constexpr std::array<AttributeDesc, 1> TPUReshardVariables::attribute_descs;

constexpr std::array<ArgumentDesc, Binary::input_arg_count + Binary::output_arg_count> Binary::argument_descs;
constexpr std::array<AttributeDesc, 1> Binary::attribute_descs;

constexpr std::array<ArgumentDesc, XlaWhile::input_arg_count + XlaWhile::output_arg_count> XlaWhile::argument_descs;
constexpr std::array<AttributeDesc, 3> XlaWhile::attribute_descs;

constexpr std::array<ArgumentDesc, BatchDatasetV2::input_arg_count + BatchDatasetV2::output_arg_count> BatchDatasetV2::argument_descs;
constexpr std::array<AttributeDesc, 4> BatchDatasetV2::attribute_descs;

constexpr std::array<ArgumentDesc, Restrict::input_arg_count + Restrict::output_arg_count> Restrict::argument_descs;
constexpr std::array<AttributeDesc, 1> Restrict::attribute_descs;

constexpr std::array<ArgumentDesc, DatasetCardinality::input_arg_count + DatasetCardinality::output_arg_count> DatasetCardinality::argument_descs;
constexpr std::array<AttributeDesc, 0> DatasetCardinality::attribute_descs;

constexpr std::array<ArgumentDesc, CholeskyGrad::input_arg_count + CholeskyGrad::output_arg_count> CholeskyGrad::argument_descs;
constexpr std::array<AttributeDesc, 1> CholeskyGrad::attribute_descs;

constexpr std::array<ArgumentDesc, TypeList::input_arg_count + TypeList::output_arg_count> TypeList::argument_descs;
constexpr std::array<AttributeDesc, 1> TypeList::attribute_descs;

constexpr std::array<ArgumentDesc, TypeListTwice::input_arg_count + TypeListTwice::output_arg_count> TypeListTwice::argument_descs;
constexpr std::array<AttributeDesc, 1> TypeListTwice::attribute_descs;

constexpr std::array<ArgumentDesc, AudioSummary::input_arg_count + AudioSummary::output_arg_count> AudioSummary::argument_descs;
constexpr std::array<AttributeDesc, 2> AudioSummary::attribute_descs;

constexpr std::array<ArgumentDesc, OutTypeList::input_arg_count + OutTypeList::output_arg_count> OutTypeList::argument_descs;
constexpr std::array<AttributeDesc, 1> OutTypeList::attribute_descs;

constexpr std::array<ArgumentDesc, NonMaxSuppressionV2::input_arg_count + NonMaxSuppressionV2::output_arg_count> NonMaxSuppressionV2::argument_descs;
constexpr std::array<AttributeDesc, 2> NonMaxSuppressionV2::attribute_descs;

constexpr std::array<ArgumentDesc, Slice::input_arg_count + Slice::output_arg_count> Slice::argument_descs;
constexpr std::array<AttributeDesc, 2> Slice::attribute_descs;

constexpr std::array<ArgumentDesc, TypeListRestrict::input_arg_count + TypeListRestrict::output_arg_count> TypeListRestrict::argument_descs;
constexpr std::array<AttributeDesc, 1> TypeListRestrict::attribute_descs;

constexpr std::array<ArgumentDesc, _RecvTPUEmbeddingDeduplicationData::input_arg_count + _RecvTPUEmbeddingDeduplicationData::output_arg_count> _RecvTPUEmbeddingDeduplicationData::argument_descs;
constexpr std::array<AttributeDesc, 1> _RecvTPUEmbeddingDeduplicationData::attribute_descs;

constexpr std::array<ArgumentDesc, OutTypeListRestrict::input_arg_count + OutTypeListRestrict::output_arg_count> OutTypeListRestrict::argument_descs;
constexpr std::array<AttributeDesc, 1> OutTypeListRestrict::attribute_descs;

constexpr std::array<ArgumentDesc, TensorArrayClose::input_arg_count + TensorArrayClose::output_arg_count> TensorArrayClose::argument_descs;
constexpr std::array<AttributeDesc, 0> TensorArrayClose::attribute_descs;

constexpr std::array<ArgumentDesc, StridedSliceAssign::input_arg_count + StridedSliceAssign::output_arg_count> StridedSliceAssign::argument_descs;
constexpr std::array<AttributeDesc, 7> StridedSliceAssign::attribute_descs;

constexpr std::array<ArgumentDesc, Min::input_arg_count + Min::output_arg_count> Min::argument_descs;
constexpr std::array<AttributeDesc, 3> Min::attribute_descs;

constexpr std::array<ArgumentDesc, UnwrapDatasetVariant::input_arg_count + UnwrapDatasetVariant::output_arg_count> UnwrapDatasetVariant::argument_descs;
constexpr std::array<AttributeDesc, 0> UnwrapDatasetVariant::attribute_descs;

constexpr std::array<ArgumentDesc, InfeedDequeueTuple::input_arg_count + InfeedDequeueTuple::output_arg_count> InfeedDequeueTuple::argument_descs;
constexpr std::array<AttributeDesc, 2> InfeedDequeueTuple::attribute_descs;

constexpr std::array<ArgumentDesc, StackPop::input_arg_count + StackPop::output_arg_count> StackPop::argument_descs;
constexpr std::array<AttributeDesc, 1> StackPop::attribute_descs;

constexpr std::array<ArgumentDesc, FusedBatchNormV2::input_arg_count + FusedBatchNormV2::output_arg_count> FusedBatchNormV2::argument_descs;
constexpr std::array<AttributeDesc, 6> FusedBatchNormV2::attribute_descs;

constexpr std::array<ArgumentDesc, CudnnRNNParamsToCanonicalV2::input_arg_count + CudnnRNNParamsToCanonicalV2::output_arg_count> CudnnRNNParamsToCanonicalV2::argument_descs;
constexpr std::array<AttributeDesc, 10> CudnnRNNParamsToCanonicalV2::attribute_descs;

constexpr std::array<ArgumentDesc, ParallelConcat::input_arg_count + ParallelConcat::output_arg_count> ParallelConcat::argument_descs;
constexpr std::array<AttributeDesc, 3> ParallelConcat::attribute_descs;

constexpr std::array<ArgumentDesc, Attr::input_arg_count + Attr::output_arg_count> Attr::argument_descs;
constexpr std::array<AttributeDesc, 1> Attr::attribute_descs;

constexpr std::array<ArgumentDesc, GatherV2::input_arg_count + GatherV2::output_arg_count> GatherV2::argument_descs;
constexpr std::array<AttributeDesc, 4> GatherV2::attribute_descs;

constexpr std::array<ArgumentDesc, AttrFloat::input_arg_count + AttrFloat::output_arg_count> AttrFloat::argument_descs;
constexpr std::array<AttributeDesc, 1> AttrFloat::attribute_descs;

constexpr std::array<ArgumentDesc, CreateSummaryFileWriter::input_arg_count + CreateSummaryFileWriter::output_arg_count> CreateSummaryFileWriter::argument_descs;
constexpr std::array<AttributeDesc, 0> CreateSummaryFileWriter::attribute_descs;

constexpr std::array<ArgumentDesc, StageClear::input_arg_count + StageClear::output_arg_count> StageClear::argument_descs;
constexpr std::array<AttributeDesc, 5> StageClear::attribute_descs;

constexpr std::array<ArgumentDesc, Asinh::input_arg_count + Asinh::output_arg_count> Asinh::argument_descs;
constexpr std::array<AttributeDesc, 1> Asinh::attribute_descs;

constexpr std::array<ArgumentDesc, ParseExampleV2::input_arg_count + ParseExampleV2::output_arg_count> ParseExampleV2::argument_descs;
constexpr std::array<AttributeDesc, 6> ParseExampleV2::attribute_descs;

constexpr std::array<ArgumentDesc, MapPeek::input_arg_count + MapPeek::output_arg_count> MapPeek::argument_descs;
constexpr std::array<AttributeDesc, 5> MapPeek::attribute_descs;

constexpr std::array<ArgumentDesc, AttrBoolList::input_arg_count + AttrBoolList::output_arg_count> AttrBoolList::argument_descs;
constexpr std::array<AttributeDesc, 1> AttrBoolList::attribute_descs;

constexpr std::array<ArgumentDesc, TensorScatterMin::input_arg_count + TensorScatterMin::output_arg_count> TensorScatterMin::argument_descs;
constexpr std::array<AttributeDesc, 2> TensorScatterMin::attribute_descs;

constexpr std::array<ArgumentDesc, AttrMin::input_arg_count + AttrMin::output_arg_count> AttrMin::argument_descs;
constexpr std::array<AttributeDesc, 1> AttrMin::attribute_descs;

constexpr std::array<ArgumentDesc, DebugIdentity::input_arg_count + DebugIdentity::output_arg_count> DebugIdentity::argument_descs;
constexpr std::array<AttributeDesc, 5> DebugIdentity::attribute_descs;

constexpr std::array<ArgumentDesc, ApplyAdagradDA::input_arg_count + ApplyAdagradDA::output_arg_count> ApplyAdagradDA::argument_descs;
constexpr std::array<AttributeDesc, 2> ApplyAdagradDA::attribute_descs;

constexpr std::array<ArgumentDesc, AttrListMin::input_arg_count + AttrListMin::output_arg_count> AttrListMin::argument_descs;
constexpr std::array<AttributeDesc, 1> AttrListMin::attribute_descs;

constexpr std::array<ArgumentDesc, TensorListPushBack::input_arg_count + TensorListPushBack::output_arg_count> TensorListPushBack::argument_descs;
constexpr std::array<AttributeDesc, 1> TensorListPushBack::attribute_descs;

constexpr std::array<ArgumentDesc, ResourceAccumulatorApplyGradient::input_arg_count + ResourceAccumulatorApplyGradient::output_arg_count> ResourceAccumulatorApplyGradient::argument_descs;
constexpr std::array<AttributeDesc, 1> ResourceAccumulatorApplyGradient::attribute_descs;

constexpr std::array<ArgumentDesc, AttrEnum::input_arg_count + AttrEnum::output_arg_count> AttrEnum::argument_descs;
constexpr std::array<AttributeDesc, 1> AttrEnum::attribute_descs;

constexpr std::array<ArgumentDesc, WindowDataset::input_arg_count + WindowDataset::output_arg_count> WindowDataset::argument_descs;
constexpr std::array<AttributeDesc, 3> WindowDataset::attribute_descs;

constexpr std::array<ArgumentDesc, InitializeTableFromDataset::input_arg_count + InitializeTableFromDataset::output_arg_count> InitializeTableFromDataset::argument_descs;
constexpr std::array<AttributeDesc, 0> InitializeTableFromDataset::attribute_descs;

constexpr std::array<ArgumentDesc, SparseSegmentMean::input_arg_count + SparseSegmentMean::output_arg_count> SparseSegmentMean::argument_descs;
constexpr std::array<AttributeDesc, 3> SparseSegmentMean::attribute_descs;

constexpr std::array<ArgumentDesc, ResourceApplyAddSign::input_arg_count + ResourceApplyAddSign::output_arg_count> ResourceApplyAddSign::argument_descs;
constexpr std::array<AttributeDesc, 2> ResourceApplyAddSign::attribute_descs;

constexpr std::array<ArgumentDesc, AttrEnumList::input_arg_count + AttrEnumList::output_arg_count> AttrEnumList::argument_descs;
constexpr std::array<AttributeDesc, 1> AttrEnumList::attribute_descs;

constexpr std::array<ArgumentDesc, EncodeWav::input_arg_count + EncodeWav::output_arg_count> EncodeWav::argument_descs;
constexpr std::array<AttributeDesc, 0> EncodeWav::attribute_descs;

constexpr std::array<ArgumentDesc, AttrShape::input_arg_count + AttrShape::output_arg_count> AttrShape::argument_descs;
constexpr std::array<AttributeDesc, 1> AttrShape::attribute_descs;

constexpr std::array<ArgumentDesc, ArgMax::input_arg_count + ArgMax::output_arg_count> ArgMax::argument_descs;
constexpr std::array<AttributeDesc, 3> ArgMax::attribute_descs;

constexpr std::array<ArgumentDesc, _RecvTPUEmbeddingActivations::input_arg_count + _RecvTPUEmbeddingActivations::output_arg_count> _RecvTPUEmbeddingActivations::argument_descs;
constexpr std::array<AttributeDesc, 2> _RecvTPUEmbeddingActivations::attribute_descs;

constexpr std::array<ArgumentDesc, AttrShapeList::input_arg_count + AttrShapeList::output_arg_count> AttrShapeList::argument_descs;
constexpr std::array<AttributeDesc, 1> AttrShapeList::attribute_descs;

constexpr std::array<ArgumentDesc, ParallelMapDataset::input_arg_count + ParallelMapDataset::output_arg_count> ParallelMapDataset::argument_descs;
constexpr std::array<AttributeDesc, 8> ParallelMapDataset::attribute_descs;

constexpr std::array<ArgumentDesc, QueueClose::input_arg_count + QueueClose::output_arg_count> QueueClose::argument_descs;
constexpr std::array<AttributeDesc, 1> QueueClose::attribute_descs;

constexpr std::array<ArgumentDesc, TensorListPopBack::input_arg_count + TensorListPopBack::output_arg_count> TensorListPopBack::argument_descs;
constexpr std::array<AttributeDesc, 1> TensorListPopBack::attribute_descs;

constexpr std::array<ArgumentDesc, AttrPartialShape::input_arg_count + AttrPartialShape::output_arg_count> AttrPartialShape::argument_descs;
constexpr std::array<AttributeDesc, 1> AttrPartialShape::attribute_descs;

constexpr std::array<ArgumentDesc, LeftShift::input_arg_count + LeftShift::output_arg_count> LeftShift::argument_descs;
constexpr std::array<AttributeDesc, 1> LeftShift::attribute_descs;

constexpr std::array<ArgumentDesc, DecodeImage::input_arg_count + DecodeImage::output_arg_count> DecodeImage::argument_descs;
constexpr std::array<AttributeDesc, 3> DecodeImage::attribute_descs;

constexpr std::array<ArgumentDesc, _MklSquaredDifference::input_arg_count + _MklSquaredDifference::output_arg_count> _MklSquaredDifference::argument_descs;
constexpr std::array<AttributeDesc, 1> _MklSquaredDifference::attribute_descs;

constexpr std::array<ArgumentDesc, AttrPartialShapeList::input_arg_count + AttrPartialShapeList::output_arg_count> AttrPartialShapeList::argument_descs;
constexpr std::array<AttributeDesc, 1> AttrPartialShapeList::attribute_descs;

constexpr std::array<ArgumentDesc, AttrDefault::input_arg_count + AttrDefault::output_arg_count> AttrDefault::argument_descs;
constexpr std::array<AttributeDesc, 1> AttrDefault::attribute_descs;

constexpr std::array<ArgumentDesc, ShardedFilespec::input_arg_count + ShardedFilespec::output_arg_count> ShardedFilespec::argument_descs;
constexpr std::array<AttributeDesc, 0> ShardedFilespec::attribute_descs;

constexpr std::array<ArgumentDesc, AccumulatorApplyGradient::input_arg_count + AccumulatorApplyGradient::output_arg_count> AccumulatorApplyGradient::argument_descs;
constexpr std::array<AttributeDesc, 1> AccumulatorApplyGradient::attribute_descs;

constexpr std::array<ArgumentDesc, AttrListDefault::input_arg_count + AttrListDefault::output_arg_count> AttrListDefault::argument_descs;
constexpr std::array<AttributeDesc, 1> AttrListDefault::attribute_descs;

constexpr std::array<ArgumentDesc, BlockLSTMGradV2::input_arg_count + BlockLSTMGradV2::output_arg_count> BlockLSTMGradV2::argument_descs;
constexpr std::array<AttributeDesc, 2> BlockLSTMGradV2::attribute_descs;

constexpr std::array<ArgumentDesc, GuaranteeConst::input_arg_count + GuaranteeConst::output_arg_count> GuaranteeConst::argument_descs;
constexpr std::array<AttributeDesc, 1> GuaranteeConst::attribute_descs;

constexpr std::array<ArgumentDesc, MapIncompleteSize::input_arg_count + MapIncompleteSize::output_arg_count> MapIncompleteSize::argument_descs;
constexpr std::array<AttributeDesc, 5> MapIncompleteSize::attribute_descs;

constexpr std::array<ArgumentDesc, XlaRecv::input_arg_count + XlaRecv::output_arg_count> XlaRecv::argument_descs;
constexpr std::array<AttributeDesc, 3> XlaRecv::attribute_descs;

constexpr std::array<ArgumentDesc, Sum::input_arg_count + Sum::output_arg_count> Sum::argument_descs;
constexpr std::array<AttributeDesc, 3> Sum::attribute_descs;

constexpr std::array<ArgumentDesc, AttrEmptyListDefault::input_arg_count + AttrEmptyListDefault::output_arg_count> AttrEmptyListDefault::argument_descs;
constexpr std::array<AttributeDesc, 1> AttrEmptyListDefault::attribute_descs;

constexpr std::array<ArgumentDesc, ModelDataset::input_arg_count + ModelDataset::output_arg_count> ModelDataset::argument_descs;
constexpr std::array<AttributeDesc, 5> ModelDataset::attribute_descs;

constexpr std::array<ArgumentDesc, CollectiveGatherV2::input_arg_count + CollectiveGatherV2::output_arg_count> CollectiveGatherV2::argument_descs;
constexpr std::array<AttributeDesc, 4> CollectiveGatherV2::attribute_descs;

constexpr std::array<ArgumentDesc, ReservedAttr::input_arg_count + ReservedAttr::output_arg_count> ReservedAttr::argument_descs;
constexpr std::array<AttributeDesc, 1> ReservedAttr::attribute_descs;

constexpr std::array<ArgumentDesc, HashTableV2::input_arg_count + HashTableV2::output_arg_count> HashTableV2::argument_descs;
constexpr std::array<AttributeDesc, 5> HashTableV2::attribute_descs;

constexpr std::array<ArgumentDesc, CustomAggregator::input_arg_count + CustomAggregator::output_arg_count> CustomAggregator::argument_descs;
constexpr std::array<AttributeDesc, 1> CustomAggregator::attribute_descs;

constexpr std::array<ArgumentDesc, AttrTypeDefault::input_arg_count + AttrTypeDefault::output_arg_count> AttrTypeDefault::argument_descs;
constexpr std::array<AttributeDesc, 1> AttrTypeDefault::attribute_descs;

constexpr std::array<ArgumentDesc, Sqrt::input_arg_count + Sqrt::output_arg_count> Sqrt::argument_descs;
constexpr std::array<AttributeDesc, 1> Sqrt::attribute_descs;

constexpr std::array<ArgumentDesc, AttrListTypeDefault::input_arg_count + AttrListTypeDefault::output_arg_count> AttrListTypeDefault::argument_descs;
constexpr std::array<AttributeDesc, 2> AttrListTypeDefault::attribute_descs;

constexpr std::array<ArgumentDesc, TensorArrayCloseV3::input_arg_count + TensorArrayCloseV3::output_arg_count> TensorArrayCloseV3::argument_descs;
constexpr std::array<AttributeDesc, 0> TensorArrayCloseV3::attribute_descs;

constexpr std::array<ArgumentDesc, RealDiv::input_arg_count + RealDiv::output_arg_count> RealDiv::argument_descs;
constexpr std::array<AttributeDesc, 1> RealDiv::attribute_descs;

constexpr std::array<ArgumentDesc, NIntsIn::input_arg_count + NIntsIn::output_arg_count> NIntsIn::argument_descs;
constexpr std::array<AttributeDesc, 1> NIntsIn::attribute_descs;

constexpr std::array<ArgumentDesc, RegisterDataset::input_arg_count + RegisterDataset::output_arg_count> RegisterDataset::argument_descs;
constexpr std::array<AttributeDesc, 3> RegisterDataset::attribute_descs;

constexpr std::array<ArgumentDesc, Ceil::input_arg_count + Ceil::output_arg_count> Ceil::argument_descs;
constexpr std::array<AttributeDesc, 1> Ceil::attribute_descs;

constexpr std::array<ArgumentDesc, XlaOptimizationBarrier::input_arg_count + XlaOptimizationBarrier::output_arg_count> XlaOptimizationBarrier::argument_descs;
constexpr std::array<AttributeDesc, 1> XlaOptimizationBarrier::attribute_descs;

constexpr std::array<ArgumentDesc, EnqueueTPUEmbeddingSparseTensorBatch::input_arg_count + EnqueueTPUEmbeddingSparseTensorBatch::output_arg_count> EnqueueTPUEmbeddingSparseTensorBatch::argument_descs;
constexpr std::array<AttributeDesc, 9> EnqueueTPUEmbeddingSparseTensorBatch::attribute_descs;

constexpr std::array<ArgumentDesc, ResourceScatterAdd::input_arg_count + ResourceScatterAdd::output_arg_count> ResourceScatterAdd::argument_descs;
constexpr std::array<AttributeDesc, 2> ResourceScatterAdd::attribute_descs;

constexpr std::array<ArgumentDesc, ReverseSequence::input_arg_count + ReverseSequence::output_arg_count> ReverseSequence::argument_descs;
constexpr std::array<AttributeDesc, 4> ReverseSequence::attribute_descs;

constexpr std::array<ArgumentDesc, NPolymorphicIn::input_arg_count + NPolymorphicIn::output_arg_count> NPolymorphicIn::argument_descs;
constexpr std::array<AttributeDesc, 2> NPolymorphicIn::attribute_descs;

constexpr std::array<ArgumentDesc, NInPolymorphicTwice::input_arg_count + NInPolymorphicTwice::output_arg_count> NInPolymorphicTwice::argument_descs;
constexpr std::array<AttributeDesc, 2> NInPolymorphicTwice::attribute_descs;

constexpr std::array<ArgumentDesc, TensorArrayGradV3::input_arg_count + TensorArrayGradV3::output_arg_count> TensorArrayGradV3::argument_descs;
constexpr std::array<AttributeDesc, 1> TensorArrayGradV3::attribute_descs;

constexpr std::array<ArgumentDesc, _MklNativeFusedDepthwiseConv2dNative::input_arg_count + _MklNativeFusedDepthwiseConv2dNative::output_arg_count> _MklNativeFusedDepthwiseConv2dNative::argument_descs;
constexpr std::array<AttributeDesc, 10> _MklNativeFusedDepthwiseConv2dNative::attribute_descs;

constexpr std::array<ArgumentDesc, _MklConcatV2::input_arg_count + _MklConcatV2::output_arg_count> _MklConcatV2::argument_descs;
constexpr std::array<AttributeDesc, 3> _MklConcatV2::attribute_descs;

constexpr std::array<ArgumentDesc, NInTwoTypeVariables::input_arg_count + NInTwoTypeVariables::output_arg_count> NInTwoTypeVariables::argument_descs;
constexpr std::array<AttributeDesc, 3> NInTwoTypeVariables::attribute_descs;

constexpr std::array<ArgumentDesc, BatchSvd::input_arg_count + BatchSvd::output_arg_count> BatchSvd::argument_descs;
constexpr std::array<AttributeDesc, 3> BatchSvd::attribute_descs;

constexpr std::array<ArgumentDesc, IsNan::input_arg_count + IsNan::output_arg_count> IsNan::argument_descs;
constexpr std::array<AttributeDesc, 1> IsNan::attribute_descs;

constexpr std::array<ArgumentDesc, InPolymorphicTwice::input_arg_count + InPolymorphicTwice::output_arg_count> InPolymorphicTwice::argument_descs;
constexpr std::array<AttributeDesc, 3> InPolymorphicTwice::attribute_descs;

constexpr std::array<ArgumentDesc, NIntsOut::input_arg_count + NIntsOut::output_arg_count> NIntsOut::argument_descs;
constexpr std::array<AttributeDesc, 1> NIntsOut::attribute_descs;

constexpr std::array<ArgumentDesc, SparseSliceGrad::input_arg_count + SparseSliceGrad::output_arg_count> SparseSliceGrad::argument_descs;
constexpr std::array<AttributeDesc, 1> SparseSliceGrad::attribute_descs;

constexpr std::array<ArgumentDesc, Reshape::input_arg_count + Reshape::output_arg_count> Reshape::argument_descs;
constexpr std::array<AttributeDesc, 2> Reshape::attribute_descs;

constexpr std::array<ArgumentDesc, StringStrip::input_arg_count + StringStrip::output_arg_count> StringStrip::argument_descs;
constexpr std::array<AttributeDesc, 0> StringStrip::attribute_descs;

constexpr std::array<ArgumentDesc, ParallelInterleaveDatasetV3::input_arg_count + ParallelInterleaveDatasetV3::output_arg_count> ParallelInterleaveDatasetV3::argument_descs;
constexpr std::array<AttributeDesc, 6> ParallelInterleaveDatasetV3::attribute_descs;

constexpr std::array<ArgumentDesc, Roll::input_arg_count + Roll::output_arg_count> Roll::argument_descs;
constexpr std::array<AttributeDesc, 3> Roll::attribute_descs;

constexpr std::array<ArgumentDesc, MapAndBatchDataset::input_arg_count + MapAndBatchDataset::output_arg_count> MapAndBatchDataset::argument_descs;
constexpr std::array<AttributeDesc, 6> MapAndBatchDataset::attribute_descs;

constexpr std::array<ArgumentDesc, ListDiff::input_arg_count + ListDiff::output_arg_count> ListDiff::argument_descs;
constexpr std::array<AttributeDesc, 2> ListDiff::attribute_descs;

constexpr std::array<ArgumentDesc, NIntsOutDefault::input_arg_count + NIntsOutDefault::output_arg_count> NIntsOutDefault::argument_descs;
constexpr std::array<AttributeDesc, 1> NIntsOutDefault::attribute_descs;

constexpr std::array<ArgumentDesc, FIFOQueue::input_arg_count + FIFOQueue::output_arg_count> FIFOQueue::argument_descs;
constexpr std::array<AttributeDesc, 5> FIFOQueue::attribute_descs;

constexpr std::array<ArgumentDesc, Snapshot::input_arg_count + Snapshot::output_arg_count> Snapshot::argument_descs;
constexpr std::array<AttributeDesc, 1> Snapshot::attribute_descs;

constexpr std::array<ArgumentDesc, NPolymorphicOut::input_arg_count + NPolymorphicOut::output_arg_count> NPolymorphicOut::argument_descs;
constexpr std::array<AttributeDesc, 2> NPolymorphicOut::attribute_descs;

constexpr std::array<ArgumentDesc, RaggedTensorToTensor::input_arg_count + RaggedTensorToTensor::output_arg_count> RaggedTensorToTensor::argument_descs;
constexpr std::array<AttributeDesc, 5> RaggedTensorToTensor::attribute_descs;

constexpr std::array<ArgumentDesc, MatrixBandPart::input_arg_count + MatrixBandPart::output_arg_count> MatrixBandPart::argument_descs;
constexpr std::array<AttributeDesc, 2> MatrixBandPart::attribute_descs;

constexpr std::array<ArgumentDesc, NPolymorphicOutDefault::input_arg_count + NPolymorphicOutDefault::output_arg_count> NPolymorphicOutDefault::argument_descs;
constexpr std::array<AttributeDesc, 2> NPolymorphicOutDefault::attribute_descs;

constexpr std::array<ArgumentDesc, QuantizedResizeBilinear::input_arg_count + QuantizedResizeBilinear::output_arg_count> QuantizedResizeBilinear::argument_descs;
constexpr std::array<AttributeDesc, 3> QuantizedResizeBilinear::attribute_descs;

constexpr std::array<ArgumentDesc, NPolymorphicRestrictOut::input_arg_count + NPolymorphicRestrictOut::output_arg_count> NPolymorphicRestrictOut::argument_descs;
constexpr std::array<AttributeDesc, 2> NPolymorphicRestrictOut::attribute_descs;

constexpr std::array<ArgumentDesc, TwoRefsIn::input_arg_count + TwoRefsIn::output_arg_count> TwoRefsIn::argument_descs;
constexpr std::array<AttributeDesc, 1> TwoRefsIn::attribute_descs;

constexpr std::array<ArgumentDesc, TensorArrayGatherV2::input_arg_count + TensorArrayGatherV2::output_arg_count> TensorArrayGatherV2::argument_descs;
constexpr std::array<AttributeDesc, 2> TensorArrayGatherV2::attribute_descs;

constexpr std::array<ArgumentDesc, BoostedTreesCalculateBestFeatureSplitV2::input_arg_count + BoostedTreesCalculateBestFeatureSplitV2::output_arg_count> BoostedTreesCalculateBestFeatureSplitV2::argument_descs;
constexpr std::array<AttributeDesc, 2> BoostedTreesCalculateBestFeatureSplitV2::attribute_descs;

constexpr std::array<ArgumentDesc, ShuffleDatasetV2::input_arg_count + ShuffleDatasetV2::output_arg_count> ShuffleDatasetV2::argument_descs;
constexpr std::array<AttributeDesc, 3> ShuffleDatasetV2::attribute_descs;

constexpr std::array<ArgumentDesc, RefOut::input_arg_count + RefOut::output_arg_count> RefOut::argument_descs;
constexpr std::array<AttributeDesc, 1> RefOut::attribute_descs;

constexpr std::array<ArgumentDesc, LearnedUnigramCandidateSampler::input_arg_count + LearnedUnigramCandidateSampler::output_arg_count> LearnedUnigramCandidateSampler::argument_descs;
constexpr std::array<AttributeDesc, 6> LearnedUnigramCandidateSampler::attribute_descs;

constexpr std::array<ArgumentDesc, SimpleStruct::input_arg_count + SimpleStruct::output_arg_count> SimpleStruct::argument_descs;
constexpr std::array<AttributeDesc, 1> SimpleStruct::attribute_descs;

constexpr std::array<ArgumentDesc, SparseApplyRMSProp::input_arg_count + SparseApplyRMSProp::output_arg_count> SparseApplyRMSProp::argument_descs;
constexpr std::array<AttributeDesc, 3> SparseApplyRMSProp::attribute_descs;

constexpr std::array<ArgumentDesc, ConcatenateDataset::input_arg_count + ConcatenateDataset::output_arg_count> ConcatenateDataset::argument_descs;
constexpr std::array<AttributeDesc, 3> ConcatenateDataset::attribute_descs;

constexpr std::array<ArgumentDesc, ShardDataset::input_arg_count + ShardDataset::output_arg_count> ShardDataset::argument_descs;
constexpr std::array<AttributeDesc, 4> ShardDataset::attribute_descs;

constexpr std::array<ArgumentDesc, ExtractJpegShape::input_arg_count + ExtractJpegShape::output_arg_count> ExtractJpegShape::argument_descs;
constexpr std::array<AttributeDesc, 1> ExtractJpegShape::attribute_descs;

constexpr std::array<ArgumentDesc, MixedStruct::input_arg_count + MixedStruct::output_arg_count> MixedStruct::argument_descs;
constexpr std::array<AttributeDesc, 1> MixedStruct::attribute_descs;

constexpr std::array<ArgumentDesc, IsFinite::input_arg_count + IsFinite::output_arg_count> IsFinite::argument_descs;
constexpr std::array<AttributeDesc, 1> IsFinite::attribute_descs;

constexpr std::array<ArgumentDesc, ResourceAccumulatorSetGlobalStep::input_arg_count + ResourceAccumulatorSetGlobalStep::output_arg_count> ResourceAccumulatorSetGlobalStep::argument_descs;
constexpr std::array<AttributeDesc, 0> ResourceAccumulatorSetGlobalStep::attribute_descs;

constexpr std::array<ArgumentDesc, ComplexStruct::input_arg_count + ComplexStruct::output_arg_count> ComplexStruct::argument_descs;
constexpr std::array<AttributeDesc, 3> ComplexStruct::attribute_descs;

constexpr std::array<ArgumentDesc, DevicePlacementOp::input_arg_count + DevicePlacementOp::output_arg_count> DevicePlacementOp::argument_descs;
constexpr std::array<AttributeDesc, 0> DevicePlacementOp::attribute_descs;

constexpr std::array<ArgumentDesc, QueueSizeV2::input_arg_count + QueueSizeV2::output_arg_count> QueueSizeV2::argument_descs;
constexpr std::array<AttributeDesc, 0> QueueSizeV2::attribute_descs;

constexpr std::array<ArgumentDesc, DtypeWithDefaultOp::input_arg_count + DtypeWithDefaultOp::output_arg_count> DtypeWithDefaultOp::argument_descs;
constexpr std::array<AttributeDesc, 1> DtypeWithDefaultOp::attribute_descs;

constexpr std::array<ArgumentDesc, ToBool::input_arg_count + ToBool::output_arg_count> ToBool::argument_descs;
constexpr std::array<AttributeDesc, 1> ToBool::attribute_descs;

constexpr std::array<ArgumentDesc, IsTensorFloat32Enabled::input_arg_count + IsTensorFloat32Enabled::output_arg_count> IsTensorFloat32Enabled::argument_descs;
constexpr std::array<AttributeDesc, 0> IsTensorFloat32Enabled::attribute_descs;

constexpr std::array<ArgumentDesc, _XlaAotOnlyVarHandleOp::input_arg_count + _XlaAotOnlyVarHandleOp::output_arg_count> _XlaAotOnlyVarHandleOp::argument_descs;
constexpr std::array<AttributeDesc, 4> _XlaAotOnlyVarHandleOp::attribute_descs;

constexpr std::array<ArgumentDesc, FakeQuantWithMinMaxVars::input_arg_count + FakeQuantWithMinMaxVars::output_arg_count> FakeQuantWithMinMaxVars::argument_descs;
constexpr std::array<AttributeDesc, 2> FakeQuantWithMinMaxVars::attribute_descs;

constexpr std::array<ArgumentDesc, XlaLaunch::input_arg_count + XlaLaunch::output_arg_count> XlaLaunch::argument_descs;
constexpr std::array<AttributeDesc, 5> XlaLaunch::attribute_descs;

constexpr std::array<ArgumentDesc, SparseApplyFtrlV2::input_arg_count + SparseApplyFtrlV2::output_arg_count> SparseApplyFtrlV2::argument_descs;
constexpr std::array<AttributeDesc, 4> SparseApplyFtrlV2::attribute_descs;

constexpr std::array<ArgumentDesc, Cumsum::input_arg_count + Cumsum::output_arg_count> Cumsum::argument_descs;
constexpr std::array<AttributeDesc, 4> Cumsum::attribute_descs;

constexpr std::array<ArgumentDesc, BatchNormWithGlobalNormalization::input_arg_count + BatchNormWithGlobalNormalization::output_arg_count> BatchNormWithGlobalNormalization::argument_descs;
constexpr std::array<AttributeDesc, 3> BatchNormWithGlobalNormalization::attribute_descs;

constexpr std::array<ArgumentDesc, XlaClusterOutput::input_arg_count + XlaClusterOutput::output_arg_count> XlaClusterOutput::argument_descs;
constexpr std::array<AttributeDesc, 1> XlaClusterOutput::attribute_descs;

constexpr std::array<ArgumentDesc, _XlaCompile::input_arg_count + _XlaCompile::output_arg_count> _XlaCompile::argument_descs;
constexpr std::array<AttributeDesc, 5> _XlaCompile::attribute_descs;

constexpr std::array<ArgumentDesc, _XlaRun::input_arg_count + _XlaRun::output_arg_count> _XlaRun::argument_descs;
constexpr std::array<AttributeDesc, 2> _XlaRun::attribute_descs;

constexpr std::array<ArgumentDesc, RandomShuffle::input_arg_count + RandomShuffle::output_arg_count> RandomShuffle::argument_descs;
constexpr std::array<AttributeDesc, 3> RandomShuffle::attribute_descs;

constexpr std::array<ArgumentDesc, _XlaMerge::input_arg_count + _XlaMerge::output_arg_count> _XlaMerge::argument_descs;
constexpr std::array<AttributeDesc, 1> _XlaMerge::attribute_descs;

constexpr std::array<ArgumentDesc, AssertCardinalityDataset::input_arg_count + AssertCardinalityDataset::output_arg_count> AssertCardinalityDataset::argument_descs;
constexpr std::array<AttributeDesc, 2> AssertCardinalityDataset::attribute_descs;

constexpr std::array<ArgumentDesc, Requantize::input_arg_count + Requantize::output_arg_count> Requantize::argument_descs;
constexpr std::array<AttributeDesc, 2> Requantize::attribute_descs;

constexpr std::array<ArgumentDesc, BatchSelfAdjointEig::input_arg_count + BatchSelfAdjointEig::output_arg_count> BatchSelfAdjointEig::argument_descs;
constexpr std::array<AttributeDesc, 1> BatchSelfAdjointEig::attribute_descs;

constexpr std::array<ArgumentDesc, Fact::input_arg_count + Fact::output_arg_count> Fact::argument_descs;
constexpr std::array<AttributeDesc, 0> Fact::attribute_descs;

constexpr std::array<ArgumentDesc, RaggedCountSparseOutput::input_arg_count + RaggedCountSparseOutput::output_arg_count> RaggedCountSparseOutput::argument_descs;
constexpr std::array<AttributeDesc, 5> RaggedCountSparseOutput::attribute_descs;

constexpr std::array<ArgumentDesc, XlaVariadicReduce::input_arg_count + XlaVariadicReduce::output_arg_count> XlaVariadicReduce::argument_descs;
constexpr std::array<AttributeDesc, 4> XlaVariadicReduce::attribute_descs;

constexpr std::array<ArgumentDesc, _DisconnectHostFromDistributedTPUSystem::input_arg_count + _DisconnectHostFromDistributedTPUSystem::output_arg_count> _DisconnectHostFromDistributedTPUSystem::argument_descs;
constexpr std::array<AttributeDesc, 0> _DisconnectHostFromDistributedTPUSystem::attribute_descs;

constexpr std::array<ArgumentDesc, OrderedMapPeek::input_arg_count + OrderedMapPeek::output_arg_count> OrderedMapPeek::argument_descs;
constexpr std::array<AttributeDesc, 5> OrderedMapPeek::attribute_descs;

constexpr std::array<ArgumentDesc, Bitcast::input_arg_count + Bitcast::output_arg_count> Bitcast::argument_descs;
constexpr std::array<AttributeDesc, 2> Bitcast::attribute_descs;

constexpr std::array<ArgumentDesc, LogUniformCandidateSampler::input_arg_count + LogUniformCandidateSampler::output_arg_count> LogUniformCandidateSampler::argument_descs;
constexpr std::array<AttributeDesc, 6> LogUniformCandidateSampler::attribute_descs;

constexpr std::array<ArgumentDesc, ReaderNumRecordsProducedV2::input_arg_count + ReaderNumRecordsProducedV2::output_arg_count> ReaderNumRecordsProducedV2::argument_descs;
constexpr std::array<AttributeDesc, 0> ReaderNumRecordsProducedV2::attribute_descs;

constexpr std::array<ArgumentDesc, Lgamma::input_arg_count + Lgamma::output_arg_count> Lgamma::argument_descs;
constexpr std::array<AttributeDesc, 1> Lgamma::attribute_descs;

constexpr std::array<ArgumentDesc, CTCBeamSearchDecoder::input_arg_count + CTCBeamSearchDecoder::output_arg_count> CTCBeamSearchDecoder::argument_descs;
constexpr std::array<AttributeDesc, 4> CTCBeamSearchDecoder::attribute_descs;

constexpr std::array<ArgumentDesc, Pack::input_arg_count + Pack::output_arg_count> Pack::argument_descs;
constexpr std::array<AttributeDesc, 3> Pack::attribute_descs;

constexpr std::array<ArgumentDesc, TPUReplicateMetadata::input_arg_count + TPUReplicateMetadata::output_arg_count> TPUReplicateMetadata::argument_descs;
constexpr std::array<AttributeDesc, 11> TPUReplicateMetadata::attribute_descs;

constexpr std::array<ArgumentDesc, QuantizedMaxPool::input_arg_count + QuantizedMaxPool::output_arg_count> QuantizedMaxPool::argument_descs;
constexpr std::array<AttributeDesc, 4> QuantizedMaxPool::attribute_descs;

constexpr std::array<ArgumentDesc, DeepCopy::input_arg_count + DeepCopy::output_arg_count> DeepCopy::argument_descs;
constexpr std::array<AttributeDesc, 1> DeepCopy::attribute_descs;

constexpr std::array<ArgumentDesc, _MklMaxPoolGrad::input_arg_count + _MklMaxPoolGrad::output_arg_count> _MklMaxPoolGrad::argument_descs;
constexpr std::array<AttributeDesc, 7> _MklMaxPoolGrad::attribute_descs;

constexpr std::array<ArgumentDesc, RefNextIteration::input_arg_count + RefNextIteration::output_arg_count> RefNextIteration::argument_descs;
constexpr std::array<AttributeDesc, 1> RefNextIteration::attribute_descs;

constexpr std::array<ArgumentDesc, InplaceUpdate::input_arg_count + InplaceUpdate::output_arg_count> InplaceUpdate::argument_descs;
constexpr std::array<AttributeDesc, 1> InplaceUpdate::attribute_descs;

constexpr std::array<ArgumentDesc, RefIdentity::input_arg_count + RefIdentity::output_arg_count> RefIdentity::argument_descs;
constexpr std::array<AttributeDesc, 1> RefIdentity::attribute_descs;

constexpr std::array<ArgumentDesc, InplaceAdd::input_arg_count + InplaceAdd::output_arg_count> InplaceAdd::argument_descs;
constexpr std::array<AttributeDesc, 1> InplaceAdd::attribute_descs;

constexpr std::array<ArgumentDesc, SamplingDataset::input_arg_count + SamplingDataset::output_arg_count> SamplingDataset::argument_descs;
constexpr std::array<AttributeDesc, 2> SamplingDataset::attribute_descs;

constexpr std::array<ArgumentDesc, FixedLengthRecordDatasetV2::input_arg_count + FixedLengthRecordDatasetV2::output_arg_count> FixedLengthRecordDatasetV2::argument_descs;
constexpr std::array<AttributeDesc, 1> FixedLengthRecordDatasetV2::attribute_descs;

constexpr std::array<ArgumentDesc, _NcclBroadcastRecv::input_arg_count + _NcclBroadcastRecv::output_arg_count> _NcclBroadcastRecv::argument_descs;
constexpr std::array<AttributeDesc, 3> _NcclBroadcastRecv::attribute_descs;

constexpr std::array<ArgumentDesc, CudnnRNNBackpropV2::input_arg_count + CudnnRNNBackpropV2::output_arg_count> CudnnRNNBackpropV2::argument_descs;
constexpr std::array<AttributeDesc, 7> CudnnRNNBackpropV2::attribute_descs;

constexpr std::array<ArgumentDesc, InplaceSub::input_arg_count + InplaceSub::output_arg_count> InplaceSub::argument_descs;
constexpr std::array<AttributeDesc, 1> InplaceSub::attribute_descs;

constexpr std::array<ArgumentDesc, _MklPadWithFusedConv2D::input_arg_count + _MklPadWithFusedConv2D::output_arg_count> _MklPadWithFusedConv2D::argument_descs;
constexpr std::array<AttributeDesc, 11> _MklPadWithFusedConv2D::attribute_descs;

constexpr std::array<ArgumentDesc, AvgPool3DGrad::input_arg_count + AvgPool3DGrad::output_arg_count> AvgPool3DGrad::argument_descs;
constexpr std::array<AttributeDesc, 5> AvgPool3DGrad::attribute_descs;

constexpr std::array<ArgumentDesc, StridedSlice::input_arg_count + StridedSlice::output_arg_count> StridedSlice::argument_descs;
constexpr std::array<AttributeDesc, 7> StridedSlice::attribute_descs;

constexpr std::array<ArgumentDesc, Empty::input_arg_count + Empty::output_arg_count> Empty::argument_descs;
constexpr std::array<AttributeDesc, 2> Empty::attribute_descs;

constexpr std::array<ArgumentDesc, ReaderReadV2::input_arg_count + ReaderReadV2::output_arg_count> ReaderReadV2::argument_descs;
constexpr std::array<AttributeDesc, 0> ReaderReadV2::attribute_descs;

constexpr std::array<ArgumentDesc, OrderedMapUnstageNoKey::input_arg_count + OrderedMapUnstageNoKey::output_arg_count> OrderedMapUnstageNoKey::argument_descs;
constexpr std::array<AttributeDesc, 5> OrderedMapUnstageNoKey::attribute_descs;

constexpr std::array<ArgumentDesc, Unpack::input_arg_count + Unpack::output_arg_count> Unpack::argument_descs;
constexpr std::array<AttributeDesc, 3> Unpack::attribute_descs;

constexpr std::array<ArgumentDesc, RiscSort::input_arg_count + RiscSort::output_arg_count> RiscSort::argument_descs;
constexpr std::array<AttributeDesc, 3> RiscSort::attribute_descs;

constexpr std::array<ArgumentDesc, UnravelIndex::input_arg_count + UnravelIndex::output_arg_count> UnravelIndex::argument_descs;
constexpr std::array<AttributeDesc, 1> UnravelIndex::attribute_descs;

constexpr std::array<ArgumentDesc, BroadcastTo::input_arg_count + BroadcastTo::output_arg_count> BroadcastTo::argument_descs;
constexpr std::array<AttributeDesc, 2> BroadcastTo::attribute_descs;

constexpr std::array<ArgumentDesc, BytesProducedStatsDataset::input_arg_count + BytesProducedStatsDataset::output_arg_count> BytesProducedStatsDataset::argument_descs;
constexpr std::array<AttributeDesc, 2> BytesProducedStatsDataset::attribute_descs;

constexpr std::array<ArgumentDesc, ExperimentalStatsAggregatorSummary::input_arg_count + ExperimentalStatsAggregatorSummary::output_arg_count> ExperimentalStatsAggregatorSummary::argument_descs;
constexpr std::array<AttributeDesc, 0> ExperimentalStatsAggregatorSummary::attribute_descs;

constexpr std::array<ArgumentDesc, QuantizeAndDequantizeV4::input_arg_count + QuantizeAndDequantizeV4::output_arg_count> QuantizeAndDequantizeV4::argument_descs;
constexpr std::array<AttributeDesc, 7> QuantizeAndDequantizeV4::attribute_descs;

constexpr std::array<ArgumentDesc, RetrieveTPUEmbeddingFrequencyEstimatorParameters::input_arg_count + RetrieveTPUEmbeddingFrequencyEstimatorParameters::output_arg_count> RetrieveTPUEmbeddingFrequencyEstimatorParameters::argument_descs;
constexpr std::array<AttributeDesc, 5> RetrieveTPUEmbeddingFrequencyEstimatorParameters::attribute_descs;

constexpr std::array<ArgumentDesc, Concat::input_arg_count + Concat::output_arg_count> Concat::argument_descs;
constexpr std::array<AttributeDesc, 2> Concat::attribute_descs;

constexpr std::array<ArgumentDesc, TFRecordReader::input_arg_count + TFRecordReader::output_arg_count> TFRecordReader::argument_descs;
constexpr std::array<AttributeDesc, 3> TFRecordReader::attribute_descs;

constexpr std::array<ArgumentDesc, _FusedDepthwiseConv2dNative::input_arg_count + _FusedDepthwiseConv2dNative::output_arg_count> _FusedDepthwiseConv2dNative::argument_descs;
constexpr std::array<AttributeDesc, 9> _FusedDepthwiseConv2dNative::attribute_descs;

constexpr std::array<ArgumentDesc, ConcatV2::input_arg_count + ConcatV2::output_arg_count> ConcatV2::argument_descs;
constexpr std::array<AttributeDesc, 3> ConcatV2::attribute_descs;

constexpr std::array<ArgumentDesc, ConcatOffset::input_arg_count + ConcatOffset::output_arg_count> ConcatOffset::argument_descs;
constexpr std::array<AttributeDesc, 1> ConcatOffset::attribute_descs;

constexpr std::array<ArgumentDesc, _MklMatMul::input_arg_count + _MklMatMul::output_arg_count> _MklMatMul::argument_descs;
constexpr std::array<AttributeDesc, 3> _MklMatMul::attribute_descs;

constexpr std::array<ArgumentDesc, CumulativeLogsumexp::input_arg_count + CumulativeLogsumexp::output_arg_count> CumulativeLogsumexp::argument_descs;
constexpr std::array<AttributeDesc, 4> CumulativeLogsumexp::attribute_descs;

constexpr std::array<ArgumentDesc, MaxPool::input_arg_count + MaxPool::output_arg_count> MaxPool::argument_descs;
constexpr std::array<AttributeDesc, 6> MaxPool::attribute_descs;

constexpr std::array<ArgumentDesc, TensorArraySplit::input_arg_count + TensorArraySplit::output_arg_count> TensorArraySplit::argument_descs;
constexpr std::array<AttributeDesc, 1> TensorArraySplit::attribute_descs;

constexpr std::array<ArgumentDesc, BatchSelfAdjointEigV2::input_arg_count + BatchSelfAdjointEigV2::output_arg_count> BatchSelfAdjointEigV2::argument_descs;
constexpr std::array<AttributeDesc, 2> BatchSelfAdjointEigV2::attribute_descs;

constexpr std::array<ArgumentDesc, RandomIndexShuffle::input_arg_count + RandomIndexShuffle::output_arg_count> RandomIndexShuffle::argument_descs;
constexpr std::array<AttributeDesc, 2> RandomIndexShuffle::attribute_descs;

constexpr std::array<ArgumentDesc, MatrixDeterminant::input_arg_count + MatrixDeterminant::output_arg_count> MatrixDeterminant::argument_descs;
constexpr std::array<AttributeDesc, 1> MatrixDeterminant::attribute_descs;

constexpr std::array<ArgumentDesc, PrelinearizeTuple::input_arg_count + PrelinearizeTuple::output_arg_count> PrelinearizeTuple::argument_descs;
constexpr std::array<AttributeDesc, 3> PrelinearizeTuple::attribute_descs;

constexpr std::array<ArgumentDesc, SplitV::input_arg_count + SplitV::output_arg_count> SplitV::argument_descs;
constexpr std::array<AttributeDesc, 3> SplitV::attribute_descs;

constexpr std::array<ArgumentDesc, ResizeNearestNeighborGrad::input_arg_count + ResizeNearestNeighborGrad::output_arg_count> ResizeNearestNeighborGrad::argument_descs;
constexpr std::array<AttributeDesc, 3> ResizeNearestNeighborGrad::attribute_descs;

constexpr std::array<ArgumentDesc, EmptyTensorList::input_arg_count + EmptyTensorList::output_arg_count> EmptyTensorList::argument_descs;
constexpr std::array<AttributeDesc, 2> EmptyTensorList::attribute_descs;

constexpr std::array<ArgumentDesc, Const::input_arg_count + Const::output_arg_count> Const::argument_descs;
constexpr std::array<AttributeDesc, 2> Const::attribute_descs;

constexpr std::array<ArgumentDesc, HostConst::input_arg_count + HostConst::output_arg_count> HostConst::argument_descs;
constexpr std::array<AttributeDesc, 2> HostConst::attribute_descs;

constexpr std::array<ArgumentDesc, RiscSqueeze::input_arg_count + RiscSqueeze::output_arg_count> RiscSqueeze::argument_descs;
constexpr std::array<AttributeDesc, 2> RiscSqueeze::attribute_descs;

constexpr std::array<ArgumentDesc, QuantizedMatMulWithBiasAndRelu::input_arg_count + QuantizedMatMulWithBiasAndRelu::output_arg_count> QuantizedMatMulWithBiasAndRelu::argument_descs;
constexpr std::array<AttributeDesc, 6> QuantizedMatMulWithBiasAndRelu::attribute_descs;

constexpr std::array<ArgumentDesc, BoostedTreesQuantileStreamResourceAddSummaries::input_arg_count + BoostedTreesQuantileStreamResourceAddSummaries::output_arg_count> BoostedTreesQuantileStreamResourceAddSummaries::argument_descs;
constexpr std::array<AttributeDesc, 1> BoostedTreesQuantileStreamResourceAddSummaries::attribute_descs;

constexpr std::array<ArgumentDesc, _EagerConst::input_arg_count + _EagerConst::output_arg_count> _EagerConst::argument_descs;
constexpr std::array<AttributeDesc, 1> _EagerConst::attribute_descs;

constexpr std::array<ArgumentDesc, ImmutableConst::input_arg_count + ImmutableConst::output_arg_count> ImmutableConst::argument_descs;
constexpr std::array<AttributeDesc, 3> ImmutableConst::attribute_descs;

constexpr std::array<ArgumentDesc, ZerosLike::input_arg_count + ZerosLike::output_arg_count> ZerosLike::argument_descs;
constexpr std::array<AttributeDesc, 1> ZerosLike::attribute_descs;

constexpr std::array<ArgumentDesc, While::input_arg_count + While::output_arg_count> While::argument_descs;
constexpr std::array<AttributeDesc, 5> While::attribute_descs;

constexpr std::array<ArgumentDesc, AddN::input_arg_count + AddN::output_arg_count> AddN::argument_descs;
constexpr std::array<AttributeDesc, 2> AddN::attribute_descs;

constexpr std::array<ArgumentDesc, Rint::input_arg_count + Rint::output_arg_count> Rint::argument_descs;
constexpr std::array<AttributeDesc, 1> Rint::attribute_descs;

constexpr std::array<ArgumentDesc, Diag::input_arg_count + Diag::output_arg_count> Diag::argument_descs;
constexpr std::array<AttributeDesc, 1> Diag::attribute_descs;

constexpr std::array<ArgumentDesc, BatchMatrixInverse::input_arg_count + BatchMatrixInverse::output_arg_count> BatchMatrixInverse::argument_descs;
constexpr std::array<AttributeDesc, 2> BatchMatrixInverse::attribute_descs;

constexpr std::array<ArgumentDesc, RightShift::input_arg_count + RightShift::output_arg_count> RightShift::argument_descs;
constexpr std::array<AttributeDesc, 1> RightShift::attribute_descs;

constexpr std::array<ArgumentDesc, _MklConv2DBackpropFilterWithBias::input_arg_count + _MklConv2DBackpropFilterWithBias::output_arg_count> _MklConv2DBackpropFilterWithBias::argument_descs;
constexpr std::array<AttributeDesc, 6> _MklConv2DBackpropFilterWithBias::attribute_descs;

constexpr std::array<ArgumentDesc, Rank::input_arg_count + Rank::output_arg_count> Rank::argument_descs;
constexpr std::array<AttributeDesc, 1> Rank::attribute_descs;

constexpr std::array<ArgumentDesc, DiagPart::input_arg_count + DiagPart::output_arg_count> DiagPart::argument_descs;
constexpr std::array<AttributeDesc, 1> DiagPart::attribute_descs;

constexpr std::array<ArgumentDesc, MatrixDiag::input_arg_count + MatrixDiag::output_arg_count> MatrixDiag::argument_descs;
constexpr std::array<AttributeDesc, 1> MatrixDiag::attribute_descs;

constexpr std::array<ArgumentDesc, ResizeNearestNeighbor::input_arg_count + ResizeNearestNeighbor::output_arg_count> ResizeNearestNeighbor::argument_descs;
constexpr std::array<AttributeDesc, 3> ResizeNearestNeighbor::attribute_descs;

constexpr std::array<ArgumentDesc, AnonymousIterator::input_arg_count + AnonymousIterator::output_arg_count> AnonymousIterator::argument_descs;
constexpr std::array<AttributeDesc, 2> AnonymousIterator::attribute_descs;

constexpr std::array<ArgumentDesc, MatrixDiagV3::input_arg_count + MatrixDiagV3::output_arg_count> MatrixDiagV3::argument_descs;
constexpr std::array<AttributeDesc, 2> MatrixDiagV3::attribute_descs;

constexpr std::array<ArgumentDesc, SparseCross::input_arg_count + SparseCross::output_arg_count> SparseCross::argument_descs;
constexpr std::array<AttributeDesc, 8> SparseCross::attribute_descs;

constexpr std::array<ArgumentDesc, StackPush::input_arg_count + StackPush::output_arg_count> StackPush::argument_descs;
constexpr std::array<AttributeDesc, 2> StackPush::attribute_descs;

constexpr std::array<ArgumentDesc, MatrixSetDiag::input_arg_count + MatrixSetDiag::output_arg_count> MatrixSetDiag::argument_descs;
constexpr std::array<AttributeDesc, 1> MatrixSetDiag::attribute_descs;

constexpr std::array<ArgumentDesc, SparseCrossV2::input_arg_count + SparseCrossV2::output_arg_count> SparseCrossV2::argument_descs;
constexpr std::array<AttributeDesc, 3> SparseCrossV2::attribute_descs;

constexpr std::array<ArgumentDesc, StackPushV2::input_arg_count + StackPushV2::output_arg_count> StackPushV2::argument_descs;
constexpr std::array<AttributeDesc, 2> StackPushV2::attribute_descs;

constexpr std::array<ArgumentDesc, MatrixSetDiagV2::input_arg_count + MatrixSetDiagV2::output_arg_count> MatrixSetDiagV2::argument_descs;
constexpr std::array<AttributeDesc, 1> MatrixSetDiagV2::attribute_descs;

constexpr std::array<ArgumentDesc, BoostedTreesMakeQuantileSummaries::input_arg_count + BoostedTreesMakeQuantileSummaries::output_arg_count> BoostedTreesMakeQuantileSummaries::argument_descs;
constexpr std::array<AttributeDesc, 1> BoostedTreesMakeQuantileSummaries::attribute_descs;

constexpr std::array<ArgumentDesc, ExperimentalParseExampleDataset::input_arg_count + ExperimentalParseExampleDataset::output_arg_count> ExperimentalParseExampleDataset::argument_descs;
constexpr std::array<AttributeDesc, 8> ExperimentalParseExampleDataset::attribute_descs;

constexpr std::array<ArgumentDesc, _MklNativeConv3D::input_arg_count + _MklNativeConv3D::output_arg_count> _MklNativeConv3D::argument_descs;
constexpr std::array<AttributeDesc, 6> _MklNativeConv3D::attribute_descs;

constexpr std::array<ArgumentDesc, MatrixSetDiagV3::input_arg_count + MatrixSetDiagV3::output_arg_count> MatrixSetDiagV3::argument_descs;
constexpr std::array<AttributeDesc, 2> MatrixSetDiagV3::attribute_descs;

constexpr std::array<ArgumentDesc, ResourceSparseApplyRMSProp::input_arg_count + ResourceSparseApplyRMSProp::output_arg_count> ResourceSparseApplyRMSProp::argument_descs;
constexpr std::array<AttributeDesc, 3> ResourceSparseApplyRMSProp::attribute_descs;

constexpr std::array<ArgumentDesc, ResourceAccumulatorTakeGradient::input_arg_count + ResourceAccumulatorTakeGradient::output_arg_count> ResourceAccumulatorTakeGradient::argument_descs;
constexpr std::array<AttributeDesc, 1> ResourceAccumulatorTakeGradient::attribute_descs;

constexpr std::array<ArgumentDesc, SparseSegmentSqrtNGrad::input_arg_count + SparseSegmentSqrtNGrad::output_arg_count> SparseSegmentSqrtNGrad::argument_descs;
constexpr std::array<AttributeDesc, 3> SparseSegmentSqrtNGrad::attribute_descs;

constexpr std::array<ArgumentDesc, TPUPartitionedOutput::input_arg_count + TPUPartitionedOutput::output_arg_count> TPUPartitionedOutput::argument_descs;
constexpr std::array<AttributeDesc, 3> TPUPartitionedOutput::attribute_descs;

constexpr std::array<ArgumentDesc, MatrixDiagPartV2::input_arg_count + MatrixDiagPartV2::output_arg_count> MatrixDiagPartV2::argument_descs;
constexpr std::array<AttributeDesc, 1> MatrixDiagPartV2::attribute_descs;

constexpr std::array<ArgumentDesc, BatchMatrixDiag::input_arg_count + BatchMatrixDiag::output_arg_count> BatchMatrixDiag::argument_descs;
constexpr std::array<AttributeDesc, 1> BatchMatrixDiag::attribute_descs;

constexpr std::array<ArgumentDesc, MatrixDiagPartV3::input_arg_count + MatrixDiagPartV3::output_arg_count> MatrixDiagPartV3::argument_descs;
constexpr std::array<AttributeDesc, 2> MatrixDiagPartV3::attribute_descs;

constexpr std::array<ArgumentDesc, Reverse::input_arg_count + Reverse::output_arg_count> Reverse::argument_descs;
constexpr std::array<AttributeDesc, 1> Reverse::attribute_descs;

constexpr std::array<ArgumentDesc, ReverseV2::input_arg_count + ReverseV2::output_arg_count> ReverseV2::argument_descs;
constexpr std::array<AttributeDesc, 2> ReverseV2::attribute_descs;

constexpr std::array<ArgumentDesc, _ParallelConcatStart::input_arg_count + _ParallelConcatStart::output_arg_count> _ParallelConcatStart::argument_descs;
constexpr std::array<AttributeDesc, 2> _ParallelConcatStart::attribute_descs;

constexpr std::array<ArgumentDesc, TensorListSetItem::input_arg_count + TensorListSetItem::output_arg_count> TensorListSetItem::argument_descs;
constexpr std::array<AttributeDesc, 1> TensorListSetItem::attribute_descs;

constexpr std::array<ArgumentDesc, Identity::input_arg_count + Identity::output_arg_count> Identity::argument_descs;
constexpr std::array<AttributeDesc, 1> Identity::attribute_descs;

constexpr std::array<ArgumentDesc, SnapshotDataset::input_arg_count + SnapshotDataset::output_arg_count> SnapshotDataset::argument_descs;
constexpr std::array<AttributeDesc, 16> SnapshotDataset::attribute_descs;

constexpr std::array<ArgumentDesc, BlockLSTMGrad::input_arg_count + BlockLSTMGrad::output_arg_count> BlockLSTMGrad::argument_descs;
constexpr std::array<AttributeDesc, 2> BlockLSTMGrad::attribute_descs;

constexpr std::array<ArgumentDesc, BatchMatrixDiagPart::input_arg_count + BatchMatrixDiagPart::output_arg_count> BatchMatrixDiagPart::argument_descs;
constexpr std::array<AttributeDesc, 1> BatchMatrixDiagPart::attribute_descs;

constexpr std::array<ArgumentDesc, _ParallelConcatUpdate::input_arg_count + _ParallelConcatUpdate::output_arg_count> _ParallelConcatUpdate::argument_descs;
constexpr std::array<AttributeDesc, 2> _ParallelConcatUpdate::attribute_descs;

constexpr std::array<ArgumentDesc, Gather::input_arg_count + Gather::output_arg_count> Gather::argument_descs;
constexpr std::array<AttributeDesc, 3> Gather::attribute_descs;

constexpr std::array<ArgumentDesc, FusedBatchNorm::input_arg_count + FusedBatchNorm::output_arg_count> FusedBatchNorm::argument_descs;
constexpr std::array<AttributeDesc, 5> FusedBatchNorm::attribute_descs;

constexpr std::array<ArgumentDesc, CudnnRNNParamsToCanonical::input_arg_count + CudnnRNNParamsToCanonical::output_arg_count> CudnnRNNParamsToCanonical::argument_descs;
constexpr std::array<AttributeDesc, 8> CudnnRNNParamsToCanonical::attribute_descs;

constexpr std::array<ArgumentDesc, LoadTPUEmbeddingCenteredRMSPropParameters::input_arg_count + LoadTPUEmbeddingCenteredRMSPropParameters::output_arg_count> LoadTPUEmbeddingCenteredRMSPropParameters::argument_descs;
constexpr std::array<AttributeDesc, 5> LoadTPUEmbeddingCenteredRMSPropParameters::attribute_descs;

constexpr std::array<ArgumentDesc, GatherNd::input_arg_count + GatherNd::output_arg_count> GatherNd::argument_descs;
constexpr std::array<AttributeDesc, 2> GatherNd::attribute_descs;

constexpr std::array<ArgumentDesc, TensorArraySplitV3::input_arg_count + TensorArraySplitV3::output_arg_count> TensorArraySplitV3::argument_descs;
constexpr std::array<AttributeDesc, 1> TensorArraySplitV3::attribute_descs;

constexpr std::array<ArgumentDesc, LookupTableImport::input_arg_count + LookupTableImport::output_arg_count> LookupTableImport::argument_descs;
constexpr std::array<AttributeDesc, 2> LookupTableImport::attribute_descs;

constexpr std::array<ArgumentDesc, ResourceScatterDiv::input_arg_count + ResourceScatterDiv::output_arg_count> ResourceScatterDiv::argument_descs;
constexpr std::array<AttributeDesc, 2> ResourceScatterDiv::attribute_descs;

constexpr std::array<ArgumentDesc, _MklIdentity::input_arg_count + _MklIdentity::output_arg_count> _MklIdentity::argument_descs;
constexpr std::array<AttributeDesc, 1> _MklIdentity::attribute_descs;

constexpr std::array<ArgumentDesc, BatchMatrixSolveLs::input_arg_count + BatchMatrixSolveLs::output_arg_count> BatchMatrixSolveLs::argument_descs;
constexpr std::array<AttributeDesc, 2> BatchMatrixSolveLs::attribute_descs;

constexpr std::array<ArgumentDesc, IdentityN::input_arg_count + IdentityN::output_arg_count> IdentityN::argument_descs;
constexpr std::array<AttributeDesc, 1> IdentityN::attribute_descs;

constexpr std::array<ArgumentDesc, LoadDataset::input_arg_count + LoadDataset::output_arg_count> LoadDataset::argument_descs;
constexpr std::array<AttributeDesc, 5> LoadDataset::attribute_descs;

constexpr std::array<ArgumentDesc, DebugGradientIdentity::input_arg_count + DebugGradientIdentity::output_arg_count> DebugGradientIdentity::argument_descs;
constexpr std::array<AttributeDesc, 1> DebugGradientIdentity::attribute_descs;

constexpr std::array<ArgumentDesc, ExpandDims::input_arg_count + ExpandDims::output_arg_count> ExpandDims::argument_descs;
constexpr std::array<AttributeDesc, 2> ExpandDims::attribute_descs;

constexpr std::array<ArgumentDesc, DebugGradientRefIdentity::input_arg_count + DebugGradientRefIdentity::output_arg_count> DebugGradientRefIdentity::argument_descs;
constexpr std::array<AttributeDesc, 1> DebugGradientRefIdentity::attribute_descs;

constexpr std::array<ArgumentDesc, PlaceholderWithDefault::input_arg_count + PlaceholderWithDefault::output_arg_count> PlaceholderWithDefault::argument_descs;
constexpr std::array<AttributeDesc, 2> PlaceholderWithDefault::attribute_descs;

constexpr std::array<ArgumentDesc, ApproximateEqual::input_arg_count + ApproximateEqual::output_arg_count> ApproximateEqual::argument_descs;
constexpr std::array<AttributeDesc, 2> ApproximateEqual::attribute_descs;

constexpr std::array<ArgumentDesc, AllCandidateSampler::input_arg_count + AllCandidateSampler::output_arg_count> AllCandidateSampler::argument_descs;
constexpr std::array<AttributeDesc, 5> AllCandidateSampler::attribute_descs;

constexpr std::array<ArgumentDesc, TensorScatterUpdate::input_arg_count + TensorScatterUpdate::output_arg_count> TensorScatterUpdate::argument_descs;
constexpr std::array<AttributeDesc, 2> TensorScatterUpdate::attribute_descs;

constexpr std::array<ArgumentDesc, ScalarSummary::input_arg_count + ScalarSummary::output_arg_count> ScalarSummary::argument_descs;
constexpr std::array<AttributeDesc, 1> ScalarSummary::attribute_descs;

constexpr std::array<ArgumentDesc, Sigmoid::input_arg_count + Sigmoid::output_arg_count> Sigmoid::argument_descs;
constexpr std::array<AttributeDesc, 1> Sigmoid::attribute_descs;

constexpr std::array<ArgumentDesc, FloorMod::input_arg_count + FloorMod::output_arg_count> FloorMod::argument_descs;
constexpr std::array<AttributeDesc, 1> FloorMod::attribute_descs;

constexpr std::array<ArgumentDesc, _MklNativeFusedBatchNormEx::input_arg_count + _MklNativeFusedBatchNormEx::output_arg_count> _MklNativeFusedBatchNormEx::argument_descs;
constexpr std::array<AttributeDesc, 8> _MklNativeFusedBatchNormEx::attribute_descs;

constexpr std::array<ArgumentDesc, CheckNumerics::input_arg_count + CheckNumerics::output_arg_count> CheckNumerics::argument_descs;
constexpr std::array<AttributeDesc, 2> CheckNumerics::attribute_descs;

constexpr std::array<ArgumentDesc, XlaBroadcastHelper::input_arg_count + XlaBroadcastHelper::output_arg_count> XlaBroadcastHelper::argument_descs;
constexpr std::array<AttributeDesc, 2> XlaBroadcastHelper::attribute_descs;

constexpr std::array<ArgumentDesc, CheckNumericsV2::input_arg_count + CheckNumericsV2::output_arg_count> CheckNumericsV2::argument_descs;
constexpr std::array<AttributeDesc, 2> CheckNumericsV2::attribute_descs;

constexpr std::array<ArgumentDesc, Send::input_arg_count + Send::output_arg_count> Send::argument_descs;
constexpr std::array<AttributeDesc, 6> Send::attribute_descs;

constexpr std::array<ArgumentDesc, GetSessionHandle::input_arg_count + GetSessionHandle::output_arg_count> GetSessionHandle::argument_descs;
constexpr std::array<AttributeDesc, 1> GetSessionHandle::attribute_descs;

constexpr std::array<ArgumentDesc, StatefulUniform::input_arg_count + StatefulUniform::output_arg_count> StatefulUniform::argument_descs;
constexpr std::array<AttributeDesc, 2> StatefulUniform::attribute_descs;

constexpr std::array<ArgumentDesc, StatsAggregatorHandleV2::input_arg_count + StatsAggregatorHandleV2::output_arg_count> StatsAggregatorHandleV2::argument_descs;
constexpr std::array<AttributeDesc, 2> StatsAggregatorHandleV2::attribute_descs;

constexpr std::array<ArgumentDesc, ResourceStridedSliceAssign::input_arg_count + ResourceStridedSliceAssign::output_arg_count> ResourceStridedSliceAssign::argument_descs;
constexpr std::array<AttributeDesc, 7> ResourceStridedSliceAssign::attribute_descs;

constexpr std::array<ArgumentDesc, MatrixLogarithm::input_arg_count + MatrixLogarithm::output_arg_count> MatrixLogarithm::argument_descs;
constexpr std::array<AttributeDesc, 1> MatrixLogarithm::attribute_descs;

constexpr std::array<ArgumentDesc, InvertPermutation::input_arg_count + InvertPermutation::output_arg_count> InvertPermutation::argument_descs;
constexpr std::array<AttributeDesc, 1> InvertPermutation::attribute_descs;

constexpr std::array<ArgumentDesc, AnonymousMultiDeviceIterator::input_arg_count + AnonymousMultiDeviceIterator::output_arg_count> AnonymousMultiDeviceIterator::argument_descs;
constexpr std::array<AttributeDesc, 3> AnonymousMultiDeviceIterator::attribute_descs;

constexpr std::array<ArgumentDesc, Transpose::input_arg_count + Transpose::output_arg_count> Transpose::argument_descs;
constexpr std::array<AttributeDesc, 2> Transpose::attribute_descs;

constexpr std::array<ArgumentDesc, Barrier::input_arg_count + Barrier::output_arg_count> Barrier::argument_descs;
constexpr std::array<AttributeDesc, 5> Barrier::attribute_descs;

constexpr std::array<ArgumentDesc, Sin::input_arg_count + Sin::output_arg_count> Sin::argument_descs;
constexpr std::array<AttributeDesc, 1> Sin::attribute_descs;

constexpr std::array<ArgumentDesc, Enter::input_arg_count + Enter::output_arg_count> Enter::argument_descs;
constexpr std::array<AttributeDesc, 4> Enter::attribute_descs;

constexpr std::array<ArgumentDesc, _MklQuantizeV2::input_arg_count + _MklQuantizeV2::output_arg_count> _MklQuantizeV2::argument_descs;
constexpr std::array<AttributeDesc, 6> _MklQuantizeV2::attribute_descs;

constexpr std::array<ArgumentDesc, TensorListFromTensor::input_arg_count + TensorListFromTensor::output_arg_count> TensorListFromTensor::argument_descs;
constexpr std::array<AttributeDesc, 2> TensorListFromTensor::attribute_descs;

constexpr std::array<ArgumentDesc, SparseSegmentSumGrad::input_arg_count + SparseSegmentSumGrad::output_arg_count> SparseSegmentSumGrad::argument_descs;
constexpr std::array<AttributeDesc, 3> SparseSegmentSumGrad::attribute_descs;

constexpr std::array<ArgumentDesc, _SendTPUEmbeddingGradients::input_arg_count + _SendTPUEmbeddingGradients::output_arg_count> _SendTPUEmbeddingGradients::argument_descs;
constexpr std::array<AttributeDesc, 3> _SendTPUEmbeddingGradients::attribute_descs;

constexpr std::array<ArgumentDesc, _MklTranspose::input_arg_count + _MklTranspose::output_arg_count> _MklTranspose::argument_descs;
constexpr std::array<AttributeDesc, 2> _MklTranspose::attribute_descs;

constexpr std::array<ArgumentDesc, DebugNumericSummary::input_arg_count + DebugNumericSummary::output_arg_count> DebugNumericSummary::argument_descs;
constexpr std::array<AttributeDesc, 8> DebugNumericSummary::attribute_descs;

constexpr std::array<ArgumentDesc, ConjugateTranspose::input_arg_count + ConjugateTranspose::output_arg_count> ConjugateTranspose::argument_descs;
constexpr std::array<AttributeDesc, 2> ConjugateTranspose::attribute_descs;

constexpr std::array<ArgumentDesc, _MklConjugateTranspose::input_arg_count + _MklConjugateTranspose::output_arg_count> _MklConjugateTranspose::argument_descs;
constexpr std::array<AttributeDesc, 2> _MklConjugateTranspose::attribute_descs;

constexpr std::array<ArgumentDesc, ExperimentalParallelInterleaveDataset::input_arg_count + ExperimentalParallelInterleaveDataset::output_arg_count> ExperimentalParallelInterleaveDataset::argument_descs;
constexpr std::array<AttributeDesc, 4> ExperimentalParallelInterleaveDataset::attribute_descs;

constexpr std::array<ArgumentDesc, Unique::input_arg_count + Unique::output_arg_count> Unique::argument_descs;
constexpr std::array<AttributeDesc, 2> Unique::attribute_descs;

constexpr std::array<ArgumentDesc, UniqueV2::input_arg_count + UniqueV2::output_arg_count> UniqueV2::argument_descs;
constexpr std::array<AttributeDesc, 3> UniqueV2::attribute_descs;

constexpr std::array<ArgumentDesc, TakeDataset::input_arg_count + TakeDataset::output_arg_count> TakeDataset::argument_descs;
constexpr std::array<AttributeDesc, 3> TakeDataset::attribute_descs;

constexpr std::array<ArgumentDesc, UniqueWithCounts::input_arg_count + UniqueWithCounts::output_arg_count> UniqueWithCounts::argument_descs;
constexpr std::array<AttributeDesc, 2> UniqueWithCounts::attribute_descs;

constexpr std::array<ArgumentDesc, UniqueWithCountsV2::input_arg_count + UniqueWithCountsV2::output_arg_count> UniqueWithCountsV2::argument_descs;
constexpr std::array<AttributeDesc, 3> UniqueWithCountsV2::attribute_descs;

constexpr std::array<ArgumentDesc, AdjustSaturation::input_arg_count + AdjustSaturation::output_arg_count> AdjustSaturation::argument_descs;
constexpr std::array<AttributeDesc, 1> AdjustSaturation::attribute_descs;

constexpr std::array<ArgumentDesc, VariableShape::input_arg_count + VariableShape::output_arg_count> VariableShape::argument_descs;
constexpr std::array<AttributeDesc, 1> VariableShape::attribute_descs;

constexpr std::array<ArgumentDesc, ParallelDynamicStitch::input_arg_count + ParallelDynamicStitch::output_arg_count> ParallelDynamicStitch::argument_descs;
constexpr std::array<AttributeDesc, 2> ParallelDynamicStitch::attribute_descs;

constexpr std::array<ArgumentDesc, _MklEluGrad::input_arg_count + _MklEluGrad::output_arg_count> _MklEluGrad::argument_descs;
constexpr std::array<AttributeDesc, 1> _MklEluGrad::attribute_descs;

constexpr std::array<ArgumentDesc, Shape::input_arg_count + Shape::output_arg_count> Shape::argument_descs;
constexpr std::array<AttributeDesc, 2> Shape::attribute_descs;

constexpr std::array<ArgumentDesc, KmeansPlusPlusInitialization::input_arg_count + KmeansPlusPlusInitialization::output_arg_count> KmeansPlusPlusInitialization::argument_descs;
constexpr std::array<AttributeDesc, 0> KmeansPlusPlusInitialization::attribute_descs;

constexpr std::array<ArgumentDesc, Tan::input_arg_count + Tan::output_arg_count> Tan::argument_descs;
constexpr std::array<AttributeDesc, 1> Tan::attribute_descs;

constexpr std::array<ArgumentDesc, RiscReduce::input_arg_count + RiscReduce::output_arg_count> RiscReduce::argument_descs;
constexpr std::array<AttributeDesc, 3> RiscReduce::attribute_descs;

constexpr std::array<ArgumentDesc, Inv::input_arg_count + Inv::output_arg_count> Inv::argument_descs;
constexpr std::array<AttributeDesc, 1> Inv::attribute_descs;

constexpr std::array<ArgumentDesc, ShapeN::input_arg_count + ShapeN::output_arg_count> ShapeN::argument_descs;
constexpr std::array<AttributeDesc, 3> ShapeN::attribute_descs;

constexpr std::array<ArgumentDesc, QueueIsClosedV2::input_arg_count + QueueIsClosedV2::output_arg_count> QueueIsClosedV2::argument_descs;
constexpr std::array<AttributeDesc, 0> QueueIsClosedV2::attribute_descs;

constexpr std::array<ArgumentDesc, ScatterUpdate::input_arg_count + ScatterUpdate::output_arg_count> ScatterUpdate::argument_descs;
constexpr std::array<AttributeDesc, 3> ScatterUpdate::attribute_descs;

constexpr std::array<ArgumentDesc, EnsureShape::input_arg_count + EnsureShape::output_arg_count> EnsureShape::argument_descs;
constexpr std::array<AttributeDesc, 2> EnsureShape::attribute_descs;

constexpr std::array<ArgumentDesc, AvgPool3D::input_arg_count + AvgPool3D::output_arg_count> AvgPool3D::argument_descs;
constexpr std::array<AttributeDesc, 5> AvgPool3D::attribute_descs;

constexpr std::array<ArgumentDesc, BoostedTreesQuantileStreamResourceDeserialize::input_arg_count + BoostedTreesQuantileStreamResourceDeserialize::output_arg_count> BoostedTreesQuantileStreamResourceDeserialize::argument_descs;
constexpr std::array<AttributeDesc, 1> BoostedTreesQuantileStreamResourceDeserialize::attribute_descs;

constexpr std::array<ArgumentDesc, StridedSliceGrad::input_arg_count + StridedSliceGrad::output_arg_count> StridedSliceGrad::argument_descs;
constexpr std::array<AttributeDesc, 7> StridedSliceGrad::attribute_descs;

constexpr std::array<ArgumentDesc, RiscFft::input_arg_count + RiscFft::output_arg_count> RiscFft::argument_descs;
constexpr std::array<AttributeDesc, 1> RiscFft::attribute_descs;

constexpr std::array<ArgumentDesc, TensorStridedSliceUpdate::input_arg_count + TensorStridedSliceUpdate::output_arg_count> TensorStridedSliceUpdate::argument_descs;
constexpr std::array<AttributeDesc, 7> TensorStridedSliceUpdate::attribute_descs;

constexpr std::array<ArgumentDesc, DummyMemoryCache::input_arg_count + DummyMemoryCache::output_arg_count> DummyMemoryCache::argument_descs;
constexpr std::array<AttributeDesc, 0> DummyMemoryCache::attribute_descs;

constexpr std::array<ArgumentDesc, ReaderReadUpToV2::input_arg_count + ReaderReadUpToV2::output_arg_count> ReaderReadUpToV2::argument_descs;
constexpr std::array<AttributeDesc, 0> ReaderReadUpToV2::attribute_descs;

constexpr std::array<ArgumentDesc, _MklMaxPool::input_arg_count + _MklMaxPool::output_arg_count> _MklMaxPool::argument_descs;
constexpr std::array<AttributeDesc, 7> _MklMaxPool::attribute_descs;

constexpr std::array<ArgumentDesc, MultiDeviceIterator::input_arg_count + MultiDeviceIterator::output_arg_count> MultiDeviceIterator::argument_descs;
constexpr std::array<AttributeDesc, 5> MultiDeviceIterator::attribute_descs;

constexpr std::array<ArgumentDesc, TileGrad::input_arg_count + TileGrad::output_arg_count> TileGrad::argument_descs;
constexpr std::array<AttributeDesc, 1> TileGrad::attribute_descs;

constexpr std::array<ArgumentDesc, TensorMapHasKey::input_arg_count + TensorMapHasKey::output_arg_count> TensorMapHasKey::argument_descs;
constexpr std::array<AttributeDesc, 1> TensorMapHasKey::attribute_descs;

constexpr std::array<ArgumentDesc, BroadcastArgs::input_arg_count + BroadcastArgs::output_arg_count> BroadcastArgs::argument_descs;
constexpr std::array<AttributeDesc, 1> BroadcastArgs::attribute_descs;

constexpr std::array<ArgumentDesc, _MklFusedMatMul::input_arg_count + _MklFusedMatMul::output_arg_count> _MklFusedMatMul::argument_descs;
constexpr std::array<AttributeDesc, 8> _MklFusedMatMul::attribute_descs;

constexpr std::array<ArgumentDesc, BroadcastGradientArgs::input_arg_count + BroadcastGradientArgs::output_arg_count> BroadcastGradientArgs::argument_descs;
constexpr std::array<AttributeDesc, 1> BroadcastGradientArgs::attribute_descs;

constexpr std::array<ArgumentDesc, PadV2::input_arg_count + PadV2::output_arg_count> PadV2::argument_descs;
constexpr std::array<AttributeDesc, 2> PadV2::attribute_descs;

constexpr std::array<ArgumentDesc, MirrorPad::input_arg_count + MirrorPad::output_arg_count> MirrorPad::argument_descs;
constexpr std::array<AttributeDesc, 3> MirrorPad::attribute_descs;

constexpr std::array<ArgumentDesc, MirrorPadGrad::input_arg_count + MirrorPadGrad::output_arg_count> MirrorPadGrad::argument_descs;
constexpr std::array<AttributeDesc, 3> MirrorPadGrad::attribute_descs;

constexpr std::array<ArgumentDesc, Placeholder::input_arg_count + Placeholder::output_arg_count> Placeholder::argument_descs;
constexpr std::array<AttributeDesc, 2> Placeholder::attribute_descs;

constexpr std::array<ArgumentDesc, GeneratorDataset::input_arg_count + GeneratorDataset::output_arg_count> GeneratorDataset::argument_descs;
constexpr std::array<AttributeDesc, 9> GeneratorDataset::attribute_descs;

constexpr std::array<ArgumentDesc, RiscLogicalNot::input_arg_count + RiscLogicalNot::output_arg_count> RiscLogicalNot::argument_descs;
constexpr std::array<AttributeDesc, 0> RiscLogicalNot::attribute_descs;

constexpr std::array<ArgumentDesc, PlaceholderV2::input_arg_count + PlaceholderV2::output_arg_count> PlaceholderV2::argument_descs;
constexpr std::array<AttributeDesc, 2> PlaceholderV2::attribute_descs;

constexpr std::array<ArgumentDesc, Squeeze::input_arg_count + Squeeze::output_arg_count> Squeeze::argument_descs;
constexpr std::array<AttributeDesc, 2> Squeeze::attribute_descs;

constexpr std::array<ArgumentDesc, EmptyTensorMap::input_arg_count + EmptyTensorMap::output_arg_count> EmptyTensorMap::argument_descs;
constexpr std::array<AttributeDesc, 0> EmptyTensorMap::attribute_descs;

constexpr std::array<ArgumentDesc, AccumulatorSetGlobalStep::input_arg_count + AccumulatorSetGlobalStep::output_arg_count> AccumulatorSetGlobalStep::argument_descs;
constexpr std::array<AttributeDesc, 0> AccumulatorSetGlobalStep::attribute_descs;

constexpr std::array<ArgumentDesc, MergeV2Checkpoints::input_arg_count + MergeV2Checkpoints::output_arg_count> MergeV2Checkpoints::argument_descs;
constexpr std::array<AttributeDesc, 1> MergeV2Checkpoints::attribute_descs;

constexpr std::array<ArgumentDesc, ImageSummary::input_arg_count + ImageSummary::output_arg_count> ImageSummary::argument_descs;
constexpr std::array<AttributeDesc, 3> ImageSummary::attribute_descs;

constexpr std::array<ArgumentDesc, SpaceToBatchND::input_arg_count + SpaceToBatchND::output_arg_count> SpaceToBatchND::argument_descs;
constexpr std::array<AttributeDesc, 3> SpaceToBatchND::attribute_descs;

constexpr std::array<ArgumentDesc, InterleaveDataset::input_arg_count + InterleaveDataset::output_arg_count> InterleaveDataset::argument_descs;
constexpr std::array<AttributeDesc, 5> InterleaveDataset::attribute_descs;

constexpr std::array<ArgumentDesc, SpaceToBatch::input_arg_count + SpaceToBatch::output_arg_count> SpaceToBatch::argument_descs;
constexpr std::array<AttributeDesc, 3> SpaceToBatch::attribute_descs;

constexpr std::array<ArgumentDesc, PriorityQueueV2::input_arg_count + PriorityQueueV2::output_arg_count> PriorityQueueV2::argument_descs;
constexpr std::array<AttributeDesc, 5> PriorityQueueV2::attribute_descs;

constexpr std::array<ArgumentDesc, BatchToSpaceND::input_arg_count + BatchToSpaceND::output_arg_count> BatchToSpaceND::argument_descs;
constexpr std::array<AttributeDesc, 3> BatchToSpaceND::attribute_descs;

constexpr std::array<ArgumentDesc, DivNoNan::input_arg_count + DivNoNan::output_arg_count> DivNoNan::argument_descs;
constexpr std::array<AttributeDesc, 1> DivNoNan::attribute_descs;

constexpr std::array<ArgumentDesc, ResourceApplyAdam::input_arg_count + ResourceApplyAdam::output_arg_count> ResourceApplyAdam::argument_descs;
constexpr std::array<AttributeDesc, 3> ResourceApplyAdam::attribute_descs;

constexpr std::array<ArgumentDesc, TensorArrayWriteV2::input_arg_count + TensorArrayWriteV2::output_arg_count> TensorArrayWriteV2::argument_descs;
constexpr std::array<AttributeDesc, 1> TensorArrayWriteV2::attribute_descs;

constexpr std::array<ArgumentDesc, BatchToSpace::input_arg_count + BatchToSpace::output_arg_count> BatchToSpace::argument_descs;
constexpr std::array<AttributeDesc, 3> BatchToSpace::attribute_descs;

constexpr std::array<ArgumentDesc, SpaceToDepth::input_arg_count + SpaceToDepth::output_arg_count> SpaceToDepth::argument_descs;
constexpr std::array<AttributeDesc, 3> SpaceToDepth::attribute_descs;

constexpr std::array<ArgumentDesc, ResizeBilinear::input_arg_count + ResizeBilinear::output_arg_count> ResizeBilinear::argument_descs;
constexpr std::array<AttributeDesc, 3> ResizeBilinear::attribute_descs;

constexpr std::array<ArgumentDesc, DepthToSpace::input_arg_count + DepthToSpace::output_arg_count> DepthToSpace::argument_descs;
constexpr std::array<AttributeDesc, 3> DepthToSpace::attribute_descs;

constexpr std::array<ArgumentDesc, LoadTPUEmbeddingMomentumParameters::input_arg_count + LoadTPUEmbeddingMomentumParameters::output_arg_count> LoadTPUEmbeddingMomentumParameters::argument_descs;
constexpr std::array<AttributeDesc, 5> LoadTPUEmbeddingMomentumParameters::attribute_descs;

constexpr std::array<ArgumentDesc, SparseReduceMax::input_arg_count + SparseReduceMax::output_arg_count> SparseReduceMax::argument_descs;
constexpr std::array<AttributeDesc, 2> SparseReduceMax::attribute_descs;

constexpr std::array<ArgumentDesc, ExtractImagePatches::input_arg_count + ExtractImagePatches::output_arg_count> ExtractImagePatches::argument_descs;
constexpr std::array<AttributeDesc, 5> ExtractImagePatches::attribute_descs;

constexpr std::array<ArgumentDesc, _UnaryOpsComposition::input_arg_count + _UnaryOpsComposition::output_arg_count> _UnaryOpsComposition::argument_descs;
constexpr std::array<AttributeDesc, 2> _UnaryOpsComposition::attribute_descs;

constexpr std::array<ArgumentDesc, ExtractVolumePatches::input_arg_count + ExtractVolumePatches::output_arg_count> ExtractVolumePatches::argument_descs;
constexpr std::array<AttributeDesc, 4> ExtractVolumePatches::attribute_descs;

constexpr std::array<ArgumentDesc, FractionalMaxPool::input_arg_count + FractionalMaxPool::output_arg_count> FractionalMaxPool::argument_descs;
constexpr std::array<AttributeDesc, 7> FractionalMaxPool::attribute_descs;

constexpr std::array<ArgumentDesc, BoostedTreesBucketize::input_arg_count + BoostedTreesBucketize::output_arg_count> BoostedTreesBucketize::argument_descs;
constexpr std::array<AttributeDesc, 1> BoostedTreesBucketize::attribute_descs;

constexpr std::array<ArgumentDesc, OneHot::input_arg_count + OneHot::output_arg_count> OneHot::argument_descs;
constexpr std::array<AttributeDesc, 3> OneHot::attribute_descs;

constexpr std::array<ArgumentDesc, CollectiveReduce::input_arg_count + CollectiveReduce::output_arg_count> CollectiveReduce::argument_descs;
constexpr std::array<AttributeDesc, 10> CollectiveReduce::attribute_descs;

constexpr std::array<ArgumentDesc, QuantizeAndDequantize::input_arg_count + QuantizeAndDequantize::output_arg_count> QuantizeAndDequantize::argument_descs;
constexpr std::array<AttributeDesc, 6> QuantizeAndDequantize::attribute_descs;

constexpr std::array<ArgumentDesc, CollectiveReduceV2::input_arg_count + CollectiveReduceV2::output_arg_count> CollectiveReduceV2::argument_descs;
constexpr std::array<AttributeDesc, 7> CollectiveReduceV2::attribute_descs;

constexpr std::array<ArgumentDesc, QuantizeAndDequantizeV2::input_arg_count + QuantizeAndDequantizeV2::output_arg_count> QuantizeAndDequantizeV2::argument_descs;
constexpr std::array<AttributeDesc, 7> QuantizeAndDequantizeV2::attribute_descs;

constexpr std::array<ArgumentDesc, QuantizeAndDequantizeV4Grad::input_arg_count + QuantizeAndDequantizeV4Grad::output_arg_count> QuantizeAndDequantizeV4Grad::argument_descs;
constexpr std::array<AttributeDesc, 2> QuantizeAndDequantizeV4Grad::attribute_descs;

constexpr std::array<ArgumentDesc, QueueDequeueManyV2::input_arg_count + QueueDequeueManyV2::output_arg_count> QueueDequeueManyV2::argument_descs;
constexpr std::array<AttributeDesc, 2> QueueDequeueManyV2::attribute_descs;

constexpr std::array<ArgumentDesc, ScatterNdNonAliasingAdd::input_arg_count + ScatterNdNonAliasingAdd::output_arg_count> ScatterNdNonAliasingAdd::argument_descs;
constexpr std::array<AttributeDesc, 2> ScatterNdNonAliasingAdd::attribute_descs;

constexpr std::array<ArgumentDesc, PrintV2::input_arg_count + PrintV2::output_arg_count> PrintV2::argument_descs;
constexpr std::array<AttributeDesc, 2> PrintV2::attribute_descs;

constexpr std::array<ArgumentDesc, CollectiveReduceV3::input_arg_count + CollectiveReduceV3::output_arg_count> CollectiveReduceV3::argument_descs;
constexpr std::array<AttributeDesc, 3> CollectiveReduceV3::attribute_descs;

constexpr std::array<ArgumentDesc, BoostedTreesAggregateStats::input_arg_count + BoostedTreesAggregateStats::output_arg_count> BoostedTreesAggregateStats::argument_descs;
constexpr std::array<AttributeDesc, 2> BoostedTreesAggregateStats::attribute_descs;

constexpr std::array<ArgumentDesc, QuantizeAndDequantizeV3::input_arg_count + QuantizeAndDequantizeV3::output_arg_count> QuantizeAndDequantizeV3::argument_descs;
constexpr std::array<AttributeDesc, 5> QuantizeAndDequantizeV3::attribute_descs;

constexpr std::array<ArgumentDesc, UnicodeDecodeWithOffsets::input_arg_count + UnicodeDecodeWithOffsets::output_arg_count> UnicodeDecodeWithOffsets::argument_descs;
constexpr std::array<AttributeDesc, 5> UnicodeDecodeWithOffsets::attribute_descs;

constexpr std::array<ArgumentDesc, NcclAllReduce::input_arg_count + NcclAllReduce::output_arg_count> NcclAllReduce::argument_descs;
constexpr std::array<AttributeDesc, 4> NcclAllReduce::attribute_descs;

constexpr std::array<ArgumentDesc, QuantizeV2::input_arg_count + QuantizeV2::output_arg_count> QuantizeV2::argument_descs;
constexpr std::array<AttributeDesc, 6> QuantizeV2::attribute_descs;

constexpr std::array<ArgumentDesc, TensorArrayGrad::input_arg_count + TensorArrayGrad::output_arg_count> TensorArrayGrad::argument_descs;
constexpr std::array<AttributeDesc, 1> TensorArrayGrad::attribute_descs;

constexpr std::array<ArgumentDesc, BitwiseOr::input_arg_count + BitwiseOr::output_arg_count> BitwiseOr::argument_descs;
constexpr std::array<AttributeDesc, 1> BitwiseOr::attribute_descs;

constexpr std::array<ArgumentDesc, Dequantize::input_arg_count + Dequantize::output_arg_count> Dequantize::argument_descs;
constexpr std::array<AttributeDesc, 5> Dequantize::attribute_descs;

constexpr std::array<ArgumentDesc, RiscMin::input_arg_count + RiscMin::output_arg_count> RiscMin::argument_descs;
constexpr std::array<AttributeDesc, 1> RiscMin::attribute_descs;

constexpr std::array<ArgumentDesc, QuantizedConcat::input_arg_count + QuantizedConcat::output_arg_count> QuantizedConcat::argument_descs;
constexpr std::array<AttributeDesc, 2> QuantizedConcat::attribute_descs;

constexpr std::array<ArgumentDesc, EncodeJpegVariableQuality::input_arg_count + EncodeJpegVariableQuality::output_arg_count> EncodeJpegVariableQuality::argument_descs;
constexpr std::array<AttributeDesc, 0> EncodeJpegVariableQuality::attribute_descs;

constexpr std::array<ArgumentDesc, QuantizedReshape::input_arg_count + QuantizedReshape::output_arg_count> QuantizedReshape::argument_descs;
constexpr std::array<AttributeDesc, 2> QuantizedReshape::attribute_descs;

constexpr std::array<ArgumentDesc, SparseSegmentMeanWithNumSegments::input_arg_count + SparseSegmentMeanWithNumSegments::output_arg_count> SparseSegmentMeanWithNumSegments::argument_descs;
constexpr std::array<AttributeDesc, 4> SparseSegmentMeanWithNumSegments::attribute_descs;

constexpr std::array<ArgumentDesc, TensorArraySizeV3::input_arg_count + TensorArraySizeV3::output_arg_count> TensorArraySizeV3::argument_descs;
constexpr std::array<AttributeDesc, 0> TensorArraySizeV3::attribute_descs;

constexpr std::array<ArgumentDesc, MapDefun::input_arg_count + MapDefun::output_arg_count> MapDefun::argument_descs;
constexpr std::array<AttributeDesc, 6> MapDefun::attribute_descs;

constexpr std::array<ArgumentDesc, QuantizedInstanceNorm::input_arg_count + QuantizedInstanceNorm::output_arg_count> QuantizedInstanceNorm::argument_descs;
constexpr std::array<AttributeDesc, 6> QuantizedInstanceNorm::attribute_descs;

constexpr std::array<ArgumentDesc, FilterDataset::input_arg_count + FilterDataset::output_arg_count> FilterDataset::argument_descs;
constexpr std::array<AttributeDesc, 5> FilterDataset::attribute_descs;

constexpr std::array<ArgumentDesc, UpperBound::input_arg_count + UpperBound::output_arg_count> UpperBound::argument_descs;
constexpr std::array<AttributeDesc, 2> UpperBound::attribute_descs;

constexpr std::array<ArgumentDesc, TensorListConcatLists::input_arg_count + TensorListConcatLists::output_arg_count> TensorListConcatLists::argument_descs;
constexpr std::array<AttributeDesc, 1> TensorListConcatLists::attribute_descs;

constexpr std::array<ArgumentDesc, XlaDynamicSlice::input_arg_count + XlaDynamicSlice::output_arg_count> XlaDynamicSlice::argument_descs;
constexpr std::array<AttributeDesc, 2> XlaDynamicSlice::attribute_descs;

constexpr std::array<ArgumentDesc, AnonymousRandomSeedGenerator::input_arg_count + AnonymousRandomSeedGenerator::output_arg_count> AnonymousRandomSeedGenerator::argument_descs;
constexpr std::array<AttributeDesc, 0> AnonymousRandomSeedGenerator::attribute_descs;

constexpr std::array<ArgumentDesc, LowerBound::input_arg_count + LowerBound::output_arg_count> LowerBound::argument_descs;
constexpr std::array<AttributeDesc, 2> LowerBound::attribute_descs;

constexpr std::array<ArgumentDesc, Relu6Grad::input_arg_count + Relu6Grad::output_arg_count> Relu6Grad::argument_descs;
constexpr std::array<AttributeDesc, 1> Relu6Grad::attribute_descs;

constexpr std::array<ArgumentDesc, ScatterNd::input_arg_count + ScatterNd::output_arg_count> ScatterNd::argument_descs;
constexpr std::array<AttributeDesc, 2> ScatterNd::attribute_descs;

constexpr std::array<ArgumentDesc, TensorScatterAdd::input_arg_count + TensorScatterAdd::output_arg_count> TensorScatterAdd::argument_descs;
constexpr std::array<AttributeDesc, 2> TensorScatterAdd::attribute_descs;

constexpr std::array<ArgumentDesc, SelectV2::input_arg_count + SelectV2::output_arg_count> SelectV2::argument_descs;
constexpr std::array<AttributeDesc, 1> SelectV2::attribute_descs;

constexpr std::array<ArgumentDesc, GetSessionTensor::input_arg_count + GetSessionTensor::output_arg_count> GetSessionTensor::argument_descs;
constexpr std::array<AttributeDesc, 1> GetSessionTensor::attribute_descs;

constexpr std::array<ArgumentDesc, ScaleAndTranslateGrad::input_arg_count + ScaleAndTranslateGrad::output_arg_count> ScaleAndTranslateGrad::argument_descs;
constexpr std::array<AttributeDesc, 3> ScaleAndTranslateGrad::attribute_descs;

constexpr std::array<ArgumentDesc, InfeedEnqueuePrelinearizedBuffer::input_arg_count + InfeedEnqueuePrelinearizedBuffer::output_arg_count> InfeedEnqueuePrelinearizedBuffer::argument_descs;
constexpr std::array<AttributeDesc, 1> InfeedEnqueuePrelinearizedBuffer::attribute_descs;

constexpr std::array<ArgumentDesc, DeserializeSparse::input_arg_count + DeserializeSparse::output_arg_count> DeserializeSparse::argument_descs;
constexpr std::array<AttributeDesc, 2> DeserializeSparse::attribute_descs;

constexpr std::array<ArgumentDesc, TensorScatterSub::input_arg_count + TensorScatterSub::output_arg_count> TensorScatterSub::argument_descs;
constexpr std::array<AttributeDesc, 2> TensorScatterSub::attribute_descs;

constexpr std::array<ArgumentDesc, MatrixInverse::input_arg_count + MatrixInverse::output_arg_count> MatrixInverse::argument_descs;
constexpr std::array<AttributeDesc, 2> MatrixInverse::attribute_descs;

constexpr std::array<ArgumentDesc, AdjustContrastv2::input_arg_count + AdjustContrastv2::output_arg_count> AdjustContrastv2::argument_descs;
constexpr std::array<AttributeDesc, 1> AdjustContrastv2::attribute_descs;

constexpr std::array<ArgumentDesc, ParallelMapDatasetV2::input_arg_count + ParallelMapDatasetV2::output_arg_count> ParallelMapDatasetV2::argument_descs;
constexpr std::array<AttributeDesc, 8> ParallelMapDatasetV2::attribute_descs;

constexpr std::array<ArgumentDesc, QueueCloseV2::input_arg_count + QueueCloseV2::output_arg_count> QueueCloseV2::argument_descs;
constexpr std::array<AttributeDesc, 1> QueueCloseV2::attribute_descs;

constexpr std::array<ArgumentDesc, TensorScatterMax::input_arg_count + TensorScatterMax::output_arg_count> TensorScatterMax::argument_descs;
constexpr std::array<AttributeDesc, 2> TensorScatterMax::attribute_descs;

constexpr std::array<ArgumentDesc, RiscNeg::input_arg_count + RiscNeg::output_arg_count> RiscNeg::argument_descs;
constexpr std::array<AttributeDesc, 1> RiscNeg::attribute_descs;

constexpr std::array<ArgumentDesc, FakeQuantWithMinMaxArgsGradient::input_arg_count + FakeQuantWithMinMaxArgsGradient::output_arg_count> FakeQuantWithMinMaxArgsGradient::argument_descs;
constexpr std::array<AttributeDesc, 4> FakeQuantWithMinMaxArgsGradient::attribute_descs;

constexpr std::array<ArgumentDesc, BatchCholesky::input_arg_count + BatchCholesky::output_arg_count> BatchCholesky::argument_descs;
constexpr std::array<AttributeDesc, 1> BatchCholesky::attribute_descs;

constexpr std::array<ArgumentDesc, OutfeedEnqueue::input_arg_count + OutfeedEnqueue::output_arg_count> OutfeedEnqueue::argument_descs;
constexpr std::array<AttributeDesc, 1> OutfeedEnqueue::attribute_descs;

constexpr std::array<ArgumentDesc, TPUPartitionedCall::input_arg_count + TPUPartitionedCall::output_arg_count> TPUPartitionedCall::argument_descs;
constexpr std::array<AttributeDesc, 4> TPUPartitionedCall::attribute_descs;

constexpr std::array<ArgumentDesc, FakeQuantWithMinMaxVarsGradient::input_arg_count + FakeQuantWithMinMaxVarsGradient::output_arg_count> FakeQuantWithMinMaxVarsGradient::argument_descs;
constexpr std::array<AttributeDesc, 2> FakeQuantWithMinMaxVarsGradient::attribute_descs;

constexpr std::array<ArgumentDesc, CSVDatasetV2::input_arg_count + CSVDatasetV2::output_arg_count> CSVDatasetV2::argument_descs;
constexpr std::array<AttributeDesc, 2> CSVDatasetV2::attribute_descs;

constexpr std::array<ArgumentDesc, FakeQuantWithMinMaxVarsPerChannel::input_arg_count + FakeQuantWithMinMaxVarsPerChannel::output_arg_count> FakeQuantWithMinMaxVarsPerChannel::argument_descs;
constexpr std::array<AttributeDesc, 2> FakeQuantWithMinMaxVarsPerChannel::attribute_descs;

constexpr std::array<ArgumentDesc, FakeQuantWithMinMaxVarsPerChannelGradient::input_arg_count + FakeQuantWithMinMaxVarsPerChannelGradient::output_arg_count> FakeQuantWithMinMaxVarsPerChannelGradient::argument_descs;
constexpr std::array<AttributeDesc, 2> FakeQuantWithMinMaxVarsPerChannelGradient::attribute_descs;

constexpr std::array<ArgumentDesc, UnsortedSegmentProd::input_arg_count + UnsortedSegmentProd::output_arg_count> UnsortedSegmentProd::argument_descs;
constexpr std::array<AttributeDesc, 3> UnsortedSegmentProd::attribute_descs;

constexpr std::array<ArgumentDesc, ResourceSparseApplyAdagrad::input_arg_count + ResourceSparseApplyAdagrad::output_arg_count> ResourceSparseApplyAdagrad::argument_descs;
constexpr std::array<AttributeDesc, 4> ResourceSparseApplyAdagrad::attribute_descs;

constexpr std::array<ArgumentDesc, GroupByReducerDataset::input_arg_count + GroupByReducerDataset::output_arg_count> GroupByReducerDataset::argument_descs;
constexpr std::array<AttributeDesc, 10> GroupByReducerDataset::attribute_descs;

constexpr std::array<ArgumentDesc, Fingerprint::input_arg_count + Fingerprint::output_arg_count> Fingerprint::argument_descs;
constexpr std::array<AttributeDesc, 1> Fingerprint::attribute_descs;

constexpr std::array<ArgumentDesc, ResizeBicubic::input_arg_count + ResizeBicubic::output_arg_count> ResizeBicubic::argument_descs;
constexpr std::array<AttributeDesc, 3> ResizeBicubic::attribute_descs;

constexpr std::array<ArgumentDesc, _MklConcat::input_arg_count + _MklConcat::output_arg_count> _MklConcat::argument_descs;
constexpr std::array<AttributeDesc, 2> _MklConcat::attribute_descs;

constexpr std::array<ArgumentDesc, TensorArrayConcat::input_arg_count + TensorArrayConcat::output_arg_count> TensorArrayConcat::argument_descs;
constexpr std::array<AttributeDesc, 2> TensorArrayConcat::attribute_descs;

constexpr std::array<ArgumentDesc, BatchMatrixSetDiag::input_arg_count + BatchMatrixSetDiag::output_arg_count> BatchMatrixSetDiag::argument_descs;
constexpr std::array<AttributeDesc, 1> BatchMatrixSetDiag::attribute_descs;

constexpr std::array<ArgumentDesc, AnonymousMemoryCache::input_arg_count + AnonymousMemoryCache::output_arg_count> AnonymousMemoryCache::argument_descs;
constexpr std::array<AttributeDesc, 0> AnonymousMemoryCache::attribute_descs;

constexpr std::array<ArgumentDesc, BatchMatrixBandPart::input_arg_count + BatchMatrixBandPart::output_arg_count> BatchMatrixBandPart::argument_descs;
constexpr std::array<AttributeDesc, 1> BatchMatrixBandPart::attribute_descs;

constexpr std::array<ArgumentDesc, IteratorGetNext::input_arg_count + IteratorGetNext::output_arg_count> IteratorGetNext::argument_descs;
constexpr std::array<AttributeDesc, 2> IteratorGetNext::attribute_descs;

constexpr std::array<ArgumentDesc, ExperimentalPrivateThreadPoolDataset::input_arg_count + ExperimentalPrivateThreadPoolDataset::output_arg_count> ExperimentalPrivateThreadPoolDataset::argument_descs;
constexpr std::array<AttributeDesc, 2> ExperimentalPrivateThreadPoolDataset::attribute_descs;

constexpr std::array<ArgumentDesc, DecodeWav::input_arg_count + DecodeWav::output_arg_count> DecodeWav::argument_descs;
constexpr std::array<AttributeDesc, 2> DecodeWav::attribute_descs;

constexpr std::array<ArgumentDesc, CacheDatasetV2::input_arg_count + CacheDatasetV2::output_arg_count> CacheDatasetV2::argument_descs;
constexpr std::array<AttributeDesc, 3> CacheDatasetV2::attribute_descs;

constexpr std::array<ArgumentDesc, FractionalAvgPoolGrad::input_arg_count + FractionalAvgPoolGrad::output_arg_count> FractionalAvgPoolGrad::argument_descs;
constexpr std::array<AttributeDesc, 2> FractionalAvgPoolGrad::attribute_descs;

constexpr std::array<ArgumentDesc, Mfcc::input_arg_count + Mfcc::output_arg_count> Mfcc::argument_descs;
constexpr std::array<AttributeDesc, 4> Mfcc::attribute_descs;

constexpr std::array<ArgumentDesc, PaddedBatchDataset::input_arg_count + PaddedBatchDataset::output_arg_count> PaddedBatchDataset::argument_descs;
constexpr std::array<AttributeDesc, 4> PaddedBatchDataset::attribute_descs;

constexpr std::array<ArgumentDesc, IFFT3D::input_arg_count + IFFT3D::output_arg_count> IFFT3D::argument_descs;
constexpr std::array<AttributeDesc, 1> IFFT3D::attribute_descs;

constexpr std::array<ArgumentDesc, BatchFunction::input_arg_count + BatchFunction::output_arg_count> BatchFunction::argument_descs;
constexpr std::array<AttributeDesc, 13> BatchFunction::attribute_descs;

constexpr std::array<ArgumentDesc, SerializeIterator::input_arg_count + SerializeIterator::output_arg_count> SerializeIterator::argument_descs;
constexpr std::array<AttributeDesc, 1> SerializeIterator::attribute_descs;

constexpr std::array<ArgumentDesc, Batch::input_arg_count + Batch::output_arg_count> Batch::argument_descs;
constexpr std::array<AttributeDesc, 10> Batch::attribute_descs;

constexpr std::array<ArgumentDesc, RsqrtGrad::input_arg_count + RsqrtGrad::output_arg_count> RsqrtGrad::argument_descs;
constexpr std::array<AttributeDesc, 1> RsqrtGrad::attribute_descs;

constexpr std::array<ArgumentDesc, Unbatch::input_arg_count + Unbatch::output_arg_count> Unbatch::argument_descs;
constexpr std::array<AttributeDesc, 4> Unbatch::attribute_descs;

constexpr std::array<ArgumentDesc, InTopK::input_arg_count + InTopK::output_arg_count> InTopK::argument_descs;
constexpr std::array<AttributeDesc, 2> InTopK::attribute_descs;

constexpr std::array<ArgumentDesc, SparseTensorSliceDataset::input_arg_count + SparseTensorSliceDataset::output_arg_count> SparseTensorSliceDataset::argument_descs;
constexpr std::array<AttributeDesc, 1> SparseTensorSliceDataset::attribute_descs;

constexpr std::array<ArgumentDesc, UnbatchGrad::input_arg_count + UnbatchGrad::output_arg_count> UnbatchGrad::argument_descs;
constexpr std::array<AttributeDesc, 3> UnbatchGrad::attribute_descs;

constexpr std::array<ArgumentDesc, BoostedTreesCenterBias::input_arg_count + BoostedTreesCenterBias::output_arg_count> BoostedTreesCenterBias::argument_descs;
constexpr std::array<AttributeDesc, 0> BoostedTreesCenterBias::attribute_descs;

constexpr std::array<ArgumentDesc, SparseApplyAdagrad::input_arg_count + SparseApplyAdagrad::output_arg_count> SparseApplyAdagrad::argument_descs;
constexpr std::array<AttributeDesc, 4> SparseApplyAdagrad::attribute_descs;

constexpr std::array<ArgumentDesc, Invert::input_arg_count + Invert::output_arg_count> Invert::argument_descs;
constexpr std::array<AttributeDesc, 1> Invert::attribute_descs;

constexpr std::array<ArgumentDesc, LookupTableRemoveV2::input_arg_count + LookupTableRemoveV2::output_arg_count> LookupTableRemoveV2::argument_descs;
constexpr std::array<AttributeDesc, 1> LookupTableRemoveV2::attribute_descs;

constexpr std::array<ArgumentDesc, _If::input_arg_count + _If::output_arg_count> _If::argument_descs;
constexpr std::array<AttributeDesc, 5> _If::attribute_descs;

constexpr std::array<ArgumentDesc, PopulationCount::input_arg_count + PopulationCount::output_arg_count> PopulationCount::argument_descs;
constexpr std::array<AttributeDesc, 1> PopulationCount::attribute_descs;

constexpr std::array<ArgumentDesc, TensorArray::input_arg_count + TensorArray::output_arg_count> TensorArray::argument_descs;
constexpr std::array<AttributeDesc, 5> TensorArray::attribute_descs;

constexpr std::array<ArgumentDesc, BitwiseAnd::input_arg_count + BitwiseAnd::output_arg_count> BitwiseAnd::argument_descs;
constexpr std::array<AttributeDesc, 1> BitwiseAnd::attribute_descs;

constexpr std::array<ArgumentDesc, _MklNativeConv2DBackpropFilter::input_arg_count + _MklNativeConv2DBackpropFilter::output_arg_count> _MklNativeConv2DBackpropFilter::argument_descs;
constexpr std::array<AttributeDesc, 7> _MklNativeConv2DBackpropFilter::attribute_descs;

constexpr std::array<ArgumentDesc, BitwiseXor::input_arg_count + BitwiseXor::output_arg_count> BitwiseXor::argument_descs;
constexpr std::array<AttributeDesc, 1> BitwiseXor::attribute_descs;

constexpr std::array<ArgumentDesc, Add::input_arg_count + Add::output_arg_count> Add::argument_descs;
constexpr std::array<AttributeDesc, 1> Add::attribute_descs;

constexpr std::array<ArgumentDesc, CollectiveInitializeCommunicator::input_arg_count + CollectiveInitializeCommunicator::output_arg_count> CollectiveInitializeCommunicator::argument_descs;
constexpr std::array<AttributeDesc, 2> CollectiveInitializeCommunicator::attribute_descs;

constexpr std::array<ArgumentDesc, BoostedTreesEnsembleResourceHandleOp::input_arg_count + BoostedTreesEnsembleResourceHandleOp::output_arg_count> BoostedTreesEnsembleResourceHandleOp::argument_descs;
constexpr std::array<AttributeDesc, 2> BoostedTreesEnsembleResourceHandleOp::attribute_descs;

constexpr std::array<ArgumentDesc, IsBoostedTreesEnsembleInitialized::input_arg_count + IsBoostedTreesEnsembleInitialized::output_arg_count> IsBoostedTreesEnsembleInitialized::argument_descs;
constexpr std::array<AttributeDesc, 0> IsBoostedTreesEnsembleInitialized::attribute_descs;

constexpr std::array<ArgumentDesc, _MklFusedBatchMatMulV2::input_arg_count + _MklFusedBatchMatMulV2::output_arg_count> _MklFusedBatchMatMulV2::argument_descs;
constexpr std::array<AttributeDesc, 5> _MklFusedBatchMatMulV2::attribute_descs;

constexpr std::array<ArgumentDesc, OutfeedDequeueTupleV2::input_arg_count + OutfeedDequeueTupleV2::output_arg_count> OutfeedDequeueTupleV2::argument_descs;
constexpr std::array<AttributeDesc, 2> OutfeedDequeueTupleV2::attribute_descs;

constexpr std::array<ArgumentDesc, ResourceAccumulatorNumAccumulated::input_arg_count + ResourceAccumulatorNumAccumulated::output_arg_count> ResourceAccumulatorNumAccumulated::argument_descs;
constexpr std::array<AttributeDesc, 0> ResourceAccumulatorNumAccumulated::attribute_descs;

constexpr std::array<ArgumentDesc, BoostedTreesCalculateBestGainsPerFeature::input_arg_count + BoostedTreesCalculateBestGainsPerFeature::output_arg_count> BoostedTreesCalculateBestGainsPerFeature::argument_descs;
constexpr std::array<AttributeDesc, 2> BoostedTreesCalculateBestGainsPerFeature::attribute_descs;

constexpr std::array<ArgumentDesc, TensorArrayGather::input_arg_count + TensorArrayGather::output_arg_count> TensorArrayGather::argument_descs;
constexpr std::array<AttributeDesc, 2> TensorArrayGather::attribute_descs;

constexpr std::array<ArgumentDesc, BoostedTreesCalculateBestFeatureSplit::input_arg_count + BoostedTreesCalculateBestFeatureSplit::output_arg_count> BoostedTreesCalculateBestFeatureSplit::argument_descs;
constexpr std::array<AttributeDesc, 2> BoostedTreesCalculateBestFeatureSplit::attribute_descs;

constexpr std::array<ArgumentDesc, ShuffleDataset::input_arg_count + ShuffleDataset::output_arg_count> ShuffleDataset::argument_descs;
constexpr std::array<AttributeDesc, 4> ShuffleDataset::attribute_descs;

constexpr std::array<ArgumentDesc, BarrierInsertMany::input_arg_count + BarrierInsertMany::output_arg_count> BarrierInsertMany::argument_descs;
constexpr std::array<AttributeDesc, 2> BarrierInsertMany::attribute_descs;

constexpr std::array<ArgumentDesc, BoostedTreesSparseCalculateBestFeatureSplit::input_arg_count + BoostedTreesSparseCalculateBestFeatureSplit::output_arg_count> BoostedTreesSparseCalculateBestFeatureSplit::argument_descs;
constexpr std::array<AttributeDesc, 2> BoostedTreesSparseCalculateBestFeatureSplit::attribute_descs;

constexpr std::array<ArgumentDesc, BoostedTreesCreateEnsemble::input_arg_count + BoostedTreesCreateEnsemble::output_arg_count> BoostedTreesCreateEnsemble::argument_descs;
constexpr std::array<AttributeDesc, 0> BoostedTreesCreateEnsemble::attribute_descs;

constexpr std::array<ArgumentDesc, BiasAddV1::input_arg_count + BiasAddV1::output_arg_count> BiasAddV1::argument_descs;
constexpr std::array<AttributeDesc, 1> BiasAddV1::attribute_descs;

constexpr std::array<ArgumentDesc, BoostedTreesDeserializeEnsemble::input_arg_count + BoostedTreesDeserializeEnsemble::output_arg_count> BoostedTreesDeserializeEnsemble::argument_descs;
constexpr std::array<AttributeDesc, 0> BoostedTreesDeserializeEnsemble::attribute_descs;

constexpr std::array<ArgumentDesc, LoadTPUEmbeddingProximalYogiParameters::input_arg_count + LoadTPUEmbeddingProximalYogiParameters::output_arg_count> LoadTPUEmbeddingProximalYogiParameters::argument_descs;
constexpr std::array<AttributeDesc, 5> LoadTPUEmbeddingProximalYogiParameters::attribute_descs;

constexpr std::array<ArgumentDesc, RandomGamma::input_arg_count + RandomGamma::output_arg_count> RandomGamma::argument_descs;
constexpr std::array<AttributeDesc, 4> RandomGamma::attribute_descs;

constexpr std::array<ArgumentDesc, ResourceConditionalAccumulator::input_arg_count + ResourceConditionalAccumulator::output_arg_count> ResourceConditionalAccumulator::argument_descs;
constexpr std::array<AttributeDesc, 5> ResourceConditionalAccumulator::attribute_descs;

constexpr std::array<ArgumentDesc, BoostedTreesGetEnsembleStates::input_arg_count + BoostedTreesGetEnsembleStates::output_arg_count> BoostedTreesGetEnsembleStates::argument_descs;
constexpr std::array<AttributeDesc, 0> BoostedTreesGetEnsembleStates::attribute_descs;

constexpr std::array<ArgumentDesc, NegTrain::input_arg_count + NegTrain::output_arg_count> NegTrain::argument_descs;
constexpr std::array<AttributeDesc, 2> NegTrain::attribute_descs;

constexpr std::array<ArgumentDesc, BoostedTreesMakeStatsSummary::input_arg_count + BoostedTreesMakeStatsSummary::output_arg_count> BoostedTreesMakeStatsSummary::argument_descs;
constexpr std::array<AttributeDesc, 3> BoostedTreesMakeStatsSummary::attribute_descs;

constexpr std::array<ArgumentDesc, BoostedTreesSparseAggregateStats::input_arg_count + BoostedTreesSparseAggregateStats::output_arg_count> BoostedTreesSparseAggregateStats::argument_descs;
constexpr std::array<AttributeDesc, 2> BoostedTreesSparseAggregateStats::attribute_descs;

constexpr std::array<ArgumentDesc, BoostedTreesPredict::input_arg_count + BoostedTreesPredict::output_arg_count> BoostedTreesPredict::argument_descs;
constexpr std::array<AttributeDesc, 2> BoostedTreesPredict::attribute_descs;

constexpr std::array<ArgumentDesc, ApplyProximalAdagrad::input_arg_count + ApplyProximalAdagrad::output_arg_count> ApplyProximalAdagrad::argument_descs;
constexpr std::array<AttributeDesc, 2> ApplyProximalAdagrad::attribute_descs;

constexpr std::array<ArgumentDesc, BoostedTreesExampleDebugOutputs::input_arg_count + BoostedTreesExampleDebugOutputs::output_arg_count> BoostedTreesExampleDebugOutputs::argument_descs;
constexpr std::array<AttributeDesc, 2> BoostedTreesExampleDebugOutputs::attribute_descs;

constexpr std::array<ArgumentDesc, ComputeBatchSize::input_arg_count + ComputeBatchSize::output_arg_count> ComputeBatchSize::argument_descs;
constexpr std::array<AttributeDesc, 0> ComputeBatchSize::attribute_descs;

constexpr std::array<ArgumentDesc, ClipByValue::input_arg_count + ClipByValue::output_arg_count> ClipByValue::argument_descs;
constexpr std::array<AttributeDesc, 1> ClipByValue::attribute_descs;

constexpr std::array<ArgumentDesc, SigmoidGrad::input_arg_count + SigmoidGrad::output_arg_count> SigmoidGrad::argument_descs;
constexpr std::array<AttributeDesc, 1> SigmoidGrad::attribute_descs;

constexpr std::array<ArgumentDesc, LoadTPUEmbeddingAdagradParameters::input_arg_count + LoadTPUEmbeddingAdagradParameters::output_arg_count> LoadTPUEmbeddingAdagradParameters::argument_descs;
constexpr std::array<AttributeDesc, 5> LoadTPUEmbeddingAdagradParameters::attribute_descs;

constexpr std::array<ArgumentDesc, BoostedTreesSerializeEnsemble::input_arg_count + BoostedTreesSerializeEnsemble::output_arg_count> BoostedTreesSerializeEnsemble::argument_descs;
constexpr std::array<AttributeDesc, 0> BoostedTreesSerializeEnsemble::attribute_descs;

constexpr std::array<ArgumentDesc, BoostedTreesTrainingPredict::input_arg_count + BoostedTreesTrainingPredict::output_arg_count> BoostedTreesTrainingPredict::argument_descs;
constexpr std::array<AttributeDesc, 2> BoostedTreesTrainingPredict::attribute_descs;

constexpr std::array<ArgumentDesc, SnapshotNestedDatasetReader::input_arg_count + SnapshotNestedDatasetReader::output_arg_count> SnapshotNestedDatasetReader::argument_descs;
constexpr std::array<AttributeDesc, 3> SnapshotNestedDatasetReader::attribute_descs;

constexpr std::array<ArgumentDesc, BoostedTreesUpdateEnsemble::input_arg_count + BoostedTreesUpdateEnsemble::output_arg_count> BoostedTreesUpdateEnsemble::argument_descs;
constexpr std::array<AttributeDesc, 2> BoostedTreesUpdateEnsemble::attribute_descs;

constexpr std::array<ArgumentDesc, BoostedTreesUpdateEnsembleV2::input_arg_count + BoostedTreesUpdateEnsembleV2::output_arg_count> BoostedTreesUpdateEnsembleV2::argument_descs;
constexpr std::array<AttributeDesc, 3> BoostedTreesUpdateEnsembleV2::attribute_descs;

constexpr std::array<ArgumentDesc, RangeDataset::input_arg_count + RangeDataset::output_arg_count> RangeDataset::argument_descs;
constexpr std::array<AttributeDesc, 3> RangeDataset::attribute_descs;

constexpr std::array<ArgumentDesc, BoostedTreesQuantileStreamResourceHandleOp::input_arg_count + BoostedTreesQuantileStreamResourceHandleOp::output_arg_count> BoostedTreesQuantileStreamResourceHandleOp::argument_descs;
constexpr std::array<AttributeDesc, 2> BoostedTreesQuantileStreamResourceHandleOp::attribute_descs;

constexpr std::array<ArgumentDesc, ScatterNdSub::input_arg_count + ScatterNdSub::output_arg_count> ScatterNdSub::argument_descs;
constexpr std::array<AttributeDesc, 3> ScatterNdSub::attribute_descs;

constexpr std::array<ArgumentDesc, CollectiveAllToAllV3::input_arg_count + CollectiveAllToAllV3::output_arg_count> CollectiveAllToAllV3::argument_descs;
constexpr std::array<AttributeDesc, 2> CollectiveAllToAllV3::attribute_descs;

constexpr std::array<ArgumentDesc, IsBoostedTreesQuantileStreamResourceInitialized::input_arg_count + IsBoostedTreesQuantileStreamResourceInitialized::output_arg_count> IsBoostedTreesQuantileStreamResourceInitialized::argument_descs;
constexpr std::array<AttributeDesc, 0> IsBoostedTreesQuantileStreamResourceInitialized::attribute_descs;

constexpr std::array<ArgumentDesc, OrderedMapUnstage::input_arg_count + OrderedMapUnstage::output_arg_count> OrderedMapUnstage::argument_descs;
constexpr std::array<AttributeDesc, 5> OrderedMapUnstage::attribute_descs;

constexpr std::array<ArgumentDesc, BoostedTreesFlushQuantileSummaries::input_arg_count + BoostedTreesFlushQuantileSummaries::output_arg_count> BoostedTreesFlushQuantileSummaries::argument_descs;
constexpr std::array<AttributeDesc, 1> BoostedTreesFlushQuantileSummaries::attribute_descs;

constexpr std::array<ArgumentDesc, BoostedTreesQuantileStreamResourceFlush::input_arg_count + BoostedTreesQuantileStreamResourceFlush::output_arg_count> BoostedTreesQuantileStreamResourceFlush::argument_descs;
constexpr std::array<AttributeDesc, 1> BoostedTreesQuantileStreamResourceFlush::attribute_descs;

constexpr std::array<ArgumentDesc, BoostedTreesQuantileStreamResourceGetBucketBoundaries::input_arg_count + BoostedTreesQuantileStreamResourceGetBucketBoundaries::output_arg_count> BoostedTreesQuantileStreamResourceGetBucketBoundaries::argument_descs;
constexpr std::array<AttributeDesc, 1> BoostedTreesQuantileStreamResourceGetBucketBoundaries::attribute_descs;

constexpr std::array<ArgumentDesc, Cumprod::input_arg_count + Cumprod::output_arg_count> Cumprod::argument_descs;
constexpr std::array<AttributeDesc, 4> Cumprod::attribute_descs;

constexpr std::array<ArgumentDesc, UniformCandidateSampler::input_arg_count + UniformCandidateSampler::output_arg_count> UniformCandidateSampler::argument_descs;
constexpr std::array<AttributeDesc, 6> UniformCandidateSampler::attribute_descs;

constexpr std::array<ArgumentDesc, DebugNanCount::input_arg_count + DebugNanCount::output_arg_count> DebugNanCount::argument_descs;
constexpr std::array<AttributeDesc, 5> DebugNanCount::attribute_descs;

constexpr std::array<ArgumentDesc, MutableHashTable::input_arg_count + MutableHashTable::output_arg_count> MutableHashTable::argument_descs;
constexpr std::array<AttributeDesc, 5> MutableHashTable::attribute_descs;

constexpr std::array<ArgumentDesc, ThreadUnsafeUnigramCandidateSampler::input_arg_count + ThreadUnsafeUnigramCandidateSampler::output_arg_count> ThreadUnsafeUnigramCandidateSampler::argument_descs;
constexpr std::array<AttributeDesc, 6> ThreadUnsafeUnigramCandidateSampler::attribute_descs;

constexpr std::array<ArgumentDesc, _VarHandlesOp::input_arg_count + _VarHandlesOp::output_arg_count> _VarHandlesOp::argument_descs;
constexpr std::array<AttributeDesc, 5> _VarHandlesOp::attribute_descs;

constexpr std::array<ArgumentDesc, FixedUnigramCandidateSampler::input_arg_count + FixedUnigramCandidateSampler::output_arg_count> FixedUnigramCandidateSampler::argument_descs;
constexpr std::array<AttributeDesc, 12> FixedUnigramCandidateSampler::attribute_descs;

constexpr std::array<ArgumentDesc, ComputeAccidentalHits::input_arg_count + ComputeAccidentalHits::output_arg_count> ComputeAccidentalHits::argument_descs;
constexpr std::array<AttributeDesc, 3> ComputeAccidentalHits::attribute_descs;

constexpr std::array<ArgumentDesc, TensorArrayGradWithShape::input_arg_count + TensorArrayGradWithShape::output_arg_count> TensorArrayGradWithShape::argument_descs;
constexpr std::array<AttributeDesc, 1> TensorArrayGradWithShape::attribute_descs;

constexpr std::array<ArgumentDesc, LoadAndRemapMatrix::input_arg_count + LoadAndRemapMatrix::output_arg_count> LoadAndRemapMatrix::argument_descs;
constexpr std::array<AttributeDesc, 3> LoadAndRemapMatrix::attribute_descs;

constexpr std::array<ArgumentDesc, KMC2ChainInitialization::input_arg_count + KMC2ChainInitialization::output_arg_count> KMC2ChainInitialization::argument_descs;
constexpr std::array<AttributeDesc, 0> KMC2ChainInitialization::attribute_descs;

constexpr std::array<ArgumentDesc, NearestNeighbors::input_arg_count + NearestNeighbors::output_arg_count> NearestNeighbors::argument_descs;
constexpr std::array<AttributeDesc, 0> NearestNeighbors::attribute_descs;

constexpr std::array<ArgumentDesc, DecodeBmp::input_arg_count + DecodeBmp::output_arg_count> DecodeBmp::argument_descs;
constexpr std::array<AttributeDesc, 1> DecodeBmp::attribute_descs;

constexpr std::array<ArgumentDesc, QueueDequeueUpTo::input_arg_count + QueueDequeueUpTo::output_arg_count> QueueDequeueUpTo::argument_descs;
constexpr std::array<AttributeDesc, 2> QueueDequeueUpTo::attribute_descs;

constexpr std::array<ArgumentDesc, DecodePng::input_arg_count + DecodePng::output_arg_count> DecodePng::argument_descs;
constexpr std::array<AttributeDesc, 2> DecodePng::attribute_descs;

constexpr std::array<ArgumentDesc, DataFormatDimMap::input_arg_count + DataFormatDimMap::output_arg_count> DataFormatDimMap::argument_descs;
constexpr std::array<AttributeDesc, 3> DataFormatDimMap::attribute_descs;

constexpr std::array<ArgumentDesc, CollectiveBcastRecv::input_arg_count + CollectiveBcastRecv::output_arg_count> CollectiveBcastRecv::argument_descs;
constexpr std::array<AttributeDesc, 7> CollectiveBcastRecv::attribute_descs;

constexpr std::array<ArgumentDesc, CollectiveAssignGroupV2::input_arg_count + CollectiveAssignGroupV2::output_arg_count> CollectiveAssignGroupV2::argument_descs;
constexpr std::array<AttributeDesc, 0> CollectiveAssignGroupV2::attribute_descs;

constexpr std::array<ArgumentDesc, ExperimentalThreadPoolDataset::input_arg_count + ExperimentalThreadPoolDataset::output_arg_count> ExperimentalThreadPoolDataset::argument_descs;
constexpr std::array<AttributeDesc, 2> ExperimentalThreadPoolDataset::attribute_descs;

constexpr std::array<ArgumentDesc, TensorArrayPack::input_arg_count + TensorArrayPack::output_arg_count> TensorArrayPack::argument_descs;
constexpr std::array<AttributeDesc, 2> TensorArrayPack::attribute_descs;

constexpr std::array<ArgumentDesc, CollectiveBcastSendV2::input_arg_count + CollectiveBcastSendV2::output_arg_count> CollectiveBcastSendV2::argument_descs;
constexpr std::array<AttributeDesc, 3> CollectiveBcastSendV2::attribute_descs;

constexpr std::array<ArgumentDesc, QueueDequeueUpToV2::input_arg_count + QueueDequeueUpToV2::output_arg_count> QueueDequeueUpToV2::argument_descs;
constexpr std::array<AttributeDesc, 2> QueueDequeueUpToV2::attribute_descs;

constexpr std::array<ArgumentDesc, CollectiveBcastRecvV2::input_arg_count + CollectiveBcastRecvV2::output_arg_count> CollectiveBcastRecvV2::argument_descs;
constexpr std::array<AttributeDesc, 4> CollectiveBcastRecvV2::attribute_descs;

constexpr std::array<ArgumentDesc, Switch::input_arg_count + Switch::output_arg_count> Switch::argument_descs;
constexpr std::array<AttributeDesc, 1> Switch::attribute_descs;

constexpr std::array<ArgumentDesc, Log::input_arg_count + Log::output_arg_count> Log::argument_descs;
constexpr std::array<AttributeDesc, 1> Log::attribute_descs;

constexpr std::array<ArgumentDesc, Qr::input_arg_count + Qr::output_arg_count> Qr::argument_descs;
constexpr std::array<AttributeDesc, 2> Qr::attribute_descs;

constexpr std::array<ArgumentDesc, Einsum::input_arg_count + Einsum::output_arg_count> Einsum::argument_descs;
constexpr std::array<AttributeDesc, 3> Einsum::attribute_descs;

constexpr std::array<ArgumentDesc, _ConnectInterTPUEmbeddingCommunication::input_arg_count + _ConnectInterTPUEmbeddingCommunication::output_arg_count> _ConnectInterTPUEmbeddingCommunication::argument_descs;
constexpr std::array<AttributeDesc, 1> _ConnectInterTPUEmbeddingCommunication::attribute_descs;

constexpr std::array<ArgumentDesc, RefSwitch::input_arg_count + RefSwitch::output_arg_count> RefSwitch::argument_descs;
constexpr std::array<AttributeDesc, 1> RefSwitch::attribute_descs;

constexpr std::array<ArgumentDesc, RefSelect::input_arg_count + RefSelect::output_arg_count> RefSelect::argument_descs;
constexpr std::array<AttributeDesc, 2> RefSelect::attribute_descs;

constexpr std::array<ArgumentDesc, MaxPoolGradGrad::input_arg_count + MaxPoolGradGrad::output_arg_count> MaxPoolGradGrad::argument_descs;
constexpr std::array<AttributeDesc, 5> MaxPoolGradGrad::attribute_descs;

constexpr std::array<ArgumentDesc, TensorArrayUnpack::input_arg_count + TensorArrayUnpack::output_arg_count> TensorArrayUnpack::argument_descs;
constexpr std::array<AttributeDesc, 1> TensorArrayUnpack::attribute_descs;

constexpr std::array<ArgumentDesc, Merge::input_arg_count + Merge::output_arg_count> Merge::argument_descs;
constexpr std::array<AttributeDesc, 2> Merge::attribute_descs;

constexpr std::array<ArgumentDesc, MapDataset::input_arg_count + MapDataset::output_arg_count> MapDataset::argument_descs;
constexpr std::array<AttributeDesc, 7> MapDataset::attribute_descs;

constexpr std::array<ArgumentDesc, _MklQuantizedConv2DAndRequantize::input_arg_count + _MklQuantizedConv2DAndRequantize::output_arg_count> _MklQuantizedConv2DAndRequantize::argument_descs;
constexpr std::array<AttributeDesc, 9> _MklQuantizedConv2DAndRequantize::attribute_descs;

constexpr std::array<ArgumentDesc, ResourceApplyAdagradDA::input_arg_count + ResourceApplyAdagradDA::output_arg_count> ResourceApplyAdagradDA::argument_descs;
constexpr std::array<AttributeDesc, 2> ResourceApplyAdagradDA::attribute_descs;

constexpr std::array<ArgumentDesc, TensorArrayConcatV3::input_arg_count + TensorArrayConcatV3::output_arg_count> TensorArrayConcatV3::argument_descs;
constexpr std::array<AttributeDesc, 2> TensorArrayConcatV3::attribute_descs;

constexpr std::array<ArgumentDesc, RefMerge::input_arg_count + RefMerge::output_arg_count> RefMerge::argument_descs;
constexpr std::array<AttributeDesc, 2> RefMerge::attribute_descs;

constexpr std::array<ArgumentDesc, DecodeAndCropJpeg::input_arg_count + DecodeAndCropJpeg::output_arg_count> DecodeAndCropJpeg::argument_descs;
constexpr std::array<AttributeDesc, 6> DecodeAndCropJpeg::attribute_descs;

constexpr std::array<ArgumentDesc, RefEnter::input_arg_count + RefEnter::output_arg_count> RefEnter::argument_descs;
constexpr std::array<AttributeDesc, 4> RefEnter::attribute_descs;

constexpr std::array<ArgumentDesc, StackClose::input_arg_count + StackClose::output_arg_count> StackClose::argument_descs;
constexpr std::array<AttributeDesc, 0> StackClose::attribute_descs;

constexpr std::array<ArgumentDesc, XlaRngBitGenerator::input_arg_count + XlaRngBitGenerator::output_arg_count> XlaRngBitGenerator::argument_descs;
constexpr std::array<AttributeDesc, 2> XlaRngBitGenerator::attribute_descs;

constexpr std::array<ArgumentDesc, Exit::input_arg_count + Exit::output_arg_count> Exit::argument_descs;
constexpr std::array<AttributeDesc, 1> Exit::attribute_descs;

constexpr std::array<ArgumentDesc, DeleteSessionTensor::input_arg_count + DeleteSessionTensor::output_arg_count> DeleteSessionTensor::argument_descs;
constexpr std::array<AttributeDesc, 0> DeleteSessionTensor::attribute_descs;

constexpr std::array<ArgumentDesc, ExperimentalNonSerializableDataset::input_arg_count + ExperimentalNonSerializableDataset::output_arg_count> ExperimentalNonSerializableDataset::argument_descs;
constexpr std::array<AttributeDesc, 2> ExperimentalNonSerializableDataset::attribute_descs;

constexpr std::array<ArgumentDesc, NextIteration::input_arg_count + NextIteration::output_arg_count> NextIteration::argument_descs;
constexpr std::array<AttributeDesc, 1> NextIteration::attribute_descs;

constexpr std::array<ArgumentDesc, OutfeedDequeueTuple::input_arg_count + OutfeedDequeueTuple::output_arg_count> OutfeedDequeueTuple::argument_descs;
constexpr std::array<AttributeDesc, 3> OutfeedDequeueTuple::attribute_descs;

constexpr std::array<ArgumentDesc, LoopCond::input_arg_count + LoopCond::output_arg_count> LoopCond::argument_descs;
constexpr std::array<AttributeDesc, 0> LoopCond::attribute_descs;

constexpr std::array<ArgumentDesc, QuantizedDepthwiseConv2DWithBiasAndRelu::input_arg_count + QuantizedDepthwiseConv2DWithBiasAndRelu::output_arg_count> QuantizedDepthwiseConv2DWithBiasAndRelu::argument_descs;
constexpr std::array<AttributeDesc, 7> QuantizedDepthwiseConv2DWithBiasAndRelu::attribute_descs;

constexpr std::array<ArgumentDesc, Abort::input_arg_count + Abort::output_arg_count> Abort::argument_descs;
constexpr std::array<AttributeDesc, 2> Abort::attribute_descs;

constexpr std::array<ArgumentDesc, NonMaxSuppressionV3::input_arg_count + NonMaxSuppressionV3::output_arg_count> NonMaxSuppressionV3::argument_descs;
constexpr std::array<AttributeDesc, 2> NonMaxSuppressionV3::attribute_descs;

constexpr std::array<ArgumentDesc, DenseCountSparseOutput::input_arg_count + DenseCountSparseOutput::output_arg_count> DenseCountSparseOutput::argument_descs;
constexpr std::array<AttributeDesc, 5> DenseCountSparseOutput::attribute_descs;

constexpr std::array<ArgumentDesc, OrderedMapSize::input_arg_count + OrderedMapSize::output_arg_count> OrderedMapSize::argument_descs;
constexpr std::array<AttributeDesc, 5> OrderedMapSize::attribute_descs;

constexpr std::array<ArgumentDesc, _MklQuantizedConv2DAndReluAndRequantize::input_arg_count + _MklQuantizedConv2DAndReluAndRequantize::output_arg_count> _MklQuantizedConv2DAndReluAndRequantize::argument_descs;
constexpr std::array<AttributeDesc, 9> _MklQuantizedConv2DAndReluAndRequantize::attribute_descs;

constexpr std::array<ArgumentDesc, ParseSingleSequenceExample::input_arg_count + ParseSingleSequenceExample::output_arg_count> ParseSingleSequenceExample::argument_descs;
constexpr std::array<AttributeDesc, 10> ParseSingleSequenceExample::attribute_descs;

constexpr std::array<ArgumentDesc, SparseCountSparseOutput::input_arg_count + SparseCountSparseOutput::output_arg_count> SparseCountSparseOutput::argument_descs;
constexpr std::array<AttributeDesc, 5> SparseCountSparseOutput::attribute_descs;

constexpr std::array<ArgumentDesc, BatchMatrixSolve::input_arg_count + BatchMatrixSolve::output_arg_count> BatchMatrixSolve::argument_descs;
constexpr std::array<AttributeDesc, 2> BatchMatrixSolve::attribute_descs;

constexpr std::array<ArgumentDesc, QuantizedConv2DAndReluAndRequantize::input_arg_count + QuantizedConv2DAndReluAndRequantize::output_arg_count> QuantizedConv2DAndReluAndRequantize::argument_descs;
constexpr std::array<AttributeDesc, 7> QuantizedConv2DAndReluAndRequantize::attribute_descs;

constexpr std::array<ArgumentDesc, CTCLoss::input_arg_count + CTCLoss::output_arg_count> CTCLoss::argument_descs;
constexpr std::array<AttributeDesc, 4> CTCLoss::attribute_descs;

constexpr std::array<ArgumentDesc, Sign::input_arg_count + Sign::output_arg_count> Sign::argument_descs;
constexpr std::array<AttributeDesc, 1> Sign::attribute_descs;

constexpr std::array<ArgumentDesc, BarrierTakeMany::input_arg_count + BarrierTakeMany::output_arg_count> BarrierTakeMany::argument_descs;
constexpr std::array<AttributeDesc, 4> BarrierTakeMany::attribute_descs;

constexpr std::array<ArgumentDesc, CTCLossV2::input_arg_count + CTCLossV2::output_arg_count> CTCLossV2::argument_descs;
constexpr std::array<AttributeDesc, 3> CTCLossV2::attribute_descs;

constexpr std::array<ArgumentDesc, IsTPUEmbeddingInitialized::input_arg_count + IsTPUEmbeddingInitialized::output_arg_count> IsTPUEmbeddingInitialized::argument_descs;
constexpr std::array<AttributeDesc, 1> IsTPUEmbeddingInitialized::attribute_descs;

constexpr std::array<ArgumentDesc, WholeFileReaderV2::input_arg_count + WholeFileReaderV2::output_arg_count> WholeFileReaderV2::argument_descs;
constexpr std::array<AttributeDesc, 2> WholeFileReaderV2::attribute_descs;

constexpr std::array<ArgumentDesc, StatelessRandomUniformInt::input_arg_count + StatelessRandomUniformInt::output_arg_count> StatelessRandomUniformInt::argument_descs;
constexpr std::array<AttributeDesc, 3> StatelessRandomUniformInt::attribute_descs;

constexpr std::array<ArgumentDesc, CTCGreedyDecoder::input_arg_count + CTCGreedyDecoder::output_arg_count> CTCGreedyDecoder::argument_descs;
constexpr std::array<AttributeDesc, 3> CTCGreedyDecoder::attribute_descs;

constexpr std::array<ArgumentDesc, CudnnRNNParamsSize::input_arg_count + CudnnRNNParamsSize::output_arg_count> CudnnRNNParamsSize::argument_descs;
constexpr std::array<AttributeDesc, 9> CudnnRNNParamsSize::attribute_descs;

constexpr std::array<ArgumentDesc, ScaleAndTranslate::input_arg_count + ScaleAndTranslate::output_arg_count> ScaleAndTranslate::argument_descs;
constexpr std::array<AttributeDesc, 3> ScaleAndTranslate::attribute_descs;

constexpr std::array<ArgumentDesc, ConfigureTPUEmbedding::input_arg_count + ConfigureTPUEmbedding::output_arg_count> ConfigureTPUEmbedding::argument_descs;
constexpr std::array<AttributeDesc, 1> ConfigureTPUEmbedding::attribute_descs;

constexpr std::array<ArgumentDesc, CudnnRNN::input_arg_count + CudnnRNN::output_arg_count> CudnnRNN::argument_descs;
constexpr std::array<AttributeDesc, 8> CudnnRNN::attribute_descs;

constexpr std::array<ArgumentDesc, CudnnRNNV2::input_arg_count + CudnnRNNV2::output_arg_count> CudnnRNNV2::argument_descs;
constexpr std::array<AttributeDesc, 8> CudnnRNNV2::attribute_descs;

constexpr std::array<ArgumentDesc, BatchMatMul::input_arg_count + BatchMatMul::output_arg_count> BatchMatMul::argument_descs;
constexpr std::array<AttributeDesc, 3> BatchMatMul::attribute_descs;

constexpr std::array<ArgumentDesc, CudnnRNNBackprop::input_arg_count + CudnnRNNBackprop::output_arg_count> CudnnRNNBackprop::argument_descs;
constexpr std::array<AttributeDesc, 7> CudnnRNNBackprop::attribute_descs;

constexpr std::array<ArgumentDesc, CudnnRNNCanonicalToParams::input_arg_count + CudnnRNNCanonicalToParams::output_arg_count> CudnnRNNCanonicalToParams::argument_descs;
constexpr std::array<AttributeDesc, 8> CudnnRNNCanonicalToParams::attribute_descs;

constexpr std::array<ArgumentDesc, DynamicPartition::input_arg_count + DynamicPartition::output_arg_count> DynamicPartition::argument_descs;
constexpr std::array<AttributeDesc, 2> DynamicPartition::attribute_descs;

constexpr std::array<ArgumentDesc, ScatterNdUpdate::input_arg_count + ScatterNdUpdate::output_arg_count> ScatterNdUpdate::argument_descs;
constexpr std::array<AttributeDesc, 3> ScatterNdUpdate::attribute_descs;

constexpr std::array<ArgumentDesc, RandomShuffleQueue::input_arg_count + RandomShuffleQueue::output_arg_count> RandomShuffleQueue::argument_descs;
constexpr std::array<AttributeDesc, 8> RandomShuffleQueue::attribute_descs;

constexpr std::array<ArgumentDesc, XlaReduce::input_arg_count + XlaReduce::output_arg_count> XlaReduce::argument_descs;
constexpr std::array<AttributeDesc, 3> XlaReduce::attribute_descs;

constexpr std::array<ArgumentDesc, QuantizedMatMulWithBiasAndRequantize::input_arg_count + QuantizedMatMulWithBiasAndRequantize::output_arg_count> QuantizedMatMulWithBiasAndRequantize::argument_descs;
constexpr std::array<AttributeDesc, 7> QuantizedMatMulWithBiasAndRequantize::attribute_descs;

constexpr std::array<ArgumentDesc, AccumulatorTakeGradient::input_arg_count + AccumulatorTakeGradient::output_arg_count> AccumulatorTakeGradient::argument_descs;
constexpr std::array<AttributeDesc, 1> AccumulatorTakeGradient::attribute_descs;

constexpr std::array<ArgumentDesc, FIFOQueueV2::input_arg_count + FIFOQueueV2::output_arg_count> FIFOQueueV2::argument_descs;
constexpr std::array<AttributeDesc, 5> FIFOQueueV2::attribute_descs;

constexpr std::array<ArgumentDesc, CompositeTensorVariantFromComponents::input_arg_count + CompositeTensorVariantFromComponents::output_arg_count> CompositeTensorVariantFromComponents::argument_descs;
constexpr std::array<AttributeDesc, 2> CompositeTensorVariantFromComponents::attribute_descs;

constexpr std::array<ArgumentDesc, ExperimentalScanDataset::input_arg_count + ExperimentalScanDataset::output_arg_count> ExperimentalScanDataset::argument_descs;
constexpr std::array<AttributeDesc, 6> ExperimentalScanDataset::attribute_descs;

constexpr std::array<ArgumentDesc, MatrixExponential::input_arg_count + MatrixExponential::output_arg_count> MatrixExponential::argument_descs;
constexpr std::array<AttributeDesc, 1> MatrixExponential::attribute_descs;

constexpr std::array<ArgumentDesc, ApplyFtrlV2::input_arg_count + ApplyFtrlV2::output_arg_count> ApplyFtrlV2::argument_descs;
constexpr std::array<AttributeDesc, 3> ApplyFtrlV2::attribute_descs;

constexpr std::array<ArgumentDesc, VarIsInitializedOp::input_arg_count + VarIsInitializedOp::output_arg_count> VarIsInitializedOp::argument_descs;
constexpr std::array<AttributeDesc, 0> VarIsInitializedOp::attribute_descs;

constexpr std::array<ArgumentDesc, PaddingFIFOQueueV2::input_arg_count + PaddingFIFOQueueV2::output_arg_count> PaddingFIFOQueueV2::argument_descs;
constexpr std::array<AttributeDesc, 5> PaddingFIFOQueueV2::attribute_descs;

constexpr std::array<ArgumentDesc, Lu::input_arg_count + Lu::output_arg_count> Lu::argument_descs;
constexpr std::array<AttributeDesc, 2> Lu::attribute_descs;

constexpr std::array<ArgumentDesc, QueueEnqueue::input_arg_count + QueueEnqueue::output_arg_count> QueueEnqueue::argument_descs;
constexpr std::array<AttributeDesc, 2> QueueEnqueue::attribute_descs;

constexpr std::array<ArgumentDesc, QueueEnqueueV2::input_arg_count + QueueEnqueueV2::output_arg_count> QueueEnqueueV2::argument_descs;
constexpr std::array<AttributeDesc, 2> QueueEnqueueV2::attribute_descs;

constexpr std::array<ArgumentDesc, QueueEnqueueMany::input_arg_count + QueueEnqueueMany::output_arg_count> QueueEnqueueMany::argument_descs;
constexpr std::array<AttributeDesc, 2> QueueEnqueueMany::attribute_descs;

constexpr std::array<ArgumentDesc, ReaderResetV2::input_arg_count + ReaderResetV2::output_arg_count> ReaderResetV2::argument_descs;
constexpr std::array<AttributeDesc, 0> ReaderResetV2::attribute_descs;

constexpr std::array<ArgumentDesc, QueueEnqueueManyV2::input_arg_count + QueueEnqueueManyV2::output_arg_count> QueueEnqueueManyV2::argument_descs;
constexpr std::array<AttributeDesc, 2> QueueEnqueueManyV2::attribute_descs;

constexpr std::array<ArgumentDesc, QueueDequeue::input_arg_count + QueueDequeue::output_arg_count> QueueDequeue::argument_descs;
constexpr std::array<AttributeDesc, 2> QueueDequeue::attribute_descs;

constexpr std::array<ArgumentDesc, RandomDataset::input_arg_count + RandomDataset::output_arg_count> RandomDataset::argument_descs;
constexpr std::array<AttributeDesc, 3> RandomDataset::attribute_descs;

constexpr std::array<ArgumentDesc, UnsortedSegmentMax::input_arg_count + UnsortedSegmentMax::output_arg_count> UnsortedSegmentMax::argument_descs;
constexpr std::array<AttributeDesc, 3> UnsortedSegmentMax::attribute_descs;

constexpr std::array<ArgumentDesc, QueueDequeueV2::input_arg_count + QueueDequeueV2::output_arg_count> QueueDequeueV2::argument_descs;
constexpr std::array<AttributeDesc, 2> QueueDequeueV2::attribute_descs;

constexpr std::array<ArgumentDesc, RetrieveTPUEmbeddingFTRLParameters::input_arg_count + RetrieveTPUEmbeddingFTRLParameters::output_arg_count> RetrieveTPUEmbeddingFTRLParameters::argument_descs;
constexpr std::array<AttributeDesc, 5> RetrieveTPUEmbeddingFTRLParameters::attribute_descs;

constexpr std::array<ArgumentDesc, MultiDeviceIteratorGetNextFromShard::input_arg_count + MultiDeviceIteratorGetNextFromShard::output_arg_count> MultiDeviceIteratorGetNextFromShard::argument_descs;
constexpr std::array<AttributeDesc, 2> MultiDeviceIteratorGetNextFromShard::attribute_descs;

constexpr std::array<ArgumentDesc, QueueDequeueMany::input_arg_count + QueueDequeueMany::output_arg_count> QueueDequeueMany::argument_descs;
constexpr std::array<AttributeDesc, 2> QueueDequeueMany::attribute_descs;

constexpr std::array<ArgumentDesc, QueueIsClosed::input_arg_count + QueueIsClosed::output_arg_count> QueueIsClosed::argument_descs;
constexpr std::array<AttributeDesc, 0> QueueIsClosed::attribute_descs;

constexpr std::array<ArgumentDesc, AccumulatorNumAccumulated::input_arg_count + AccumulatorNumAccumulated::output_arg_count> AccumulatorNumAccumulated::argument_descs;
constexpr std::array<AttributeDesc, 0> AccumulatorNumAccumulated::attribute_descs;

constexpr std::array<ArgumentDesc, SegmentProd::input_arg_count + SegmentProd::output_arg_count> SegmentProd::argument_descs;
constexpr std::array<AttributeDesc, 2> SegmentProd::attribute_descs;

constexpr std::array<ArgumentDesc, _MklConv3D::input_arg_count + _MklConv3D::output_arg_count> _MklConv3D::argument_descs;
constexpr std::array<AttributeDesc, 6> _MklConv3D::attribute_descs;

constexpr std::array<ArgumentDesc, MatchingFilesDataset::input_arg_count + MatchingFilesDataset::output_arg_count> MatchingFilesDataset::argument_descs;
constexpr std::array<AttributeDesc, 0> MatchingFilesDataset::attribute_descs;

constexpr std::array<ArgumentDesc, SaveSlices::input_arg_count + SaveSlices::output_arg_count> SaveSlices::argument_descs;
constexpr std::array<AttributeDesc, 1> SaveSlices::attribute_descs;

constexpr std::array<ArgumentDesc, StringFormat::input_arg_count + StringFormat::output_arg_count> StringFormat::argument_descs;
constexpr std::array<AttributeDesc, 4> StringFormat::attribute_descs;

constexpr std::array<ArgumentDesc, SparseConditionalAccumulator::input_arg_count + SparseConditionalAccumulator::output_arg_count> SparseConditionalAccumulator::argument_descs;
constexpr std::array<AttributeDesc, 5> SparseConditionalAccumulator::attribute_descs;

constexpr std::array<ArgumentDesc, GreaterEqual::input_arg_count + GreaterEqual::output_arg_count> GreaterEqual::argument_descs;
constexpr std::array<AttributeDesc, 1> GreaterEqual::attribute_descs;

constexpr std::array<ArgumentDesc, StatelessRandomPoisson::input_arg_count + StatelessRandomPoisson::output_arg_count> StatelessRandomPoisson::argument_descs;
constexpr std::array<AttributeDesc, 4> StatelessRandomPoisson::attribute_descs;

constexpr std::array<ArgumentDesc, SparseAccumulatorApplyGradient::input_arg_count + SparseAccumulatorApplyGradient::output_arg_count> SparseAccumulatorApplyGradient::argument_descs;
constexpr std::array<AttributeDesc, 2> SparseAccumulatorApplyGradient::attribute_descs;

constexpr std::array<ArgumentDesc, ResourceScatterNdAdd::input_arg_count + ResourceScatterNdAdd::output_arg_count> ResourceScatterNdAdd::argument_descs;
constexpr std::array<AttributeDesc, 3> ResourceScatterNdAdd::attribute_descs;

constexpr std::array<ArgumentDesc, SparseAccumulatorTakeGradient::input_arg_count + SparseAccumulatorTakeGradient::output_arg_count> SparseAccumulatorTakeGradient::argument_descs;
constexpr std::array<AttributeDesc, 1> SparseAccumulatorTakeGradient::attribute_descs;

constexpr std::array<ArgumentDesc, RepeatDataset::input_arg_count + RepeatDataset::output_arg_count> RepeatDataset::argument_descs;
constexpr std::array<AttributeDesc, 3> RepeatDataset::attribute_descs;

constexpr std::array<ArgumentDesc, StackPopV2::input_arg_count + StackPopV2::output_arg_count> StackPopV2::argument_descs;
constexpr std::array<AttributeDesc, 1> StackPopV2::attribute_descs;

constexpr std::array<ArgumentDesc, StatelessSampleDistortedBoundingBox::input_arg_count + StatelessSampleDistortedBoundingBox::output_arg_count> StatelessSampleDistortedBoundingBox::argument_descs;
constexpr std::array<AttributeDesc, 6> StatelessSampleDistortedBoundingBox::attribute_descs;

constexpr std::array<ArgumentDesc, StackCloseV2::input_arg_count + StackCloseV2::output_arg_count> StackCloseV2::argument_descs;
constexpr std::array<AttributeDesc, 0> StackCloseV2::attribute_descs;

constexpr std::array<ArgumentDesc, GroupByWindowDataset::input_arg_count + GroupByWindowDataset::output_arg_count> GroupByWindowDataset::argument_descs;
constexpr std::array<AttributeDesc, 9> GroupByWindowDataset::attribute_descs;

constexpr std::array<ArgumentDesc, Stack::input_arg_count + Stack::output_arg_count> Stack::argument_descs;
constexpr std::array<AttributeDesc, 2> Stack::attribute_descs;

constexpr std::array<ArgumentDesc, TensorArrayV3::input_arg_count + TensorArrayV3::output_arg_count> TensorArrayV3::argument_descs;
constexpr std::array<AttributeDesc, 6> TensorArrayV3::attribute_descs;

constexpr std::array<ArgumentDesc, XlaSpmdShardToFullShape::input_arg_count + XlaSpmdShardToFullShape::output_arg_count> XlaSpmdShardToFullShape::argument_descs;
constexpr std::array<AttributeDesc, 5> XlaSpmdShardToFullShape::attribute_descs;

constexpr std::array<ArgumentDesc, TensorArrayWriteV3::input_arg_count + TensorArrayWriteV3::output_arg_count> TensorArrayWriteV3::argument_descs;
constexpr std::array<AttributeDesc, 1> TensorArrayWriteV3::attribute_descs;

constexpr std::array<ArgumentDesc, TensorArrayReadV3::input_arg_count + TensorArrayReadV3::output_arg_count> TensorArrayReadV3::argument_descs;
constexpr std::array<AttributeDesc, 1> TensorArrayReadV3::attribute_descs;

constexpr std::array<ArgumentDesc, TensorArrayGatherV3::input_arg_count + TensorArrayGatherV3::output_arg_count> TensorArrayGatherV3::argument_descs;
constexpr std::array<AttributeDesc, 2> TensorArrayGatherV3::attribute_descs;

constexpr std::array<ArgumentDesc, ShuffleDatasetV3::input_arg_count + ShuffleDatasetV3::output_arg_count> ShuffleDatasetV3::argument_descs;
constexpr std::array<AttributeDesc, 4> ShuffleDatasetV3::attribute_descs;

constexpr std::array<ArgumentDesc, DebugNumericSummaryV2::input_arg_count + DebugNumericSummaryV2::output_arg_count> DebugNumericSummaryV2::argument_descs;
constexpr std::array<AttributeDesc, 4> DebugNumericSummaryV2::attribute_descs;

constexpr std::array<ArgumentDesc, BarrierReadySize::input_arg_count + BarrierReadySize::output_arg_count> BarrierReadySize::argument_descs;
constexpr std::array<AttributeDesc, 0> BarrierReadySize::attribute_descs;

constexpr std::array<ArgumentDesc, TensorArrayScatterV3::input_arg_count + TensorArrayScatterV3::output_arg_count> TensorArrayScatterV3::argument_descs;
constexpr std::array<AttributeDesc, 1> TensorArrayScatterV3::attribute_descs;

constexpr std::array<ArgumentDesc, AvgPoolGrad::input_arg_count + AvgPoolGrad::output_arg_count> AvgPoolGrad::argument_descs;
constexpr std::array<AttributeDesc, 5> AvgPoolGrad::attribute_descs;

constexpr std::array<ArgumentDesc, TensorArrayV2::input_arg_count + TensorArrayV2::output_arg_count> TensorArrayV2::argument_descs;
constexpr std::array<AttributeDesc, 5> TensorArrayV2::attribute_descs;

constexpr std::array<ArgumentDesc, TensorArrayGradV2::input_arg_count + TensorArrayGradV2::output_arg_count> TensorArrayGradV2::argument_descs;
constexpr std::array<AttributeDesc, 1> TensorArrayGradV2::attribute_descs;

constexpr std::array<ArgumentDesc, TensorArrayRead::input_arg_count + TensorArrayRead::output_arg_count> TensorArrayRead::argument_descs;
constexpr std::array<AttributeDesc, 1> TensorArrayRead::attribute_descs;

constexpr std::array<ArgumentDesc, SegmentSum::input_arg_count + SegmentSum::output_arg_count> SegmentSum::argument_descs;
constexpr std::array<AttributeDesc, 2> SegmentSum::attribute_descs;

constexpr std::array<ArgumentDesc, Floor::input_arg_count + Floor::output_arg_count> Floor::argument_descs;
constexpr std::array<AttributeDesc, 1> Floor::attribute_descs;

constexpr std::array<ArgumentDesc, _MklNativeAvgPool::input_arg_count + _MklNativeAvgPool::output_arg_count> _MklNativeAvgPool::argument_descs;
constexpr std::array<AttributeDesc, 5> _MklNativeAvgPool::attribute_descs;

constexpr std::array<ArgumentDesc, TensorArrayReadV2::input_arg_count + TensorArrayReadV2::output_arg_count> TensorArrayReadV2::argument_descs;
constexpr std::array<AttributeDesc, 1> TensorArrayReadV2::attribute_descs;

constexpr std::array<ArgumentDesc, SparseReshape::input_arg_count + SparseReshape::output_arg_count> SparseReshape::argument_descs;
constexpr std::array<AttributeDesc, 0> SparseReshape::attribute_descs;

constexpr std::array<ArgumentDesc, FusedBatchNormV3::input_arg_count + FusedBatchNormV3::output_arg_count> FusedBatchNormV3::argument_descs;
constexpr std::array<AttributeDesc, 6> FusedBatchNormV3::attribute_descs;

constexpr std::array<ArgumentDesc, TensorArrayScatter::input_arg_count + TensorArrayScatter::output_arg_count> TensorArrayScatter::argument_descs;
constexpr std::array<AttributeDesc, 1> TensorArrayScatter::attribute_descs;

constexpr std::array<ArgumentDesc, _FinalizeTPUEmbeddingSystemConfiguration::input_arg_count + _FinalizeTPUEmbeddingSystemConfiguration::output_arg_count> _FinalizeTPUEmbeddingSystemConfiguration::argument_descs;
constexpr std::array<AttributeDesc, 1> _FinalizeTPUEmbeddingSystemConfiguration::attribute_descs;

constexpr std::array<ArgumentDesc, TensorArrayScatterV2::input_arg_count + TensorArrayScatterV2::output_arg_count> TensorArrayScatterV2::argument_descs;
constexpr std::array<AttributeDesc, 1> TensorArrayScatterV2::attribute_descs;

constexpr std::array<ArgumentDesc, LoadTPUEmbeddingAdagradMomentumParameters::input_arg_count + LoadTPUEmbeddingAdagradMomentumParameters::output_arg_count> LoadTPUEmbeddingAdagradMomentumParameters::argument_descs;
constexpr std::array<AttributeDesc, 5> LoadTPUEmbeddingAdagradMomentumParameters::attribute_descs;

constexpr std::array<ArgumentDesc, OrderedMapClear::input_arg_count + OrderedMapClear::output_arg_count> OrderedMapClear::argument_descs;
constexpr std::array<AttributeDesc, 5> OrderedMapClear::attribute_descs;

constexpr std::array<ArgumentDesc, TensorArrayConcatV2::input_arg_count + TensorArrayConcatV2::output_arg_count> TensorArrayConcatV2::argument_descs;
constexpr std::array<AttributeDesc, 2> TensorArrayConcatV2::attribute_descs;

constexpr std::array<ArgumentDesc, SetStatsAggregatorDataset::input_arg_count + SetStatsAggregatorDataset::output_arg_count> SetStatsAggregatorDataset::argument_descs;
constexpr std::array<AttributeDesc, 2> SetStatsAggregatorDataset::attribute_descs;

constexpr std::array<ArgumentDesc, SqrtGrad::input_arg_count + SqrtGrad::output_arg_count> SqrtGrad::argument_descs;
constexpr std::array<AttributeDesc, 1> SqrtGrad::attribute_descs;

constexpr std::array<ArgumentDesc, MaxPoolV2::input_arg_count + MaxPoolV2::output_arg_count> MaxPoolV2::argument_descs;
constexpr std::array<AttributeDesc, 3> MaxPoolV2::attribute_descs;

constexpr std::array<ArgumentDesc, TensorArraySplitV2::input_arg_count + TensorArraySplitV2::output_arg_count> TensorArraySplitV2::argument_descs;
constexpr std::array<AttributeDesc, 1> TensorArraySplitV2::attribute_descs;

constexpr std::array<ArgumentDesc, IteratorGetNextAsOptional::input_arg_count + IteratorGetNextAsOptional::output_arg_count> IteratorGetNextAsOptional::argument_descs;
constexpr std::array<AttributeDesc, 2> IteratorGetNextAsOptional::attribute_descs;

constexpr std::array<ArgumentDesc, SparseMatrixSparseMatMul::input_arg_count + SparseMatrixSparseMatMul::output_arg_count> SparseMatrixSparseMatMul::argument_descs;
constexpr std::array<AttributeDesc, 5> SparseMatrixSparseMatMul::attribute_descs;

constexpr std::array<ArgumentDesc, TensorArraySize::input_arg_count + TensorArraySize::output_arg_count> TensorArraySize::argument_descs;
constexpr std::array<AttributeDesc, 0> TensorArraySize::attribute_descs;

constexpr std::array<ArgumentDesc, TensorArraySizeV2::input_arg_count + TensorArraySizeV2::output_arg_count> TensorArraySizeV2::argument_descs;
constexpr std::array<AttributeDesc, 0> TensorArraySizeV2::attribute_descs;

constexpr std::array<ArgumentDesc, TensorArrayCloseV2::input_arg_count + TensorArrayCloseV2::output_arg_count> TensorArrayCloseV2::argument_descs;
constexpr std::array<AttributeDesc, 0> TensorArrayCloseV2::attribute_descs;

constexpr std::array<ArgumentDesc, BarrierClose::input_arg_count + BarrierClose::output_arg_count> BarrierClose::argument_descs;
constexpr std::array<AttributeDesc, 1> BarrierClose::attribute_descs;

constexpr std::array<ArgumentDesc, BarrierIncompleteSize::input_arg_count + BarrierIncompleteSize::output_arg_count> BarrierIncompleteSize::argument_descs;
constexpr std::array<AttributeDesc, 0> BarrierIncompleteSize::attribute_descs;

constexpr std::array<ArgumentDesc, GetSessionHandleV2::input_arg_count + GetSessionHandleV2::output_arg_count> GetSessionHandleV2::argument_descs;
constexpr std::array<AttributeDesc, 1> GetSessionHandleV2::attribute_descs;

constexpr std::array<ArgumentDesc, WorkerHeartbeat::input_arg_count + WorkerHeartbeat::output_arg_count> WorkerHeartbeat::argument_descs;
constexpr std::array<AttributeDesc, 0> WorkerHeartbeat::attribute_descs;

constexpr std::array<ArgumentDesc, Stage::input_arg_count + Stage::output_arg_count> Stage::argument_descs;
constexpr std::array<AttributeDesc, 5> Stage::attribute_descs;

constexpr std::array<ArgumentDesc, Select::input_arg_count + Select::output_arg_count> Select::argument_descs;
constexpr std::array<AttributeDesc, 1> Select::attribute_descs;

constexpr std::array<ArgumentDesc, Unstage::input_arg_count + Unstage::output_arg_count> Unstage::argument_descs;
constexpr std::array<AttributeDesc, 5> Unstage::attribute_descs;

constexpr std::array<ArgumentDesc, StagePeek::input_arg_count + StagePeek::output_arg_count> StagePeek::argument_descs;
constexpr std::array<AttributeDesc, 5> StagePeek::attribute_descs;

constexpr std::array<ArgumentDesc, StageSize::input_arg_count + StageSize::output_arg_count> StageSize::argument_descs;
constexpr std::array<AttributeDesc, 5> StageSize::attribute_descs;

constexpr std::array<ArgumentDesc, MapStage::input_arg_count + MapStage::output_arg_count> MapStage::argument_descs;
constexpr std::array<AttributeDesc, 6> MapStage::attribute_descs;

constexpr std::array<ArgumentDesc, RaggedGather::input_arg_count + RaggedGather::output_arg_count> RaggedGather::argument_descs;
constexpr std::array<AttributeDesc, 5> RaggedGather::attribute_descs;

constexpr std::array<ArgumentDesc, MapUnstage::input_arg_count + MapUnstage::output_arg_count> MapUnstage::argument_descs;
constexpr std::array<AttributeDesc, 5> MapUnstage::attribute_descs;

constexpr std::array<ArgumentDesc, MapUnstageNoKey::input_arg_count + MapUnstageNoKey::output_arg_count> MapUnstageNoKey::argument_descs;
constexpr std::array<AttributeDesc, 5> MapUnstageNoKey::attribute_descs;

constexpr std::array<ArgumentDesc, XlaKeyValueSort::input_arg_count + XlaKeyValueSort::output_arg_count> XlaKeyValueSort::argument_descs;
constexpr std::array<AttributeDesc, 2> XlaKeyValueSort::attribute_descs;

constexpr std::array<ArgumentDesc, SleepDataset::input_arg_count + SleepDataset::output_arg_count> SleepDataset::argument_descs;
constexpr std::array<AttributeDesc, 2> SleepDataset::attribute_descs;

constexpr std::array<ArgumentDesc, MapSize::input_arg_count + MapSize::output_arg_count> MapSize::argument_descs;
constexpr std::array<AttributeDesc, 5> MapSize::attribute_descs;

constexpr std::array<ArgumentDesc, MapClear::input_arg_count + MapClear::output_arg_count> MapClear::argument_descs;
constexpr std::array<AttributeDesc, 5> MapClear::attribute_descs;

constexpr std::array<ArgumentDesc, OrderedMapIncompleteSize::input_arg_count + OrderedMapIncompleteSize::output_arg_count> OrderedMapIncompleteSize::argument_descs;
constexpr std::array<AttributeDesc, 5> OrderedMapIncompleteSize::attribute_descs;

constexpr std::array<ArgumentDesc, TensorSliceDataset::input_arg_count + TensorSliceDataset::output_arg_count> TensorSliceDataset::argument_descs;
constexpr std::array<AttributeDesc, 4> TensorSliceDataset::attribute_descs;

constexpr std::array<ArgumentDesc, TensorMapStackKeys::input_arg_count + TensorMapStackKeys::output_arg_count> TensorMapStackKeys::argument_descs;
constexpr std::array<AttributeDesc, 1> TensorMapStackKeys::attribute_descs;

constexpr std::array<ArgumentDesc, ZipDataset::input_arg_count + ZipDataset::output_arg_count> ZipDataset::argument_descs;
constexpr std::array<AttributeDesc, 4> ZipDataset::attribute_descs;

constexpr std::array<ArgumentDesc, MultiDeviceIteratorInit::input_arg_count + MultiDeviceIteratorInit::output_arg_count> MultiDeviceIteratorInit::argument_descs;
constexpr std::array<AttributeDesc, 0> MultiDeviceIteratorInit::attribute_descs;

constexpr std::array<ArgumentDesc, ResourceSparseApplyAdagradDA::input_arg_count + ResourceSparseApplyAdagradDA::output_arg_count> ResourceSparseApplyAdagradDA::argument_descs;
constexpr std::array<AttributeDesc, 3> ResourceSparseApplyAdagradDA::attribute_descs;

constexpr std::array<ArgumentDesc, Expm1::input_arg_count + Expm1::output_arg_count> Expm1::argument_descs;
constexpr std::array<AttributeDesc, 1> Expm1::attribute_descs;

constexpr std::array<ArgumentDesc, Conv3DBackpropInputV2::input_arg_count + Conv3DBackpropInputV2::output_arg_count> Conv3DBackpropInputV2::argument_descs;
constexpr std::array<AttributeDesc, 6> Conv3DBackpropInputV2::attribute_descs;

constexpr std::array<ArgumentDesc, PrefetchDataset::input_arg_count + PrefetchDataset::output_arg_count> PrefetchDataset::argument_descs;
constexpr std::array<AttributeDesc, 6> PrefetchDataset::attribute_descs;

constexpr std::array<ArgumentDesc, ParseExampleDatasetV2::input_arg_count + ParseExampleDatasetV2::output_arg_count> ParseExampleDatasetV2::argument_descs;
constexpr std::array<AttributeDesc, 11> ParseExampleDatasetV2::attribute_descs;

constexpr std::array<ArgumentDesc, ParallelInterleaveDatasetV2::input_arg_count + ParallelInterleaveDatasetV2::output_arg_count> ParallelInterleaveDatasetV2::argument_descs;
constexpr std::array<AttributeDesc, 6> ParallelInterleaveDatasetV2::attribute_descs;

constexpr std::array<ArgumentDesc, ParallelInterleaveDatasetV4::input_arg_count + ParallelInterleaveDatasetV4::output_arg_count> ParallelInterleaveDatasetV4::argument_descs;
constexpr std::array<AttributeDesc, 6> ParallelInterleaveDatasetV4::attribute_descs;

constexpr std::array<ArgumentDesc, ParallelFilterDataset::input_arg_count + ParallelFilterDataset::output_arg_count> ParallelFilterDataset::argument_descs;
constexpr std::array<AttributeDesc, 6> ParallelFilterDataset::attribute_descs;

constexpr std::array<ArgumentDesc, WindowOp::input_arg_count + WindowOp::output_arg_count> WindowOp::argument_descs;
constexpr std::array<AttributeDesc, 3> WindowOp::attribute_descs;

constexpr std::array<ArgumentDesc, BatchDataset::input_arg_count + BatchDataset::output_arg_count> BatchDataset::argument_descs;
constexpr std::array<AttributeDesc, 3> BatchDataset::attribute_descs;

constexpr std::array<ArgumentDesc, PaddedBatchDatasetV2::input_arg_count + PaddedBatchDatasetV2::output_arg_count> PaddedBatchDatasetV2::argument_descs;
constexpr std::array<AttributeDesc, 5> PaddedBatchDatasetV2::attribute_descs;

constexpr std::array<ArgumentDesc, AnonymousSeedGenerator::input_arg_count + AnonymousSeedGenerator::output_arg_count> AnonymousSeedGenerator::argument_descs;
constexpr std::array<AttributeDesc, 0> AnonymousSeedGenerator::attribute_descs;

constexpr std::array<ArgumentDesc, DeleteSeedGenerator::input_arg_count + DeleteSeedGenerator::output_arg_count> DeleteSeedGenerator::argument_descs;
constexpr std::array<AttributeDesc, 0> DeleteSeedGenerator::attribute_descs;

constexpr std::array<ArgumentDesc, SaveDataset::input_arg_count + SaveDataset::output_arg_count> SaveDataset::argument_descs;
constexpr std::array<AttributeDesc, 4> SaveDataset::attribute_descs;

constexpr std::array<ArgumentDesc, DeleteRandomSeedGenerator::input_arg_count + DeleteRandomSeedGenerator::output_arg_count> DeleteRandomSeedGenerator::argument_descs;
constexpr std::array<AttributeDesc, 0> DeleteRandomSeedGenerator::attribute_descs;

constexpr std::array<ArgumentDesc, DummySeedGenerator::input_arg_count + DummySeedGenerator::output_arg_count> DummySeedGenerator::argument_descs;
constexpr std::array<AttributeDesc, 0> DummySeedGenerator::attribute_descs;

constexpr std::array<ArgumentDesc, ResizeArea::input_arg_count + ResizeArea::output_arg_count> ResizeArea::argument_descs;
constexpr std::array<AttributeDesc, 2> ResizeArea::attribute_descs;

constexpr std::array<ArgumentDesc, _MklQuantizedConv2DWithBiasAndReluAndRequantize::input_arg_count + _MklQuantizedConv2DWithBiasAndReluAndRequantize::output_arg_count> _MklQuantizedConv2DWithBiasAndReluAndRequantize::argument_descs;
constexpr std::array<AttributeDesc, 11> _MklQuantizedConv2DWithBiasAndReluAndRequantize::attribute_descs;

constexpr std::array<ArgumentDesc, ShuffleAndRepeatDataset::input_arg_count + ShuffleAndRepeatDataset::output_arg_count> ShuffleAndRepeatDataset::argument_descs;
constexpr std::array<AttributeDesc, 4> ShuffleAndRepeatDataset::attribute_descs;

constexpr std::array<ArgumentDesc, OptionalGetValue::input_arg_count + OptionalGetValue::output_arg_count> OptionalGetValue::argument_descs;
constexpr std::array<AttributeDesc, 2> OptionalGetValue::attribute_descs;

constexpr std::array<ArgumentDesc, ShuffleAndRepeatDatasetV2::input_arg_count + ShuffleAndRepeatDatasetV2::output_arg_count> ShuffleAndRepeatDatasetV2::argument_descs;
constexpr std::array<AttributeDesc, 4> ShuffleAndRepeatDatasetV2::attribute_descs;

constexpr std::array<ArgumentDesc, _MklMaximum::input_arg_count + _MklMaximum::output_arg_count> _MklMaximum::argument_descs;
constexpr std::array<AttributeDesc, 1> _MklMaximum::attribute_descs;

constexpr std::array<ArgumentDesc, ResourceApplyProximalGradientDescent::input_arg_count + ResourceApplyProximalGradientDescent::output_arg_count> ResourceApplyProximalGradientDescent::argument_descs;
constexpr std::array<AttributeDesc, 2> ResourceApplyProximalGradientDescent::attribute_descs;

constexpr std::array<ArgumentDesc, DeleteMemoryCache::input_arg_count + DeleteMemoryCache::output_arg_count> DeleteMemoryCache::argument_descs;
constexpr std::array<AttributeDesc, 0> DeleteMemoryCache::attribute_descs;

constexpr std::array<ArgumentDesc, _MklQuantizedConcatV2::input_arg_count + _MklQuantizedConcatV2::output_arg_count> _MklQuantizedConcatV2::argument_descs;
constexpr std::array<AttributeDesc, 3> _MklQuantizedConcatV2::attribute_descs;

constexpr std::array<ArgumentDesc, ResourceScatterNdMax::input_arg_count + ResourceScatterNdMax::output_arg_count> ResourceScatterNdMax::argument_descs;
constexpr std::array<AttributeDesc, 3> ResourceScatterNdMax::attribute_descs;

constexpr std::array<ArgumentDesc, CacheDataset::input_arg_count + CacheDataset::output_arg_count> CacheDataset::argument_descs;
constexpr std::array<AttributeDesc, 3> CacheDataset::attribute_descs;

constexpr std::array<ArgumentDesc, TextLineDataset::input_arg_count + TextLineDataset::output_arg_count> TextLineDataset::argument_descs;
constexpr std::array<AttributeDesc, 1> TextLineDataset::attribute_descs;

constexpr std::array<ArgumentDesc, FixedLengthRecordDataset::input_arg_count + FixedLengthRecordDataset::output_arg_count> FixedLengthRecordDataset::argument_descs;
constexpr std::array<AttributeDesc, 1> FixedLengthRecordDataset::attribute_descs;

constexpr std::array<ArgumentDesc, SobolSample::input_arg_count + SobolSample::output_arg_count> SobolSample::argument_descs;
constexpr std::array<AttributeDesc, 1> SobolSample::attribute_descs;

constexpr std::array<ArgumentDesc, TFRecordDataset::input_arg_count + TFRecordDataset::output_arg_count> TFRecordDataset::argument_descs;
constexpr std::array<AttributeDesc, 1> TFRecordDataset::attribute_descs;

constexpr std::array<ArgumentDesc, ReaderReadUpTo::input_arg_count + ReaderReadUpTo::output_arg_count> ReaderReadUpTo::argument_descs;
constexpr std::array<AttributeDesc, 0> ReaderReadUpTo::attribute_descs;

constexpr std::array<ArgumentDesc, Iterator::input_arg_count + Iterator::output_arg_count> Iterator::argument_descs;
constexpr std::array<AttributeDesc, 4> Iterator::attribute_descs;

constexpr std::array<ArgumentDesc, MutexLock::input_arg_count + MutexLock::output_arg_count> MutexLock::argument_descs;
constexpr std::array<AttributeDesc, 0> MutexLock::attribute_descs;

constexpr std::array<ArgumentDesc, IteratorV2::input_arg_count + IteratorV2::output_arg_count> IteratorV2::argument_descs;
constexpr std::array<AttributeDesc, 4> IteratorV2::attribute_descs;

constexpr std::array<ArgumentDesc, TPUCompileSucceededAssert::input_arg_count + TPUCompileSucceededAssert::output_arg_count> TPUCompileSucceededAssert::argument_descs;
constexpr std::array<AttributeDesc, 0> TPUCompileSucceededAssert::attribute_descs;

constexpr std::array<ArgumentDesc, AnonymousIteratorV2::input_arg_count + AnonymousIteratorV2::output_arg_count> AnonymousIteratorV2::argument_descs;
constexpr std::array<AttributeDesc, 2> AnonymousIteratorV2::attribute_descs;

constexpr std::array<ArgumentDesc, Atan::input_arg_count + Atan::output_arg_count> Atan::argument_descs;
constexpr std::array<AttributeDesc, 1> Atan::attribute_descs;

constexpr std::array<ArgumentDesc, AnonymousIteratorV3::input_arg_count + AnonymousIteratorV3::output_arg_count> AnonymousIteratorV3::argument_descs;
constexpr std::array<AttributeDesc, 2> AnonymousIteratorV3::attribute_descs;

constexpr std::array<ArgumentDesc, DeleteIterator::input_arg_count + DeleteIterator::output_arg_count> DeleteIterator::argument_descs;
constexpr std::array<AttributeDesc, 0> DeleteIterator::attribute_descs;

constexpr std::array<ArgumentDesc, DeleteMultiDeviceIterator::input_arg_count + DeleteMultiDeviceIterator::output_arg_count> DeleteMultiDeviceIterator::argument_descs;
constexpr std::array<AttributeDesc, 1> DeleteMultiDeviceIterator::attribute_descs;

constexpr std::array<ArgumentDesc, MakeIterator::input_arg_count + MakeIterator::output_arg_count> MakeIterator::argument_descs;
constexpr std::array<AttributeDesc, 0> MakeIterator::attribute_descs;

constexpr std::array<ArgumentDesc, ReduceDataset::input_arg_count + ReduceDataset::output_arg_count> ReduceDataset::argument_descs;
constexpr std::array<AttributeDesc, 7> ReduceDataset::attribute_descs;

constexpr std::array<ArgumentDesc, MaxPoolGradWithArgmax::input_arg_count + MaxPoolGradWithArgmax::output_arg_count> MaxPoolGradWithArgmax::argument_descs;
constexpr std::array<AttributeDesc, 6> MaxPoolGradWithArgmax::attribute_descs;

constexpr std::array<ArgumentDesc, OneShotIterator::input_arg_count + OneShotIterator::output_arg_count> OneShotIterator::argument_descs;
constexpr std::array<AttributeDesc, 5> OneShotIterator::attribute_descs;

constexpr std::array<ArgumentDesc, IteratorGetNextSync::input_arg_count + IteratorGetNextSync::output_arg_count> IteratorGetNextSync::argument_descs;
constexpr std::array<AttributeDesc, 2> IteratorGetNextSync::attribute_descs;

constexpr std::array<ArgumentDesc, RGBToHSV::input_arg_count + RGBToHSV::output_arg_count> RGBToHSV::argument_descs;
constexpr std::array<AttributeDesc, 1> RGBToHSV::attribute_descs;

constexpr std::array<ArgumentDesc, DatasetToSingleElement::input_arg_count + DatasetToSingleElement::output_arg_count> DatasetToSingleElement::argument_descs;
constexpr std::array<AttributeDesc, 3> DatasetToSingleElement::attribute_descs;

constexpr std::array<ArgumentDesc, IteratorToStringHandle::input_arg_count + IteratorToStringHandle::output_arg_count> IteratorToStringHandle::argument_descs;
constexpr std::array<AttributeDesc, 0> IteratorToStringHandle::attribute_descs;

constexpr std::array<ArgumentDesc, IteratorFromStringHandle::input_arg_count + IteratorFromStringHandle::output_arg_count> IteratorFromStringHandle::argument_descs;
constexpr std::array<AttributeDesc, 2> IteratorFromStringHandle::attribute_descs;

constexpr std::array<ArgumentDesc, QuantizedConv2DPerChannel::input_arg_count + QuantizedConv2DPerChannel::output_arg_count> QuantizedConv2DPerChannel::argument_descs;
constexpr std::array<AttributeDesc, 6> QuantizedConv2DPerChannel::attribute_descs;

constexpr std::array<ArgumentDesc, LatencyStatsDataset::input_arg_count + LatencyStatsDataset::output_arg_count> LatencyStatsDataset::argument_descs;
constexpr std::array<AttributeDesc, 2> LatencyStatsDataset::attribute_descs;

constexpr std::array<ArgumentDesc, KthOrderStatistic::input_arg_count + KthOrderStatistic::output_arg_count> KthOrderStatistic::argument_descs;
constexpr std::array<AttributeDesc, 1> KthOrderStatistic::attribute_descs;

constexpr std::array<ArgumentDesc, IteratorFromStringHandleV2::input_arg_count + IteratorFromStringHandleV2::output_arg_count> IteratorFromStringHandleV2::argument_descs;
constexpr std::array<AttributeDesc, 2> IteratorFromStringHandleV2::attribute_descs;

constexpr std::array<ArgumentDesc, DeserializeIterator::input_arg_count + DeserializeIterator::output_arg_count> DeserializeIterator::argument_descs;
constexpr std::array<AttributeDesc, 0> DeserializeIterator::attribute_descs;

constexpr std::array<ArgumentDesc, DatasetToGraph::input_arg_count + DatasetToGraph::output_arg_count> DatasetToGraph::argument_descs;
constexpr std::array<AttributeDesc, 3> DatasetToGraph::attribute_descs;

constexpr std::array<ArgumentDesc, CopyHost::input_arg_count + CopyHost::output_arg_count> CopyHost::argument_descs;
constexpr std::array<AttributeDesc, 3> CopyHost::attribute_descs;

constexpr std::array<ArgumentDesc, RiscCholesky::input_arg_count + RiscCholesky::output_arg_count> RiscCholesky::argument_descs;
constexpr std::array<AttributeDesc, 1> RiscCholesky::attribute_descs;

constexpr std::array<ArgumentDesc, _MklLRN::input_arg_count + _MklLRN::output_arg_count> _MklLRN::argument_descs;
constexpr std::array<AttributeDesc, 6> _MklLRN::attribute_descs;

constexpr std::array<ArgumentDesc, DatasetToGraphV2::input_arg_count + DatasetToGraphV2::output_arg_count> DatasetToGraphV2::argument_descs;
constexpr std::array<AttributeDesc, 2> DatasetToGraphV2::attribute_descs;

constexpr std::array<ArgumentDesc, ExperimentalLMDBDataset::input_arg_count + ExperimentalLMDBDataset::output_arg_count> ExperimentalLMDBDataset::argument_descs;
constexpr std::array<AttributeDesc, 2> ExperimentalLMDBDataset::attribute_descs;

constexpr std::array<ArgumentDesc, BatchMatrixTriangularSolve::input_arg_count + BatchMatrixTriangularSolve::output_arg_count> BatchMatrixTriangularSolve::argument_descs;
constexpr std::array<AttributeDesc, 3> BatchMatrixTriangularSolve::attribute_descs;

constexpr std::array<ArgumentDesc, OptimizeDataset::input_arg_count + OptimizeDataset::output_arg_count> OptimizeDataset::argument_descs;
constexpr std::array<AttributeDesc, 3> OptimizeDataset::attribute_descs;

constexpr std::array<ArgumentDesc, RiscRem::input_arg_count + RiscRem::output_arg_count> RiscRem::argument_descs;
constexpr std::array<AttributeDesc, 1> RiscRem::attribute_descs;

constexpr std::array<ArgumentDesc, OptimizeDatasetV2::input_arg_count + OptimizeDatasetV2::output_arg_count> OptimizeDatasetV2::argument_descs;
constexpr std::array<AttributeDesc, 3> OptimizeDatasetV2::attribute_descs;

constexpr std::array<ArgumentDesc, NotEqual::input_arg_count + NotEqual::output_arg_count> NotEqual::argument_descs;
constexpr std::array<AttributeDesc, 2> NotEqual::attribute_descs;

constexpr std::array<ArgumentDesc, LinSpace::input_arg_count + LinSpace::output_arg_count> LinSpace::argument_descs;
constexpr std::array<AttributeDesc, 2> LinSpace::attribute_descs;

constexpr std::array<ArgumentDesc, OptionalFromValue::input_arg_count + OptionalFromValue::output_arg_count> OptionalFromValue::argument_descs;
constexpr std::array<AttributeDesc, 1> OptionalFromValue::attribute_descs;

constexpr std::array<ArgumentDesc, OptionalNone::input_arg_count + OptionalNone::output_arg_count> OptionalNone::argument_descs;
constexpr std::array<AttributeDesc, 0> OptionalNone::attribute_descs;

constexpr std::array<ArgumentDesc, ExperimentalSqlDataset::input_arg_count + ExperimentalSqlDataset::output_arg_count> ExperimentalSqlDataset::argument_descs;
constexpr std::array<AttributeDesc, 2> ExperimentalSqlDataset::attribute_descs;

constexpr std::array<ArgumentDesc, StringUpper::input_arg_count + StringUpper::output_arg_count> StringUpper::argument_descs;
constexpr std::array<AttributeDesc, 1> StringUpper::attribute_descs;

constexpr std::array<ArgumentDesc, OptionalHasValue::input_arg_count + OptionalHasValue::output_arg_count> OptionalHasValue::argument_descs;
constexpr std::array<AttributeDesc, 0> OptionalHasValue::attribute_descs;

constexpr std::array<ArgumentDesc, SnapshotDatasetReader::input_arg_count + SnapshotDatasetReader::output_arg_count> SnapshotDatasetReader::argument_descs;
constexpr std::array<AttributeDesc, 4> SnapshotDatasetReader::attribute_descs;

constexpr std::array<ArgumentDesc, IdentityReader::input_arg_count + IdentityReader::output_arg_count> IdentityReader::argument_descs;
constexpr std::array<AttributeDesc, 2> IdentityReader::attribute_descs;

constexpr std::array<ArgumentDesc, WrapDatasetVariant::input_arg_count + WrapDatasetVariant::output_arg_count> WrapDatasetVariant::argument_descs;
constexpr std::array<AttributeDesc, 0> WrapDatasetVariant::attribute_descs;

constexpr std::array<ArgumentDesc, AnonymousMultiDeviceIteratorV3::input_arg_count + AnonymousMultiDeviceIteratorV3::output_arg_count> AnonymousMultiDeviceIteratorV3::argument_descs;
constexpr std::array<AttributeDesc, 3> AnonymousMultiDeviceIteratorV3::attribute_descs;

constexpr std::array<ArgumentDesc, StatelessMultinomial::input_arg_count + StatelessMultinomial::output_arg_count> StatelessMultinomial::argument_descs;
constexpr std::array<AttributeDesc, 3> StatelessMultinomial::attribute_descs;

constexpr std::array<ArgumentDesc, MultiDeviceIteratorToStringHandle::input_arg_count + MultiDeviceIteratorToStringHandle::output_arg_count> MultiDeviceIteratorToStringHandle::argument_descs;
constexpr std::array<AttributeDesc, 0> MultiDeviceIteratorToStringHandle::attribute_descs;

constexpr std::array<ArgumentDesc, RiscExp::input_arg_count + RiscExp::output_arg_count> RiscExp::argument_descs;
constexpr std::array<AttributeDesc, 1> RiscExp::attribute_descs;

constexpr std::array<ArgumentDesc, MultiDeviceIteratorFromStringHandle::input_arg_count + MultiDeviceIteratorFromStringHandle::output_arg_count> MultiDeviceIteratorFromStringHandle::argument_descs;
constexpr std::array<AttributeDesc, 2> MultiDeviceIteratorFromStringHandle::attribute_descs;

constexpr std::array<ArgumentDesc, _MklQuantizedMatMulWithBiasAndDequantize::input_arg_count + _MklQuantizedMatMulWithBiasAndDequantize::output_arg_count> _MklQuantizedMatMulWithBiasAndDequantize::argument_descs;
constexpr std::array<AttributeDesc, 8> _MklQuantizedMatMulWithBiasAndDequantize::attribute_descs;

constexpr std::array<ArgumentDesc, UnicodeScript::input_arg_count + UnicodeScript::output_arg_count> UnicodeScript::argument_descs;
constexpr std::array<AttributeDesc, 0> UnicodeScript::attribute_descs;

constexpr std::array<ArgumentDesc, OptionsDataset::input_arg_count + OptionsDataset::output_arg_count> OptionsDataset::argument_descs;
constexpr std::array<AttributeDesc, 4> OptionsDataset::attribute_descs;

constexpr std::array<ArgumentDesc, Timestamp::input_arg_count + Timestamp::output_arg_count> Timestamp::argument_descs;
constexpr std::array<AttributeDesc, 0> Timestamp::attribute_descs;

constexpr std::array<ArgumentDesc, GetOptions::input_arg_count + GetOptions::output_arg_count> GetOptions::argument_descs;
constexpr std::array<AttributeDesc, 0> GetOptions::attribute_descs;

constexpr std::array<ArgumentDesc, LMDBReader::input_arg_count + LMDBReader::output_arg_count> LMDBReader::argument_descs;
constexpr std::array<AttributeDesc, 2> LMDBReader::attribute_descs;

constexpr std::array<ArgumentDesc, FinalizeDataset::input_arg_count + FinalizeDataset::output_arg_count> FinalizeDataset::argument_descs;
constexpr std::array<AttributeDesc, 3> FinalizeDataset::attribute_descs;

constexpr std::array<ArgumentDesc, RFFT3D::input_arg_count + RFFT3D::output_arg_count> RFFT3D::argument_descs;
constexpr std::array<AttributeDesc, 2> RFFT3D::attribute_descs;

constexpr std::array<ArgumentDesc, Copy::input_arg_count + Copy::output_arg_count> Copy::argument_descs;
constexpr std::array<AttributeDesc, 3> Copy::attribute_descs;

constexpr std::array<ArgumentDesc, DebugIdentityV2::input_arg_count + DebugIdentityV2::output_arg_count> DebugIdentityV2::argument_descs;
constexpr std::array<AttributeDesc, 8> DebugIdentityV2::attribute_descs;

constexpr std::array<ArgumentDesc, AssertNextDataset::input_arg_count + AssertNextDataset::output_arg_count> AssertNextDataset::argument_descs;
constexpr std::array<AttributeDesc, 2> AssertNextDataset::attribute_descs;

constexpr std::array<ArgumentDesc, RandomCrop::input_arg_count + RandomCrop::output_arg_count> RandomCrop::argument_descs;
constexpr std::array<AttributeDesc, 3> RandomCrop::attribute_descs;

constexpr std::array<ArgumentDesc, ExperimentalAssertNextDataset::input_arg_count + ExperimentalAssertNextDataset::output_arg_count> ExperimentalAssertNextDataset::argument_descs;
constexpr std::array<AttributeDesc, 2> ExperimentalAssertNextDataset::attribute_descs;

constexpr std::array<ArgumentDesc, AssertPrevDataset::input_arg_count + AssertPrevDataset::output_arg_count> AssertPrevDataset::argument_descs;
constexpr std::array<AttributeDesc, 2> AssertPrevDataset::attribute_descs;

constexpr std::array<ArgumentDesc, IsotonicRegression::input_arg_count + IsotonicRegression::output_arg_count> IsotonicRegression::argument_descs;
constexpr std::array<AttributeDesc, 2> IsotonicRegression::attribute_descs;

constexpr std::array<ArgumentDesc, AutoShardDataset::input_arg_count + AutoShardDataset::output_arg_count> AutoShardDataset::argument_descs;
constexpr std::array<AttributeDesc, 4> AutoShardDataset::attribute_descs;

constexpr std::array<ArgumentDesc, ExperimentalAutoShardDataset::input_arg_count + ExperimentalAutoShardDataset::output_arg_count> ExperimentalAutoShardDataset::argument_descs;
constexpr std::array<AttributeDesc, 3> ExperimentalAutoShardDataset::attribute_descs;

constexpr std::array<ArgumentDesc, ExperimentalBytesProducedStatsDataset::input_arg_count + ExperimentalBytesProducedStatsDataset::output_arg_count> ExperimentalBytesProducedStatsDataset::argument_descs;
constexpr std::array<AttributeDesc, 2> ExperimentalBytesProducedStatsDataset::attribute_descs;

constexpr std::array<ArgumentDesc, _MklNativeConv3DBackpropInputV2::input_arg_count + _MklNativeConv3DBackpropInputV2::output_arg_count> _MklNativeConv3DBackpropInputV2::argument_descs;
constexpr std::array<AttributeDesc, 6> _MklNativeConv3DBackpropInputV2::attribute_descs;

constexpr std::array<ArgumentDesc, ExperimentalIgnoreErrorsDataset::input_arg_count + ExperimentalIgnoreErrorsDataset::output_arg_count> ExperimentalIgnoreErrorsDataset::argument_descs;
constexpr std::array<AttributeDesc, 3> ExperimentalIgnoreErrorsDataset::attribute_descs;

constexpr std::array<ArgumentDesc, ChooseFastestBranchDataset::input_arg_count + ChooseFastestBranchDataset::output_arg_count> ChooseFastestBranchDataset::argument_descs;
constexpr std::array<AttributeDesc, 6> ChooseFastestBranchDataset::attribute_descs;

constexpr std::array<ArgumentDesc, GRUBlockCell::input_arg_count + GRUBlockCell::output_arg_count> GRUBlockCell::argument_descs;
constexpr std::array<AttributeDesc, 1> GRUBlockCell::attribute_descs;

constexpr std::array<ArgumentDesc, ExperimentalChooseFastestDataset::input_arg_count + ExperimentalChooseFastestDataset::output_arg_count> ExperimentalChooseFastestDataset::argument_descs;
constexpr std::array<AttributeDesc, 4> ExperimentalChooseFastestDataset::attribute_descs;

constexpr std::array<ArgumentDesc, IgammaGradA::input_arg_count + IgammaGradA::output_arg_count> IgammaGradA::argument_descs;
constexpr std::array<AttributeDesc, 1> IgammaGradA::attribute_descs;

constexpr std::array<ArgumentDesc, CompressElement::input_arg_count + CompressElement::output_arg_count> CompressElement::argument_descs;
constexpr std::array<AttributeDesc, 1> CompressElement::attribute_descs;

constexpr std::array<ArgumentDesc, QuantizeDownAndShrinkRange::input_arg_count + QuantizeDownAndShrinkRange::output_arg_count> QuantizeDownAndShrinkRange::argument_descs;
constexpr std::array<AttributeDesc, 2> QuantizeDownAndShrinkRange::attribute_descs;

constexpr std::array<ArgumentDesc, UncompressElement::input_arg_count + UncompressElement::output_arg_count> UncompressElement::argument_descs;
constexpr std::array<AttributeDesc, 2> UncompressElement::attribute_descs;

constexpr std::array<ArgumentDesc, CSVDataset::input_arg_count + CSVDataset::output_arg_count> CSVDataset::argument_descs;
constexpr std::array<AttributeDesc, 2> CSVDataset::attribute_descs;

constexpr std::array<ArgumentDesc, ExperimentalCSVDataset::input_arg_count + ExperimentalCSVDataset::output_arg_count> ExperimentalCSVDataset::argument_descs;
constexpr std::array<AttributeDesc, 2> ExperimentalCSVDataset::attribute_descs;

constexpr std::array<ArgumentDesc, AssignAdd::input_arg_count + AssignAdd::output_arg_count> AssignAdd::argument_descs;
constexpr std::array<AttributeDesc, 2> AssignAdd::attribute_descs;

constexpr std::array<ArgumentDesc, ExperimentalDatasetCardinality::input_arg_count + ExperimentalDatasetCardinality::output_arg_count> ExperimentalDatasetCardinality::argument_descs;
constexpr std::array<AttributeDesc, 0> ExperimentalDatasetCardinality::attribute_descs;

constexpr std::array<ArgumentDesc, Complex::input_arg_count + Complex::output_arg_count> Complex::argument_descs;
constexpr std::array<AttributeDesc, 2> Complex::attribute_descs;

constexpr std::array<ArgumentDesc, ExperimentalGroupByWindowDataset::input_arg_count + ExperimentalGroupByWindowDataset::output_arg_count> ExperimentalGroupByWindowDataset::argument_descs;
constexpr std::array<AttributeDesc, 8> ExperimentalGroupByWindowDataset::attribute_descs;

constexpr std::array<ArgumentDesc, _XlaSendFromHostV2::input_arg_count + _XlaSendFromHostV2::output_arg_count> _XlaSendFromHostV2::argument_descs;
constexpr std::array<AttributeDesc, 2> _XlaSendFromHostV2::attribute_descs;

constexpr std::array<ArgumentDesc, StatelessRandomGetKeyCounter::input_arg_count + StatelessRandomGetKeyCounter::output_arg_count> StatelessRandomGetKeyCounter::argument_descs;
constexpr std::array<AttributeDesc, 1> StatelessRandomGetKeyCounter::attribute_descs;

constexpr std::array<ArgumentDesc, SparseToSparseSetOperation::input_arg_count + SparseToSparseSetOperation::output_arg_count> SparseToSparseSetOperation::argument_descs;
constexpr std::array<AttributeDesc, 3> SparseToSparseSetOperation::attribute_descs;

constexpr std::array<ArgumentDesc, DatasetFromGraph::input_arg_count + DatasetFromGraph::output_arg_count> DatasetFromGraph::argument_descs;
constexpr std::array<AttributeDesc, 0> DatasetFromGraph::attribute_descs;

constexpr std::array<ArgumentDesc, Acos::input_arg_count + Acos::output_arg_count> Acos::argument_descs;
constexpr std::array<AttributeDesc, 1> Acos::attribute_descs;

constexpr std::array<ArgumentDesc, ExperimentalDatasetToTFRecord::input_arg_count + ExperimentalDatasetToTFRecord::output_arg_count> ExperimentalDatasetToTFRecord::argument_descs;
constexpr std::array<AttributeDesc, 0> ExperimentalDatasetToTFRecord::attribute_descs;

constexpr std::array<ArgumentDesc, DenseToSparseBatchDataset::input_arg_count + DenseToSparseBatchDataset::output_arg_count> DenseToSparseBatchDataset::argument_descs;
constexpr std::array<AttributeDesc, 2> DenseToSparseBatchDataset::attribute_descs;

constexpr std::array<ArgumentDesc, SparseSegmentSumWithNumSegments::input_arg_count + SparseSegmentSumWithNumSegments::output_arg_count> SparseSegmentSumWithNumSegments::argument_descs;
constexpr std::array<AttributeDesc, 4> SparseSegmentSumWithNumSegments::attribute_descs;

constexpr std::array<ArgumentDesc, TruncateMod::input_arg_count + TruncateMod::output_arg_count> TruncateMod::argument_descs;
constexpr std::array<AttributeDesc, 1> TruncateMod::attribute_descs;

constexpr std::array<ArgumentDesc, SlidingWindowDataset::input_arg_count + SlidingWindowDataset::output_arg_count> SlidingWindowDataset::argument_descs;
constexpr std::array<AttributeDesc, 3> SlidingWindowDataset::attribute_descs;

constexpr std::array<ArgumentDesc, ExperimentalDenseToSparseBatchDataset::input_arg_count + ExperimentalDenseToSparseBatchDataset::output_arg_count> ExperimentalDenseToSparseBatchDataset::argument_descs;
constexpr std::array<AttributeDesc, 2> ExperimentalDenseToSparseBatchDataset::attribute_descs;

constexpr std::array<ArgumentDesc, DirectedInterleaveDataset::input_arg_count + DirectedInterleaveDataset::output_arg_count> DirectedInterleaveDataset::argument_descs;
constexpr std::array<AttributeDesc, 4> DirectedInterleaveDataset::attribute_descs;

constexpr std::array<ArgumentDesc, ExperimentalDirectedInterleaveDataset::input_arg_count + ExperimentalDirectedInterleaveDataset::output_arg_count> ExperimentalDirectedInterleaveDataset::argument_descs;
constexpr std::array<AttributeDesc, 3> ExperimentalDirectedInterleaveDataset::attribute_descs;

constexpr std::array<ArgumentDesc, LRN::input_arg_count + LRN::output_arg_count> LRN::argument_descs;
constexpr std::array<AttributeDesc, 5> LRN::attribute_descs;

constexpr std::array<ArgumentDesc, ExperimentalGroupByReducerDataset::input_arg_count + ExperimentalGroupByReducerDataset::output_arg_count> ExperimentalGroupByReducerDataset::argument_descs;
constexpr std::array<AttributeDesc, 10> ExperimentalGroupByReducerDataset::attribute_descs;

constexpr std::array<ArgumentDesc, GetElementAtIndex::input_arg_count + GetElementAtIndex::output_arg_count> GetElementAtIndex::argument_descs;
constexpr std::array<AttributeDesc, 2> GetElementAtIndex::attribute_descs;

constexpr std::array<ArgumentDesc, IgnoreErrorsDataset::input_arg_count + IgnoreErrorsDataset::output_arg_count> IgnoreErrorsDataset::argument_descs;
constexpr std::array<AttributeDesc, 3> IgnoreErrorsDataset::attribute_descs;

constexpr std::array<ArgumentDesc, ExperimentalIteratorGetDevice::input_arg_count + ExperimentalIteratorGetDevice::output_arg_count> ExperimentalIteratorGetDevice::argument_descs;
constexpr std::array<AttributeDesc, 0> ExperimentalIteratorGetDevice::attribute_descs;

constexpr std::array<ArgumentDesc, ExperimentalMapAndBatchDataset::input_arg_count + ExperimentalMapAndBatchDataset::output_arg_count> ExperimentalMapAndBatchDataset::argument_descs;
constexpr std::array<AttributeDesc, 5> ExperimentalMapAndBatchDataset::attribute_descs;

constexpr std::array<ArgumentDesc, QuantizedAdd::input_arg_count + QuantizedAdd::output_arg_count> QuantizedAdd::argument_descs;
constexpr std::array<AttributeDesc, 3> QuantizedAdd::attribute_descs;

constexpr std::array<ArgumentDesc, ResourceSparseApplyKerasMomentum::input_arg_count + ResourceSparseApplyKerasMomentum::output_arg_count> ResourceSparseApplyKerasMomentum::argument_descs;
constexpr std::array<AttributeDesc, 4> ResourceSparseApplyKerasMomentum::attribute_descs;

constexpr std::array<ArgumentDesc, ResourceScatterNdMin::input_arg_count + ResourceScatterNdMin::output_arg_count> ResourceScatterNdMin::argument_descs;
constexpr std::array<AttributeDesc, 3> ResourceScatterNdMin::attribute_descs;

constexpr std::array<ArgumentDesc, ExperimentalMapDataset::input_arg_count + ExperimentalMapDataset::output_arg_count> ExperimentalMapDataset::argument_descs;
constexpr std::array<AttributeDesc, 6> ExperimentalMapDataset::attribute_descs;

constexpr std::array<ArgumentDesc, ExperimentalMatchingFilesDataset::input_arg_count + ExperimentalMatchingFilesDataset::output_arg_count> ExperimentalMatchingFilesDataset::argument_descs;
constexpr std::array<AttributeDesc, 0> ExperimentalMatchingFilesDataset::attribute_descs;

constexpr std::array<ArgumentDesc, WholeFileReader::input_arg_count + WholeFileReader::output_arg_count> WholeFileReader::argument_descs;
constexpr std::array<AttributeDesc, 2> WholeFileReader::attribute_descs;

constexpr std::array<ArgumentDesc, ScatterMul::input_arg_count + ScatterMul::output_arg_count> ScatterMul::argument_descs;
constexpr std::array<AttributeDesc, 3> ScatterMul::attribute_descs;

constexpr std::array<ArgumentDesc, NonSerializableDataset::input_arg_count + NonSerializableDataset::output_arg_count> NonSerializableDataset::argument_descs;
constexpr std::array<AttributeDesc, 2> NonSerializableDataset::attribute_descs;

constexpr std::array<ArgumentDesc, MaxIntraOpParallelismDataset::input_arg_count + MaxIntraOpParallelismDataset::output_arg_count> MaxIntraOpParallelismDataset::argument_descs;
constexpr std::array<AttributeDesc, 2> MaxIntraOpParallelismDataset::attribute_descs;

constexpr std::array<ArgumentDesc, ParallelInterleaveDataset::input_arg_count + ParallelInterleaveDataset::output_arg_count> ParallelInterleaveDataset::argument_descs;
constexpr std::array<AttributeDesc, 5> ParallelInterleaveDataset::attribute_descs;

constexpr std::array<ArgumentDesc, LegacyParallelInterleaveDatasetV2::input_arg_count + LegacyParallelInterleaveDatasetV2::output_arg_count> LegacyParallelInterleaveDatasetV2::argument_descs;
constexpr std::array<AttributeDesc, 6> LegacyParallelInterleaveDatasetV2::attribute_descs;

constexpr std::array<ArgumentDesc, Erf::input_arg_count + Erf::output_arg_count> Erf::argument_descs;
constexpr std::array<AttributeDesc, 1> Erf::attribute_descs;

constexpr std::array<ArgumentDesc, TruncateDiv::input_arg_count + TruncateDiv::output_arg_count> TruncateDiv::argument_descs;
constexpr std::array<AttributeDesc, 1> TruncateDiv::attribute_descs;

constexpr std::array<ArgumentDesc, Conv3DBackpropInput::input_arg_count + Conv3DBackpropInput::output_arg_count> Conv3DBackpropInput::argument_descs;
constexpr std::array<AttributeDesc, 4> Conv3DBackpropInput::attribute_descs;

constexpr std::array<ArgumentDesc, ParseExampleDataset::input_arg_count + ParseExampleDataset::output_arg_count> ParseExampleDataset::argument_descs;
constexpr std::array<AttributeDesc, 11> ParseExampleDataset::attribute_descs;

constexpr std::array<ArgumentDesc, _TPUCompileMlirPlaceholderProgramKey::input_arg_count + _TPUCompileMlirPlaceholderProgramKey::output_arg_count> _TPUCompileMlirPlaceholderProgramKey::argument_descs;
constexpr std::array<AttributeDesc, 0> _TPUCompileMlirPlaceholderProgramKey::attribute_descs;

constexpr std::array<ArgumentDesc, _MklDequantize::input_arg_count + _MklDequantize::output_arg_count> _MklDequantize::argument_descs;
constexpr std::array<AttributeDesc, 5> _MklDequantize::attribute_descs;

constexpr std::array<ArgumentDesc, ApplyCenteredRMSProp::input_arg_count + ApplyCenteredRMSProp::output_arg_count> ApplyCenteredRMSProp::argument_descs;
constexpr std::array<AttributeDesc, 2> ApplyCenteredRMSProp::attribute_descs;

constexpr std::array<ArgumentDesc, PrivateThreadPoolDataset::input_arg_count + PrivateThreadPoolDataset::output_arg_count> PrivateThreadPoolDataset::argument_descs;
constexpr std::array<AttributeDesc, 2> PrivateThreadPoolDataset::attribute_descs;

constexpr std::array<ArgumentDesc, ResourceCountUpTo::input_arg_count + ResourceCountUpTo::output_arg_count> ResourceCountUpTo::argument_descs;
constexpr std::array<AttributeDesc, 2> ResourceCountUpTo::attribute_descs;

constexpr std::array<ArgumentDesc, Angle::input_arg_count + Angle::output_arg_count> Angle::argument_descs;
constexpr std::array<AttributeDesc, 2> Angle::attribute_descs;

constexpr std::array<ArgumentDesc, ExperimentalRandomDataset::input_arg_count + ExperimentalRandomDataset::output_arg_count> ExperimentalRandomDataset::argument_descs;
constexpr std::array<AttributeDesc, 2> ExperimentalRandomDataset::attribute_descs;

constexpr std::array<ArgumentDesc, ExperimentalRebatchDataset::input_arg_count + ExperimentalRebatchDataset::output_arg_count> ExperimentalRebatchDataset::argument_descs;
constexpr std::array<AttributeDesc, 3> ExperimentalRebatchDataset::attribute_descs;

constexpr std::array<ArgumentDesc, XlaSharding::input_arg_count + XlaSharding::output_arg_count> XlaSharding::argument_descs;
constexpr std::array<AttributeDesc, 3> XlaSharding::attribute_descs;

constexpr std::array<ArgumentDesc, RebatchDataset::input_arg_count + RebatchDataset::output_arg_count> RebatchDataset::argument_descs;
constexpr std::array<AttributeDesc, 3> RebatchDataset::attribute_descs;

constexpr std::array<ArgumentDesc, DenseBincount::input_arg_count + DenseBincount::output_arg_count> DenseBincount::argument_descs;
constexpr std::array<AttributeDesc, 3> DenseBincount::attribute_descs;

constexpr std::array<ArgumentDesc, RaggedTensorToVariant::input_arg_count + RaggedTensorToVariant::output_arg_count> RaggedTensorToVariant::argument_descs;
constexpr std::array<AttributeDesc, 4> RaggedTensorToVariant::attribute_descs;

constexpr std::array<ArgumentDesc, RebatchDatasetV2::input_arg_count + RebatchDatasetV2::output_arg_count> RebatchDatasetV2::argument_descs;
constexpr std::array<AttributeDesc, 2> RebatchDatasetV2::attribute_descs;

constexpr std::array<ArgumentDesc, ExperimentalSetStatsAggregatorDataset::input_arg_count + ExperimentalSetStatsAggregatorDataset::output_arg_count> ExperimentalSetStatsAggregatorDataset::argument_descs;
constexpr std::array<AttributeDesc, 2> ExperimentalSetStatsAggregatorDataset::attribute_descs;

constexpr std::array<ArgumentDesc, ExperimentalSleepDataset::input_arg_count + ExperimentalSleepDataset::output_arg_count> ExperimentalSleepDataset::argument_descs;
constexpr std::array<AttributeDesc, 2> ExperimentalSleepDataset::attribute_descs;

constexpr std::array<ArgumentDesc, SoftplusGrad::input_arg_count + SoftplusGrad::output_arg_count> SoftplusGrad::argument_descs;
constexpr std::array<AttributeDesc, 1> SoftplusGrad::attribute_descs;

constexpr std::array<ArgumentDesc, ExperimentalSlidingWindowDataset::input_arg_count + ExperimentalSlidingWindowDataset::output_arg_count> ExperimentalSlidingWindowDataset::argument_descs;
constexpr std::array<AttributeDesc, 2> ExperimentalSlidingWindowDataset::attribute_descs;

constexpr std::array<ArgumentDesc, SnapshotDatasetV2::input_arg_count + SnapshotDatasetV2::output_arg_count> SnapshotDatasetV2::argument_descs;
constexpr std::array<AttributeDesc, 12> SnapshotDatasetV2::attribute_descs;

constexpr std::array<ArgumentDesc, HistogramSummary::input_arg_count + HistogramSummary::output_arg_count> HistogramSummary::argument_descs;
constexpr std::array<AttributeDesc, 1> HistogramSummary::attribute_descs;

constexpr std::array<ArgumentDesc, QuantizedReluX::input_arg_count + QuantizedReluX::output_arg_count> QuantizedReluX::argument_descs;
constexpr std::array<AttributeDesc, 2> QuantizedReluX::attribute_descs;

constexpr std::array<ArgumentDesc, SaveDatasetV2::input_arg_count + SaveDatasetV2::output_arg_count> SaveDatasetV2::argument_descs;
constexpr std::array<AttributeDesc, 6> SaveDatasetV2::attribute_descs;

constexpr std::array<ArgumentDesc, Conv2DBackpropInput::input_arg_count + Conv2DBackpropInput::output_arg_count> Conv2DBackpropInput::argument_descs;
constexpr std::array<AttributeDesc, 7> Conv2DBackpropInput::attribute_descs;

constexpr std::array<ArgumentDesc, SqlDataset::input_arg_count + SqlDataset::output_arg_count> SqlDataset::argument_descs;
constexpr std::array<AttributeDesc, 2> SqlDataset::attribute_descs;

constexpr std::array<ArgumentDesc, MatchingFiles::input_arg_count + MatchingFiles::output_arg_count> MatchingFiles::argument_descs;
constexpr std::array<AttributeDesc, 0> MatchingFiles::attribute_descs;

constexpr std::array<ArgumentDesc, Betainc::input_arg_count + Betainc::output_arg_count> Betainc::argument_descs;
constexpr std::array<AttributeDesc, 1> Betainc::attribute_descs;

constexpr std::array<ArgumentDesc, ReadFile::input_arg_count + ReadFile::output_arg_count> ReadFile::argument_descs;
constexpr std::array<AttributeDesc, 0> ReadFile::attribute_descs;

constexpr std::array<ArgumentDesc, _MklQuantizedConv2DWithBiasSumAndReluAndRequantize::input_arg_count + _MklQuantizedConv2DWithBiasSumAndReluAndRequantize::output_arg_count> _MklQuantizedConv2DWithBiasSumAndReluAndRequantize::argument_descs;
constexpr std::array<AttributeDesc, 12> _MklQuantizedConv2DWithBiasSumAndReluAndRequantize::attribute_descs;

constexpr std::array<ArgumentDesc, StatsAggregatorHandle::input_arg_count + StatsAggregatorHandle::output_arg_count> StatsAggregatorHandle::argument_descs;
constexpr std::array<AttributeDesc, 2> StatsAggregatorHandle::attribute_descs;

constexpr std::array<ArgumentDesc, ExperimentalStatsAggregatorHandle::input_arg_count + ExperimentalStatsAggregatorHandle::output_arg_count> ExperimentalStatsAggregatorHandle::argument_descs;
constexpr std::array<AttributeDesc, 2> ExperimentalStatsAggregatorHandle::attribute_descs;

constexpr std::array<ArgumentDesc, StatsAggregatorSetSummaryWriter::input_arg_count + StatsAggregatorSetSummaryWriter::output_arg_count> StatsAggregatorSetSummaryWriter::argument_descs;
constexpr std::array<AttributeDesc, 0> StatsAggregatorSetSummaryWriter::attribute_descs;

constexpr std::array<ArgumentDesc, AddSparseToTensorsMap::input_arg_count + AddSparseToTensorsMap::output_arg_count> AddSparseToTensorsMap::argument_descs;
constexpr std::array<AttributeDesc, 3> AddSparseToTensorsMap::attribute_descs;

constexpr std::array<ArgumentDesc, TakeWhileDataset::input_arg_count + TakeWhileDataset::output_arg_count> TakeWhileDataset::argument_descs;
constexpr std::array<AttributeDesc, 5> TakeWhileDataset::attribute_descs;

constexpr std::array<ArgumentDesc, ExperimentalTakeWhileDataset::input_arg_count + ExperimentalTakeWhileDataset::output_arg_count> ExperimentalTakeWhileDataset::argument_descs;
constexpr std::array<AttributeDesc, 4> ExperimentalTakeWhileDataset::attribute_descs;

constexpr std::array<ArgumentDesc, ReaderRestoreState::input_arg_count + ReaderRestoreState::output_arg_count> ReaderRestoreState::argument_descs;
constexpr std::array<AttributeDesc, 0> ReaderRestoreState::attribute_descs;

constexpr std::array<ArgumentDesc, ThreadPoolHandle::input_arg_count + ThreadPoolHandle::output_arg_count> ThreadPoolHandle::argument_descs;
constexpr std::array<AttributeDesc, 5> ThreadPoolHandle::attribute_descs;

constexpr std::array<ArgumentDesc, ExperimentalThreadPoolHandle::input_arg_count + ExperimentalThreadPoolHandle::output_arg_count> ExperimentalThreadPoolHandle::argument_descs;
constexpr std::array<AttributeDesc, 5> ExperimentalThreadPoolHandle::attribute_descs;

constexpr std::array<ArgumentDesc, BesselK1e::input_arg_count + BesselK1e::output_arg_count> BesselK1e::argument_descs;
constexpr std::array<AttributeDesc, 1> BesselK1e::attribute_descs;

constexpr std::array<ArgumentDesc, QuantizedDepthwiseConv2DWithBias::input_arg_count + QuantizedDepthwiseConv2DWithBias::output_arg_count> QuantizedDepthwiseConv2DWithBias::argument_descs;
constexpr std::array<AttributeDesc, 6> QuantizedDepthwiseConv2DWithBias::attribute_descs;

constexpr std::array<ArgumentDesc, UnbatchDataset::input_arg_count + UnbatchDataset::output_arg_count> UnbatchDataset::argument_descs;
constexpr std::array<AttributeDesc, 3> UnbatchDataset::attribute_descs;

constexpr std::array<ArgumentDesc, ExperimentalUniqueDataset::input_arg_count + ExperimentalUniqueDataset::output_arg_count> ExperimentalUniqueDataset::argument_descs;
constexpr std::array<AttributeDesc, 2> ExperimentalUniqueDataset::attribute_descs;

constexpr std::array<ArgumentDesc, IdentityReaderV2::input_arg_count + IdentityReaderV2::output_arg_count> IdentityReaderV2::argument_descs;
constexpr std::array<AttributeDesc, 2> IdentityReaderV2::attribute_descs;

constexpr std::array<ArgumentDesc, DrawBoundingBoxes::input_arg_count + DrawBoundingBoxes::output_arg_count> DrawBoundingBoxes::argument_descs;
constexpr std::array<AttributeDesc, 1> DrawBoundingBoxes::attribute_descs;

constexpr std::array<ArgumentDesc, DataServiceDataset::input_arg_count + DataServiceDataset::output_arg_count> DataServiceDataset::argument_descs;
constexpr std::array<AttributeDesc, 5> DataServiceDataset::attribute_descs;

constexpr std::array<ArgumentDesc, DataServiceDatasetV2::input_arg_count + DataServiceDatasetV2::output_arg_count> DataServiceDatasetV2::argument_descs;
constexpr std::array<AttributeDesc, 5> DataServiceDatasetV2::attribute_descs;

constexpr std::array<ArgumentDesc, StatefulStandardNormal::input_arg_count + StatefulStandardNormal::output_arg_count> StatefulStandardNormal::argument_descs;
constexpr std::array<AttributeDesc, 2> StatefulStandardNormal::attribute_descs;

constexpr std::array<ArgumentDesc, ParameterizedTruncatedNormal::input_arg_count + ParameterizedTruncatedNormal::output_arg_count> ParameterizedTruncatedNormal::argument_descs;
constexpr std::array<AttributeDesc, 4> ParameterizedTruncatedNormal::attribute_descs;

constexpr std::array<ArgumentDesc, DataServiceDatasetV3::input_arg_count + DataServiceDatasetV3::output_arg_count> DataServiceDatasetV3::argument_descs;
constexpr std::array<AttributeDesc, 7> DataServiceDatasetV3::attribute_descs;

constexpr std::array<ArgumentDesc, FileSystemSetConfiguration::input_arg_count + FileSystemSetConfiguration::output_arg_count> FileSystemSetConfiguration::argument_descs;
constexpr std::array<AttributeDesc, 0> FileSystemSetConfiguration::attribute_descs;

constexpr std::array<ArgumentDesc, ResizeBicubicGrad::input_arg_count + ResizeBicubicGrad::output_arg_count> ResizeBicubicGrad::argument_descs;
constexpr std::array<AttributeDesc, 3> ResizeBicubicGrad::attribute_descs;

constexpr std::array<ArgumentDesc, ResizeBilinearGrad::input_arg_count + ResizeBilinearGrad::output_arg_count> ResizeBilinearGrad::argument_descs;
constexpr std::array<AttributeDesc, 3> ResizeBilinearGrad::attribute_descs;

constexpr std::array<ArgumentDesc, StatelessRandomUniformIntV2::input_arg_count + StatelessRandomUniformIntV2::output_arg_count> StatelessRandomUniformIntV2::argument_descs;
constexpr std::array<AttributeDesc, 2> StatelessRandomUniformIntV2::attribute_descs;

constexpr std::array<ArgumentDesc, DepthwiseConv2dNativeBackpropInput::input_arg_count + DepthwiseConv2dNativeBackpropInput::output_arg_count> DepthwiseConv2dNativeBackpropInput::argument_descs;
constexpr std::array<AttributeDesc, 6> DepthwiseConv2dNativeBackpropInput::attribute_descs;

constexpr std::array<ArgumentDesc, Real::input_arg_count + Real::output_arg_count> Real::argument_descs;
constexpr std::array<AttributeDesc, 2> Real::attribute_descs;

constexpr std::array<ArgumentDesc, DecodeJpeg::input_arg_count + DecodeJpeg::output_arg_count> DecodeJpeg::argument_descs;
constexpr std::array<AttributeDesc, 6> DecodeJpeg::attribute_descs;

constexpr std::array<ArgumentDesc, BesselI1e::input_arg_count + BesselI1e::output_arg_count> BesselI1e::argument_descs;
constexpr std::array<AttributeDesc, 1> BesselI1e::attribute_descs;

constexpr std::array<ArgumentDesc, AnonymousMutableHashTable::input_arg_count + AnonymousMutableHashTable::output_arg_count> AnonymousMutableHashTable::argument_descs;
constexpr std::array<AttributeDesc, 2> AnonymousMutableHashTable::attribute_descs;

constexpr std::array<ArgumentDesc, AdjustContrast::input_arg_count + AdjustContrast::output_arg_count> AdjustContrast::argument_descs;
constexpr std::array<AttributeDesc, 1> AdjustContrast::attribute_descs;

constexpr std::array<ArgumentDesc, TensorMapLookup::input_arg_count + TensorMapLookup::output_arg_count> TensorMapLookup::argument_descs;
constexpr std::array<AttributeDesc, 2> TensorMapLookup::attribute_descs;

constexpr std::array<ArgumentDesc, RiscScatter::input_arg_count + RiscScatter::output_arg_count> RiscScatter::argument_descs;
constexpr std::array<AttributeDesc, 2> RiscScatter::attribute_descs;

constexpr std::array<ArgumentDesc, AdjustHue::input_arg_count + AdjustHue::output_arg_count> AdjustHue::argument_descs;
constexpr std::array<AttributeDesc, 1> AdjustHue::attribute_descs;

constexpr std::array<ArgumentDesc, DecodeGif::input_arg_count + DecodeGif::output_arg_count> DecodeGif::argument_descs;
constexpr std::array<AttributeDesc, 0> DecodeGif::attribute_descs;

constexpr std::array<ArgumentDesc, HSVToRGB::input_arg_count + HSVToRGB::output_arg_count> HSVToRGB::argument_descs;
constexpr std::array<AttributeDesc, 1> HSVToRGB::attribute_descs;

constexpr std::array<ArgumentDesc, DrawBoundingBoxesV2::input_arg_count + DrawBoundingBoxesV2::output_arg_count> DrawBoundingBoxesV2::argument_descs;
constexpr std::array<AttributeDesc, 1> DrawBoundingBoxesV2::attribute_descs;

constexpr std::array<ArgumentDesc, _MklNativeDepthwiseConv2dNativeBackpropInput::input_arg_count + _MklNativeDepthwiseConv2dNativeBackpropInput::output_arg_count> _MklNativeDepthwiseConv2dNativeBackpropInput::argument_descs;
constexpr std::array<AttributeDesc, 6> _MklNativeDepthwiseConv2dNativeBackpropInput::attribute_descs;

constexpr std::array<ArgumentDesc, ExtractGlimpse::input_arg_count + ExtractGlimpse::output_arg_count> ExtractGlimpse::argument_descs;
constexpr std::array<AttributeDesc, 4> ExtractGlimpse::attribute_descs;

constexpr std::array<ArgumentDesc, ExtractGlimpseV2::input_arg_count + ExtractGlimpseV2::output_arg_count> ExtractGlimpseV2::argument_descs;
constexpr std::array<AttributeDesc, 4> ExtractGlimpseV2::attribute_descs;

constexpr std::array<ArgumentDesc, LoadTPUEmbeddingRMSPropParameters::input_arg_count + LoadTPUEmbeddingRMSPropParameters::output_arg_count> LoadTPUEmbeddingRMSPropParameters::argument_descs;
constexpr std::array<AttributeDesc, 5> LoadTPUEmbeddingRMSPropParameters::attribute_descs;

constexpr std::array<ArgumentDesc, SaveV2::input_arg_count + SaveV2::output_arg_count> SaveV2::argument_descs;
constexpr std::array<AttributeDesc, 1> SaveV2::attribute_descs;

constexpr std::array<ArgumentDesc, MutableHashTableOfTensorsV2::input_arg_count + MutableHashTableOfTensorsV2::output_arg_count> MutableHashTableOfTensorsV2::argument_descs;
constexpr std::array<AttributeDesc, 6> MutableHashTableOfTensorsV2::attribute_descs;

constexpr std::array<ArgumentDesc, IsInf::input_arg_count + IsInf::output_arg_count> IsInf::argument_descs;
constexpr std::array<AttributeDesc, 1> IsInf::attribute_descs;

constexpr std::array<ArgumentDesc, CropAndResize::input_arg_count + CropAndResize::output_arg_count> CropAndResize::argument_descs;
constexpr std::array<AttributeDesc, 3> CropAndResize::attribute_descs;

constexpr std::array<ArgumentDesc, TFRecordReaderV2::input_arg_count + TFRecordReaderV2::output_arg_count> TFRecordReaderV2::argument_descs;
constexpr std::array<AttributeDesc, 3> TFRecordReaderV2::attribute_descs;

constexpr std::array<ArgumentDesc, CropAndResizeGradImage::input_arg_count + CropAndResizeGradImage::output_arg_count> CropAndResizeGradImage::argument_descs;
constexpr std::array<AttributeDesc, 2> CropAndResizeGradImage::attribute_descs;

constexpr std::array<ArgumentDesc, XlaPad::input_arg_count + XlaPad::output_arg_count> XlaPad::argument_descs;
constexpr std::array<AttributeDesc, 2> XlaPad::attribute_descs;

constexpr std::array<ArgumentDesc, CropAndResizeGradBoxes::input_arg_count + CropAndResizeGradBoxes::output_arg_count> CropAndResizeGradBoxes::argument_descs;
constexpr std::array<AttributeDesc, 2> CropAndResizeGradBoxes::attribute_descs;

constexpr std::array<ArgumentDesc, LoadTPUEmbeddingFTRLParameters::input_arg_count + LoadTPUEmbeddingFTRLParameters::output_arg_count> LoadTPUEmbeddingFTRLParameters::argument_descs;
constexpr std::array<AttributeDesc, 5> LoadTPUEmbeddingFTRLParameters::attribute_descs;

constexpr std::array<ArgumentDesc, NonMaxSuppression::input_arg_count + NonMaxSuppression::output_arg_count> NonMaxSuppression::argument_descs;
constexpr std::array<AttributeDesc, 1> NonMaxSuppression::attribute_descs;

constexpr std::array<ArgumentDesc, Atanh::input_arg_count + Atanh::output_arg_count> Atanh::argument_descs;
constexpr std::array<AttributeDesc, 1> Atanh::attribute_descs;

constexpr std::array<ArgumentDesc, NonMaxSuppressionV4::input_arg_count + NonMaxSuppressionV4::output_arg_count> NonMaxSuppressionV4::argument_descs;
constexpr std::array<AttributeDesc, 3> NonMaxSuppressionV4::attribute_descs;

constexpr std::array<ArgumentDesc, NonMaxSuppressionV5::input_arg_count + NonMaxSuppressionV5::output_arg_count> NonMaxSuppressionV5::argument_descs;
constexpr std::array<AttributeDesc, 2> NonMaxSuppressionV5::attribute_descs;

constexpr std::array<ArgumentDesc, SparseFillEmptyRowsGrad::input_arg_count + SparseFillEmptyRowsGrad::output_arg_count> SparseFillEmptyRowsGrad::argument_descs;
constexpr std::array<AttributeDesc, 1> SparseFillEmptyRowsGrad::attribute_descs;

constexpr std::array<ArgumentDesc, CombinedNonMaxSuppression::input_arg_count + CombinedNonMaxSuppression::output_arg_count> CombinedNonMaxSuppression::argument_descs;
constexpr std::array<AttributeDesc, 2> CombinedNonMaxSuppression::attribute_descs;

constexpr std::array<ArgumentDesc, Prod::input_arg_count + Prod::output_arg_count> Prod::argument_descs;
constexpr std::array<AttributeDesc, 3> Prod::attribute_descs;

constexpr std::array<ArgumentDesc, GenerateBoundingBoxProposals::input_arg_count + GenerateBoundingBoxProposals::output_arg_count> GenerateBoundingBoxProposals::argument_descs;
constexpr std::array<AttributeDesc, 1> GenerateBoundingBoxProposals::attribute_descs;

constexpr std::array<ArgumentDesc, Acosh::input_arg_count + Acosh::output_arg_count> Acosh::argument_descs;
constexpr std::array<AttributeDesc, 1> Acosh::attribute_descs;

constexpr std::array<ArgumentDesc, ImageProjectiveTransformV2::input_arg_count + ImageProjectiveTransformV2::output_arg_count> ImageProjectiveTransformV2::argument_descs;
constexpr std::array<AttributeDesc, 3> ImageProjectiveTransformV2::attribute_descs;

constexpr std::array<ArgumentDesc, ImageProjectiveTransformV3::input_arg_count + ImageProjectiveTransformV3::output_arg_count> ImageProjectiveTransformV3::argument_descs;
constexpr std::array<AttributeDesc, 3> ImageProjectiveTransformV3::attribute_descs;

constexpr std::array<ArgumentDesc, SquaredDifference::input_arg_count + SquaredDifference::output_arg_count> SquaredDifference::argument_descs;
constexpr std::array<AttributeDesc, 1> SquaredDifference::attribute_descs;

constexpr std::array<ArgumentDesc, RestoreV2::input_arg_count + RestoreV2::output_arg_count> RestoreV2::argument_descs;
constexpr std::array<AttributeDesc, 1> RestoreV2::attribute_descs;

constexpr std::array<ArgumentDesc, Conv3DBackpropFilterV2::input_arg_count + Conv3DBackpropFilterV2::output_arg_count> Conv3DBackpropFilterV2::argument_descs;
constexpr std::array<AttributeDesc, 5> Conv3DBackpropFilterV2::attribute_descs;

constexpr std::array<ArgumentDesc, Save::input_arg_count + Save::output_arg_count> Save::argument_descs;
constexpr std::array<AttributeDesc, 1> Save::attribute_descs;

constexpr std::array<ArgumentDesc, MutableHashTableOfTensors::input_arg_count + MutableHashTableOfTensors::output_arg_count> MutableHashTableOfTensors::argument_descs;
constexpr std::array<AttributeDesc, 6> MutableHashTableOfTensors::attribute_descs;

constexpr std::array<ArgumentDesc, UnsortedSegmentMin::input_arg_count + UnsortedSegmentMin::output_arg_count> UnsortedSegmentMin::argument_descs;
constexpr std::array<AttributeDesc, 3> UnsortedSegmentMin::attribute_descs;

constexpr std::array<ArgumentDesc, _FusedMatMul::input_arg_count + _FusedMatMul::output_arg_count> _FusedMatMul::argument_descs;
constexpr std::array<AttributeDesc, 7> _FusedMatMul::attribute_descs;

constexpr std::array<ArgumentDesc, TopKV2::input_arg_count + TopKV2::output_arg_count> TopKV2::argument_descs;
constexpr std::array<AttributeDesc, 2> TopKV2::attribute_descs;

constexpr std::array<ArgumentDesc, Restore::input_arg_count + Restore::output_arg_count> Restore::argument_descs;
constexpr std::array<AttributeDesc, 2> Restore::attribute_descs;

constexpr std::array<ArgumentDesc, SparseApplyAdagradV2::input_arg_count + SparseApplyAdagradV2::output_arg_count> SparseApplyAdagradV2::argument_descs;
constexpr std::array<AttributeDesc, 4> SparseApplyAdagradV2::attribute_descs;

constexpr std::array<ArgumentDesc, RestoreSlice::input_arg_count + RestoreSlice::output_arg_count> RestoreSlice::argument_descs;
constexpr std::array<AttributeDesc, 2> RestoreSlice::attribute_descs;

constexpr std::array<ArgumentDesc, ShardedFilename::input_arg_count + ShardedFilename::output_arg_count> ShardedFilename::argument_descs;
constexpr std::array<AttributeDesc, 0> ShardedFilename::attribute_descs;

constexpr std::array<ArgumentDesc, Relu6::input_arg_count + Relu6::output_arg_count> Relu6::argument_descs;
constexpr std::array<AttributeDesc, 1> Relu6::attribute_descs;

constexpr std::array<ArgumentDesc, TextLineReader::input_arg_count + TextLineReader::output_arg_count> TextLineReader::argument_descs;
constexpr std::array<AttributeDesc, 3> TextLineReader::attribute_descs;

constexpr std::array<ArgumentDesc, StatefulUniformFullInt::input_arg_count + StatefulUniformFullInt::output_arg_count> StatefulUniformFullInt::argument_descs;
constexpr std::array<AttributeDesc, 2> StatefulUniformFullInt::attribute_descs;

constexpr std::array<ArgumentDesc, TextLineReaderV2::input_arg_count + TextLineReaderV2::output_arg_count> TextLineReaderV2::argument_descs;
constexpr std::array<AttributeDesc, 3> TextLineReaderV2::attribute_descs;

constexpr std::array<ArgumentDesc, TensorListGather::input_arg_count + TensorListGather::output_arg_count> TensorListGather::argument_descs;
constexpr std::array<AttributeDesc, 1> TensorListGather::attribute_descs;

constexpr std::array<ArgumentDesc, FixedLengthRecordReader::input_arg_count + FixedLengthRecordReader::output_arg_count> FixedLengthRecordReader::argument_descs;
constexpr std::array<AttributeDesc, 6> FixedLengthRecordReader::attribute_descs;

constexpr std::array<ArgumentDesc, MatrixSquareRoot::input_arg_count + MatrixSquareRoot::output_arg_count> MatrixSquareRoot::argument_descs;
constexpr std::array<AttributeDesc, 1> MatrixSquareRoot::attribute_descs;

constexpr std::array<ArgumentDesc, XlaSpmdFullToShardShape::input_arg_count + XlaSpmdFullToShardShape::output_arg_count> XlaSpmdFullToShardShape::argument_descs;
constexpr std::array<AttributeDesc, 4> XlaSpmdFullToShardShape::attribute_descs;

constexpr std::array<ArgumentDesc, NthElement::input_arg_count + NthElement::output_arg_count> NthElement::argument_descs;
constexpr std::array<AttributeDesc, 2> NthElement::attribute_descs;

constexpr std::array<ArgumentDesc, FixedLengthRecordReaderV2::input_arg_count + FixedLengthRecordReaderV2::output_arg_count> FixedLengthRecordReaderV2::argument_descs;
constexpr std::array<AttributeDesc, 7> FixedLengthRecordReaderV2::attribute_descs;

constexpr std::array<ArgumentDesc, ReaderRead::input_arg_count + ReaderRead::output_arg_count> ReaderRead::argument_descs;
constexpr std::array<AttributeDesc, 0> ReaderRead::attribute_descs;

constexpr std::array<ArgumentDesc, ReaderNumRecordsProduced::input_arg_count + ReaderNumRecordsProduced::output_arg_count> ReaderNumRecordsProduced::argument_descs;
constexpr std::array<AttributeDesc, 0> ReaderNumRecordsProduced::attribute_descs;

constexpr std::array<ArgumentDesc, ReaderNumWorkUnitsCompleted::input_arg_count + ReaderNumWorkUnitsCompleted::output_arg_count> ReaderNumWorkUnitsCompleted::argument_descs;
constexpr std::array<AttributeDesc, 0> ReaderNumWorkUnitsCompleted::attribute_descs;

constexpr std::array<ArgumentDesc, LogMatrixDeterminant::input_arg_count + LogMatrixDeterminant::output_arg_count> LogMatrixDeterminant::argument_descs;
constexpr std::array<AttributeDesc, 1> LogMatrixDeterminant::attribute_descs;

constexpr std::array<ArgumentDesc, ReaderNumWorkUnitsCompletedV2::input_arg_count + ReaderNumWorkUnitsCompletedV2::output_arg_count> ReaderNumWorkUnitsCompletedV2::argument_descs;
constexpr std::array<AttributeDesc, 0> ReaderNumWorkUnitsCompletedV2::attribute_descs;

constexpr std::array<ArgumentDesc, ReaderSerializeState::input_arg_count + ReaderSerializeState::output_arg_count> ReaderSerializeState::argument_descs;
constexpr std::array<AttributeDesc, 0> ReaderSerializeState::attribute_descs;

constexpr std::array<ArgumentDesc, ReaderSerializeStateV2::input_arg_count + ReaderSerializeStateV2::output_arg_count> ReaderSerializeStateV2::argument_descs;
constexpr std::array<AttributeDesc, 0> ReaderSerializeStateV2::attribute_descs;

constexpr std::array<ArgumentDesc, Xlog1py::input_arg_count + Xlog1py::output_arg_count> Xlog1py::argument_descs;
constexpr std::array<AttributeDesc, 1> Xlog1py::attribute_descs;

constexpr std::array<ArgumentDesc, AssignSubVariableOp::input_arg_count + AssignSubVariableOp::output_arg_count> AssignSubVariableOp::argument_descs;
constexpr std::array<AttributeDesc, 1> AssignSubVariableOp::attribute_descs;

constexpr std::array<ArgumentDesc, ReaderRestoreStateV2::input_arg_count + ReaderRestoreStateV2::output_arg_count> ReaderRestoreStateV2::argument_descs;
constexpr std::array<AttributeDesc, 0> ReaderRestoreStateV2::attribute_descs;

constexpr std::array<ArgumentDesc, ReaderReset::input_arg_count + ReaderReset::output_arg_count> ReaderReset::argument_descs;
constexpr std::array<AttributeDesc, 0> ReaderReset::attribute_descs;

constexpr std::array<ArgumentDesc, SparseMatrixSoftmaxGrad::input_arg_count + SparseMatrixSoftmaxGrad::output_arg_count> SparseMatrixSoftmaxGrad::argument_descs;
constexpr std::array<AttributeDesc, 1> SparseMatrixSoftmaxGrad::attribute_descs;

constexpr std::array<ArgumentDesc, Multinomial::input_arg_count + Multinomial::output_arg_count> Multinomial::argument_descs;
constexpr std::array<AttributeDesc, 4> Multinomial::attribute_descs;

constexpr std::array<ArgumentDesc, WriteFile::input_arg_count + WriteFile::output_arg_count> WriteFile::argument_descs;
constexpr std::array<AttributeDesc, 0> WriteFile::attribute_descs;

constexpr std::array<ArgumentDesc, Cholesky::input_arg_count + Cholesky::output_arg_count> Cholesky::argument_descs;
constexpr std::array<AttributeDesc, 1> Cholesky::attribute_descs;

constexpr std::array<ArgumentDesc, AnonymousMutableDenseHashTable::input_arg_count + AnonymousMutableDenseHashTable::output_arg_count> AnonymousMutableDenseHashTable::argument_descs;
constexpr std::array<AttributeDesc, 5> AnonymousMutableDenseHashTable::attribute_descs;

constexpr std::array<ArgumentDesc, ApplyPowerSign::input_arg_count + ApplyPowerSign::output_arg_count> ApplyPowerSign::argument_descs;
constexpr std::array<AttributeDesc, 2> ApplyPowerSign::attribute_descs;

constexpr std::array<ArgumentDesc, SelfAdjointEig::input_arg_count + SelfAdjointEig::output_arg_count> SelfAdjointEig::argument_descs;
constexpr std::array<AttributeDesc, 1> SelfAdjointEig::attribute_descs;

constexpr std::array<ArgumentDesc, Eig::input_arg_count + Eig::output_arg_count> Eig::argument_descs;
constexpr std::array<AttributeDesc, 3> Eig::attribute_descs;

constexpr std::array<ArgumentDesc, LookupTableSize::input_arg_count + LookupTableSize::output_arg_count> LookupTableSize::argument_descs;
constexpr std::array<AttributeDesc, 0> LookupTableSize::attribute_descs;

constexpr std::array<ArgumentDesc, SerializeSparse::input_arg_count + SerializeSparse::output_arg_count> SerializeSparse::argument_descs;
constexpr std::array<AttributeDesc, 2> SerializeSparse::attribute_descs;

constexpr std::array<ArgumentDesc, SelfAdjointEigV2::input_arg_count + SelfAdjointEigV2::output_arg_count> SelfAdjointEigV2::argument_descs;
constexpr std::array<AttributeDesc, 2> SelfAdjointEigV2::attribute_descs;

constexpr std::array<ArgumentDesc, MatrixSolve::input_arg_count + MatrixSolve::output_arg_count> MatrixSolve::argument_descs;
constexpr std::array<AttributeDesc, 2> MatrixSolve::attribute_descs;

constexpr std::array<ArgumentDesc, _MklLayerNorm::input_arg_count + _MklLayerNorm::output_arg_count> _MklLayerNorm::argument_descs;
constexpr std::array<AttributeDesc, 2> _MklLayerNorm::attribute_descs;

constexpr std::array<ArgumentDesc, _MklQuantizedConv2DWithBiasAndRequantize::input_arg_count + _MklQuantizedConv2DWithBiasAndRequantize::output_arg_count> _MklQuantizedConv2DWithBiasAndRequantize::argument_descs;
constexpr std::array<AttributeDesc, 11> _MklQuantizedConv2DWithBiasAndRequantize::attribute_descs;

constexpr std::array<ArgumentDesc, BandedTriangularSolve::input_arg_count + BandedTriangularSolve::output_arg_count> BandedTriangularSolve::argument_descs;
constexpr std::array<AttributeDesc, 3> BandedTriangularSolve::attribute_descs;

constexpr std::array<ArgumentDesc, RiscReverse::input_arg_count + RiscReverse::output_arg_count> RiscReverse::argument_descs;
constexpr std::array<AttributeDesc, 2> RiscReverse::attribute_descs;

constexpr std::array<ArgumentDesc, MatrixTriangularSolve::input_arg_count + MatrixTriangularSolve::output_arg_count> MatrixTriangularSolve::argument_descs;
constexpr std::array<AttributeDesc, 3> MatrixTriangularSolve::attribute_descs;

constexpr std::array<ArgumentDesc, If::input_arg_count + If::output_arg_count> If::argument_descs;
constexpr std::array<AttributeDesc, 6> If::attribute_descs;

constexpr std::array<ArgumentDesc, RiscBitcast::input_arg_count + RiscBitcast::output_arg_count> RiscBitcast::argument_descs;
constexpr std::array<AttributeDesc, 2> RiscBitcast::attribute_descs;

constexpr std::array<ArgumentDesc, MatrixSolveLs::input_arg_count + MatrixSolveLs::output_arg_count> MatrixSolveLs::argument_descs;
constexpr std::array<AttributeDesc, 2> MatrixSolveLs::attribute_descs;

constexpr std::array<ArgumentDesc, _MklNativeConv2DBackpropFilterWithBias::input_arg_count + _MklNativeConv2DBackpropFilterWithBias::output_arg_count> _MklNativeConv2DBackpropFilterWithBias::argument_descs;
constexpr std::array<AttributeDesc, 6> _MklNativeConv2DBackpropFilterWithBias::attribute_descs;

constexpr std::array<ArgumentDesc, CountUpTo::input_arg_count + CountUpTo::output_arg_count> CountUpTo::argument_descs;
constexpr std::array<AttributeDesc, 2> CountUpTo::attribute_descs;

constexpr std::array<ArgumentDesc, Svd::input_arg_count + Svd::output_arg_count> Svd::argument_descs;
constexpr std::array<AttributeDesc, 3> Svd::attribute_descs;

constexpr std::array<ArgumentDesc, Exp::input_arg_count + Exp::output_arg_count> Exp::argument_descs;
constexpr std::array<AttributeDesc, 1> Exp::attribute_descs;

constexpr std::array<ArgumentDesc, TridiagonalMatMul::input_arg_count + TridiagonalMatMul::output_arg_count> TridiagonalMatMul::argument_descs;
constexpr std::array<AttributeDesc, 1> TridiagonalMatMul::attribute_descs;

constexpr std::array<ArgumentDesc, TridiagonalSolve::input_arg_count + TridiagonalSolve::output_arg_count> TridiagonalSolve::argument_descs;
constexpr std::array<AttributeDesc, 3> TridiagonalSolve::attribute_descs;

constexpr std::array<ArgumentDesc, BatchMatrixDeterminant::input_arg_count + BatchMatrixDeterminant::output_arg_count> BatchMatrixDeterminant::argument_descs;
constexpr std::array<AttributeDesc, 1> BatchMatrixDeterminant::attribute_descs;

constexpr std::array<ArgumentDesc, RiscSign::input_arg_count + RiscSign::output_arg_count> RiscSign::argument_descs;
constexpr std::array<AttributeDesc, 1> RiscSign::attribute_descs;

constexpr std::array<ArgumentDesc, BatchCholeskyGrad::input_arg_count + BatchCholeskyGrad::output_arg_count> BatchCholeskyGrad::argument_descs;
constexpr std::array<AttributeDesc, 1> BatchCholeskyGrad::attribute_descs;

constexpr std::array<ArgumentDesc, ResourceApplyAdagradV2::input_arg_count + ResourceApplyAdagradV2::output_arg_count> ResourceApplyAdagradV2::argument_descs;
constexpr std::array<AttributeDesc, 3> ResourceApplyAdagradV2::attribute_descs;

constexpr std::array<ArgumentDesc, TensorListPushBackBatch::input_arg_count + TensorListPushBackBatch::output_arg_count> TensorListPushBackBatch::argument_descs;
constexpr std::array<AttributeDesc, 1> TensorListPushBackBatch::attribute_descs;

constexpr std::array<ArgumentDesc, TensorListLength::input_arg_count + TensorListLength::output_arg_count> TensorListLength::argument_descs;
constexpr std::array<AttributeDesc, 0> TensorListLength::attribute_descs;

constexpr std::array<ArgumentDesc, Mean::input_arg_count + Mean::output_arg_count> Mean::argument_descs;
constexpr std::array<AttributeDesc, 3> Mean::attribute_descs;

constexpr std::array<ArgumentDesc, TensorListStack::input_arg_count + TensorListStack::output_arg_count> TensorListStack::argument_descs;
constexpr std::array<AttributeDesc, 2> TensorListStack::attribute_descs;

constexpr std::array<ArgumentDesc, _Retval::input_arg_count + _Retval::output_arg_count> _Retval::argument_descs;
constexpr std::array<AttributeDesc, 2> _Retval::attribute_descs;

constexpr std::array<ArgumentDesc, TensorListConcat::input_arg_count + TensorListConcat::output_arg_count> TensorListConcat::argument_descs;
constexpr std::array<AttributeDesc, 2> TensorListConcat::attribute_descs;

constexpr std::array<ArgumentDesc, TensorListConcatV2::input_arg_count + TensorListConcatV2::output_arg_count> TensorListConcatV2::argument_descs;
constexpr std::array<AttributeDesc, 2> TensorListConcatV2::attribute_descs;

constexpr std::array<ArgumentDesc, _MklQuantizedMatMulWithBiasAndReluAndRequantize::input_arg_count + _MklQuantizedMatMulWithBiasAndReluAndRequantize::output_arg_count> _MklQuantizedMatMulWithBiasAndReluAndRequantize::argument_descs;
constexpr std::array<AttributeDesc, 8> _MklQuantizedMatMulWithBiasAndReluAndRequantize::attribute_descs;

constexpr std::array<ArgumentDesc, TensorListSplit::input_arg_count + TensorListSplit::output_arg_count> TensorListSplit::argument_descs;
constexpr std::array<AttributeDesc, 2> TensorListSplit::attribute_descs;

constexpr std::array<ArgumentDesc, TensorSummaryV2::input_arg_count + TensorSummaryV2::output_arg_count> TensorSummaryV2::argument_descs;
constexpr std::array<AttributeDesc, 1> TensorSummaryV2::attribute_descs;

constexpr std::array<ArgumentDesc, TensorListElementShape::input_arg_count + TensorListElementShape::output_arg_count> TensorListElementShape::argument_descs;
constexpr std::array<AttributeDesc, 1> TensorListElementShape::attribute_descs;

constexpr std::array<ArgumentDesc, _HostCast::input_arg_count + _HostCast::output_arg_count> _HostCast::argument_descs;
constexpr std::array<AttributeDesc, 3> _HostCast::attribute_descs;

constexpr std::array<ArgumentDesc, TensorListGetItem::input_arg_count + TensorListGetItem::output_arg_count> TensorListGetItem::argument_descs;
constexpr std::array<AttributeDesc, 1> TensorListGetItem::attribute_descs;

constexpr std::array<ArgumentDesc, TensorListResize::input_arg_count + TensorListResize::output_arg_count> TensorListResize::argument_descs;
constexpr std::array<AttributeDesc, 0> TensorListResize::attribute_descs;

constexpr std::array<ArgumentDesc, TensorListScatter::input_arg_count + TensorListScatter::output_arg_count> TensorListScatter::argument_descs;
constexpr std::array<AttributeDesc, 2> TensorListScatter::attribute_descs;

constexpr std::array<ArgumentDesc, NextAfter::input_arg_count + NextAfter::output_arg_count> NextAfter::argument_descs;
constexpr std::array<AttributeDesc, 1> NextAfter::attribute_descs;

constexpr std::array<ArgumentDesc, TensorListScatterV2::input_arg_count + TensorListScatterV2::output_arg_count> TensorListScatterV2::argument_descs;
constexpr std::array<AttributeDesc, 2> TensorListScatterV2::attribute_descs;

constexpr std::array<ArgumentDesc, TensorListScatterIntoExistingList::input_arg_count + TensorListScatterIntoExistingList::output_arg_count> TensorListScatterIntoExistingList::argument_descs;
constexpr std::array<AttributeDesc, 1> TensorListScatterIntoExistingList::attribute_descs;

constexpr std::array<ArgumentDesc, TensorMapSize::input_arg_count + TensorMapSize::output_arg_count> TensorMapSize::argument_descs;
constexpr std::array<AttributeDesc, 0> TensorMapSize::attribute_descs;

constexpr std::array<ArgumentDesc, AudioSummaryV2::input_arg_count + AudioSummaryV2::output_arg_count> AudioSummaryV2::argument_descs;
constexpr std::array<AttributeDesc, 1> AudioSummaryV2::attribute_descs;

constexpr std::array<ArgumentDesc, TensorMapErase::input_arg_count + TensorMapErase::output_arg_count> TensorMapErase::argument_descs;
constexpr std::array<AttributeDesc, 2> TensorMapErase::attribute_descs;

constexpr std::array<ArgumentDesc, Assert::input_arg_count + Assert::output_arg_count> Assert::argument_descs;
constexpr std::array<AttributeDesc, 2> Assert::attribute_descs;

constexpr std::array<ArgumentDesc, Print::input_arg_count + Print::output_arg_count> Print::argument_descs;
constexpr std::array<AttributeDesc, 5> Print::attribute_descs;

constexpr std::array<ArgumentDesc, TensorSummary::input_arg_count + TensorSummary::output_arg_count> TensorSummary::argument_descs;
constexpr std::array<AttributeDesc, 4> TensorSummary::attribute_descs;

constexpr std::array<ArgumentDesc, LookupTableFind::input_arg_count + LookupTableFind::output_arg_count> LookupTableFind::argument_descs;
constexpr std::array<AttributeDesc, 2> LookupTableFind::attribute_descs;

constexpr std::array<ArgumentDesc, LookupTableFindV2::input_arg_count + LookupTableFindV2::output_arg_count> LookupTableFindV2::argument_descs;
constexpr std::array<AttributeDesc, 2> LookupTableFindV2::attribute_descs;

constexpr std::array<ArgumentDesc, SerializeTensor::input_arg_count + SerializeTensor::output_arg_count> SerializeTensor::argument_descs;
constexpr std::array<AttributeDesc, 1> SerializeTensor::attribute_descs;

constexpr std::array<ArgumentDesc, _FusedBatchNormGradEx::input_arg_count + _FusedBatchNormGradEx::output_arg_count> _FusedBatchNormGradEx::argument_descs;
constexpr std::array<AttributeDesc, 7> _FusedBatchNormGradEx::attribute_descs;

constexpr std::array<ArgumentDesc, LookupTableInsert::input_arg_count + LookupTableInsert::output_arg_count> LookupTableInsert::argument_descs;
constexpr std::array<AttributeDesc, 2> LookupTableInsert::attribute_descs;

constexpr std::array<ArgumentDesc, InvGrad::input_arg_count + InvGrad::output_arg_count> InvGrad::argument_descs;
constexpr std::array<AttributeDesc, 1> InvGrad::attribute_descs;

constexpr std::array<ArgumentDesc, LookupTableInsertV2::input_arg_count + LookupTableInsertV2::output_arg_count> LookupTableInsertV2::argument_descs;
constexpr std::array<AttributeDesc, 2> LookupTableInsertV2::attribute_descs;

constexpr std::array<ArgumentDesc, BatchMatMulV3::input_arg_count + BatchMatMulV3::output_arg_count> BatchMatMulV3::argument_descs;
constexpr std::array<AttributeDesc, 5> BatchMatMulV3::attribute_descs;

constexpr std::array<ArgumentDesc, LookupTableExport::input_arg_count + LookupTableExport::output_arg_count> LookupTableExport::argument_descs;
constexpr std::array<AttributeDesc, 2> LookupTableExport::attribute_descs;

constexpr std::array<ArgumentDesc, LookupTableExportV2::input_arg_count + LookupTableExportV2::output_arg_count> LookupTableExportV2::argument_descs;
constexpr std::array<AttributeDesc, 2> LookupTableExportV2::attribute_descs;

constexpr std::array<ArgumentDesc, Rsqrt::input_arg_count + Rsqrt::output_arg_count> Rsqrt::argument_descs;
constexpr std::array<AttributeDesc, 1> Rsqrt::attribute_descs;

constexpr std::array<ArgumentDesc, LookupTableImportV2::input_arg_count + LookupTableImportV2::output_arg_count> LookupTableImportV2::argument_descs;
constexpr std::array<AttributeDesc, 2> LookupTableImportV2::attribute_descs;

constexpr std::array<ArgumentDesc, HashTable::input_arg_count + HashTable::output_arg_count> HashTable::argument_descs;
constexpr std::array<AttributeDesc, 5> HashTable::attribute_descs;

constexpr std::array<ArgumentDesc, Asin::input_arg_count + Asin::output_arg_count> Asin::argument_descs;
constexpr std::array<AttributeDesc, 1> Asin::attribute_descs;

constexpr std::array<ArgumentDesc, RequantizationRange::input_arg_count + RequantizationRange::output_arg_count> RequantizationRange::argument_descs;
constexpr std::array<AttributeDesc, 1> RequantizationRange::attribute_descs;

constexpr std::array<ArgumentDesc, QuantizedConv2DWithBiasAndRequantize::input_arg_count + QuantizedConv2DWithBiasAndRequantize::output_arg_count> QuantizedConv2DWithBiasAndRequantize::argument_descs;
constexpr std::array<AttributeDesc, 8> QuantizedConv2DWithBiasAndRequantize::attribute_descs;

constexpr std::array<ArgumentDesc, AnonymousHashTable::input_arg_count + AnonymousHashTable::output_arg_count> AnonymousHashTable::argument_descs;
constexpr std::array<AttributeDesc, 2> AnonymousHashTable::attribute_descs;

constexpr std::array<ArgumentDesc, MutableHashTableV2::input_arg_count + MutableHashTableV2::output_arg_count> MutableHashTableV2::argument_descs;
constexpr std::array<AttributeDesc, 5> MutableHashTableV2::attribute_descs;

constexpr std::array<ArgumentDesc, Any::input_arg_count + Any::output_arg_count> Any::argument_descs;
constexpr std::array<AttributeDesc, 2> Any::attribute_descs;

constexpr std::array<ArgumentDesc, AnonymousMutableHashTableOfTensors::input_arg_count + AnonymousMutableHashTableOfTensors::output_arg_count> AnonymousMutableHashTableOfTensors::argument_descs;
constexpr std::array<AttributeDesc, 3> AnonymousMutableHashTableOfTensors::attribute_descs;

constexpr std::array<ArgumentDesc, SparseMatrixSparseCholesky::input_arg_count + SparseMatrixSparseCholesky::output_arg_count> SparseMatrixSparseCholesky::argument_descs;
constexpr std::array<AttributeDesc, 1> SparseMatrixSparseCholesky::attribute_descs;

constexpr std::array<ArgumentDesc, MutableDenseHashTable::input_arg_count + MutableDenseHashTable::output_arg_count> MutableDenseHashTable::argument_descs;
constexpr std::array<AttributeDesc, 8> MutableDenseHashTable::attribute_descs;

constexpr std::array<ArgumentDesc, LSTMBlockCellGrad::input_arg_count + LSTMBlockCellGrad::output_arg_count> LSTMBlockCellGrad::argument_descs;
constexpr std::array<AttributeDesc, 2> LSTMBlockCellGrad::attribute_descs;

constexpr std::array<ArgumentDesc, MutableDenseHashTableV2::input_arg_count + MutableDenseHashTableV2::output_arg_count> MutableDenseHashTableV2::argument_descs;
constexpr std::array<AttributeDesc, 8> MutableDenseHashTableV2::attribute_descs;

constexpr std::array<ArgumentDesc, DenseToDenseSetOperation::input_arg_count + DenseToDenseSetOperation::output_arg_count> DenseToDenseSetOperation::argument_descs;
constexpr std::array<AttributeDesc, 3> DenseToDenseSetOperation::attribute_descs;

constexpr std::array<ArgumentDesc, RandomPoisson::input_arg_count + RandomPoisson::output_arg_count> RandomPoisson::argument_descs;
constexpr std::array<AttributeDesc, 4> RandomPoisson::attribute_descs;

constexpr std::array<ArgumentDesc, InitializeTable::input_arg_count + InitializeTable::output_arg_count> InitializeTable::argument_descs;
constexpr std::array<AttributeDesc, 2> InitializeTable::attribute_descs;

constexpr std::array<ArgumentDesc, _MklFusedConv2D::input_arg_count + _MklFusedConv2D::output_arg_count> _MklFusedConv2D::argument_descs;
constexpr std::array<AttributeDesc, 12> _MklFusedConv2D::attribute_descs;

constexpr std::array<ArgumentDesc, SparseMatMul::input_arg_count + SparseMatMul::output_arg_count> SparseMatMul::argument_descs;
constexpr std::array<AttributeDesc, 6> SparseMatMul::attribute_descs;

constexpr std::array<ArgumentDesc, RandomPoissonV2::input_arg_count + RandomPoissonV2::output_arg_count> RandomPoissonV2::argument_descs;
constexpr std::array<AttributeDesc, 5> RandomPoissonV2::attribute_descs;

constexpr std::array<ArgumentDesc, InitializeTableV2::input_arg_count + InitializeTableV2::output_arg_count> InitializeTableV2::argument_descs;
constexpr std::array<AttributeDesc, 2> InitializeTableV2::attribute_descs;

constexpr std::array<ArgumentDesc, ApplyAdam::input_arg_count + ApplyAdam::output_arg_count> ApplyAdam::argument_descs;
constexpr std::array<AttributeDesc, 3> ApplyAdam::attribute_descs;

constexpr std::array<ArgumentDesc, InitializeTableFromTextFile::input_arg_count + InitializeTableFromTextFile::output_arg_count> InitializeTableFromTextFile::argument_descs;
constexpr std::array<AttributeDesc, 5> InitializeTableFromTextFile::attribute_descs;

constexpr std::array<ArgumentDesc, For::input_arg_count + For::output_arg_count> For::argument_descs;
constexpr std::array<AttributeDesc, 2> For::attribute_descs;

constexpr std::array<ArgumentDesc, Tanh::input_arg_count + Tanh::output_arg_count> Tanh::argument_descs;
constexpr std::array<AttributeDesc, 1> Tanh::attribute_descs;

constexpr std::array<ArgumentDesc, InitializeTableFromTextFileV2::input_arg_count + InitializeTableFromTextFileV2::output_arg_count> InitializeTableFromTextFileV2::argument_descs;
constexpr std::array<AttributeDesc, 5> InitializeTableFromTextFileV2::attribute_descs;

constexpr std::array<ArgumentDesc, ResourceSparseApplyFtrlV2::input_arg_count + ResourceSparseApplyFtrlV2::output_arg_count> ResourceSparseApplyFtrlV2::argument_descs;
constexpr std::array<AttributeDesc, 4> ResourceSparseApplyFtrlV2::attribute_descs;

constexpr std::array<ArgumentDesc, AccumulateNV2::input_arg_count + AccumulateNV2::output_arg_count> AccumulateNV2::argument_descs;
constexpr std::array<AttributeDesc, 3> AccumulateNV2::attribute_descs;

constexpr std::array<ArgumentDesc, BatchMatMulV2::input_arg_count + BatchMatMulV2::output_arg_count> BatchMatMulV2::argument_descs;
constexpr std::array<AttributeDesc, 3> BatchMatMulV2::attribute_descs;

constexpr std::array<ArgumentDesc, RiscAbs::input_arg_count + RiscAbs::output_arg_count> RiscAbs::argument_descs;
constexpr std::array<AttributeDesc, 1> RiscAbs::attribute_descs;

constexpr std::array<ArgumentDesc, _MklBatchMatMulV2::input_arg_count + _MklBatchMatMulV2::output_arg_count> _MklBatchMatMulV2::argument_descs;
constexpr std::array<AttributeDesc, 3> _MklBatchMatMulV2::attribute_descs;

constexpr std::array<ArgumentDesc, _MklLeakyReluGrad::input_arg_count + _MklLeakyReluGrad::output_arg_count> _MklLeakyReluGrad::argument_descs;
constexpr std::array<AttributeDesc, 2> _MklLeakyReluGrad::attribute_descs;

constexpr std::array<ArgumentDesc, Cast::input_arg_count + Cast::output_arg_count> Cast::argument_descs;
constexpr std::array<AttributeDesc, 3> Cast::attribute_descs;

constexpr std::array<ArgumentDesc, Abs::input_arg_count + Abs::output_arg_count> Abs::argument_descs;
constexpr std::array<AttributeDesc, 1> Abs::attribute_descs;

constexpr std::array<ArgumentDesc, ComplexAbs::input_arg_count + ComplexAbs::output_arg_count> ComplexAbs::argument_descs;
constexpr std::array<AttributeDesc, 2> ComplexAbs::attribute_descs;

constexpr std::array<ArgumentDesc, TPUCompile::input_arg_count + TPUCompile::output_arg_count> TPUCompile::argument_descs;
constexpr std::array<AttributeDesc, 5> TPUCompile::attribute_descs;

constexpr std::array<ArgumentDesc, RiscGather::input_arg_count + RiscGather::output_arg_count> RiscGather::argument_descs;
constexpr std::array<AttributeDesc, 4> RiscGather::attribute_descs;

constexpr std::array<ArgumentDesc, Neg::input_arg_count + Neg::output_arg_count> Neg::argument_descs;
constexpr std::array<AttributeDesc, 1> Neg::attribute_descs;

constexpr std::array<ArgumentDesc, Softmax::input_arg_count + Softmax::output_arg_count> Softmax::argument_descs;
constexpr std::array<AttributeDesc, 1> Softmax::attribute_descs;

constexpr std::array<ArgumentDesc, Erfc::input_arg_count + Erfc::output_arg_count> Erfc::argument_descs;
constexpr std::array<AttributeDesc, 1> Erfc::attribute_descs;

constexpr std::array<ArgumentDesc, Expint::input_arg_count + Expint::output_arg_count> Expint::argument_descs;
constexpr std::array<AttributeDesc, 1> Expint::attribute_descs;

constexpr std::array<ArgumentDesc, ReciprocalGrad::input_arg_count + ReciprocalGrad::output_arg_count> ReciprocalGrad::argument_descs;
constexpr std::array<AttributeDesc, 1> ReciprocalGrad::attribute_descs;

constexpr std::array<ArgumentDesc, StatelessRandomUniformFullIntV2::input_arg_count + StatelessRandomUniformFullIntV2::output_arg_count> StatelessRandomUniformFullIntV2::argument_descs;
constexpr std::array<AttributeDesc, 2> StatelessRandomUniformFullIntV2::attribute_descs;

constexpr std::array<ArgumentDesc, DataFormatVecPermute::input_arg_count + DataFormatVecPermute::output_arg_count> DataFormatVecPermute::argument_descs;
constexpr std::array<AttributeDesc, 3> DataFormatVecPermute::attribute_descs;

constexpr std::array<ArgumentDesc, Square::input_arg_count + Square::output_arg_count> Square::argument_descs;
constexpr std::array<AttributeDesc, 1> Square::attribute_descs;

constexpr std::array<ArgumentDesc, _MklQuantizedConv2D::input_arg_count + _MklQuantizedConv2D::output_arg_count> _MklQuantizedConv2D::argument_descs;
constexpr std::array<AttributeDesc, 9> _MklQuantizedConv2D::attribute_descs;

constexpr std::array<ArgumentDesc, StringToHashBucket::input_arg_count + StringToHashBucket::output_arg_count> StringToHashBucket::argument_descs;
constexpr std::array<AttributeDesc, 1> StringToHashBucket::attribute_descs;

constexpr std::array<ArgumentDesc, Sinh::input_arg_count + Sinh::output_arg_count> Sinh::argument_descs;
constexpr std::array<AttributeDesc, 1> Sinh::attribute_descs;

constexpr std::array<ArgumentDesc, Cosh::input_arg_count + Cosh::output_arg_count> Cosh::argument_descs;
constexpr std::array<AttributeDesc, 1> Cosh::attribute_descs;

constexpr std::array<ArgumentDesc, StatelessRandomUniform::input_arg_count + StatelessRandomUniform::output_arg_count> StatelessRandomUniform::argument_descs;
constexpr std::array<AttributeDesc, 3> StatelessRandomUniform::attribute_descs;

constexpr std::array<ArgumentDesc, TanhGrad::input_arg_count + TanhGrad::output_arg_count> TanhGrad::argument_descs;
constexpr std::array<AttributeDesc, 1> TanhGrad::attribute_descs;

constexpr std::array<ArgumentDesc, Digamma::input_arg_count + Digamma::output_arg_count> Digamma::argument_descs;
constexpr std::array<AttributeDesc, 1> Digamma::attribute_descs;

constexpr std::array<ArgumentDesc, Ndtri::input_arg_count + Ndtri::output_arg_count> Ndtri::argument_descs;
constexpr std::array<AttributeDesc, 1> Ndtri::attribute_descs;

constexpr std::array<ArgumentDesc, Cos::input_arg_count + Cos::output_arg_count> Cos::argument_descs;
constexpr std::array<AttributeDesc, 1> Cos::attribute_descs;

constexpr std::array<ArgumentDesc, Polygamma::input_arg_count + Polygamma::output_arg_count> Polygamma::argument_descs;
constexpr std::array<AttributeDesc, 1> Polygamma::attribute_descs;

constexpr std::array<ArgumentDesc, _ListToArray::input_arg_count + _ListToArray::output_arg_count> _ListToArray::argument_descs;
constexpr std::array<AttributeDesc, 3> _ListToArray::attribute_descs;

constexpr std::array<ArgumentDesc, AddV2::input_arg_count + AddV2::output_arg_count> AddV2::argument_descs;
constexpr std::array<AttributeDesc, 1> AddV2::attribute_descs;

constexpr std::array<ArgumentDesc, XlaSelectAndScatter::input_arg_count + XlaSelectAndScatter::output_arg_count> XlaSelectAndScatter::argument_descs;
constexpr std::array<AttributeDesc, 4> XlaSelectAndScatter::attribute_descs;

constexpr std::array<ArgumentDesc, _MklAdd::input_arg_count + _MklAdd::output_arg_count> _MklAdd::argument_descs;
constexpr std::array<AttributeDesc, 1> _MklAdd::attribute_descs;

constexpr std::array<ArgumentDesc, SparseSoftmax::input_arg_count + SparseSoftmax::output_arg_count> SparseSoftmax::argument_descs;
constexpr std::array<AttributeDesc, 1> SparseSoftmax::attribute_descs;

constexpr std::array<ArgumentDesc, Conv2DBackpropFilter::input_arg_count + Conv2DBackpropFilter::output_arg_count> Conv2DBackpropFilter::argument_descs;
constexpr std::array<AttributeDesc, 7> Conv2DBackpropFilter::attribute_descs;

constexpr std::array<ArgumentDesc, _MklAddV2::input_arg_count + _MklAddV2::output_arg_count> _MklAddV2::argument_descs;
constexpr std::array<AttributeDesc, 1> _MklAddV2::attribute_descs;

constexpr std::array<ArgumentDesc, Sub::input_arg_count + Sub::output_arg_count> Sub::argument_descs;
constexpr std::array<AttributeDesc, 1> Sub::attribute_descs;

constexpr std::array<ArgumentDesc, _MklSub::input_arg_count + _MklSub::output_arg_count> _MklSub::argument_descs;
constexpr std::array<AttributeDesc, 1> _MklSub::attribute_descs;

constexpr std::array<ArgumentDesc, LoadTPUEmbeddingProximalAdagradParameters::input_arg_count + LoadTPUEmbeddingProximalAdagradParameters::output_arg_count> LoadTPUEmbeddingProximalAdagradParameters::argument_descs;
constexpr std::array<AttributeDesc, 5> LoadTPUEmbeddingProximalAdagradParameters::attribute_descs;

constexpr std::array<ArgumentDesc, ParseSequenceExampleV2::input_arg_count + ParseSequenceExampleV2::output_arg_count> ParseSequenceExampleV2::argument_descs;
constexpr std::array<AttributeDesc, 13> ParseSequenceExampleV2::attribute_descs;

constexpr std::array<ArgumentDesc, Mul::input_arg_count + Mul::output_arg_count> Mul::argument_descs;
constexpr std::array<AttributeDesc, 1> Mul::attribute_descs;

constexpr std::array<ArgumentDesc, _MklMul::input_arg_count + _MklMul::output_arg_count> _MklMul::argument_descs;
constexpr std::array<AttributeDesc, 1> _MklMul::attribute_descs;

constexpr std::array<ArgumentDesc, Xlogy::input_arg_count + Xlogy::output_arg_count> Xlogy::argument_descs;
constexpr std::array<AttributeDesc, 1> Xlogy::attribute_descs;

constexpr std::array<ArgumentDesc, Xdivy::input_arg_count + Xdivy::output_arg_count> Xdivy::argument_descs;
constexpr std::array<AttributeDesc, 1> Xdivy::attribute_descs;

constexpr std::array<ArgumentDesc, Maximum::input_arg_count + Maximum::output_arg_count> Maximum::argument_descs;
constexpr std::array<AttributeDesc, 1> Maximum::attribute_descs;

constexpr std::array<ArgumentDesc, Minimum::input_arg_count + Minimum::output_arg_count> Minimum::argument_descs;
constexpr std::array<AttributeDesc, 1> Minimum::attribute_descs;

constexpr std::array<ArgumentDesc, QuantizedMatMulWithBiasAndReluAndRequantize::input_arg_count + QuantizedMatMulWithBiasAndReluAndRequantize::output_arg_count> QuantizedMatMulWithBiasAndReluAndRequantize::argument_descs;
constexpr std::array<AttributeDesc, 7> QuantizedMatMulWithBiasAndReluAndRequantize::attribute_descs;

constexpr std::array<ArgumentDesc, Pow::input_arg_count + Pow::output_arg_count> Pow::argument_descs;
constexpr std::array<AttributeDesc, 1> Pow::attribute_descs;

constexpr std::array<ArgumentDesc, Igamma::input_arg_count + Igamma::output_arg_count> Igamma::argument_descs;
constexpr std::array<AttributeDesc, 1> Igamma::attribute_descs;

constexpr std::array<ArgumentDesc, ApplyGradientDescent::input_arg_count + ApplyGradientDescent::output_arg_count> ApplyGradientDescent::argument_descs;
constexpr std::array<AttributeDesc, 2> ApplyGradientDescent::attribute_descs;

constexpr std::array<ArgumentDesc, Zeta::input_arg_count + Zeta::output_arg_count> Zeta::argument_descs;
constexpr std::array<AttributeDesc, 1> Zeta::attribute_descs;

constexpr std::array<ArgumentDesc, Atan2::input_arg_count + Atan2::output_arg_count> Atan2::argument_descs;
constexpr std::array<AttributeDesc, 1> Atan2::attribute_descs;

constexpr std::array<ArgumentDesc, Less::input_arg_count + Less::output_arg_count> Less::argument_descs;
constexpr std::array<AttributeDesc, 1> Less::attribute_descs;

constexpr std::array<ArgumentDesc, LessEqual::input_arg_count + LessEqual::output_arg_count> LessEqual::argument_descs;
constexpr std::array<AttributeDesc, 1> LessEqual::attribute_descs;

constexpr std::array<ArgumentDesc, _MklQuantizedConv2DPerChannel::input_arg_count + _MklQuantizedConv2DPerChannel::output_arg_count> _MklQuantizedConv2DPerChannel::argument_descs;
constexpr std::array<AttributeDesc, 9> _MklQuantizedConv2DPerChannel::attribute_descs;

constexpr std::array<ArgumentDesc, Greater::input_arg_count + Greater::output_arg_count> Greater::argument_descs;
constexpr std::array<AttributeDesc, 1> Greater::attribute_descs;

constexpr std::array<ArgumentDesc, __MklDummyPadWithFusedConv2D::input_arg_count + __MklDummyPadWithFusedConv2D::output_arg_count> __MklDummyPadWithFusedConv2D::argument_descs;
constexpr std::array<AttributeDesc, 10> __MklDummyPadWithFusedConv2D::attribute_descs;

constexpr std::array<ArgumentDesc, ResourceApplyAdamWithAmsgrad::input_arg_count + ResourceApplyAdamWithAmsgrad::output_arg_count> ResourceApplyAdamWithAmsgrad::argument_descs;
constexpr std::array<AttributeDesc, 2> ResourceApplyAdamWithAmsgrad::attribute_descs;

constexpr std::array<ArgumentDesc, ResourceApplyGradientDescent::input_arg_count + ResourceApplyGradientDescent::output_arg_count> ResourceApplyGradientDescent::argument_descs;
constexpr std::array<AttributeDesc, 2> ResourceApplyGradientDescent::attribute_descs;

constexpr std::array<ArgumentDesc, _MklPadWithConv2D::input_arg_count + _MklPadWithConv2D::output_arg_count> _MklPadWithConv2D::argument_descs;
constexpr std::array<AttributeDesc, 8> _MklPadWithConv2D::attribute_descs;

constexpr std::array<ArgumentDesc, Equal::input_arg_count + Equal::output_arg_count> Equal::argument_descs;
constexpr std::array<AttributeDesc, 2> Equal::attribute_descs;

constexpr std::array<ArgumentDesc, _MklQuantizedMatMulWithBiasAndRelu::input_arg_count + _MklQuantizedMatMulWithBiasAndRelu::output_arg_count> _MklQuantizedMatMulWithBiasAndRelu::argument_descs;
constexpr std::array<AttributeDesc, 7> _MklQuantizedMatMulWithBiasAndRelu::attribute_descs;

constexpr std::array<ArgumentDesc, Variable::input_arg_count + Variable::output_arg_count> Variable::argument_descs;
constexpr std::array<AttributeDesc, 4> Variable::attribute_descs;

constexpr std::array<ArgumentDesc, LogicalNot::input_arg_count + LogicalNot::output_arg_count> LogicalNot::argument_descs;
constexpr std::array<AttributeDesc, 0> LogicalNot::attribute_descs;

constexpr std::array<ArgumentDesc, _MklNativeDepthwiseConv2dNativeBackpropFilter::input_arg_count + _MklNativeDepthwiseConv2dNativeBackpropFilter::output_arg_count> _MklNativeDepthwiseConv2dNativeBackpropFilter::argument_descs;
constexpr std::array<AttributeDesc, 6> _MklNativeDepthwiseConv2dNativeBackpropFilter::attribute_descs;

constexpr std::array<ArgumentDesc, EuclideanNorm::input_arg_count + EuclideanNorm::output_arg_count> EuclideanNorm::argument_descs;
constexpr std::array<AttributeDesc, 3> EuclideanNorm::attribute_descs;

constexpr std::array<ArgumentDesc, LogicalAnd::input_arg_count + LogicalAnd::output_arg_count> LogicalAnd::argument_descs;
constexpr std::array<AttributeDesc, 0> LogicalAnd::attribute_descs;

constexpr std::array<ArgumentDesc, LogicalOr::input_arg_count + LogicalOr::output_arg_count> LogicalOr::argument_descs;
constexpr std::array<AttributeDesc, 0> LogicalOr::attribute_descs;

constexpr std::array<ArgumentDesc, MatMul::input_arg_count + MatMul::output_arg_count> MatMul::argument_descs;
constexpr std::array<AttributeDesc, 3> MatMul::attribute_descs;

constexpr std::array<ArgumentDesc, Max::input_arg_count + Max::output_arg_count> Max::argument_descs;
constexpr std::array<AttributeDesc, 3> Max::attribute_descs;

constexpr std::array<ArgumentDesc, ArgMin::input_arg_count + ArgMin::output_arg_count> ArgMin::argument_descs;
constexpr std::array<AttributeDesc, 3> ArgMin::attribute_descs;

constexpr std::array<ArgumentDesc, LRNGrad::input_arg_count + LRNGrad::output_arg_count> LRNGrad::argument_descs;
constexpr std::array<AttributeDesc, 5> LRNGrad::attribute_descs;

constexpr std::array<ArgumentDesc, SegmentMean::input_arg_count + SegmentMean::output_arg_count> SegmentMean::argument_descs;
constexpr std::array<AttributeDesc, 2> SegmentMean::attribute_descs;

constexpr std::array<ArgumentDesc, CloseSummaryWriter::input_arg_count + CloseSummaryWriter::output_arg_count> CloseSummaryWriter::argument_descs;
constexpr std::array<AttributeDesc, 0> CloseSummaryWriter::attribute_descs;

constexpr std::array<ArgumentDesc, SegmentMin::input_arg_count + SegmentMin::output_arg_count> SegmentMin::argument_descs;
constexpr std::array<AttributeDesc, 2> SegmentMin::attribute_descs;

constexpr std::array<ArgumentDesc, SegmentMax::input_arg_count + SegmentMax::output_arg_count> SegmentMax::argument_descs;
constexpr std::array<AttributeDesc, 2> SegmentMax::attribute_descs;

constexpr std::array<ArgumentDesc, XlaIf::input_arg_count + XlaIf::output_arg_count> XlaIf::argument_descs;
constexpr std::array<AttributeDesc, 5> XlaIf::attribute_descs;

constexpr std::array<ArgumentDesc, UnsortedSegmentSum::input_arg_count + UnsortedSegmentSum::output_arg_count> UnsortedSegmentSum::argument_descs;
constexpr std::array<AttributeDesc, 3> UnsortedSegmentSum::attribute_descs;

constexpr std::array<ArgumentDesc, SparseSegmentSum::input_arg_count + SparseSegmentSum::output_arg_count> SparseSegmentSum::argument_descs;
constexpr std::array<AttributeDesc, 3> SparseSegmentSum::attribute_descs;

constexpr std::array<ArgumentDesc, SparseApplyProximalAdagrad::input_arg_count + SparseApplyProximalAdagrad::output_arg_count> SparseApplyProximalAdagrad::argument_descs;
constexpr std::array<AttributeDesc, 3> SparseApplyProximalAdagrad::attribute_descs;

constexpr std::array<ArgumentDesc, SparseSegmentMeanGrad::input_arg_count + SparseSegmentMeanGrad::output_arg_count> SparseSegmentMeanGrad::argument_descs;
constexpr std::array<AttributeDesc, 3> SparseSegmentMeanGrad::attribute_descs;

constexpr std::array<ArgumentDesc, SparseSegmentSqrtN::input_arg_count + SparseSegmentSqrtN::output_arg_count> SparseSegmentSqrtN::argument_descs;
constexpr std::array<AttributeDesc, 3> SparseSegmentSqrtN::attribute_descs;

constexpr std::array<ArgumentDesc, SparseSegmentSqrtNWithNumSegments::input_arg_count + SparseSegmentSqrtNWithNumSegments::output_arg_count> SparseSegmentSqrtNWithNumSegments::argument_descs;
constexpr std::array<AttributeDesc, 4> SparseSegmentSqrtNWithNumSegments::attribute_descs;

constexpr std::array<ArgumentDesc, RiscConcat::input_arg_count + RiscConcat::output_arg_count> RiscConcat::argument_descs;
constexpr std::array<AttributeDesc, 3> RiscConcat::attribute_descs;

constexpr std::array<ArgumentDesc, Range::input_arg_count + Range::output_arg_count> Range::argument_descs;
constexpr std::array<AttributeDesc, 1> Range::attribute_descs;

constexpr std::array<ArgumentDesc, AssignVariableOp::input_arg_count + AssignVariableOp::output_arg_count> AssignVariableOp::argument_descs;
constexpr std::array<AttributeDesc, 2> AssignVariableOp::attribute_descs;

constexpr std::array<ArgumentDesc, Conj::input_arg_count + Conj::output_arg_count> Conj::argument_descs;
constexpr std::array<AttributeDesc, 1> Conj::attribute_descs;

constexpr std::array<ArgumentDesc, Cross::input_arg_count + Cross::output_arg_count> Cross::argument_descs;
constexpr std::array<AttributeDesc, 1> Cross::attribute_descs;

constexpr std::array<ArgumentDesc, HistogramFixedWidth::input_arg_count + HistogramFixedWidth::output_arg_count> HistogramFixedWidth::argument_descs;
constexpr std::array<AttributeDesc, 2> HistogramFixedWidth::attribute_descs;

constexpr std::array<ArgumentDesc, Bincount::input_arg_count + Bincount::output_arg_count> Bincount::argument_descs;
constexpr std::array<AttributeDesc, 1> Bincount::attribute_descs;

constexpr std::array<ArgumentDesc, SparseBincount::input_arg_count + SparseBincount::output_arg_count> SparseBincount::argument_descs;
constexpr std::array<AttributeDesc, 3> SparseBincount::attribute_descs;

constexpr std::array<ArgumentDesc, RaggedBincount::input_arg_count + RaggedBincount::output_arg_count> RaggedBincount::argument_descs;
constexpr std::array<AttributeDesc, 3> RaggedBincount::attribute_descs;

constexpr std::array<ArgumentDesc, _NcclReduceSend::input_arg_count + _NcclReduceSend::output_arg_count> _NcclReduceSend::argument_descs;
constexpr std::array<AttributeDesc, 4> _NcclReduceSend::attribute_descs;

constexpr std::array<ArgumentDesc, QuantizedMatMul::input_arg_count + QuantizedMatMul::output_arg_count> QuantizedMatMul::argument_descs;
constexpr std::array<AttributeDesc, 6> QuantizedMatMul::attribute_descs;

constexpr std::array<ArgumentDesc, LSTMBlockCell::input_arg_count + LSTMBlockCell::output_arg_count> LSTMBlockCell::argument_descs;
constexpr std::array<AttributeDesc, 4> LSTMBlockCell::attribute_descs;

constexpr std::array<ArgumentDesc, SoftmaxCrossEntropyWithLogits::input_arg_count + SoftmaxCrossEntropyWithLogits::output_arg_count> SoftmaxCrossEntropyWithLogits::argument_descs;
constexpr std::array<AttributeDesc, 1> SoftmaxCrossEntropyWithLogits::attribute_descs;

constexpr std::array<ArgumentDesc, QuantizedMul::input_arg_count + QuantizedMul::output_arg_count> QuantizedMul::argument_descs;
constexpr std::array<AttributeDesc, 3> QuantizedMul::attribute_descs;

constexpr std::array<ArgumentDesc, _MklAddN::input_arg_count + _MklAddN::output_arg_count> _MklAddN::argument_descs;
constexpr std::array<AttributeDesc, 2> _MklAddN::attribute_descs;

constexpr std::array<ArgumentDesc, Dilation2DBackpropInput::input_arg_count + Dilation2DBackpropInput::output_arg_count> Dilation2DBackpropInput::argument_descs;
constexpr std::array<AttributeDesc, 4> Dilation2DBackpropInput::attribute_descs;

constexpr std::array<ArgumentDesc, RequantizePerChannel::input_arg_count + RequantizePerChannel::output_arg_count> RequantizePerChannel::argument_descs;
constexpr std::array<AttributeDesc, 2> RequantizePerChannel::attribute_descs;

constexpr std::array<ArgumentDesc, RequantizationRangePerChannel::input_arg_count + RequantizationRangePerChannel::output_arg_count> RequantizationRangePerChannel::argument_descs;
constexpr std::array<AttributeDesc, 2> RequantizationRangePerChannel::attribute_descs;

constexpr std::array<ArgumentDesc, NcclReduce::input_arg_count + NcclReduce::output_arg_count> NcclReduce::argument_descs;
constexpr std::array<AttributeDesc, 3> NcclReduce::attribute_descs;

constexpr std::array<ArgumentDesc, _NcclReduceRecv::input_arg_count + _NcclReduceRecv::output_arg_count> _NcclReduceRecv::argument_descs;
constexpr std::array<AttributeDesc, 4> _NcclReduceRecv::attribute_descs;

constexpr std::array<ArgumentDesc, NcclBroadcast::input_arg_count + NcclBroadcast::output_arg_count> NcclBroadcast::argument_descs;
constexpr std::array<AttributeDesc, 2> NcclBroadcast::attribute_descs;

constexpr std::array<ArgumentDesc, StatefulStandardNormalV2::input_arg_count + StatefulStandardNormalV2::output_arg_count> StatefulStandardNormalV2::argument_descs;
constexpr std::array<AttributeDesc, 2> StatefulStandardNormalV2::attribute_descs;

constexpr std::array<ArgumentDesc, _NcclBroadcastSend::input_arg_count + _NcclBroadcastSend::output_arg_count> _NcclBroadcastSend::argument_descs;
constexpr std::array<AttributeDesc, 3> _NcclBroadcastSend::attribute_descs;

constexpr std::array<ArgumentDesc, AvgPool::input_arg_count + AvgPool::output_arg_count> AvgPool::argument_descs;
constexpr std::array<AttributeDesc, 5> AvgPool::attribute_descs;

constexpr std::array<ArgumentDesc, BatchNormWithGlobalNormalizationGrad::input_arg_count + BatchNormWithGlobalNormalizationGrad::output_arg_count> BatchNormWithGlobalNormalizationGrad::argument_descs;
constexpr std::array<AttributeDesc, 3> BatchNormWithGlobalNormalizationGrad::attribute_descs;

constexpr std::array<ArgumentDesc, _FusedBatchNormEx::input_arg_count + _FusedBatchNormEx::output_arg_count> _FusedBatchNormEx::argument_descs;
constexpr std::array<AttributeDesc, 8> _FusedBatchNormEx::attribute_descs;

constexpr std::array<ArgumentDesc, FusedBatchNormGrad::input_arg_count + FusedBatchNormGrad::output_arg_count> FusedBatchNormGrad::argument_descs;
constexpr std::array<AttributeDesc, 4> FusedBatchNormGrad::attribute_descs;

constexpr std::array<ArgumentDesc, FusedBatchNormGradV2::input_arg_count + FusedBatchNormGradV2::output_arg_count> FusedBatchNormGradV2::argument_descs;
constexpr std::array<AttributeDesc, 5> FusedBatchNormGradV2::attribute_descs;

constexpr std::array<ArgumentDesc, FusedBatchNormGradV3::input_arg_count + FusedBatchNormGradV3::output_arg_count> FusedBatchNormGradV3::argument_descs;
constexpr std::array<AttributeDesc, 5> FusedBatchNormGradV3::attribute_descs;

constexpr std::array<ArgumentDesc, BiasAdd::input_arg_count + BiasAdd::output_arg_count> BiasAdd::argument_descs;
constexpr std::array<AttributeDesc, 2> BiasAdd::attribute_descs;

constexpr std::array<ArgumentDesc, FresnelCos::input_arg_count + FresnelCos::output_arg_count> FresnelCos::argument_descs;
constexpr std::array<AttributeDesc, 1> FresnelCos::attribute_descs;

constexpr std::array<ArgumentDesc, BiasAddGrad::input_arg_count + BiasAddGrad::output_arg_count> BiasAddGrad::argument_descs;
constexpr std::array<AttributeDesc, 2> BiasAddGrad::attribute_descs;

constexpr std::array<ArgumentDesc, Conv2D::input_arg_count + Conv2D::output_arg_count> Conv2D::argument_descs;
constexpr std::array<AttributeDesc, 7> Conv2D::attribute_descs;

constexpr std::array<ArgumentDesc, _FusedConv2D::input_arg_count + _FusedConv2D::output_arg_count> _FusedConv2D::argument_descs;
constexpr std::array<AttributeDesc, 11> _FusedConv2D::attribute_descs;

constexpr std::array<ArgumentDesc, FusedResizeAndPadConv2D::input_arg_count + FusedResizeAndPadConv2D::output_arg_count> FusedResizeAndPadConv2D::argument_descs;
constexpr std::array<AttributeDesc, 5> FusedResizeAndPadConv2D::attribute_descs;

constexpr std::array<ArgumentDesc, _MklEinsum::input_arg_count + _MklEinsum::output_arg_count> _MklEinsum::argument_descs;
constexpr std::array<AttributeDesc, 3> _MklEinsum::attribute_descs;

constexpr std::array<ArgumentDesc, SparseDenseCwiseMul::input_arg_count + SparseDenseCwiseMul::output_arg_count> SparseDenseCwiseMul::argument_descs;
constexpr std::array<AttributeDesc, 1> SparseDenseCwiseMul::attribute_descs;

constexpr std::array<ArgumentDesc, FusedPadConv2D::input_arg_count + FusedPadConv2D::output_arg_count> FusedPadConv2D::argument_descs;
constexpr std::array<AttributeDesc, 4> FusedPadConv2D::attribute_descs;

constexpr std::array<ArgumentDesc, ApplyAdaMax::input_arg_count + ApplyAdaMax::output_arg_count> ApplyAdaMax::argument_descs;
constexpr std::array<AttributeDesc, 2> ApplyAdaMax::attribute_descs;

constexpr std::array<ArgumentDesc, DepthwiseConv2dNative::input_arg_count + DepthwiseConv2dNative::output_arg_count> DepthwiseConv2dNative::argument_descs;
constexpr std::array<AttributeDesc, 6> DepthwiseConv2dNative::attribute_descs;

constexpr std::array<ArgumentDesc, SparseApplyAdadelta::input_arg_count + SparseApplyAdadelta::output_arg_count> SparseApplyAdadelta::argument_descs;
constexpr std::array<AttributeDesc, 3> SparseApplyAdadelta::attribute_descs;

constexpr std::array<ArgumentDesc, DepthwiseConv2dNativeBackpropFilter::input_arg_count + DepthwiseConv2dNativeBackpropFilter::output_arg_count> DepthwiseConv2dNativeBackpropFilter::argument_descs;
constexpr std::array<AttributeDesc, 6> DepthwiseConv2dNativeBackpropFilter::attribute_descs;

constexpr std::array<ArgumentDesc, Conv3DBackpropFilter::input_arg_count + Conv3DBackpropFilter::output_arg_count> Conv3DBackpropFilter::argument_descs;
constexpr std::array<AttributeDesc, 4> Conv3DBackpropFilter::attribute_descs;

constexpr std::array<ArgumentDesc, MaxPool3D::input_arg_count + MaxPool3D::output_arg_count> MaxPool3D::argument_descs;
constexpr std::array<AttributeDesc, 5> MaxPool3D::attribute_descs;

constexpr std::array<ArgumentDesc, MaxPool3DGrad::input_arg_count + MaxPool3DGrad::output_arg_count> MaxPool3DGrad::argument_descs;
constexpr std::array<AttributeDesc, 6> MaxPool3DGrad::attribute_descs;

constexpr std::array<ArgumentDesc, MaxPool3DGradGrad::input_arg_count + MaxPool3DGradGrad::output_arg_count> MaxPool3DGradGrad::argument_descs;
constexpr std::array<AttributeDesc, 5> MaxPool3DGradGrad::attribute_descs;

constexpr std::array<ArgumentDesc, StatelessIf::input_arg_count + StatelessIf::output_arg_count> StatelessIf::argument_descs;
constexpr std::array<AttributeDesc, 6> StatelessIf::attribute_descs;

constexpr std::array<ArgumentDesc, ScatterDiv::input_arg_count + ScatterDiv::output_arg_count> ScatterDiv::argument_descs;
constexpr std::array<AttributeDesc, 3> ScatterDiv::attribute_descs;

constexpr std::array<ArgumentDesc, L2Loss::input_arg_count + L2Loss::output_arg_count> L2Loss::argument_descs;
constexpr std::array<AttributeDesc, 1> L2Loss::attribute_descs;

constexpr std::array<ArgumentDesc, MaxPoolGrad::input_arg_count + MaxPoolGrad::output_arg_count> MaxPoolGrad::argument_descs;
constexpr std::array<AttributeDesc, 6> MaxPoolGrad::attribute_descs;

constexpr std::array<ArgumentDesc, MaxPoolGradV2::input_arg_count + MaxPoolGradV2::output_arg_count> MaxPoolGradV2::argument_descs;
constexpr std::array<AttributeDesc, 3> MaxPoolGradV2::attribute_descs;

constexpr std::array<ArgumentDesc, MaxPoolGradGradV2::input_arg_count + MaxPoolGradGradV2::output_arg_count> MaxPoolGradGradV2::argument_descs;
constexpr std::array<AttributeDesc, 3> MaxPoolGradGradV2::attribute_descs;

constexpr std::array<ArgumentDesc, _ScopedAllocatorSplit::input_arg_count + _ScopedAllocatorSplit::output_arg_count> _ScopedAllocatorSplit::argument_descs;
constexpr std::array<AttributeDesc, 5> _ScopedAllocatorSplit::attribute_descs;

constexpr std::array<ArgumentDesc, MaxPoolWithArgmax::input_arg_count + MaxPoolWithArgmax::output_arg_count> MaxPoolWithArgmax::argument_descs;
constexpr std::array<AttributeDesc, 6> MaxPoolWithArgmax::attribute_descs;

constexpr std::array<ArgumentDesc, MaxPoolGradGradWithArgmax::input_arg_count + MaxPoolGradGradWithArgmax::output_arg_count> MaxPoolGradGradWithArgmax::argument_descs;
constexpr std::array<AttributeDesc, 6> MaxPoolGradGradWithArgmax::attribute_descs;

constexpr std::array<ArgumentDesc, Dilation2D::input_arg_count + Dilation2D::output_arg_count> Dilation2D::argument_descs;
constexpr std::array<AttributeDesc, 4> Dilation2D::attribute_descs;

constexpr std::array<ArgumentDesc, RiscTranspose::input_arg_count + RiscTranspose::output_arg_count> RiscTranspose::argument_descs;
constexpr std::array<AttributeDesc, 2> RiscTranspose::attribute_descs;

constexpr std::array<ArgumentDesc, _MklMaxPool3D::input_arg_count + _MklMaxPool3D::output_arg_count> _MklMaxPool3D::argument_descs;
constexpr std::array<AttributeDesc, 6> _MklMaxPool3D::attribute_descs;

constexpr std::array<ArgumentDesc, Dilation2DBackpropFilter::input_arg_count + Dilation2DBackpropFilter::output_arg_count> Dilation2DBackpropFilter::argument_descs;
constexpr std::array<AttributeDesc, 4> Dilation2DBackpropFilter::attribute_descs;

constexpr std::array<ArgumentDesc, StatelessCase::input_arg_count + StatelessCase::output_arg_count> StatelessCase::argument_descs;
constexpr std::array<AttributeDesc, 4> StatelessCase::attribute_descs;

constexpr std::array<ArgumentDesc, Relu::input_arg_count + Relu::output_arg_count> Relu::argument_descs;
constexpr std::array<AttributeDesc, 1> Relu::attribute_descs;

constexpr std::array<ArgumentDesc, ReluGrad::input_arg_count + ReluGrad::output_arg_count> ReluGrad::argument_descs;
constexpr std::array<AttributeDesc, 1> ReluGrad::attribute_descs;

constexpr std::array<ArgumentDesc, LeakyRelu::input_arg_count + LeakyRelu::output_arg_count> LeakyRelu::argument_descs;
constexpr std::array<AttributeDesc, 2> LeakyRelu::attribute_descs;

constexpr std::array<ArgumentDesc, LeakyReluGrad::input_arg_count + LeakyReluGrad::output_arg_count> LeakyReluGrad::argument_descs;
constexpr std::array<AttributeDesc, 2> LeakyReluGrad::attribute_descs;

constexpr std::array<ArgumentDesc, Elu::input_arg_count + Elu::output_arg_count> Elu::argument_descs;
constexpr std::array<AttributeDesc, 1> Elu::attribute_descs;

constexpr std::array<ArgumentDesc, ConfigureDistributedTPU::input_arg_count + ConfigureDistributedTPU::output_arg_count> ConfigureDistributedTPU::argument_descs;
constexpr std::array<AttributeDesc, 6> ConfigureDistributedTPU::attribute_descs;

constexpr std::array<ArgumentDesc, EluGrad::input_arg_count + EluGrad::output_arg_count> EluGrad::argument_descs;
constexpr std::array<AttributeDesc, 1> EluGrad::attribute_descs;

constexpr std::array<ArgumentDesc, Selu::input_arg_count + Selu::output_arg_count> Selu::argument_descs;
constexpr std::array<AttributeDesc, 1> Selu::attribute_descs;

constexpr std::array<ArgumentDesc, SeluGrad::input_arg_count + SeluGrad::output_arg_count> SeluGrad::argument_descs;
constexpr std::array<AttributeDesc, 1> SeluGrad::attribute_descs;

constexpr std::array<ArgumentDesc, Softsign::input_arg_count + Softsign::output_arg_count> Softsign::argument_descs;
constexpr std::array<AttributeDesc, 1> Softsign::attribute_descs;

constexpr std::array<ArgumentDesc, SoftsignGrad::input_arg_count + SoftsignGrad::output_arg_count> SoftsignGrad::argument_descs;
constexpr std::array<AttributeDesc, 1> SoftsignGrad::attribute_descs;

constexpr std::array<ArgumentDesc, _MklFusedBatchNormV2::input_arg_count + _MklFusedBatchNormV2::output_arg_count> _MklFusedBatchNormV2::argument_descs;
constexpr std::array<AttributeDesc, 6> _MklFusedBatchNormV2::attribute_descs;

constexpr std::array<ArgumentDesc, LogSoftmax::input_arg_count + LogSoftmax::output_arg_count> LogSoftmax::argument_descs;
constexpr std::array<AttributeDesc, 1> LogSoftmax::attribute_descs;

constexpr std::array<ArgumentDesc, SparseSoftmaxCrossEntropyWithLogits::input_arg_count + SparseSoftmaxCrossEntropyWithLogits::output_arg_count> SparseSoftmaxCrossEntropyWithLogits::argument_descs;
constexpr std::array<AttributeDesc, 2> SparseSoftmaxCrossEntropyWithLogits::attribute_descs;

constexpr std::array<ArgumentDesc, XlaConv::input_arg_count + XlaConv::output_arg_count> XlaConv::argument_descs;
constexpr std::array<AttributeDesc, 4> XlaConv::attribute_descs;

constexpr std::array<ArgumentDesc, InTopKV2::input_arg_count + InTopKV2::output_arg_count> InTopKV2::argument_descs;
constexpr std::array<AttributeDesc, 1> InTopKV2::attribute_descs;

constexpr std::array<ArgumentDesc, TopK::input_arg_count + TopK::output_arg_count> TopK::argument_descs;
constexpr std::array<AttributeDesc, 3> TopK::attribute_descs;

constexpr std::array<ArgumentDesc, FractionalMaxPoolGrad::input_arg_count + FractionalMaxPoolGrad::output_arg_count> FractionalMaxPoolGrad::argument_descs;
constexpr std::array<AttributeDesc, 2> FractionalMaxPoolGrad::attribute_descs;

constexpr std::array<ArgumentDesc, TPUOrdinalSelector::input_arg_count + TPUOrdinalSelector::output_arg_count> TPUOrdinalSelector::argument_descs;
constexpr std::array<AttributeDesc, 0> TPUOrdinalSelector::attribute_descs;

constexpr std::array<ArgumentDesc, BlockLSTMV2::input_arg_count + BlockLSTMV2::output_arg_count> BlockLSTMV2::argument_descs;
constexpr std::array<AttributeDesc, 3> BlockLSTMV2::attribute_descs;

constexpr std::array<ArgumentDesc, FractionalAvgPool::input_arg_count + FractionalAvgPool::output_arg_count> FractionalAvgPool::argument_descs;
constexpr std::array<AttributeDesc, 7> FractionalAvgPool::attribute_descs;

constexpr std::array<ArgumentDesc, PartitionedCall::input_arg_count + PartitionedCall::output_arg_count> PartitionedCall::argument_descs;
constexpr std::array<AttributeDesc, 6> PartitionedCall::attribute_descs;

constexpr std::array<ArgumentDesc, QuantizedAvgPool::input_arg_count + QuantizedAvgPool::output_arg_count> QuantizedAvgPool::argument_descs;
constexpr std::array<AttributeDesc, 4> QuantizedAvgPool::attribute_descs;

constexpr std::array<ArgumentDesc, QuantizedBiasAdd::input_arg_count + QuantizedBiasAdd::output_arg_count> QuantizedBiasAdd::argument_descs;
constexpr std::array<AttributeDesc, 3> QuantizedBiasAdd::attribute_descs;

constexpr std::array<ArgumentDesc, QuantizedConv2D::input_arg_count + QuantizedConv2D::output_arg_count> QuantizedConv2D::argument_descs;
constexpr std::array<AttributeDesc, 6> QuantizedConv2D::attribute_descs;

constexpr std::array<ArgumentDesc, QuantizedRelu::input_arg_count + QuantizedRelu::output_arg_count> QuantizedRelu::argument_descs;
constexpr std::array<AttributeDesc, 2> QuantizedRelu::attribute_descs;

constexpr std::array<ArgumentDesc, SparseTensorToCSRSparseMatrix::input_arg_count + SparseTensorToCSRSparseMatrix::output_arg_count> SparseTensorToCSRSparseMatrix::argument_descs;
constexpr std::array<AttributeDesc, 1> SparseTensorToCSRSparseMatrix::attribute_descs;

constexpr std::array<ArgumentDesc, QuantizedRelu6::input_arg_count + QuantizedRelu6::output_arg_count> QuantizedRelu6::argument_descs;
constexpr std::array<AttributeDesc, 2> QuantizedRelu6::attribute_descs;

constexpr std::array<ArgumentDesc, _MklDepthwiseConv2dNative::input_arg_count + _MklDepthwiseConv2dNative::output_arg_count> _MklDepthwiseConv2dNative::argument_descs;
constexpr std::array<AttributeDesc, 7> _MklDepthwiseConv2dNative::attribute_descs;

constexpr std::array<ArgumentDesc, _MklConv2D::input_arg_count + _MklConv2D::output_arg_count> _MklConv2D::argument_descs;
constexpr std::array<AttributeDesc, 8> _MklConv2D::attribute_descs;

constexpr std::array<ArgumentDesc, _MklNativeConv2D::input_arg_count + _MklNativeConv2D::output_arg_count> _MklNativeConv2D::argument_descs;
constexpr std::array<AttributeDesc, 8> _MklNativeConv2D::attribute_descs;

constexpr std::array<ArgumentDesc, _MklRelu6Grad::input_arg_count + _MklRelu6Grad::output_arg_count> _MklRelu6Grad::argument_descs;
constexpr std::array<AttributeDesc, 1> _MklRelu6Grad::attribute_descs;

constexpr std::array<ArgumentDesc, __MklDummyConv2DWithBias::input_arg_count + __MklDummyConv2DWithBias::output_arg_count> __MklDummyConv2DWithBias::argument_descs;
constexpr std::array<AttributeDesc, 8> __MklDummyConv2DWithBias::attribute_descs;

constexpr std::array<ArgumentDesc, _MklConv2DWithBias::input_arg_count + _MklConv2DWithBias::output_arg_count> _MklConv2DWithBias::argument_descs;
constexpr std::array<AttributeDesc, 8> _MklConv2DWithBias::attribute_descs;

constexpr std::array<ArgumentDesc, __MklDummyPadWithConv2D::input_arg_count + __MklDummyPadWithConv2D::output_arg_count> __MklDummyPadWithConv2D::argument_descs;
constexpr std::array<AttributeDesc, 7> __MklDummyPadWithConv2D::attribute_descs;

constexpr std::array<ArgumentDesc, TPUCompilationResult::input_arg_count + TPUCompilationResult::output_arg_count> TPUCompilationResult::argument_descs;
constexpr std::array<AttributeDesc, 0> TPUCompilationResult::attribute_descs;

constexpr std::array<ArgumentDesc, _MklConv2DBackpropFilter::input_arg_count + _MklConv2DBackpropFilter::output_arg_count> _MklConv2DBackpropFilter::argument_descs;
constexpr std::array<AttributeDesc, 7> _MklConv2DBackpropFilter::attribute_descs;

constexpr std::array<ArgumentDesc, BesselI0::input_arg_count + BesselI0::output_arg_count> BesselI0::argument_descs;
constexpr std::array<AttributeDesc, 1> BesselI0::attribute_descs;

constexpr std::array<ArgumentDesc, __MklDummyConv2DBackpropFilterWithBias::input_arg_count + __MklDummyConv2DBackpropFilterWithBias::output_arg_count> __MklDummyConv2DBackpropFilterWithBias::argument_descs;
constexpr std::array<AttributeDesc, 6> __MklDummyConv2DBackpropFilterWithBias::attribute_descs;

constexpr std::array<ArgumentDesc, _MklConv2DBackpropInput::input_arg_count + _MklConv2DBackpropInput::output_arg_count> _MklConv2DBackpropInput::argument_descs;
constexpr std::array<AttributeDesc, 7> _MklConv2DBackpropInput::attribute_descs;

constexpr std::array<ArgumentDesc, _MklNativeConv2DBackpropInput::input_arg_count + _MklNativeConv2DBackpropInput::output_arg_count> _MklNativeConv2DBackpropInput::argument_descs;
constexpr std::array<AttributeDesc, 7> _MklNativeConv2DBackpropInput::attribute_descs;

constexpr std::array<ArgumentDesc, _MklNativeMaxPoolGrad::input_arg_count + _MklNativeMaxPoolGrad::output_arg_count> _MklNativeMaxPoolGrad::argument_descs;
constexpr std::array<AttributeDesc, 7> _MklNativeMaxPoolGrad::attribute_descs;

constexpr std::array<ArgumentDesc, _MklConv3DBackpropInputV2::input_arg_count + _MklConv3DBackpropInputV2::output_arg_count> _MklConv3DBackpropInputV2::argument_descs;
constexpr std::array<AttributeDesc, 6> _MklConv3DBackpropInputV2::attribute_descs;

constexpr std::array<ArgumentDesc, _MklConv3DBackpropFilterV2::input_arg_count + _MklConv3DBackpropFilterV2::output_arg_count> _MklConv3DBackpropFilterV2::argument_descs;
constexpr std::array<AttributeDesc, 5> _MklConv3DBackpropFilterV2::attribute_descs;

constexpr std::array<ArgumentDesc, RiscLog::input_arg_count + RiscLog::output_arg_count> RiscLog::argument_descs;
constexpr std::array<AttributeDesc, 1> RiscLog::attribute_descs;

constexpr std::array<ArgumentDesc, _MklRelu::input_arg_count + _MklRelu::output_arg_count> _MklRelu::argument_descs;
constexpr std::array<AttributeDesc, 1> _MklRelu::attribute_descs;

constexpr std::array<ArgumentDesc, _MklReluGrad::input_arg_count + _MklReluGrad::output_arg_count> _MklReluGrad::argument_descs;
constexpr std::array<AttributeDesc, 1> _MklReluGrad::attribute_descs;

constexpr std::array<ArgumentDesc, _MklRelu6::input_arg_count + _MklRelu6::output_arg_count> _MklRelu6::argument_descs;
constexpr std::array<AttributeDesc, 1> _MklRelu6::attribute_descs;

constexpr std::array<ArgumentDesc, _MklLeakyRelu::input_arg_count + _MklLeakyRelu::output_arg_count> _MklLeakyRelu::argument_descs;
constexpr std::array<AttributeDesc, 2> _MklLeakyRelu::attribute_descs;

constexpr std::array<ArgumentDesc, TPUEmbeddingActivations::input_arg_count + TPUEmbeddingActivations::output_arg_count> TPUEmbeddingActivations::argument_descs;
constexpr std::array<AttributeDesc, 2> TPUEmbeddingActivations::attribute_descs;

constexpr std::array<ArgumentDesc, _MklElu::input_arg_count + _MklElu::output_arg_count> _MklElu::argument_descs;
constexpr std::array<AttributeDesc, 1> _MklElu::attribute_descs;

constexpr std::array<ArgumentDesc, _MklSoftmax::input_arg_count + _MklSoftmax::output_arg_count> _MklSoftmax::argument_descs;
constexpr std::array<AttributeDesc, 1> _MklSoftmax::attribute_descs;

constexpr std::array<ArgumentDesc, _MklTanh::input_arg_count + _MklTanh::output_arg_count> _MklTanh::argument_descs;
constexpr std::array<AttributeDesc, 1> _MklTanh::attribute_descs;

constexpr std::array<ArgumentDesc, _MklTanhGrad::input_arg_count + _MklTanhGrad::output_arg_count> _MklTanhGrad::argument_descs;
constexpr std::array<AttributeDesc, 1> _MklTanhGrad::attribute_descs;

constexpr std::array<ArgumentDesc, _MklAvgPool::input_arg_count + _MklAvgPool::output_arg_count> _MklAvgPool::argument_descs;
constexpr std::array<AttributeDesc, 5> _MklAvgPool::attribute_descs;

constexpr std::array<ArgumentDesc, _MklQuantizedAvgPool::input_arg_count + _MklQuantizedAvgPool::output_arg_count> _MklQuantizedAvgPool::argument_descs;
constexpr std::array<AttributeDesc, 4> _MklQuantizedAvgPool::attribute_descs;

constexpr std::array<ArgumentDesc, _MklAvgPool3D::input_arg_count + _MklAvgPool3D::output_arg_count> _MklAvgPool3D::argument_descs;
constexpr std::array<AttributeDesc, 5> _MklAvgPool3D::attribute_descs;

constexpr std::array<ArgumentDesc, SparseMatrixTranspose::input_arg_count + SparseMatrixTranspose::output_arg_count> SparseMatrixTranspose::argument_descs;
constexpr std::array<AttributeDesc, 2> SparseMatrixTranspose::attribute_descs;

constexpr std::array<ArgumentDesc, _MklAvgPool3DGrad::input_arg_count + _MklAvgPool3DGrad::output_arg_count> _MklAvgPool3DGrad::argument_descs;
constexpr std::array<AttributeDesc, 5> _MklAvgPool3DGrad::attribute_descs;

constexpr std::array<ArgumentDesc, _MklNativePadWithFusedConv2D::input_arg_count + _MklNativePadWithFusedConv2D::output_arg_count> _MklNativePadWithFusedConv2D::argument_descs;
constexpr std::array<AttributeDesc, 11> _MklNativePadWithFusedConv2D::attribute_descs;

constexpr std::array<ArgumentDesc, Recv::input_arg_count + Recv::output_arg_count> Recv::argument_descs;
constexpr std::array<AttributeDesc, 6> Recv::attribute_descs;

constexpr std::array<ArgumentDesc, _MklMaxPool3DGrad::input_arg_count + _MklMaxPool3DGrad::output_arg_count> _MklMaxPool3DGrad::argument_descs;
constexpr std::array<AttributeDesc, 7> _MklMaxPool3DGrad::attribute_descs;

constexpr std::array<ArgumentDesc, _MklLRNGrad::input_arg_count + _MklLRNGrad::output_arg_count> _MklLRNGrad::argument_descs;
constexpr std::array<AttributeDesc, 6> _MklLRNGrad::attribute_descs;

constexpr std::array<ArgumentDesc, _MklFusedBatchNorm::input_arg_count + _MklFusedBatchNorm::output_arg_count> _MklFusedBatchNorm::argument_descs;
constexpr std::array<AttributeDesc, 5> _MklFusedBatchNorm::attribute_descs;

constexpr std::array<ArgumentDesc, _MklFusedBatchNormGrad::input_arg_count + _MklFusedBatchNormGrad::output_arg_count> _MklFusedBatchNormGrad::argument_descs;
constexpr std::array<AttributeDesc, 4> _MklFusedBatchNormGrad::attribute_descs;

constexpr std::array<ArgumentDesc, _MklFusedBatchNormGradV2::input_arg_count + _MklFusedBatchNormGradV2::output_arg_count> _MklFusedBatchNormGradV2::argument_descs;
constexpr std::array<AttributeDesc, 5> _MklFusedBatchNormGradV2::attribute_descs;

constexpr std::array<ArgumentDesc, _MklToTf::input_arg_count + _MklToTf::output_arg_count> _MklToTf::argument_descs;
constexpr std::array<AttributeDesc, 2> _MklToTf::attribute_descs;

constexpr std::array<ArgumentDesc, _MklInputConversion::input_arg_count + _MklInputConversion::output_arg_count> _MklInputConversion::argument_descs;
constexpr std::array<AttributeDesc, 2> _MklInputConversion::attribute_descs;

constexpr std::array<ArgumentDesc, ResourceScatterMax::input_arg_count + ResourceScatterMax::output_arg_count> ResourceScatterMax::argument_descs;
constexpr std::array<AttributeDesc, 2> ResourceScatterMax::attribute_descs;

constexpr std::array<ArgumentDesc, QuantizedConv2DAndRequantize::input_arg_count + QuantizedConv2DAndRequantize::output_arg_count> QuantizedConv2DAndRequantize::argument_descs;
constexpr std::array<AttributeDesc, 7> QuantizedConv2DAndRequantize::attribute_descs;

constexpr std::array<ArgumentDesc, RiscBroadcast::input_arg_count + RiscBroadcast::output_arg_count> RiscBroadcast::argument_descs;
constexpr std::array<AttributeDesc, 2> RiscBroadcast::attribute_descs;

constexpr std::array<ArgumentDesc, QuantizedConv2DAndRelu::input_arg_count + QuantizedConv2DAndRelu::output_arg_count> QuantizedConv2DAndRelu::argument_descs;
constexpr std::array<AttributeDesc, 7> QuantizedConv2DAndRelu::attribute_descs;

constexpr std::array<ArgumentDesc, QuantizedConv2DWithBiasAndRelu::input_arg_count + QuantizedConv2DWithBiasAndRelu::output_arg_count> QuantizedConv2DWithBiasAndRelu::argument_descs;
constexpr std::array<AttributeDesc, 7> QuantizedConv2DWithBiasAndRelu::attribute_descs;

constexpr std::array<ArgumentDesc, QuantizedConv2DWithBiasAndReluAndRequantize::input_arg_count + QuantizedConv2DWithBiasAndReluAndRequantize::output_arg_count> QuantizedConv2DWithBiasAndReluAndRequantize::argument_descs;
constexpr std::array<AttributeDesc, 8> QuantizedConv2DWithBiasAndReluAndRequantize::attribute_descs;

constexpr std::array<ArgumentDesc, QuantizedConv2DWithBiasSumAndRelu::input_arg_count + QuantizedConv2DWithBiasSumAndRelu::output_arg_count> QuantizedConv2DWithBiasSumAndRelu::argument_descs;
constexpr std::array<AttributeDesc, 7> QuantizedConv2DWithBiasSumAndRelu::attribute_descs;

constexpr std::array<ArgumentDesc, QuantizedConv2DWithBiasSumAndReluAndRequantize::input_arg_count + QuantizedConv2DWithBiasSumAndReluAndRequantize::output_arg_count> QuantizedConv2DWithBiasSumAndReluAndRequantize::argument_descs;
constexpr std::array<AttributeDesc, 9> QuantizedConv2DWithBiasSumAndReluAndRequantize::attribute_descs;

constexpr std::array<ArgumentDesc, _ReadVariablesOp::input_arg_count + _ReadVariablesOp::output_arg_count> _ReadVariablesOp::argument_descs;
constexpr std::array<AttributeDesc, 2> _ReadVariablesOp::attribute_descs;

constexpr std::array<ArgumentDesc, QuantizedConv2DWithBiasSignedSumAndReluAndRequantize::input_arg_count + QuantizedConv2DWithBiasSignedSumAndReluAndRequantize::output_arg_count> QuantizedConv2DWithBiasSignedSumAndReluAndRequantize::argument_descs;
constexpr std::array<AttributeDesc, 9> QuantizedConv2DWithBiasSignedSumAndReluAndRequantize::attribute_descs;

constexpr std::array<ArgumentDesc, QuantizedMatMulWithBias::input_arg_count + QuantizedMatMulWithBias::output_arg_count> QuantizedMatMulWithBias::argument_descs;
constexpr std::array<AttributeDesc, 7> QuantizedMatMulWithBias::attribute_descs;

constexpr std::array<ArgumentDesc, QuantizedMatMulWithBiasAndDequantize::input_arg_count + QuantizedMatMulWithBiasAndDequantize::output_arg_count> QuantizedMatMulWithBiasAndDequantize::argument_descs;
constexpr std::array<AttributeDesc, 7> QuantizedMatMulWithBiasAndDequantize::attribute_descs;

constexpr std::array<ArgumentDesc, EnqueueTPUEmbeddingBatch::input_arg_count + EnqueueTPUEmbeddingBatch::output_arg_count> EnqueueTPUEmbeddingBatch::argument_descs;
constexpr std::array<AttributeDesc, 3> EnqueueTPUEmbeddingBatch::attribute_descs;

constexpr std::array<ArgumentDesc, QuantizedDepthwiseConv2D::input_arg_count + QuantizedDepthwiseConv2D::output_arg_count> QuantizedDepthwiseConv2D::argument_descs;
constexpr std::array<AttributeDesc, 6> QuantizedDepthwiseConv2D::attribute_descs;

constexpr std::array<ArgumentDesc, QuantizedDepthwiseConv2DWithBiasAndReluAndRequantize::input_arg_count + QuantizedDepthwiseConv2DWithBiasAndReluAndRequantize::output_arg_count> QuantizedDepthwiseConv2DWithBiasAndReluAndRequantize::argument_descs;
constexpr std::array<AttributeDesc, 8> QuantizedDepthwiseConv2DWithBiasAndReluAndRequantize::attribute_descs;

constexpr std::array<ArgumentDesc, RiscUnary::input_arg_count + RiscUnary::output_arg_count> RiscUnary::argument_descs;
constexpr std::array<AttributeDesc, 2> RiscUnary::attribute_descs;

constexpr std::array<ArgumentDesc, ResourceScatterMul::input_arg_count + ResourceScatterMul::output_arg_count> ResourceScatterMul::argument_descs;
constexpr std::array<AttributeDesc, 2> ResourceScatterMul::attribute_descs;

constexpr std::array<ArgumentDesc, NoOp::input_arg_count + NoOp::output_arg_count> NoOp::argument_descs;
constexpr std::array<AttributeDesc, 0> NoOp::attribute_descs;

constexpr std::array<ArgumentDesc, DecodeRaw::input_arg_count + DecodeRaw::output_arg_count> DecodeRaw::argument_descs;
constexpr std::array<AttributeDesc, 2> DecodeRaw::attribute_descs;

constexpr std::array<ArgumentDesc, ResourceApplyAdadelta::input_arg_count + ResourceApplyAdadelta::output_arg_count> ResourceApplyAdadelta::argument_descs;
constexpr std::array<AttributeDesc, 2> ResourceApplyAdadelta::attribute_descs;

constexpr std::array<ArgumentDesc, DecodePaddedRaw::input_arg_count + DecodePaddedRaw::output_arg_count> DecodePaddedRaw::argument_descs;
constexpr std::array<AttributeDesc, 2> DecodePaddedRaw::attribute_descs;

constexpr std::array<ArgumentDesc, DecodeCompressed::input_arg_count + DecodeCompressed::output_arg_count> DecodeCompressed::argument_descs;
constexpr std::array<AttributeDesc, 1> DecodeCompressed::attribute_descs;

constexpr std::array<ArgumentDesc, ParseExample::input_arg_count + ParseExample::output_arg_count> ParseExample::argument_descs;
constexpr std::array<AttributeDesc, 5> ParseExample::attribute_descs;

constexpr std::array<ArgumentDesc, ParseSingleExample::input_arg_count + ParseSingleExample::output_arg_count> ParseSingleExample::argument_descs;
constexpr std::array<AttributeDesc, 6> ParseSingleExample::attribute_descs;

constexpr std::array<ArgumentDesc, ParseSequenceExample::input_arg_count + ParseSequenceExample::output_arg_count> ParseSequenceExample::argument_descs;
constexpr std::array<AttributeDesc, 15> ParseSequenceExample::attribute_descs;

constexpr std::array<ArgumentDesc, DecodeJSONExample::input_arg_count + DecodeJSONExample::output_arg_count> DecodeJSONExample::argument_descs;
constexpr std::array<AttributeDesc, 0> DecodeJSONExample::attribute_descs;

constexpr std::array<ArgumentDesc, DecodeCSV::input_arg_count + DecodeCSV::output_arg_count> DecodeCSV::argument_descs;
constexpr std::array<AttributeDesc, 5> DecodeCSV::attribute_descs;

constexpr std::array<ArgumentDesc, StringToNumber::input_arg_count + StringToNumber::output_arg_count> StringToNumber::argument_descs;
constexpr std::array<AttributeDesc, 1> StringToNumber::attribute_descs;

constexpr std::array<ArgumentDesc, RaggedCross::input_arg_count + RaggedCross::output_arg_count> RaggedCross::argument_descs;
constexpr std::array<AttributeDesc, 11> RaggedCross::attribute_descs;

constexpr std::array<ArgumentDesc, RaggedTensorToSparse::input_arg_count + RaggedTensorToSparse::output_arg_count> RaggedTensorToSparse::argument_descs;
constexpr std::array<AttributeDesc, 3> RaggedTensorToSparse::attribute_descs;

constexpr std::array<ArgumentDesc, RaggedTensorFromVariant::input_arg_count + RaggedTensorFromVariant::output_arg_count> RaggedTensorFromVariant::argument_descs;
constexpr std::array<AttributeDesc, 4> RaggedTensorFromVariant::attribute_descs;

constexpr std::array<ArgumentDesc, RaggedTensorToVariantGradient::input_arg_count + RaggedTensorToVariantGradient::output_arg_count> RaggedTensorToVariantGradient::argument_descs;
constexpr std::array<AttributeDesc, 2> RaggedTensorToVariantGradient::attribute_descs;

constexpr std::array<ArgumentDesc, RaggedRange::input_arg_count + RaggedRange::output_arg_count> RaggedRange::argument_descs;
constexpr std::array<AttributeDesc, 2> RaggedRange::attribute_descs;

constexpr std::array<ArgumentDesc, RandomUniform::input_arg_count + RandomUniform::output_arg_count> RandomUniform::argument_descs;
constexpr std::array<AttributeDesc, 4> RandomUniform::attribute_descs;

constexpr std::array<ArgumentDesc, RandomUniformInt::input_arg_count + RandomUniformInt::output_arg_count> RandomUniformInt::argument_descs;
constexpr std::array<AttributeDesc, 4> RandomUniformInt::attribute_descs;

constexpr std::array<ArgumentDesc, RandomStandardNormal::input_arg_count + RandomStandardNormal::output_arg_count> RandomStandardNormal::argument_descs;
constexpr std::array<AttributeDesc, 4> RandomStandardNormal::attribute_descs;

constexpr std::array<ArgumentDesc, TruncatedNormal::input_arg_count + TruncatedNormal::output_arg_count> TruncatedNormal::argument_descs;
constexpr std::array<AttributeDesc, 4> TruncatedNormal::attribute_descs;

constexpr std::array<ArgumentDesc, RandomGammaGrad::input_arg_count + RandomGammaGrad::output_arg_count> RandomGammaGrad::argument_descs;
constexpr std::array<AttributeDesc, 1> RandomGammaGrad::attribute_descs;

constexpr std::array<ArgumentDesc, VarHandleOp::input_arg_count + VarHandleOp::output_arg_count> VarHandleOp::argument_descs;
constexpr std::array<AttributeDesc, 5> VarHandleOp::attribute_descs;

constexpr std::array<ArgumentDesc, ReadVariableOp::input_arg_count + ReadVariableOp::output_arg_count> ReadVariableOp::argument_descs;
constexpr std::array<AttributeDesc, 1> ReadVariableOp::attribute_descs;

constexpr std::array<ArgumentDesc, DestroyResourceOp::input_arg_count + DestroyResourceOp::output_arg_count> DestroyResourceOp::argument_descs;
constexpr std::array<AttributeDesc, 1> DestroyResourceOp::attribute_descs;

constexpr std::array<ArgumentDesc, AssignAddVariableOp::input_arg_count + AssignAddVariableOp::output_arg_count> AssignAddVariableOp::argument_descs;
constexpr std::array<AttributeDesc, 1> AssignAddVariableOp::attribute_descs;

constexpr std::array<ArgumentDesc, ResourceGather::input_arg_count + ResourceGather::output_arg_count> ResourceGather::argument_descs;
constexpr std::array<AttributeDesc, 4> ResourceGather::attribute_descs;

constexpr std::array<ArgumentDesc, StringSplit::input_arg_count + StringSplit::output_arg_count> StringSplit::argument_descs;
constexpr std::array<AttributeDesc, 1> StringSplit::attribute_descs;

constexpr std::array<ArgumentDesc, ResourceGatherNd::input_arg_count + ResourceGatherNd::output_arg_count> ResourceGatherNd::argument_descs;
constexpr std::array<AttributeDesc, 2> ResourceGatherNd::attribute_descs;

constexpr std::array<ArgumentDesc, ResourceScatterSub::input_arg_count + ResourceScatterSub::output_arg_count> ResourceScatterSub::argument_descs;
constexpr std::array<AttributeDesc, 2> ResourceScatterSub::attribute_descs;

constexpr std::array<ArgumentDesc, ResourceScatterMin::input_arg_count + ResourceScatterMin::output_arg_count> ResourceScatterMin::argument_descs;
constexpr std::array<AttributeDesc, 2> ResourceScatterMin::attribute_descs;

constexpr std::array<ArgumentDesc, ResourceScatterUpdate::input_arg_count + ResourceScatterUpdate::output_arg_count> ResourceScatterUpdate::argument_descs;
constexpr std::array<AttributeDesc, 2> ResourceScatterUpdate::attribute_descs;

constexpr std::array<ArgumentDesc, MakeUnique::input_arg_count + MakeUnique::output_arg_count> MakeUnique::argument_descs;
constexpr std::array<AttributeDesc, 0> MakeUnique::attribute_descs;

constexpr std::array<ArgumentDesc, MutexV2::input_arg_count + MutexV2::output_arg_count> MutexV2::argument_descs;
constexpr std::array<AttributeDesc, 2> MutexV2::attribute_descs;

constexpr std::array<ArgumentDesc, ConsumeMutexLock::input_arg_count + ConsumeMutexLock::output_arg_count> ConsumeMutexLock::argument_descs;
constexpr std::array<AttributeDesc, 0> ConsumeMutexLock::attribute_descs;

constexpr std::array<ArgumentDesc, _MklNativeFusedConv3D::input_arg_count + _MklNativeFusedConv3D::output_arg_count> _MklNativeFusedConv3D::argument_descs;
constexpr std::array<AttributeDesc, 11> _MklNativeFusedConv3D::attribute_descs;

constexpr std::array<ArgumentDesc, RiscAdd::input_arg_count + RiscAdd::output_arg_count> RiscAdd::argument_descs;
constexpr std::array<AttributeDesc, 1> RiscAdd::attribute_descs;

constexpr std::array<ArgumentDesc, RiscBinaryComparison::input_arg_count + RiscBinaryComparison::output_arg_count> RiscBinaryComparison::argument_descs;
constexpr std::array<AttributeDesc, 2> RiscBinaryComparison::attribute_descs;

constexpr std::array<ArgumentDesc, RiscCast::input_arg_count + RiscCast::output_arg_count> RiscCast::argument_descs;
constexpr std::array<AttributeDesc, 2> RiscCast::attribute_descs;

constexpr std::array<ArgumentDesc, RiscConv::input_arg_count + RiscConv::output_arg_count> RiscConv::argument_descs;
constexpr std::array<AttributeDesc, 4> RiscConv::attribute_descs;

constexpr std::array<ArgumentDesc, RiscCos::input_arg_count + RiscCos::output_arg_count> RiscCos::argument_descs;
constexpr std::array<AttributeDesc, 1> RiscCos::attribute_descs;

constexpr std::array<ArgumentDesc, _MklNativeDepthwiseConv2dNative::input_arg_count + _MklNativeDepthwiseConv2dNative::output_arg_count> _MklNativeDepthwiseConv2dNative::argument_descs;
constexpr std::array<AttributeDesc, 7> _MklNativeDepthwiseConv2dNative::attribute_descs;

constexpr std::array<ArgumentDesc, RiscDiv::input_arg_count + RiscDiv::output_arg_count> RiscDiv::argument_descs;
constexpr std::array<AttributeDesc, 1> RiscDiv::attribute_descs;

constexpr std::array<ArgumentDesc, RiscDot::input_arg_count + RiscDot::output_arg_count> RiscDot::argument_descs;
constexpr std::array<AttributeDesc, 3> RiscDot::attribute_descs;

constexpr std::array<ArgumentDesc, RiscFloor::input_arg_count + RiscFloor::output_arg_count> RiscFloor::argument_descs;
constexpr std::array<AttributeDesc, 1> RiscFloor::attribute_descs;

constexpr std::array<ArgumentDesc, RiscImag::input_arg_count + RiscImag::output_arg_count> RiscImag::argument_descs;
constexpr std::array<AttributeDesc, 2> RiscImag::attribute_descs;

constexpr std::array<ArgumentDesc, _ConfigureTPUEmbeddingMemory::input_arg_count + _ConfigureTPUEmbeddingMemory::output_arg_count> _ConfigureTPUEmbeddingMemory::argument_descs;
constexpr std::array<AttributeDesc, 1> _ConfigureTPUEmbeddingMemory::attribute_descs;

constexpr std::array<ArgumentDesc, RiscIsFinite::input_arg_count + RiscIsFinite::output_arg_count> RiscIsFinite::argument_descs;
constexpr std::array<AttributeDesc, 1> RiscIsFinite::attribute_descs;

constexpr std::array<ArgumentDesc, RiscLogicalAnd::input_arg_count + RiscLogicalAnd::output_arg_count> RiscLogicalAnd::argument_descs;
constexpr std::array<AttributeDesc, 0> RiscLogicalAnd::attribute_descs;

constexpr std::array<ArgumentDesc, RiscLogicalOr::input_arg_count + RiscLogicalOr::output_arg_count> RiscLogicalOr::argument_descs;
constexpr std::array<AttributeDesc, 0> RiscLogicalOr::attribute_descs;

constexpr std::array<ArgumentDesc, FFT3D::input_arg_count + FFT3D::output_arg_count> FFT3D::argument_descs;
constexpr std::array<AttributeDesc, 1> FFT3D::attribute_descs;

constexpr std::array<ArgumentDesc, RiscMax::input_arg_count + RiscMax::output_arg_count> RiscMax::argument_descs;
constexpr std::array<AttributeDesc, 1> RiscMax::attribute_descs;

constexpr std::array<ArgumentDesc, RiscMul::input_arg_count + RiscMul::output_arg_count> RiscMul::argument_descs;
constexpr std::array<AttributeDesc, 1> RiscMul::attribute_descs;

constexpr std::array<ArgumentDesc, RiscPad::input_arg_count + RiscPad::output_arg_count> RiscPad::argument_descs;
constexpr std::array<AttributeDesc, 2> RiscPad::attribute_descs;

constexpr std::array<ArgumentDesc, SendTPUEmbeddingGradients::input_arg_count + SendTPUEmbeddingGradients::output_arg_count> SendTPUEmbeddingGradients::argument_descs;
constexpr std::array<AttributeDesc, 3> SendTPUEmbeddingGradients::attribute_descs;

constexpr std::array<ArgumentDesc, RiscPool::input_arg_count + RiscPool::output_arg_count> RiscPool::argument_descs;
constexpr std::array<AttributeDesc, 5> RiscPool::attribute_descs;

constexpr std::array<ArgumentDesc, TemporaryVariable::input_arg_count + TemporaryVariable::output_arg_count> TemporaryVariable::argument_descs;
constexpr std::array<AttributeDesc, 3> TemporaryVariable::attribute_descs;

constexpr std::array<ArgumentDesc, RiscPow::input_arg_count + RiscPow::output_arg_count> RiscPow::argument_descs;
constexpr std::array<AttributeDesc, 1> RiscPow::attribute_descs;

constexpr std::array<ArgumentDesc, RiscRandomUniform::input_arg_count + RiscRandomUniform::output_arg_count> RiscRandomUniform::argument_descs;
constexpr std::array<AttributeDesc, 2> RiscRandomUniform::attribute_descs;

constexpr std::array<ArgumentDesc, RiscReshape::input_arg_count + RiscReshape::output_arg_count> RiscReshape::argument_descs;
constexpr std::array<AttributeDesc, 2> RiscReshape::attribute_descs;

constexpr std::array<ArgumentDesc, RiscShape::input_arg_count + RiscShape::output_arg_count> RiscShape::argument_descs;
constexpr std::array<AttributeDesc, 2> RiscShape::attribute_descs;

constexpr std::array<ArgumentDesc, WriteRawProtoSummary::input_arg_count + WriteRawProtoSummary::output_arg_count> WriteRawProtoSummary::argument_descs;
constexpr std::array<AttributeDesc, 0> WriteRawProtoSummary::attribute_descs;

constexpr std::array<ArgumentDesc, RiscSub::input_arg_count + RiscSub::output_arg_count> RiscSub::argument_descs;
constexpr std::array<AttributeDesc, 1> RiscSub::attribute_descs;

constexpr std::array<ArgumentDesc, RiscTriangularSolve::input_arg_count + RiscTriangularSolve::output_arg_count> RiscTriangularSolve::argument_descs;
constexpr std::array<AttributeDesc, 3> RiscTriangularSolve::attribute_descs;

constexpr std::array<ArgumentDesc, RiscWhile::input_arg_count + RiscWhile::output_arg_count> RiscWhile::argument_descs;
constexpr std::array<AttributeDesc, 5> RiscWhile::attribute_descs;

constexpr std::array<ArgumentDesc, GRUBlockCellGrad::input_arg_count + GRUBlockCellGrad::output_arg_count> GRUBlockCellGrad::argument_descs;
constexpr std::array<AttributeDesc, 1> GRUBlockCellGrad::attribute_descs;

constexpr std::array<ArgumentDesc, StringNGrams::input_arg_count + StringNGrams::output_arg_count> StringNGrams::argument_descs;
constexpr std::array<AttributeDesc, 7> StringNGrams::attribute_descs;

constexpr std::array<ArgumentDesc, Dawsn::input_arg_count + Dawsn::output_arg_count> Dawsn::argument_descs;
constexpr std::array<AttributeDesc, 1> Dawsn::attribute_descs;

constexpr std::array<ArgumentDesc, FresnelSin::input_arg_count + FresnelSin::output_arg_count> FresnelSin::argument_descs;
constexpr std::array<AttributeDesc, 1> FresnelSin::attribute_descs;

constexpr std::array<ArgumentDesc, Spence::input_arg_count + Spence::output_arg_count> Spence::argument_descs;
constexpr std::array<AttributeDesc, 1> Spence::attribute_descs;

constexpr std::array<ArgumentDesc, BesselI1::input_arg_count + BesselI1::output_arg_count> BesselI1::argument_descs;
constexpr std::array<AttributeDesc, 1> BesselI1::attribute_descs;

constexpr std::array<ArgumentDesc, BesselI0e::input_arg_count + BesselI0e::output_arg_count> BesselI0e::argument_descs;
constexpr std::array<AttributeDesc, 1> BesselI0e::attribute_descs;

constexpr std::array<ArgumentDesc, BesselK0::input_arg_count + BesselK0::output_arg_count> BesselK0::argument_descs;
constexpr std::array<AttributeDesc, 1> BesselK0::attribute_descs;

constexpr std::array<ArgumentDesc, BesselK0e::input_arg_count + BesselK0e::output_arg_count> BesselK0e::argument_descs;
constexpr std::array<AttributeDesc, 1> BesselK0e::attribute_descs;

constexpr std::array<ArgumentDesc, BatchFFT3D::input_arg_count + BatchFFT3D::output_arg_count> BatchFFT3D::argument_descs;
constexpr std::array<AttributeDesc, 0> BatchFFT3D::attribute_descs;

constexpr std::array<ArgumentDesc, BesselJ1::input_arg_count + BesselJ1::output_arg_count> BesselJ1::argument_descs;
constexpr std::array<AttributeDesc, 1> BesselJ1::attribute_descs;

constexpr std::array<ArgumentDesc, BesselY1::input_arg_count + BesselY1::output_arg_count> BesselY1::argument_descs;
constexpr std::array<AttributeDesc, 1> BesselY1::attribute_descs;

constexpr std::array<ArgumentDesc, StatefulTruncatedNormal::input_arg_count + StatefulTruncatedNormal::output_arg_count> StatefulTruncatedNormal::argument_descs;
constexpr std::array<AttributeDesc, 2> StatefulTruncatedNormal::attribute_descs;

constexpr std::array<ArgumentDesc, XlaSelfAdjointEig::input_arg_count + XlaSelfAdjointEig::output_arg_count> XlaSelfAdjointEig::argument_descs;
constexpr std::array<AttributeDesc, 4> XlaSelfAdjointEig::attribute_descs;

constexpr std::array<ArgumentDesc, StatefulUniformInt::input_arg_count + StatefulUniformInt::output_arg_count> StatefulUniformInt::argument_descs;
constexpr std::array<AttributeDesc, 2> StatefulUniformInt::attribute_descs;

constexpr std::array<ArgumentDesc, _MklQuantizedConv2DWithBiasAndRelu::input_arg_count + _MklQuantizedConv2DWithBiasAndRelu::output_arg_count> _MklQuantizedConv2DWithBiasAndRelu::argument_descs;
constexpr std::array<AttributeDesc, 10> _MklQuantizedConv2DWithBiasAndRelu::attribute_descs;

constexpr std::array<ArgumentDesc, RngSkip::input_arg_count + RngSkip::output_arg_count> RngSkip::argument_descs;
constexpr std::array<AttributeDesc, 0> RngSkip::attribute_descs;

constexpr std::array<ArgumentDesc, RngReadAndSkip::input_arg_count + RngReadAndSkip::output_arg_count> RngReadAndSkip::argument_descs;
constexpr std::array<AttributeDesc, 0> RngReadAndSkip::attribute_descs;

constexpr std::array<ArgumentDesc, NonDeterministicInts::input_arg_count + NonDeterministicInts::output_arg_count> NonDeterministicInts::argument_descs;
constexpr std::array<AttributeDesc, 2> NonDeterministicInts::attribute_descs;

constexpr std::array<ArgumentDesc, ApplyAddSign::input_arg_count + ApplyAddSign::output_arg_count> ApplyAddSign::argument_descs;
constexpr std::array<AttributeDesc, 2> ApplyAddSign::attribute_descs;

constexpr std::array<ArgumentDesc, StatefulRandomBinomial::input_arg_count + StatefulRandomBinomial::output_arg_count> StatefulRandomBinomial::argument_descs;
constexpr std::array<AttributeDesc, 3> StatefulRandomBinomial::attribute_descs;

constexpr std::array<ArgumentDesc, _ScopedAllocator::input_arg_count + _ScopedAllocator::output_arg_count> _ScopedAllocator::argument_descs;
constexpr std::array<AttributeDesc, 6> _ScopedAllocator::attribute_descs;

constexpr std::array<ArgumentDesc, _ScopedAllocatorConcat::input_arg_count + _ScopedAllocatorConcat::output_arg_count> _ScopedAllocatorConcat::argument_descs;
constexpr std::array<AttributeDesc, 6> _ScopedAllocatorConcat::attribute_descs;

constexpr std::array<ArgumentDesc, _ShutdownDistributedTPU::input_arg_count + _ShutdownDistributedTPU::output_arg_count> _ShutdownDistributedTPU::argument_descs;
constexpr std::array<AttributeDesc, 0> _ShutdownDistributedTPU::attribute_descs;

constexpr std::array<ArgumentDesc, PyFunc::input_arg_count + PyFunc::output_arg_count> PyFunc::argument_descs;
constexpr std::array<AttributeDesc, 3> PyFunc::attribute_descs;

constexpr std::array<ArgumentDesc, PyFuncStateless::input_arg_count + PyFuncStateless::output_arg_count> PyFuncStateless::argument_descs;
constexpr std::array<AttributeDesc, 3> PyFuncStateless::attribute_descs;

constexpr std::array<ArgumentDesc, EagerPyFunc::input_arg_count + EagerPyFunc::output_arg_count> EagerPyFunc::argument_descs;
constexpr std::array<AttributeDesc, 4> EagerPyFunc::attribute_descs;

constexpr std::array<ArgumentDesc, SdcaOptimizer::input_arg_count + SdcaOptimizer::output_arg_count> SdcaOptimizer::argument_descs;
constexpr std::array<AttributeDesc, 9> SdcaOptimizer::attribute_descs;

constexpr std::array<ArgumentDesc, SdcaOptimizerV2::input_arg_count + SdcaOptimizerV2::output_arg_count> SdcaOptimizerV2::argument_descs;
constexpr std::array<AttributeDesc, 9> SdcaOptimizerV2::attribute_descs;

constexpr std::array<ArgumentDesc, SdcaShrinkL1::input_arg_count + SdcaShrinkL1::output_arg_count> SdcaShrinkL1::argument_descs;
constexpr std::array<AttributeDesc, 3> SdcaShrinkL1::attribute_descs;

constexpr std::array<ArgumentDesc, SdcaFprint::input_arg_count + SdcaFprint::output_arg_count> SdcaFprint::argument_descs;
constexpr std::array<AttributeDesc, 0> SdcaFprint::attribute_descs;

constexpr std::array<ArgumentDesc, _Send::input_arg_count + _Send::output_arg_count> _Send::argument_descs;
constexpr std::array<AttributeDesc, 6> _Send::attribute_descs;

constexpr std::array<ArgumentDesc, _HostSend::input_arg_count + _HostSend::output_arg_count> _HostSend::argument_descs;
constexpr std::array<AttributeDesc, 6> _HostSend::attribute_descs;

constexpr std::array<ArgumentDesc, _HostRecv::input_arg_count + _HostRecv::output_arg_count> _HostRecv::argument_descs;
constexpr std::array<AttributeDesc, 6> _HostRecv::attribute_descs;

constexpr std::array<ArgumentDesc, DenseToSparseSetOperation::input_arg_count + DenseToSparseSetOperation::output_arg_count> DenseToSparseSetOperation::argument_descs;
constexpr std::array<AttributeDesc, 3> DenseToSparseSetOperation::attribute_descs;

constexpr std::array<ArgumentDesc, BatchIFFT::input_arg_count + BatchIFFT::output_arg_count> BatchIFFT::argument_descs;
constexpr std::array<AttributeDesc, 0> BatchIFFT::attribute_descs;

constexpr std::array<ArgumentDesc, CSRSparseMatrixToSparseTensor::input_arg_count + CSRSparseMatrixToSparseTensor::output_arg_count> CSRSparseMatrixToSparseTensor::argument_descs;
constexpr std::array<AttributeDesc, 1> CSRSparseMatrixToSparseTensor::attribute_descs;

constexpr std::array<ArgumentDesc, DenseToCSRSparseMatrix::input_arg_count + DenseToCSRSparseMatrix::output_arg_count> DenseToCSRSparseMatrix::argument_descs;
constexpr std::array<AttributeDesc, 1> DenseToCSRSparseMatrix::attribute_descs;

constexpr std::array<ArgumentDesc, CSRSparseMatrixToDense::input_arg_count + CSRSparseMatrixToDense::output_arg_count> CSRSparseMatrixToDense::argument_descs;
constexpr std::array<AttributeDesc, 1> CSRSparseMatrixToDense::attribute_descs;

constexpr std::array<ArgumentDesc, CSRSparseMatrixComponents::input_arg_count + CSRSparseMatrixComponents::output_arg_count> CSRSparseMatrixComponents::argument_descs;
constexpr std::array<AttributeDesc, 1> CSRSparseMatrixComponents::attribute_descs;

constexpr std::array<ArgumentDesc, SparseMatrixNNZ::input_arg_count + SparseMatrixNNZ::output_arg_count> SparseMatrixNNZ::argument_descs;
constexpr std::array<AttributeDesc, 0> SparseMatrixNNZ::attribute_descs;

constexpr std::array<ArgumentDesc, SparseMatrixMatMul::input_arg_count + SparseMatrixMatMul::output_arg_count> SparseMatrixMatMul::argument_descs;
constexpr std::array<AttributeDesc, 7> SparseMatrixMatMul::attribute_descs;

constexpr std::array<ArgumentDesc, SparseMatrixMul::input_arg_count + SparseMatrixMul::output_arg_count> SparseMatrixMul::argument_descs;
constexpr std::array<AttributeDesc, 1> SparseMatrixMul::attribute_descs;

constexpr std::array<ArgumentDesc, SparseMatrixAdd::input_arg_count + SparseMatrixAdd::output_arg_count> SparseMatrixAdd::argument_descs;
constexpr std::array<AttributeDesc, 1> SparseMatrixAdd::attribute_descs;

constexpr std::array<ArgumentDesc, StatelessRandomNormalV2::input_arg_count + StatelessRandomNormalV2::output_arg_count> StatelessRandomNormalV2::argument_descs;
constexpr std::array<AttributeDesc, 2> StatelessRandomNormalV2::attribute_descs;

constexpr std::array<ArgumentDesc, SparseMatrixZeros::input_arg_count + SparseMatrixZeros::output_arg_count> SparseMatrixZeros::argument_descs;
constexpr std::array<AttributeDesc, 1> SparseMatrixZeros::attribute_descs;

constexpr std::array<ArgumentDesc, SparseMatrixSoftmax::input_arg_count + SparseMatrixSoftmax::output_arg_count> SparseMatrixSoftmax::argument_descs;
constexpr std::array<AttributeDesc, 1> SparseMatrixSoftmax::attribute_descs;

constexpr std::array<ArgumentDesc, SparseMatrixOrderingAMD::input_arg_count + SparseMatrixOrderingAMD::output_arg_count> SparseMatrixOrderingAMD::argument_descs;
constexpr std::array<AttributeDesc, 0> SparseMatrixOrderingAMD::attribute_descs;

constexpr std::array<ArgumentDesc, SparseAddGrad::input_arg_count + SparseAddGrad::output_arg_count> SparseAddGrad::argument_descs;
constexpr std::array<AttributeDesc, 1> SparseAddGrad::attribute_descs;

constexpr std::array<ArgumentDesc, XlaDequantize::input_arg_count + XlaDequantize::output_arg_count> XlaDequantize::argument_descs;
constexpr std::array<AttributeDesc, 4> XlaDequantize::attribute_descs;

constexpr std::array<ArgumentDesc, SparseAdd::input_arg_count + SparseAdd::output_arg_count> SparseAdd::argument_descs;
constexpr std::array<AttributeDesc, 2> SparseAdd::attribute_descs;

constexpr std::array<ArgumentDesc, SparseTensorDenseMatMul::input_arg_count + SparseTensorDenseMatMul::output_arg_count> SparseTensorDenseMatMul::argument_descs;
constexpr std::array<AttributeDesc, 4> SparseTensorDenseMatMul::attribute_descs;

constexpr std::array<ArgumentDesc, SerializeManySparse::input_arg_count + SerializeManySparse::output_arg_count> SerializeManySparse::argument_descs;
constexpr std::array<AttributeDesc, 2> SerializeManySparse::attribute_descs;

constexpr std::array<ArgumentDesc, DeserializeManySparse::input_arg_count + DeserializeManySparse::output_arg_count> DeserializeManySparse::argument_descs;
constexpr std::array<AttributeDesc, 1> DeserializeManySparse::attribute_descs;

constexpr std::array<ArgumentDesc, SparseToDense::input_arg_count + SparseToDense::output_arg_count> SparseToDense::argument_descs;
constexpr std::array<AttributeDesc, 3> SparseToDense::attribute_descs;

constexpr std::array<ArgumentDesc, SparseConcat::input_arg_count + SparseConcat::output_arg_count> SparseConcat::argument_descs;
constexpr std::array<AttributeDesc, 3> SparseConcat::attribute_descs;

constexpr std::array<ArgumentDesc, SparseCrossHashed::input_arg_count + SparseCrossHashed::output_arg_count> SparseCrossHashed::argument_descs;
constexpr std::array<AttributeDesc, 3> SparseCrossHashed::attribute_descs;

constexpr std::array<ArgumentDesc, OutfeedEnqueueTuple::input_arg_count + OutfeedEnqueueTuple::output_arg_count> OutfeedEnqueueTuple::argument_descs;
constexpr std::array<AttributeDesc, 1> OutfeedEnqueueTuple::attribute_descs;

constexpr std::array<ArgumentDesc, SparseSplit::input_arg_count + SparseSplit::output_arg_count> SparseSplit::argument_descs;
constexpr std::array<AttributeDesc, 2> SparseSplit::attribute_descs;

constexpr std::array<ArgumentDesc, SparseSlice::input_arg_count + SparseSlice::output_arg_count> SparseSlice::argument_descs;
constexpr std::array<AttributeDesc, 1> SparseSlice::attribute_descs;

constexpr std::array<ArgumentDesc, SparseReorder::input_arg_count + SparseReorder::output_arg_count> SparseReorder::argument_descs;
constexpr std::array<AttributeDesc, 1> SparseReorder::attribute_descs;

constexpr std::array<ArgumentDesc, SparseTensorDenseAdd::input_arg_count + SparseTensorDenseAdd::output_arg_count> SparseTensorDenseAdd::argument_descs;
constexpr std::array<AttributeDesc, 2> SparseTensorDenseAdd::attribute_descs;

constexpr std::array<ArgumentDesc, SparseReduceMaxSparse::input_arg_count + SparseReduceMaxSparse::output_arg_count> SparseReduceMaxSparse::argument_descs;
constexpr std::array<AttributeDesc, 2> SparseReduceMaxSparse::attribute_descs;

constexpr std::array<ArgumentDesc, SparseReduceSum::input_arg_count + SparseReduceSum::output_arg_count> SparseReduceSum::argument_descs;
constexpr std::array<AttributeDesc, 2> SparseReduceSum::attribute_descs;

constexpr std::array<ArgumentDesc, SparseDenseCwiseDiv::input_arg_count + SparseDenseCwiseDiv::output_arg_count> SparseDenseCwiseDiv::argument_descs;
constexpr std::array<AttributeDesc, 1> SparseDenseCwiseDiv::attribute_descs;

constexpr std::array<ArgumentDesc, XlaReplicaId::input_arg_count + XlaReplicaId::output_arg_count> XlaReplicaId::argument_descs;
constexpr std::array<AttributeDesc, 0> XlaReplicaId::attribute_descs;

constexpr std::array<ArgumentDesc, SparseSparseMaximum::input_arg_count + SparseSparseMaximum::output_arg_count> SparseSparseMaximum::argument_descs;
constexpr std::array<AttributeDesc, 1> SparseSparseMaximum::attribute_descs;

constexpr std::array<ArgumentDesc, SparseSparseMinimum::input_arg_count + SparseSparseMinimum::output_arg_count> SparseSparseMinimum::argument_descs;
constexpr std::array<AttributeDesc, 1> SparseSparseMinimum::attribute_descs;

constexpr std::array<ArgumentDesc, AddManySparseToTensorsMap::input_arg_count + AddManySparseToTensorsMap::output_arg_count> AddManySparseToTensorsMap::argument_descs;
constexpr std::array<AttributeDesc, 3> AddManySparseToTensorsMap::attribute_descs;

constexpr std::array<ArgumentDesc, TakeManySparseFromTensorsMap::input_arg_count + TakeManySparseFromTensorsMap::output_arg_count> TakeManySparseFromTensorsMap::argument_descs;
constexpr std::array<AttributeDesc, 3> TakeManySparseFromTensorsMap::attribute_descs;

constexpr std::array<ArgumentDesc, SparseFillEmptyRows::input_arg_count + SparseFillEmptyRows::output_arg_count> SparseFillEmptyRows::argument_descs;
constexpr std::array<AttributeDesc, 1> SparseFillEmptyRows::attribute_descs;

constexpr std::array<ArgumentDesc, SummaryWriter::input_arg_count + SummaryWriter::output_arg_count> SummaryWriter::argument_descs;
constexpr std::array<AttributeDesc, 2> SummaryWriter::attribute_descs;

constexpr std::array<ArgumentDesc, FlushSummaryWriter::input_arg_count + FlushSummaryWriter::output_arg_count> FlushSummaryWriter::argument_descs;
constexpr std::array<AttributeDesc, 0> FlushSummaryWriter::attribute_descs;

constexpr std::array<ArgumentDesc, StatelessTruncatedNormalV2::input_arg_count + StatelessTruncatedNormalV2::output_arg_count> StatelessTruncatedNormalV2::argument_descs;
constexpr std::array<AttributeDesc, 2> StatelessTruncatedNormalV2::attribute_descs;

constexpr std::array<ArgumentDesc, WriteSummary::input_arg_count + WriteSummary::output_arg_count> WriteSummary::argument_descs;
constexpr std::array<AttributeDesc, 1> WriteSummary::attribute_descs;

constexpr std::array<ArgumentDesc, ImportEvent::input_arg_count + ImportEvent::output_arg_count> ImportEvent::argument_descs;
constexpr std::array<AttributeDesc, 0> ImportEvent::attribute_descs;

constexpr std::array<ArgumentDesc, WriteHistogramSummary::input_arg_count + WriteHistogramSummary::output_arg_count> WriteHistogramSummary::argument_descs;
constexpr std::array<AttributeDesc, 1> WriteHistogramSummary::attribute_descs;

constexpr std::array<ArgumentDesc, WriteImageSummary::input_arg_count + WriteImageSummary::output_arg_count> WriteImageSummary::argument_descs;
constexpr std::array<AttributeDesc, 2> WriteImageSummary::attribute_descs;

constexpr std::array<ArgumentDesc, WriteGraphSummary::input_arg_count + WriteGraphSummary::output_arg_count> WriteGraphSummary::argument_descs;
constexpr std::array<AttributeDesc, 0> WriteGraphSummary::attribute_descs;

constexpr std::array<ArgumentDesc, FFT::input_arg_count + FFT::output_arg_count> FFT::argument_descs;
constexpr std::array<AttributeDesc, 1> FFT::attribute_descs;

constexpr std::array<ArgumentDesc, IFFT::input_arg_count + IFFT::output_arg_count> IFFT::argument_descs;
constexpr std::array<AttributeDesc, 1> IFFT::attribute_descs;

constexpr std::array<ArgumentDesc, FFT2D::input_arg_count + FFT2D::output_arg_count> FFT2D::argument_descs;
constexpr std::array<AttributeDesc, 1> FFT2D::attribute_descs;

constexpr std::array<ArgumentDesc, IFFT2D::input_arg_count + IFFT2D::output_arg_count> IFFT2D::argument_descs;
constexpr std::array<AttributeDesc, 1> IFFT2D::attribute_descs;

constexpr std::array<ArgumentDesc, RetrieveTPUEmbeddingProximalYogiParameters::input_arg_count + RetrieveTPUEmbeddingProximalYogiParameters::output_arg_count> RetrieveTPUEmbeddingProximalYogiParameters::argument_descs;
constexpr std::array<AttributeDesc, 5> RetrieveTPUEmbeddingProximalYogiParameters::attribute_descs;

constexpr std::array<ArgumentDesc, RFFT::input_arg_count + RFFT::output_arg_count> RFFT::argument_descs;
constexpr std::array<AttributeDesc, 2> RFFT::attribute_descs;

constexpr std::array<ArgumentDesc, IRFFT::input_arg_count + IRFFT::output_arg_count> IRFFT::argument_descs;
constexpr std::array<AttributeDesc, 2> IRFFT::attribute_descs;

constexpr std::array<ArgumentDesc, ScatterNdMin::input_arg_count + ScatterNdMin::output_arg_count> ScatterNdMin::argument_descs;
constexpr std::array<AttributeDesc, 3> ScatterNdMin::attribute_descs;

constexpr std::array<ArgumentDesc, RFFT2D::input_arg_count + RFFT2D::output_arg_count> RFFT2D::argument_descs;
constexpr std::array<AttributeDesc, 2> RFFT2D::attribute_descs;

constexpr std::array<ArgumentDesc, IRFFT2D::input_arg_count + IRFFT2D::output_arg_count> IRFFT2D::argument_descs;
constexpr std::array<AttributeDesc, 2> IRFFT2D::attribute_descs;

constexpr std::array<ArgumentDesc, IRFFT3D::input_arg_count + IRFFT3D::output_arg_count> IRFFT3D::argument_descs;
constexpr std::array<AttributeDesc, 2> IRFFT3D::attribute_descs;

constexpr std::array<ArgumentDesc, BatchFFT::input_arg_count + BatchFFT::output_arg_count> BatchFFT::argument_descs;
constexpr std::array<AttributeDesc, 0> BatchFFT::attribute_descs;

constexpr std::array<ArgumentDesc, BatchFFT2D::input_arg_count + BatchFFT2D::output_arg_count> BatchFFT2D::argument_descs;
constexpr std::array<AttributeDesc, 0> BatchFFT2D::attribute_descs;

constexpr std::array<ArgumentDesc, BatchIFFT2D::input_arg_count + BatchIFFT2D::output_arg_count> BatchIFFT2D::argument_descs;
constexpr std::array<AttributeDesc, 0> BatchIFFT2D::attribute_descs;

constexpr std::array<ArgumentDesc, ScatterMax::input_arg_count + ScatterMax::output_arg_count> ScatterMax::argument_descs;
constexpr std::array<AttributeDesc, 3> ScatterMax::attribute_descs;

constexpr std::array<ArgumentDesc, BatchIFFT3D::input_arg_count + BatchIFFT3D::output_arg_count> BatchIFFT3D::argument_descs;
constexpr std::array<AttributeDesc, 0> BatchIFFT3D::attribute_descs;

constexpr std::array<ArgumentDesc, VariableV2::input_arg_count + VariableV2::output_arg_count> VariableV2::argument_descs;
constexpr std::array<AttributeDesc, 4> VariableV2::attribute_descs;

constexpr std::array<ArgumentDesc, IsVariableInitialized::input_arg_count + IsVariableInitialized::output_arg_count> IsVariableInitialized::argument_descs;
constexpr std::array<AttributeDesc, 1> IsVariableInitialized::attribute_descs;

constexpr std::array<ArgumentDesc, _MklNativeFusedBatchNormGradV2::input_arg_count + _MklNativeFusedBatchNormGradV2::output_arg_count> _MklNativeFusedBatchNormGradV2::argument_descs;
constexpr std::array<AttributeDesc, 5> _MklNativeFusedBatchNormGradV2::attribute_descs;

constexpr std::array<ArgumentDesc, AssignSub::input_arg_count + AssignSub::output_arg_count> AssignSub::argument_descs;
constexpr std::array<AttributeDesc, 2> AssignSub::attribute_descs;

constexpr std::array<ArgumentDesc, InfeedEnqueue::input_arg_count + InfeedEnqueue::output_arg_count> InfeedEnqueue::argument_descs;
constexpr std::array<AttributeDesc, 4> InfeedEnqueue::attribute_descs;

constexpr std::array<ArgumentDesc, ScatterAdd::input_arg_count + ScatterAdd::output_arg_count> ScatterAdd::argument_descs;
constexpr std::array<AttributeDesc, 3> ScatterAdd::attribute_descs;

constexpr std::array<ArgumentDesc, _MklNativeFusedBatchNormV3::input_arg_count + _MklNativeFusedBatchNormV3::output_arg_count> _MklNativeFusedBatchNormV3::argument_descs;
constexpr std::array<AttributeDesc, 6> _MklNativeFusedBatchNormV3::attribute_descs;

constexpr std::array<ArgumentDesc, ScatterSub::input_arg_count + ScatterSub::output_arg_count> ScatterSub::argument_descs;
constexpr std::array<AttributeDesc, 3> ScatterSub::attribute_descs;

constexpr std::array<ArgumentDesc, ResourceScatterNdUpdate::input_arg_count + ResourceScatterNdUpdate::output_arg_count> ResourceScatterNdUpdate::argument_descs;
constexpr std::array<AttributeDesc, 3> ResourceScatterNdUpdate::attribute_descs;

constexpr std::array<ArgumentDesc, ResourceScatterNdSub::input_arg_count + ResourceScatterNdSub::output_arg_count> ResourceScatterNdSub::argument_descs;
constexpr std::array<AttributeDesc, 3> ResourceScatterNdSub::attribute_descs;

constexpr std::array<ArgumentDesc, ScatterNdAdd::input_arg_count + ScatterNdAdd::output_arg_count> ScatterNdAdd::argument_descs;
constexpr std::array<AttributeDesc, 3> ScatterNdAdd::attribute_descs;

constexpr std::array<ArgumentDesc, ScatterNdMax::input_arg_count + ScatterNdMax::output_arg_count> ScatterNdMax::argument_descs;
constexpr std::array<AttributeDesc, 3> ScatterNdMax::attribute_descs;

constexpr std::array<ArgumentDesc, StatelessRandomNormal::input_arg_count + StatelessRandomNormal::output_arg_count> StatelessRandomNormal::argument_descs;
constexpr std::array<AttributeDesc, 3> StatelessRandomNormal::attribute_descs;

constexpr std::array<ArgumentDesc, StatelessTruncatedNormal::input_arg_count + StatelessTruncatedNormal::output_arg_count> StatelessTruncatedNormal::argument_descs;
constexpr std::array<AttributeDesc, 3> StatelessTruncatedNormal::attribute_descs;

constexpr std::array<ArgumentDesc, StatelessRandomUniformFullInt::input_arg_count + StatelessRandomUniformFullInt::output_arg_count> StatelessRandomUniformFullInt::argument_descs;
constexpr std::array<AttributeDesc, 3> StatelessRandomUniformFullInt::attribute_descs;

constexpr std::array<ArgumentDesc, StatelessRandomBinomial::input_arg_count + StatelessRandomBinomial::output_arg_count> StatelessRandomBinomial::argument_descs;
constexpr std::array<AttributeDesc, 4> StatelessRandomBinomial::attribute_descs;

constexpr std::array<ArgumentDesc, StatelessParameterizedTruncatedNormal::input_arg_count + StatelessParameterizedTruncatedNormal::output_arg_count> StatelessParameterizedTruncatedNormal::argument_descs;
constexpr std::array<AttributeDesc, 3> StatelessParameterizedTruncatedNormal::attribute_descs;

constexpr std::array<ArgumentDesc, StatelessRandomUniformV2::input_arg_count + StatelessRandomUniformV2::output_arg_count> StatelessRandomUniformV2::argument_descs;
constexpr std::array<AttributeDesc, 2> StatelessRandomUniformV2::attribute_descs;

constexpr std::array<ArgumentDesc, StatelessRandomGetKeyCounterAlg::input_arg_count + StatelessRandomGetKeyCounterAlg::output_arg_count> StatelessRandomGetKeyCounterAlg::argument_descs;
constexpr std::array<AttributeDesc, 1> StatelessRandomGetKeyCounterAlg::attribute_descs;

constexpr std::array<ArgumentDesc, StatelessRandomGetAlg::input_arg_count + StatelessRandomGetAlg::output_arg_count> StatelessRandomGetAlg::argument_descs;
constexpr std::array<AttributeDesc, 0> StatelessRandomGetAlg::attribute_descs;

constexpr std::array<ArgumentDesc, RegexReplace::input_arg_count + RegexReplace::output_arg_count> RegexReplace::argument_descs;
constexpr std::array<AttributeDesc, 1> RegexReplace::attribute_descs;

constexpr std::array<ArgumentDesc, StaticRegexReplace::input_arg_count + StaticRegexReplace::output_arg_count> StaticRegexReplace::argument_descs;
constexpr std::array<AttributeDesc, 3> StaticRegexReplace::attribute_descs;

constexpr std::array<ArgumentDesc, RecvTPUEmbeddingActivations::input_arg_count + RecvTPUEmbeddingActivations::output_arg_count> RecvTPUEmbeddingActivations::argument_descs;
constexpr std::array<AttributeDesc, 2> RecvTPUEmbeddingActivations::attribute_descs;

constexpr std::array<ArgumentDesc, RegexFullMatch::input_arg_count + RegexFullMatch::output_arg_count> RegexFullMatch::argument_descs;
constexpr std::array<AttributeDesc, 0> RegexFullMatch::attribute_descs;

constexpr std::array<ArgumentDesc, StaticRegexFullMatch::input_arg_count + StaticRegexFullMatch::output_arg_count> StaticRegexFullMatch::argument_descs;
constexpr std::array<AttributeDesc, 1> StaticRegexFullMatch::attribute_descs;

constexpr std::array<ArgumentDesc, StringToHashBucketFast::input_arg_count + StringToHashBucketFast::output_arg_count> StringToHashBucketFast::argument_descs;
constexpr std::array<AttributeDesc, 1> StringToHashBucketFast::attribute_descs;

constexpr std::array<ArgumentDesc, LoadTPUEmbeddingStochasticGradientDescentParameters::input_arg_count + LoadTPUEmbeddingStochasticGradientDescentParameters::output_arg_count> LoadTPUEmbeddingStochasticGradientDescentParameters::argument_descs;
constexpr std::array<AttributeDesc, 5> LoadTPUEmbeddingStochasticGradientDescentParameters::attribute_descs;

constexpr std::array<ArgumentDesc, ResourceApplyRMSProp::input_arg_count + ResourceApplyRMSProp::output_arg_count> ResourceApplyRMSProp::argument_descs;
constexpr std::array<AttributeDesc, 2> ResourceApplyRMSProp::attribute_descs;

constexpr std::array<ArgumentDesc, _TensorToHashBucketFast::input_arg_count + _TensorToHashBucketFast::output_arg_count> _TensorToHashBucketFast::argument_descs;
constexpr std::array<AttributeDesc, 2> _TensorToHashBucketFast::attribute_descs;

constexpr std::array<ArgumentDesc, StringToHashBucketStrong::input_arg_count + StringToHashBucketStrong::output_arg_count> StringToHashBucketStrong::argument_descs;
constexpr std::array<AttributeDesc, 2> StringToHashBucketStrong::attribute_descs;

constexpr std::array<ArgumentDesc, ReduceJoin::input_arg_count + ReduceJoin::output_arg_count> ReduceJoin::argument_descs;
constexpr std::array<AttributeDesc, 2> ReduceJoin::attribute_descs;

constexpr std::array<ArgumentDesc, UnsortedSegmentJoin::input_arg_count + UnsortedSegmentJoin::output_arg_count> UnsortedSegmentJoin::argument_descs;
constexpr std::array<AttributeDesc, 3> UnsortedSegmentJoin::attribute_descs;

constexpr std::array<ArgumentDesc, AsString::input_arg_count + AsString::output_arg_count> AsString::argument_descs;
constexpr std::array<AttributeDesc, 6> AsString::attribute_descs;

constexpr std::array<ArgumentDesc, StringJoin::input_arg_count + StringJoin::output_arg_count> StringJoin::argument_descs;
constexpr std::array<AttributeDesc, 2> StringJoin::attribute_descs;

constexpr std::array<ArgumentDesc, StringSplitV2::input_arg_count + StringSplitV2::output_arg_count> StringSplitV2::argument_descs;
constexpr std::array<AttributeDesc, 1> StringSplitV2::attribute_descs;

constexpr std::array<ArgumentDesc, StringLower::input_arg_count + StringLower::output_arg_count> StringLower::argument_descs;
constexpr std::array<AttributeDesc, 1> StringLower::attribute_descs;

constexpr std::array<ArgumentDesc, StringLength::input_arg_count + StringLength::output_arg_count> StringLength::argument_descs;
constexpr std::array<AttributeDesc, 1> StringLength::attribute_descs;

constexpr std::array<ArgumentDesc, EncodeBase64::input_arg_count + EncodeBase64::output_arg_count> EncodeBase64::argument_descs;
constexpr std::array<AttributeDesc, 1> EncodeBase64::attribute_descs;

constexpr std::array<ArgumentDesc, _TPUCompileMlir::input_arg_count + _TPUCompileMlir::output_arg_count> _TPUCompileMlir::argument_descs;
constexpr std::array<AttributeDesc, 4> _TPUCompileMlir::attribute_descs;

constexpr std::array<ArgumentDesc, DecodeBase64::input_arg_count + DecodeBase64::output_arg_count> DecodeBase64::argument_descs;
constexpr std::array<AttributeDesc, 0> DecodeBase64::attribute_descs;

constexpr std::array<ArgumentDesc, _MklNativeFusedConv2D::input_arg_count + _MklNativeFusedConv2D::output_arg_count> _MklNativeFusedConv2D::argument_descs;
constexpr std::array<AttributeDesc, 12> _MklNativeFusedConv2D::attribute_descs;

constexpr std::array<ArgumentDesc, Substr::input_arg_count + Substr::output_arg_count> Substr::argument_descs;
constexpr std::array<AttributeDesc, 2> Substr::attribute_descs;

constexpr std::array<ArgumentDesc, UnicodeEncode::input_arg_count + UnicodeEncode::output_arg_count> UnicodeEncode::argument_descs;
constexpr std::array<AttributeDesc, 4> UnicodeEncode::attribute_descs;

constexpr std::array<ArgumentDesc, UnicodeTranscode::input_arg_count + UnicodeTranscode::output_arg_count> UnicodeTranscode::argument_descs;
constexpr std::array<AttributeDesc, 5> UnicodeTranscode::attribute_descs;

constexpr std::array<ArgumentDesc, UnicodeDecode::input_arg_count + UnicodeDecode::output_arg_count> UnicodeDecode::argument_descs;
constexpr std::array<AttributeDesc, 5> UnicodeDecode::attribute_descs;

constexpr std::array<ArgumentDesc, SparseApplyProximalGradientDescent::input_arg_count + SparseApplyProximalGradientDescent::output_arg_count> SparseApplyProximalGradientDescent::argument_descs;
constexpr std::array<AttributeDesc, 3> SparseApplyProximalGradientDescent::attribute_descs;

constexpr std::array<ArgumentDesc, ResourceSparseApplyProximalGradientDescent::input_arg_count + ResourceSparseApplyProximalGradientDescent::output_arg_count> ResourceSparseApplyProximalGradientDescent::argument_descs;
constexpr std::array<AttributeDesc, 3> ResourceSparseApplyProximalGradientDescent::attribute_descs;

constexpr std::array<ArgumentDesc, ApplyAdadelta::input_arg_count + ApplyAdadelta::output_arg_count> ApplyAdadelta::argument_descs;
constexpr std::array<AttributeDesc, 2> ApplyAdadelta::attribute_descs;

constexpr std::array<ArgumentDesc, ResourceSparseApplyAdadelta::input_arg_count + ResourceSparseApplyAdadelta::output_arg_count> ResourceSparseApplyAdadelta::argument_descs;
constexpr std::array<AttributeDesc, 3> ResourceSparseApplyAdadelta::attribute_descs;

constexpr std::array<ArgumentDesc, ApplyAdagrad::input_arg_count + ApplyAdagrad::output_arg_count> ApplyAdagrad::argument_descs;
constexpr std::array<AttributeDesc, 3> ApplyAdagrad::attribute_descs;

constexpr std::array<ArgumentDesc, ResourceApplyAdagrad::input_arg_count + ResourceApplyAdagrad::output_arg_count> ResourceApplyAdagrad::argument_descs;
constexpr std::array<AttributeDesc, 3> ResourceApplyAdagrad::attribute_descs;

constexpr std::array<ArgumentDesc, _InitializeHostForDistributedTPU::input_arg_count + _InitializeHostForDistributedTPU::output_arg_count> _InitializeHostForDistributedTPU::argument_descs;
constexpr std::array<AttributeDesc, 2> _InitializeHostForDistributedTPU::attribute_descs;

constexpr std::array<ArgumentDesc, ApplyMomentum::input_arg_count + ApplyMomentum::output_arg_count> ApplyMomentum::argument_descs;
constexpr std::array<AttributeDesc, 3> ApplyMomentum::attribute_descs;

constexpr std::array<ArgumentDesc, ApplyAdagradV2::input_arg_count + ApplyAdagradV2::output_arg_count> ApplyAdagradV2::argument_descs;
constexpr std::array<AttributeDesc, 3> ApplyAdagradV2::attribute_descs;

constexpr std::array<ArgumentDesc, ResourceSparseApplyAdagradV2::input_arg_count + ResourceSparseApplyAdagradV2::output_arg_count> ResourceSparseApplyAdagradV2::argument_descs;
constexpr std::array<AttributeDesc, 4> ResourceSparseApplyAdagradV2::attribute_descs;

constexpr std::array<ArgumentDesc, ResourceApplyProximalAdagrad::input_arg_count + ResourceApplyProximalAdagrad::output_arg_count> ResourceApplyProximalAdagrad::argument_descs;
constexpr std::array<AttributeDesc, 2> ResourceApplyProximalAdagrad::attribute_descs;

constexpr std::array<ArgumentDesc, ResourceSparseApplyProximalAdagrad::input_arg_count + ResourceSparseApplyProximalAdagrad::output_arg_count> ResourceSparseApplyProximalAdagrad::argument_descs;
constexpr std::array<AttributeDesc, 3> ResourceSparseApplyProximalAdagrad::attribute_descs;

constexpr std::array<ArgumentDesc, TopKWithUnique::input_arg_count + TopKWithUnique::output_arg_count> TopKWithUnique::argument_descs;
constexpr std::array<AttributeDesc, 1> TopKWithUnique::attribute_descs;

constexpr std::array<ArgumentDesc, SparseApplyCenteredRMSProp::input_arg_count + SparseApplyCenteredRMSProp::output_arg_count> SparseApplyCenteredRMSProp::argument_descs;
constexpr std::array<AttributeDesc, 3> SparseApplyCenteredRMSProp::attribute_descs;

constexpr std::array<ArgumentDesc, SparseApplyAdagradDA::input_arg_count + SparseApplyAdagradDA::output_arg_count> SparseApplyAdagradDA::argument_descs;
constexpr std::array<AttributeDesc, 3> SparseApplyAdagradDA::attribute_descs;

constexpr std::array<ArgumentDesc, SparseApplyFtrl::input_arg_count + SparseApplyFtrl::output_arg_count> SparseApplyFtrl::argument_descs;
constexpr std::array<AttributeDesc, 4> SparseApplyFtrl::attribute_descs;

constexpr std::array<ArgumentDesc, ResourceApplyFtrl::input_arg_count + ResourceApplyFtrl::output_arg_count> ResourceApplyFtrl::argument_descs;
constexpr std::array<AttributeDesc, 3> ResourceApplyFtrl::attribute_descs;

constexpr std::array<ArgumentDesc, ResourceSparseApplyFtrl::input_arg_count + ResourceSparseApplyFtrl::output_arg_count> ResourceSparseApplyFtrl::argument_descs;
constexpr std::array<AttributeDesc, 4> ResourceSparseApplyFtrl::attribute_descs;

constexpr std::array<ArgumentDesc, ResourceApplyFtrlV2::input_arg_count + ResourceApplyFtrlV2::output_arg_count> ResourceApplyFtrlV2::argument_descs;
constexpr std::array<AttributeDesc, 3> ResourceApplyFtrlV2::attribute_descs;

constexpr std::array<ArgumentDesc, SparseApplyMomentum::input_arg_count + SparseApplyMomentum::output_arg_count> SparseApplyMomentum::argument_descs;
constexpr std::array<AttributeDesc, 4> SparseApplyMomentum::attribute_descs;

constexpr std::array<ArgumentDesc, ResourceApplyMomentum::input_arg_count + ResourceApplyMomentum::output_arg_count> ResourceApplyMomentum::argument_descs;
constexpr std::array<AttributeDesc, 3> ResourceApplyMomentum::attribute_descs;

constexpr std::array<ArgumentDesc, ResourceSparseApplyMomentum::input_arg_count + ResourceSparseApplyMomentum::output_arg_count> ResourceSparseApplyMomentum::argument_descs;
constexpr std::array<AttributeDesc, 4> ResourceSparseApplyMomentum::attribute_descs;

constexpr std::array<ArgumentDesc, ResourceApplyCenteredRMSProp::input_arg_count + ResourceApplyCenteredRMSProp::output_arg_count> ResourceApplyCenteredRMSProp::argument_descs;
constexpr std::array<AttributeDesc, 2> ResourceApplyCenteredRMSProp::attribute_descs;

constexpr std::array<ArgumentDesc, ResourceSparseApplyCenteredRMSProp::input_arg_count + ResourceSparseApplyCenteredRMSProp::output_arg_count> ResourceSparseApplyCenteredRMSProp::argument_descs;
constexpr std::array<AttributeDesc, 3> ResourceSparseApplyCenteredRMSProp::attribute_descs;

constexpr std::array<ArgumentDesc, ResourceApplyPowerSign::input_arg_count + ResourceApplyPowerSign::output_arg_count> ResourceApplyPowerSign::argument_descs;
constexpr std::array<AttributeDesc, 2> ResourceApplyPowerSign::attribute_descs;

constexpr std::array<ArgumentDesc, Skipgram::input_arg_count + Skipgram::output_arg_count> Skipgram::argument_descs;
constexpr std::array<AttributeDesc, 5> Skipgram::attribute_descs;

constexpr std::array<ArgumentDesc, _WaitForDistributedTPU::input_arg_count + _WaitForDistributedTPU::output_arg_count> _WaitForDistributedTPU::argument_descs;
constexpr std::array<AttributeDesc, 2> _WaitForDistributedTPU::attribute_descs;

constexpr std::array<ArgumentDesc, _SetGlobalTPUArray::input_arg_count + _SetGlobalTPUArray::output_arg_count> _SetGlobalTPUArray::argument_descs;
constexpr std::array<AttributeDesc, 0> _SetGlobalTPUArray::attribute_descs;

constexpr std::array<ArgumentDesc, ShutdownDistributedTPU::input_arg_count + ShutdownDistributedTPU::output_arg_count> ShutdownDistributedTPU::argument_descs;
constexpr std::array<AttributeDesc, 0> ShutdownDistributedTPU::attribute_descs;

constexpr std::array<ArgumentDesc, AllToAll::input_arg_count + AllToAll::output_arg_count> AllToAll::argument_descs;
constexpr std::array<AttributeDesc, 4> AllToAll::attribute_descs;

constexpr std::array<ArgumentDesc, CollectivePermute::input_arg_count + CollectivePermute::output_arg_count> CollectivePermute::argument_descs;
constexpr std::array<AttributeDesc, 1> CollectivePermute::attribute_descs;

constexpr std::array<ArgumentDesc, EnqueueTPUEmbeddingIntegerBatch::input_arg_count + EnqueueTPUEmbeddingIntegerBatch::output_arg_count> EnqueueTPUEmbeddingIntegerBatch::argument_descs;
constexpr std::array<AttributeDesc, 2> EnqueueTPUEmbeddingIntegerBatch::attribute_descs;

constexpr std::array<ArgumentDesc, EnqueueTPUEmbeddingSparseBatch::input_arg_count + EnqueueTPUEmbeddingSparseBatch::output_arg_count> EnqueueTPUEmbeddingSparseBatch::argument_descs;
constexpr std::array<AttributeDesc, 6> EnqueueTPUEmbeddingSparseBatch::attribute_descs;

constexpr std::array<ArgumentDesc, EnqueueTPUEmbeddingRaggedTensorBatch::input_arg_count + EnqueueTPUEmbeddingRaggedTensorBatch::output_arg_count> EnqueueTPUEmbeddingRaggedTensorBatch::argument_descs;
constexpr std::array<AttributeDesc, 9> EnqueueTPUEmbeddingRaggedTensorBatch::attribute_descs;

constexpr std::array<ArgumentDesc, DynamicEnqueueTPUEmbeddingArbitraryTensorBatch::input_arg_count + DynamicEnqueueTPUEmbeddingArbitraryTensorBatch::output_arg_count> DynamicEnqueueTPUEmbeddingArbitraryTensorBatch::argument_descs;
constexpr std::array<AttributeDesc, 5> DynamicEnqueueTPUEmbeddingArbitraryTensorBatch::attribute_descs;

constexpr std::array<ArgumentDesc, XlaEinsum::input_arg_count + XlaEinsum::output_arg_count> XlaEinsum::argument_descs;
constexpr std::array<AttributeDesc, 2> XlaEinsum::attribute_descs;

constexpr std::array<ArgumentDesc, RetrieveTPUEmbeddingAdagradParameters::input_arg_count + RetrieveTPUEmbeddingAdagradParameters::output_arg_count> RetrieveTPUEmbeddingAdagradParameters::argument_descs;
constexpr std::array<AttributeDesc, 5> RetrieveTPUEmbeddingAdagradParameters::attribute_descs;

constexpr std::array<ArgumentDesc, RetrieveTPUEmbeddingAdagradMomentumParameters::input_arg_count + RetrieveTPUEmbeddingAdagradMomentumParameters::output_arg_count> RetrieveTPUEmbeddingAdagradMomentumParameters::argument_descs;
constexpr std::array<AttributeDesc, 5> RetrieveTPUEmbeddingAdagradMomentumParameters::attribute_descs;

constexpr std::array<ArgumentDesc, RetrieveTPUEmbeddingStochasticGradientDescentParameters::input_arg_count + RetrieveTPUEmbeddingStochasticGradientDescentParameters::output_arg_count> RetrieveTPUEmbeddingStochasticGradientDescentParameters::argument_descs;
constexpr std::array<AttributeDesc, 5> RetrieveTPUEmbeddingStochasticGradientDescentParameters::attribute_descs;

constexpr std::array<ArgumentDesc, LoadTPUEmbeddingADAMParameters::input_arg_count + LoadTPUEmbeddingADAMParameters::output_arg_count> LoadTPUEmbeddingADAMParameters::argument_descs;
constexpr std::array<AttributeDesc, 5> LoadTPUEmbeddingADAMParameters::attribute_descs;

constexpr std::array<ArgumentDesc, _MklQuantizedMatMulWithBias::input_arg_count + _MklQuantizedMatMulWithBias::output_arg_count> _MklQuantizedMatMulWithBias::argument_descs;
constexpr std::array<AttributeDesc, 8> _MklQuantizedMatMulWithBias::attribute_descs;

constexpr std::array<ArgumentDesc, RetrieveTPUEmbeddingADAMParameters::input_arg_count + RetrieveTPUEmbeddingADAMParameters::output_arg_count> RetrieveTPUEmbeddingADAMParameters::argument_descs;
constexpr std::array<AttributeDesc, 5> RetrieveTPUEmbeddingADAMParameters::attribute_descs;

constexpr std::array<ArgumentDesc, RetrieveTPUEmbeddingMomentumParameters::input_arg_count + RetrieveTPUEmbeddingMomentumParameters::output_arg_count> RetrieveTPUEmbeddingMomentumParameters::argument_descs;
constexpr std::array<AttributeDesc, 5> RetrieveTPUEmbeddingMomentumParameters::attribute_descs;

constexpr std::array<ArgumentDesc, _MklNativeConv2DWithBias::input_arg_count + _MklNativeConv2DWithBias::output_arg_count> _MklNativeConv2DWithBias::argument_descs;
constexpr std::array<AttributeDesc, 8> _MklNativeConv2DWithBias::attribute_descs;

constexpr std::array<ArgumentDesc, RetrieveTPUEmbeddingRMSPropParameters::input_arg_count + RetrieveTPUEmbeddingRMSPropParameters::output_arg_count> RetrieveTPUEmbeddingRMSPropParameters::argument_descs;
constexpr std::array<AttributeDesc, 5> RetrieveTPUEmbeddingRMSPropParameters::attribute_descs;

constexpr std::array<ArgumentDesc, RetrieveTPUEmbeddingCenteredRMSPropParameters::input_arg_count + RetrieveTPUEmbeddingCenteredRMSPropParameters::output_arg_count> RetrieveTPUEmbeddingCenteredRMSPropParameters::argument_descs;
constexpr std::array<AttributeDesc, 5> RetrieveTPUEmbeddingCenteredRMSPropParameters::attribute_descs;

constexpr std::array<ArgumentDesc, LoadTPUEmbeddingMDLAdagradLightParameters::input_arg_count + LoadTPUEmbeddingMDLAdagradLightParameters::output_arg_count> LoadTPUEmbeddingMDLAdagradLightParameters::argument_descs;
constexpr std::array<AttributeDesc, 5> LoadTPUEmbeddingMDLAdagradLightParameters::attribute_descs;

constexpr std::array<ArgumentDesc, RetrieveTPUEmbeddingMDLAdagradLightParameters::input_arg_count + RetrieveTPUEmbeddingMDLAdagradLightParameters::output_arg_count> RetrieveTPUEmbeddingMDLAdagradLightParameters::argument_descs;
constexpr std::array<AttributeDesc, 5> RetrieveTPUEmbeddingMDLAdagradLightParameters::attribute_descs;

constexpr std::array<ArgumentDesc, LoadTPUEmbeddingAdadeltaParameters::input_arg_count + LoadTPUEmbeddingAdadeltaParameters::output_arg_count> LoadTPUEmbeddingAdadeltaParameters::argument_descs;
constexpr std::array<AttributeDesc, 5> LoadTPUEmbeddingAdadeltaParameters::attribute_descs;

constexpr std::array<ArgumentDesc, RetrieveTPUEmbeddingAdadeltaParameters::input_arg_count + RetrieveTPUEmbeddingAdadeltaParameters::output_arg_count> RetrieveTPUEmbeddingAdadeltaParameters::argument_descs;
constexpr std::array<AttributeDesc, 5> RetrieveTPUEmbeddingAdadeltaParameters::attribute_descs;

constexpr std::array<ArgumentDesc, RetrieveTPUEmbeddingProximalAdagradParameters::input_arg_count + RetrieveTPUEmbeddingProximalAdagradParameters::output_arg_count> RetrieveTPUEmbeddingProximalAdagradParameters::argument_descs;
constexpr std::array<AttributeDesc, 5> RetrieveTPUEmbeddingProximalAdagradParameters::attribute_descs;

constexpr std::array<ArgumentDesc, LoadTPUEmbeddingFrequencyEstimatorParameters::input_arg_count + LoadTPUEmbeddingFrequencyEstimatorParameters::output_arg_count> LoadTPUEmbeddingFrequencyEstimatorParameters::argument_descs;
constexpr std::array<AttributeDesc, 5> LoadTPUEmbeddingFrequencyEstimatorParameters::attribute_descs;

constexpr std::array<ArgumentDesc, _XlaRecvAtHost::input_arg_count + _XlaRecvAtHost::output_arg_count> _XlaRecvAtHost::argument_descs;
constexpr std::array<AttributeDesc, 3> _XlaRecvAtHost::attribute_descs;

constexpr std::array<ArgumentDesc, _XlaRecvAtHostV2::input_arg_count + _XlaRecvAtHostV2::output_arg_count> _XlaRecvAtHostV2::argument_descs;
constexpr std::array<AttributeDesc, 2> _XlaRecvAtHostV2::attribute_descs;

constexpr std::array<ArgumentDesc, XlaConcatND::input_arg_count + XlaConcatND::output_arg_count> XlaConcatND::argument_descs;
constexpr std::array<AttributeDesc, 4> XlaConcatND::attribute_descs;

constexpr std::array<ArgumentDesc, ReadVariableXlaSplitND::input_arg_count + ReadVariableXlaSplitND::output_arg_count> ReadVariableXlaSplitND::argument_descs;
constexpr std::array<AttributeDesc, 4> ReadVariableXlaSplitND::attribute_descs;

constexpr std::array<ArgumentDesc, InfeedDequeue::input_arg_count + InfeedDequeue::output_arg_count> InfeedDequeue::argument_descs;
constexpr std::array<AttributeDesc, 2> InfeedDequeue::attribute_descs;

constexpr std::array<ArgumentDesc, InfeedEnqueueTuple::input_arg_count + InfeedEnqueueTuple::output_arg_count> InfeedEnqueueTuple::argument_descs;
constexpr std::array<AttributeDesc, 4> InfeedEnqueueTuple::attribute_descs;

constexpr std::array<ArgumentDesc, Prelinearize::input_arg_count + Prelinearize::output_arg_count> Prelinearize::argument_descs;
constexpr std::array<AttributeDesc, 3> Prelinearize::attribute_descs;

constexpr std::array<ArgumentDesc, XlaDot::input_arg_count + XlaDot::output_arg_count> XlaDot::argument_descs;
constexpr std::array<AttributeDesc, 3> XlaDot::attribute_descs;

constexpr std::array<ArgumentDesc, OutfeedDequeue::input_arg_count + OutfeedDequeue::output_arg_count> OutfeedDequeue::argument_descs;
constexpr std::array<AttributeDesc, 3> OutfeedDequeue::attribute_descs;

constexpr std::array<ArgumentDesc, TPUReplicatedInput::input_arg_count + TPUReplicatedInput::output_arg_count> TPUReplicatedInput::argument_descs;
constexpr std::array<AttributeDesc, 5> TPUReplicatedInput::attribute_descs;

constexpr std::array<ArgumentDesc, TPUReplicatedOutput::input_arg_count + TPUReplicatedOutput::output_arg_count> TPUReplicatedOutput::argument_descs;
constexpr std::array<AttributeDesc, 2> TPUReplicatedOutput::attribute_descs;

constexpr std::array<ArgumentDesc, _TPUReplicate::input_arg_count + _TPUReplicate::output_arg_count> _TPUReplicate::argument_descs;
constexpr std::array<AttributeDesc, 17> _TPUReplicate::attribute_descs;

constexpr std::array<ArgumentDesc, _MklNativeConv3DBackpropFilterV2::input_arg_count + _MklNativeConv3DBackpropFilterV2::output_arg_count> _MklNativeConv3DBackpropFilterV2::argument_descs;
constexpr std::array<AttributeDesc, 5> _MklNativeConv3DBackpropFilterV2::attribute_descs;

constexpr std::array<ArgumentDesc, _MklFusedDepthwiseConv2dNative::input_arg_count + _MklFusedDepthwiseConv2dNative::output_arg_count> _MklFusedDepthwiseConv2dNative::argument_descs;
constexpr std::array<AttributeDesc, 10> _MklFusedDepthwiseConv2dNative::attribute_descs;

constexpr std::array<ArgumentDesc, _MklNativeFusedMatMul::input_arg_count + _MklNativeFusedMatMul::output_arg_count> _MklNativeFusedMatMul::argument_descs;
constexpr std::array<AttributeDesc, 8> _MklNativeFusedMatMul::attribute_descs;

constexpr std::array<ArgumentDesc, _MklNativePadWithConv2D::input_arg_count + _MklNativePadWithConv2D::output_arg_count> _MklNativePadWithConv2D::argument_descs;
constexpr std::array<AttributeDesc, 8> _MklNativePadWithConv2D::attribute_descs;

constexpr std::array<ArgumentDesc, _MklNativeAvgPoolGrad::input_arg_count + _MklNativeAvgPoolGrad::output_arg_count> _MklNativeAvgPoolGrad::argument_descs;
constexpr std::array<AttributeDesc, 5> _MklNativeAvgPoolGrad::attribute_descs;

constexpr std::array<ArgumentDesc, _MklNativeAvgPool3D::input_arg_count + _MklNativeAvgPool3D::output_arg_count> _MklNativeAvgPool3D::argument_descs;
constexpr std::array<AttributeDesc, 5> _MklNativeAvgPool3D::attribute_descs;

constexpr std::array<ArgumentDesc, _MklNativeAvgPool3DGrad::input_arg_count + _MklNativeAvgPool3DGrad::output_arg_count> _MklNativeAvgPool3DGrad::argument_descs;
constexpr std::array<AttributeDesc, 5> _MklNativeAvgPool3DGrad::attribute_descs;

constexpr std::array<ArgumentDesc, _MklNativeMaxPool::input_arg_count + _MklNativeMaxPool::output_arg_count> _MklNativeMaxPool::argument_descs;
constexpr std::array<AttributeDesc, 7> _MklNativeMaxPool::attribute_descs;

constexpr std::array<ArgumentDesc, _MklNativeMaxPool3D::input_arg_count + _MklNativeMaxPool3D::output_arg_count> _MklNativeMaxPool3D::argument_descs;
constexpr std::array<AttributeDesc, 6> _MklNativeMaxPool3D::attribute_descs;

constexpr std::array<ArgumentDesc, _MklNativeMaxPool3DGrad::input_arg_count + _MklNativeMaxPool3DGrad::output_arg_count> _MklNativeMaxPool3DGrad::argument_descs;
constexpr std::array<AttributeDesc, 7> _MklNativeMaxPool3DGrad::attribute_descs;

constexpr std::array<ArgumentDesc, _MklQuantizedMaxPool::input_arg_count + _MklQuantizedMaxPool::output_arg_count> _MklQuantizedMaxPool::argument_descs;
constexpr std::array<AttributeDesc, 4> _MklQuantizedMaxPool::attribute_descs;

constexpr std::array<ArgumentDesc, _MklNativeFusedBatchNormGrad::input_arg_count + _MklNativeFusedBatchNormGrad::output_arg_count> _MklNativeFusedBatchNormGrad::argument_descs;
constexpr std::array<AttributeDesc, 4> _MklNativeFusedBatchNormGrad::attribute_descs;

constexpr std::array<ArgumentDesc, _MklQuantizedConv2DWithBias::input_arg_count + _MklQuantizedConv2DWithBias::output_arg_count> _MklQuantizedConv2DWithBias::argument_descs;
constexpr std::array<AttributeDesc, 10> _MklQuantizedConv2DWithBias::attribute_descs;

constexpr std::array<ArgumentDesc, _MklQuantizedConv2DAndRelu::input_arg_count + _MklQuantizedConv2DAndRelu::output_arg_count> _MklQuantizedConv2DAndRelu::argument_descs;
constexpr std::array<AttributeDesc, 9> _MklQuantizedConv2DAndRelu::attribute_descs;

constexpr std::array<ArgumentDesc, _MklQuantizedConv2DWithBiasSumAndRelu::input_arg_count + _MklQuantizedConv2DWithBiasSumAndRelu::output_arg_count> _MklQuantizedConv2DWithBiasSumAndRelu::argument_descs;
constexpr std::array<AttributeDesc, 10> _MklQuantizedConv2DWithBiasSumAndRelu::attribute_descs;

constexpr std::array<ArgumentDesc, _MklQuantizedConv2DWithBiasSignedSumAndReluAndRequantize::input_arg_count + _MklQuantizedConv2DWithBiasSignedSumAndReluAndRequantize::output_arg_count> _MklQuantizedConv2DWithBiasSignedSumAndReluAndRequantize::argument_descs;
constexpr std::array<AttributeDesc, 12> _MklQuantizedConv2DWithBiasSignedSumAndReluAndRequantize::attribute_descs;

constexpr std::array<ArgumentDesc, _MklDepthwiseConv2dNativeBackpropInput::input_arg_count + _MklDepthwiseConv2dNativeBackpropInput::output_arg_count> _MklDepthwiseConv2dNativeBackpropInput::argument_descs;
constexpr std::array<AttributeDesc, 6> _MklDepthwiseConv2dNativeBackpropInput::attribute_descs;

constexpr std::array<ArgumentDesc, _MklFusedBatchNormEx::input_arg_count + _MklFusedBatchNormEx::output_arg_count> _MklFusedBatchNormEx::argument_descs;
constexpr std::array<AttributeDesc, 8> _MklFusedBatchNormEx::attribute_descs;

constexpr std::array<ArgumentDesc, _MklDepthwiseConv2dNativeBackpropFilter::input_arg_count + _MklDepthwiseConv2dNativeBackpropFilter::output_arg_count> _MklDepthwiseConv2dNativeBackpropFilter::argument_descs;
constexpr std::array<AttributeDesc, 6> _MklDepthwiseConv2dNativeBackpropFilter::attribute_descs;

constexpr std::array<ArgumentDesc, _MklQuantizedMatMulWithBiasAndRequantize::input_arg_count + _MklQuantizedMatMulWithBiasAndRequantize::output_arg_count> _MklQuantizedMatMulWithBiasAndRequantize::argument_descs;
constexpr std::array<AttributeDesc, 8> _MklQuantizedMatMulWithBiasAndRequantize::attribute_descs;

constexpr std::array<ArgumentDesc, _MklQuantizedDepthwiseConv2DWithBias::input_arg_count + _MklQuantizedDepthwiseConv2DWithBias::output_arg_count> _MklQuantizedDepthwiseConv2DWithBias::argument_descs;
constexpr std::array<AttributeDesc, 9> _MklQuantizedDepthwiseConv2DWithBias::attribute_descs;

constexpr std::array<ArgumentDesc, _MklQuantizedDepthwiseConv2DWithBiasAndRelu::input_arg_count + _MklQuantizedDepthwiseConv2DWithBiasAndRelu::output_arg_count> _MklQuantizedDepthwiseConv2DWithBiasAndRelu::argument_descs;
constexpr std::array<AttributeDesc, 10> _MklQuantizedDepthwiseConv2DWithBiasAndRelu::attribute_descs;

constexpr std::array<ArgumentDesc, _MklQuantizedDepthwiseConv2DWithBiasAndReluAndRequantize::input_arg_count + _MklQuantizedDepthwiseConv2DWithBiasAndReluAndRequantize::output_arg_count> _MklQuantizedDepthwiseConv2DWithBiasAndReluAndRequantize::argument_descs;
constexpr std::array<AttributeDesc, 11> _MklQuantizedDepthwiseConv2DWithBiasAndReluAndRequantize::attribute_descs;

constexpr std::array<ArgumentDesc, _MklFusedBatchNormV3::input_arg_count + _MklFusedBatchNormV3::output_arg_count> _MklFusedBatchNormV3::argument_descs;
constexpr std::array<AttributeDesc, 6> _MklFusedBatchNormV3::attribute_descs;

constexpr std::array<ArgumentDesc, _MklNativeFusedBatchNorm::input_arg_count + _MklNativeFusedBatchNorm::output_arg_count> _MklNativeFusedBatchNorm::argument_descs;
constexpr std::array<AttributeDesc, 5> _MklNativeFusedBatchNorm::attribute_descs;

constexpr std::array<ArgumentDesc, _MklNativeFusedBatchNormV2::input_arg_count + _MklNativeFusedBatchNormV2::output_arg_count> _MklNativeFusedBatchNormV2::argument_descs;
constexpr std::array<AttributeDesc, 6> _MklNativeFusedBatchNormV2::attribute_descs;

constexpr std::array<ArgumentDesc, _MklNativeFusedBatchNormGradV3::input_arg_count + _MklNativeFusedBatchNormGradV3::output_arg_count> _MklNativeFusedBatchNormGradV3::argument_descs;
constexpr std::array<AttributeDesc, 5> _MklNativeFusedBatchNormGradV3::attribute_descs;

constexpr std::array<ArgumentDesc, _MklFusedMish::input_arg_count + _MklFusedMish::output_arg_count> _MklFusedMish::argument_descs;
constexpr std::array<AttributeDesc, 1> _MklFusedMish::attribute_descs;

constexpr std::array<ArgumentDesc, _MklSwish::input_arg_count + _MklSwish::output_arg_count> _MklSwish::argument_descs;
constexpr std::array<AttributeDesc, 1> _MklSwish::attribute_descs;

constexpr std::array<ArgumentDesc, XlaHostCompute::input_arg_count + XlaHostCompute::output_arg_count> XlaHostCompute::argument_descs;
constexpr std::array<AttributeDesc, 10> XlaHostCompute::attribute_descs;

constexpr std::array<ArgumentDesc, XlaSendToHost::input_arg_count + XlaSendToHost::output_arg_count> XlaSendToHost::argument_descs;
constexpr std::array<AttributeDesc, 2> XlaSendToHost::attribute_descs;

constexpr std::array<ArgumentDesc, TopKUnique::input_arg_count + TopKUnique::output_arg_count> TopKUnique::argument_descs;
constexpr std::array<AttributeDesc, 1> TopKUnique::attribute_descs;

constexpr std::array<ArgumentDesc, _ExecuteTPUEmbeddingPartitioner::input_arg_count + _ExecuteTPUEmbeddingPartitioner::output_arg_count> _ExecuteTPUEmbeddingPartitioner::argument_descs;
constexpr std::array<AttributeDesc, 1> _ExecuteTPUEmbeddingPartitioner::attribute_descs;

constexpr std::array<ArgumentDesc, _ConfigureTPUEmbeddingHost::input_arg_count + _ConfigureTPUEmbeddingHost::output_arg_count> _ConfigureTPUEmbeddingHost::argument_descs;
constexpr std::array<AttributeDesc, 2> _ConfigureTPUEmbeddingHost::attribute_descs;

constexpr std::array<ArgumentDesc, LoadAllTPUEmbeddingParameters::input_arg_count + LoadAllTPUEmbeddingParameters::output_arg_count> LoadAllTPUEmbeddingParameters::argument_descs;
constexpr std::array<AttributeDesc, 4> LoadAllTPUEmbeddingParameters::attribute_descs;

constexpr std::array<ArgumentDesc, RetrieveAllTPUEmbeddingParameters::input_arg_count + RetrieveAllTPUEmbeddingParameters::output_arg_count> RetrieveAllTPUEmbeddingParameters::argument_descs;
constexpr std::array<AttributeDesc, 4> RetrieveAllTPUEmbeddingParameters::attribute_descs;

constexpr std::array<ArgumentDesc, TPUExecute::input_arg_count + TPUExecute::output_arg_count> TPUExecute::argument_descs;
constexpr std::array<AttributeDesc, 2> TPUExecute::attribute_descs;

constexpr std::array<ArgumentDesc, TPUExecuteAndUpdateVariables::input_arg_count + TPUExecuteAndUpdateVariables::output_arg_count> TPUExecuteAndUpdateVariables::argument_descs;
constexpr std::array<AttributeDesc, 4> TPUExecuteAndUpdateVariables::attribute_descs;

constexpr std::array<ArgumentDesc, TPUPartitionedInput::input_arg_count + TPUPartitionedInput::output_arg_count> TPUPartitionedInput::argument_descs;
constexpr std::array<AttributeDesc, 3> TPUPartitionedInput::attribute_descs;

constexpr std::array<ArgumentDesc, MlirPassthroughOp::input_arg_count + MlirPassthroughOp::output_arg_count> MlirPassthroughOp::argument_descs;
constexpr std::array<AttributeDesc, 3> MlirPassthroughOp::attribute_descs;

constexpr std::array<ArgumentDesc, XlaSvd::input_arg_count + XlaSvd::output_arg_count> XlaSvd::argument_descs;
constexpr std::array<AttributeDesc, 4> XlaSvd::attribute_descs;

constexpr std::array<ArgumentDesc, XlaConvV2::input_arg_count + XlaConvV2::output_arg_count> XlaConvV2::argument_descs;
constexpr std::array<AttributeDesc, 7> XlaConvV2::attribute_descs;

constexpr std::array<ArgumentDesc, XlaSetDynamicDimensionSize::input_arg_count + XlaSetDynamicDimensionSize::output_arg_count> XlaSetDynamicDimensionSize::argument_descs;
constexpr std::array<AttributeDesc, 1> XlaSetDynamicDimensionSize::attribute_descs;

constexpr std::array<ArgumentDesc, XlaRemoveDynamicDimensionSize::input_arg_count + XlaRemoveDynamicDimensionSize::output_arg_count> XlaRemoveDynamicDimensionSize::argument_descs;
constexpr std::array<AttributeDesc, 1> XlaRemoveDynamicDimensionSize::attribute_descs;

constexpr std::array<ArgumentDesc, XlaDynamicUpdateSlice::input_arg_count + XlaDynamicUpdateSlice::output_arg_count> XlaDynamicUpdateSlice::argument_descs;
constexpr std::array<AttributeDesc, 2> XlaDynamicUpdateSlice::attribute_descs;

constexpr std::array<ArgumentDesc, XlaReduceWindow::input_arg_count + XlaReduceWindow::output_arg_count> XlaReduceWindow::argument_descs;
constexpr std::array<AttributeDesc, 3> XlaReduceWindow::attribute_descs;

constexpr std::array<ArgumentDesc, XlaSend::input_arg_count + XlaSend::output_arg_count> XlaSend::argument_descs;
constexpr std::array<AttributeDesc, 2> XlaSend::attribute_descs;

constexpr std::array<ArgumentDesc, XlaSort::input_arg_count + XlaSort::output_arg_count> XlaSort::argument_descs;
constexpr std::array<AttributeDesc, 1> XlaSort::attribute_descs;

constexpr std::array<ArgumentDesc, XlaVariadicSort::input_arg_count + XlaVariadicSort::output_arg_count> XlaVariadicSort::argument_descs;
constexpr std::array<AttributeDesc, 3> XlaVariadicSort::attribute_descs;

constexpr std::array<ArgumentDesc, XlaGather::input_arg_count + XlaGather::output_arg_count> XlaGather::argument_descs;
constexpr std::array<AttributeDesc, 4> XlaGather::attribute_descs;

constexpr std::array<ArgumentDesc, XlaScatter::input_arg_count + XlaScatter::output_arg_count> XlaScatter::argument_descs;
constexpr std::array<AttributeDesc, 5> XlaScatter::attribute_descs;

constexpr std::array<ArgumentDesc, XlaAllReduce::input_arg_count + XlaAllReduce::output_arg_count> XlaAllReduce::argument_descs;
constexpr std::array<AttributeDesc, 3> XlaAllReduce::attribute_descs;

constexpr std::array<ArgumentDesc, XlaReduceScatter::input_arg_count + XlaReduceScatter::output_arg_count> XlaReduceScatter::argument_descs;
constexpr std::array<AttributeDesc, 2> XlaReduceScatter::attribute_descs;

constexpr std::array<ArgumentDesc, _Arg::input_arg_count + _Arg::output_arg_count> _Arg::argument_descs;
constexpr std::array<AttributeDesc, 2> _Arg::attribute_descs;

constexpr std::array<ArgumentDesc, _DeviceArg::input_arg_count + _DeviceArg::output_arg_count> _DeviceArg::argument_descs;
constexpr std::array<AttributeDesc, 2> _DeviceArg::attribute_descs;

constexpr std::array<ArgumentDesc, _DeviceRetval::input_arg_count + _DeviceRetval::output_arg_count> _DeviceRetval::argument_descs;
constexpr std::array<AttributeDesc, 2> _DeviceRetval::attribute_descs;

constexpr std::array<ArgumentDesc, SymbolicGradient::input_arg_count + SymbolicGradient::output_arg_count> SymbolicGradient::argument_descs;
constexpr std::array<AttributeDesc, 3> SymbolicGradient::attribute_descs;

constexpr std::array<ArgumentDesc, Case::input_arg_count + Case::output_arg_count> Case::argument_descs;
constexpr std::array<AttributeDesc, 4> Case::attribute_descs;

constexpr std::array<ArgumentDesc, _While::input_arg_count + _While::output_arg_count> _While::argument_descs;
constexpr std::array<AttributeDesc, 3> _While::attribute_descs;

constexpr std::array<ArgumentDesc, StatefulPartitionedCall::input_arg_count + StatefulPartitionedCall::output_arg_count> StatefulPartitionedCall::argument_descs;
constexpr std::array<AttributeDesc, 6> StatefulPartitionedCall::attribute_descs;

constexpr std::array<ArgumentDesc, FakeParam::input_arg_count + FakeParam::output_arg_count> FakeParam::argument_descs;
constexpr std::array<AttributeDesc, 2> FakeParam::attribute_descs;

constexpr std::array<ArgumentDesc, DeviceIndex::input_arg_count + DeviceIndex::output_arg_count> DeviceIndex::argument_descs;
constexpr std::array<AttributeDesc, 1> DeviceIndex::attribute_descs;

constexpr std::array<ArgumentDesc, AudioMicrofrontend::input_arg_count + AudioMicrofrontend::output_arg_count> AudioMicrofrontend::argument_descs;
constexpr std::array<AttributeDesc, 22> AudioMicrofrontend::attribute_descs;
} // namespace tfdml
} // namespace ops
