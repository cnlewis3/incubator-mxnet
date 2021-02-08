/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

// mxnet.cc

#define MSHADOW_FORCE_STREAM

#ifndef MSHADOW_USE_CBLAS
#if (__MIN__ == 1)
#define MSHADOW_USE_CBLAS   0
#else
#define MSHADOW_USE_CBLAS   1
#endif
#endif

#define MSHADOW_USE_CUDA    0
#define MSHADOW_USE_MKL     0
#define MSHADOW_RABIT_PS    0
#define MSHADOW_DIST_PS     0
#define MSHADOW_INT64_TENSOR_SIZE 1

#if defined(__ANDROID__) || defined(__MXNET_JS__)
#define MSHADOW_USE_SSE         0
#endif

#define MXNET_USE_OPENCV    0
#define MXNET_PREDICT_ONLY  1
#define DISABLE_OPENMP 1
#define DMLC_LOG_STACK_TRACE 0

#include "src/libinfo.cc"
#include "src/common/utils.cc"

#include "src/ndarray/ndarray_function.cc"
#include "src/ndarray/ndarray.cc"

#include "src/imperative/imperative.cc"
#include "src/imperative/imperative_utils.cc"
#include "src/imperative/eliminate_common_expr_pass.cc"
#include "src/imperative/pointwise_fusion_pass.cc"
#include "src/imperative/cached_op_threadsafe.cc"
#include "src/imperative/cached_op.cc"
#include "src/imperative/attach_op_execs_pass.cc"
#include "src/imperative/attach_op_resource_pass.cc"
#include "src/imperative/inplace_addto_detect_pass.cc"

#include "src/engine/engine.cc"
#include "src/engine/naive_engine.cc"
#include "src/engine/openmp.cc"

#include "src/profiler/profiler.cc"
#include "src/profiler/aggregate_stats.cc"

#include "src/executor/graph_executor.cc"
#include "src/executor/infer_graph_attr_pass.cc"

#include "src/miniz/miniz.c"

#include "src/nnvm/legacy_json_util.cc"
#include "src/nnvm/legacy_op_util.cc"
#include "src/nnvm/graph_editor.cc"

#include "src/operator/operator.cc"
#include "src/operator/operator_util.cc"
#include "src/operator/leaky_relu.cc"
#include "src/operator/nn/activation.cc"
#include "src/operator/nn/batch_norm.cc"
#include "src/operator/nn/concat.cc"
#include "src/operator/nn/convolution.cc"
#include "src/operator/nn/deconvolution.cc"
#include "src/operator/nn/dropout.cc"
#include "src/operator/nn/fully_connected.cc"
#include "src/operator/nn/layer_norm.cc"
#include "src/operator/nn/pooling.cc"
#include "src/operator/nn/softmax_activation.cc"
#include "src/operator/nn/softmax.cc"
#include "src/operator/nn/log_softmax.cc"
#include "src/operator/numpy/np_elemwise_broadcast_op.cc"
#include "src/operator/numpy/np_elemwise_broadcast_logic_op.cc"
#include "src/operator/numpy/np_elemwise_unary_op_basic.cc"
#include "src/operator/numpy/np_matrix_op.cc"
#include "src/operator/numpy/np_true_divide.cc"
#include "src/operator/numpy/np_where_op.cc"
#include "src/operator/numpy/np_init_op.cc"
#include "src/operator/softmax_output.cc"
#include "src/operator/swapaxis.cc"
#include "src/operator/sequence_mask.cc"
#include "src/operator/tensor/elemwise_binary_broadcast_op_basic.cc"
#include "src/operator/tensor/elemwise_binary_op.cc"
#include "src/operator/tensor/elemwise_binary_op_basic.cc"
#include "src/operator/tensor/elemwise_binary_scalar_op_basic.cc"
#include "src/operator/tensor/elemwise_binary_broadcast_op_logic.cc"
#include "src/operator/tensor/elemwise_unary_op_basic.cc"
#include "src/operator/tensor/elemwise_unary_op_trig.cc"
#include "src/operator/tensor/matrix_op.cc"
#include "src/operator/tensor/indexing_op.cc"
#include "src/operator/tensor/init_op.cc"
#include "src/operator/tensor/dot.cc"
#include "src/operator/tensor/broadcast_reduce_op_value.cc"

#include "src/serialization/cnpy.cc"

#include "src/api/operator/ufunc_helper.cc"
#include "src/api/operator/utils.cc"

#include "src/storage/storage.cc"

#include "src/runtime/registry.cc"

#include "src/resource.cc"
#include "src/initialize.cc"

#include "src/c_api/c_predict_api.cc"
#include "src/c_api/c_api_symbolic.cc"
#include "src/c_api/c_api_ndarray.cc"
#include "src/c_api/c_api_error.cc"
#include "src/c_api/c_api_profile.cc"


