/************************************************************
*
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
*
*************************************************************/


#include "hidden_layer.h"

#include "mshadow/tensor.h"
#include "mshadow/cxxnet_op.h"

namespace singa {
using namespace mshadow;
using mshadow::cpu;
using mshadow::Shape1;
using mshadow::Shape2;

inline Tensor<cpu, 2> NewTensor2(Blob<float>* blob) {
  const std::vector<int>& shape = blob->shape();
  Tensor<cpu, 2> tensor(blob->mutable_cpu_data(),
      Shape2(shape[0], blob->count() / shape[0]));
  return tensor;
}

inline Tensor<cpu, 1> NewTensor1(Blob<float>* blob) {
  Tensor<cpu, 1> tensor(blob->mutable_cpu_data(), Shape1(blob->count()));
  return tensor;
}


HiddenLayer::~HiddenLayer() {
  delete weight_;
  delete bias_;
}

void HiddenLayer::Setup(const LayerProto& proto, int npartitions) {
  Layer::Setup(proto, npartitions);
  CHECK_EQ(srclayers_.size(), 1);
  const auto& src = srclayers_[0]->data(this);
  batchsize_ = src.shape()[0];
  vdim_ = src.count() / batchsize_;
  hdim_ = layer_proto_.GetExtension(hidden_conf).num_output();
  data_.Reshape(std::vector<int>{batchsize_, hdim_});
  grad_.ReshapeLike(data_);
  weight_ = Param::Create(proto.param(0));
  bias_ = Param::Create(proto.param(1));
  weight_->Setup(std::vector<int>{hdim_, vdim_});
  bias_->Setup(std::vector<int>{hdim_});
}

void HiddenLayer::ComputeFeature(int flag, Metric* perf) {
  auto data = NewTensor2(&data_);
  auto src = NewTensor2(srclayers_[0]->mutable_data(this));
  auto weight = NewTensor2(weight_->mutable_data());
  auto bias = NewTensor1(bias_->mutable_data());
  data = dot(src, weight.T());
  // repmat: repeat bias vector into batchsize rows
  data += expr::repmat(bias, batchsize_);
  data = expr::F<op::stanh>(data);
}

void HiddenLayer::ComputeGradient(int flag, Metric* perf) {
  auto data = NewTensor2(&data_);
  auto src = NewTensor2(srclayers_[0]->mutable_data(this));
  auto grad = NewTensor2(&grad_);
  auto weight = NewTensor2(weight_->mutable_data());
  auto gweight = NewTensor2(weight_->mutable_grad());
  auto gbias = NewTensor1(bias_->mutable_grad());

  grad = expr::F<op::stanh_grad>(data) * grad;
  gbias = expr::sum_rows(grad);
  gweight = dot(grad.T(), src);
  if (srclayers_[0]->mutable_grad(this) != nullptr) {
    auto gsrc = NewTensor2(srclayers_[0]->mutable_grad(this));
    gsrc = dot(grad, weight);
  }
}
}  // namespace singa
