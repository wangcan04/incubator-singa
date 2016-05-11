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


#include <glog/logging.h>
#include <algorithm>
#include "singa/neuralnet/loss_layer.h"
#include "singa/utils/math_blob.h"
#include "singa/utils/confusion.h"

namespace singa {

using std::vector;

void LineXLossLayer::Setup(const LayerProto& proto,
    const vector<Layer*>& srclayers) {
  CHECK_EQ(srclayers.size(), 2);
  LossLayer::Setup(proto, srclayers);
  const auto& src = srclayers[0]->data(this);
  batchsize_ = src.shape()[0];
  CHECK_EQ(src.count(), batchsize_);
  alpha_ = proto.linex_conf().alpha();
  confusion_ = new ConfusionMatrix(2);
}

void LineXLossLayer::ComputeFeature(int flag,
    const vector<Layer*>& srclayers) {
  const auto& label = srclayers[1]->aux_data(this);
  const float* score = srclayers[0]->data(this).cpu_data();
  float loss = 0, precision = 0;
  for (int n = 0; n < batchsize_; n++) {
    int ilabel = static_cast<int>(label[n]);
    //  CHECK_LT(ilabel,10);
    CHECK_GE(ilabel, 0);
    if (ilabel == 0 && score[n] >= 0) {
      loss += exp(score[n] + 1) - score[n] -2;
    } else if (ilabel == 1 && score[n] <= 0) {
      loss += - score[n] + exp(score[n] - 1);
    } else {
      precision++;
    }
    confusion_->Add(ilabel, score[n]>=0);
  }
  loss_ += loss / (1.0f * batchsize_);
  accuracy_ += precision / (1.0f * batchsize_);
  counter_++;
}

void LineXLossLayer::ComputeGradient(int flag,
    const vector<Layer*>& srclayers) {
  const auto& label = srclayers[1]->aux_data(this);
  const float* score = srclayers[0]->data(this).cpu_data();
  float* gscore = srclayers[0]->mutable_grad(this)->mutable_cpu_data();
  for (int n = 0; n < batchsize_; n++) {
    int ilabel = static_cast<int>(label[n]);
    if (score[n] * (ilabel * 2 -1) > 1) {
      gscore[n] = 0;
    } else if (ilabel == 0) {
      gscore[n] = (exp(score[n]+1) -1)/batchsize_;
    } else {
      gscore[n] = (exp(score[n]-1) -1)/batchsize_;
    }
  }
}

const std::string LineXLossLayer::ToString(bool debug, int flag) {
  if (debug)
    return Layer::ToString(debug, flag);

  string disp = "Loss = " + std::to_string(loss_ / counter_)
    + ", accuracy = " + std::to_string(accuracy_ / counter_);
  counter_ = 0;
  loss_ = accuracy_ = 0;

  if (flag == kTest){
    disp = disp + "\n" + confusion_->ToString();
    confusion_->Reset();
  }
  return disp;
}
}  // namespace singa
