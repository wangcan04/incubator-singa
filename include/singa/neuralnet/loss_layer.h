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

#ifndef SINGA_NEURALNET_LOSS_LAYER_H_
#define SINGA_NEURALNET_LOSS_LAYER_H_

#include <vector>
#include <string>
#include "singa/neuralnet/layer.h"
#include "singa/neuralnet/neuron_layer.h"

namespace singa {
using std::vector;
class ConfusionMatrix;

/**
 * Linear Exponential Loss for binary classification
 *               e^(y+1) - y -2  if t = 0, y>-1
 * loss(t, y) =  -y+e^(y-1)     if t = 1, y<1
 *               0               else
 */
class LineXLossLayer : public LossLayer {
 public:
   void Setup(const LayerProto& conf, const vector<Layer*>& srclayers) override;
  void ComputeFeature(int flag, const vector<Layer*>& srclayers) override;
  void ComputeGradient(int flag, const vector<Layer*>& srclayers) override;
  const std::string ToString(bool debug, int flag) override;

 private:
  int batchsize_, counter_ = 0;
  float loss_ = 0, accuracy_ = 0;
  float alpha_;
  ConfusionMatrix *confusion_;
};
/**
 * Squared Euclidean loss as @f$0.5 ||p - t||^2@f$, where p is prediction
 * result, t is the ground truth.
 */
class EuclideanLossLayer : public LossLayer {
 public:
  void Setup(const LayerProto& conf, const vector<Layer*>& srclayers) override;
  void ComputeFeature(int flag, const vector<Layer*>& srclayers) override;
  void ComputeGradient(int flag, const vector<Layer*>& srclayers) override;
  const std::string ToString(bool debug, int flag) override;

 private:
  int counter_ = 0;
  float loss_ = 0.0f;
};
/**
 * Cross-entropy loss applied to the probabilities computed from Softmax.
 * @f$ L_i = -log P_{t_i}, t_i\in [0, C] @f$ is the label for the i-th object,
 * C is the total number of classes.
 */
class SoftmaxLossLayer : public LossLayer {
 public:
  void Setup(const LayerProto& conf, const vector<Layer*>& srclayers) override;
  void ComputeFeature(int flag, const vector<Layer*>& srclayers) override;
  void ComputeGradient(int flag, const vector<Layer*>& srclayers) override;
  const std::string ToString(bool debug, int flag) override;

 private:
  int batchsize_, topk_, dim_, counter_ = 0;
  float scale_;
  float loss_ = 0.0f, accuracy_ = 0.0f;
  float current_precision_ = 1.0f, current_recall_ = 1.0f;
  ConfusionMatrix *confusion_;
};

#ifdef USE_CUDNN
class CudnnSoftmaxLossLayer : public LossLayer{
 public:
  void Setup(const LayerProto& conf, const vector<Layer*>& srclayers) override;
  void ComputeFeature(int flag, const vector<Layer*>& srclayers) override;
  void ComputeGradient(int flag, const vector<Layer*>& srclayers) override;
  const std::string ToString(bool debug, int flag) override;

 private:
  int batchsize_, dim_;
  int counter_ = 0;
  float loss_ = 0.0f;

  CudnnSoftmaxLayer softmax_;
};
#endif
}  // namespace singa

#endif  // SINGA_NEURALNET_LOSS_LAYER_H_
