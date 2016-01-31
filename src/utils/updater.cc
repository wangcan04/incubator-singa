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

#include "singa/utils/updater.h"

#include "mshadow/cxxnet_op.h"
#include "mshadow/tensor.h"
#include "singa/utils/singleton.h"
#include "singa/utils/factory.h"
#include "singa/utils/math_blob.h"
#include <chrono>

namespace singa {

using mshadow::cpu;
using mshadow::expr::F;
using mshadow::op::sqrtop;
using mshadow::op::square;
using mshadow::Shape;
using mshadow::Shape1;
using mshadow::Tensor;
using mshadow::TensorContainer;

ms update_time;
using get_time = std::chrono::steady_clock;


LRGenerator* LRGenerator::Create(const LRGenProto& proto) {
  auto factory = Singleton<Factory<LRGenerator>>::Instance();
  LRGenerator* gen = nullptr;
  if (proto.has_user_type())
    gen = factory->Create(proto.user_type());
  else
    gen = factory->Create(proto.type());
  gen->Init(proto);
  return gen;
}

float FixedStepLRGen::Get(int step) {
  if (last_idx_ < proto_.fixedstep_conf().step_size() - 1
      && step >= proto_.fixedstep_conf().step(last_idx_ + 1)) {
      last_idx_++;
    }
  return proto_.fixedstep_conf().step_lr(last_idx_);
}

float StepLRGen::Get(int step) {
  // do not cast int to float
  int freq = proto_.step_conf().change_freq();
  float lr = proto_.base_lr() * pow(proto_.step_conf().gamma(), step / freq);
  LOG_IF(INFO, step % freq == 0) << "Update learning rate to " << lr
    << " @ step " << step;
  return lr;
}

float LinearLRGen::Get(int step) {
  int freq = proto_.linear_conf().change_freq();
  float r = step * 1.0 / freq;
  return (1.0 - r) * proto_.base_lr() + r * proto_.linear_conf().final_lr();
}

float ExpLRGen::Get(int step) {
  int freq = proto_.exponential_conf().change_freq();
  return proto_.base_lr() / pow(2, step * 1. / freq);
}

float InvLRGen::Get(int step) {
  return proto_.base_lr() * pow(1.f + proto_.inverse_conf().gamma() * step,
           - proto_.inverse_conf().pow());
}

float InvTLRGen::Get(int step) {
  return proto_.base_lr() / (1 + step * 1. / proto_.inverset_conf().final_lr());
}

Updater* Updater::Create(const UpdaterProto& proto) {
  auto factory = Singleton<Factory<Updater>>::Instance();
  Updater* updater = nullptr;
  if (proto.has_user_type())
    updater = factory->Create(proto.user_type());
  else
    updater = factory->Create(proto.type());
  updater->Init(proto);
  return updater;
}

/***********************SGD with momentum******************************/
void Updater::Init(const UpdaterProto& proto) {
  momentum_ = proto.momentum();
  weight_decay_ = proto.weight_decay();
  lr_gen_ = LRGenerator::Create(proto.learning_rate());
  clip_low_ = proto.clip_low();
  clip_high_ = proto.clip_high();
}

void Updater::Clip(const float low, const float high, Param* param) {
  Blob<float>* grad = param->mutable_grad();
  float* ptr = grad->mutable_cpu_data();
  for (int i = 0; i < grad->count(); i++) {
    if (ptr[i] > high)
      ptr[i] = high;
    else if (ptr[i] < low)
      ptr[i] = low;
  }
}

void SGDUpdater::Update(int step, Param* param, float grad_scale) {
  if (clip_high_ > clip_low_)
    Clip(clip_low_, clip_high_, param);
  auto start_tick = get_time::now();
  CHECK(!param->grad().HeadAtGPU());
  CHECK(!param->data().HeadAtGPU());

  float lr = lr_gen_->Get(step) * param->lr_scale();
  float wd = weight_decay_ * param->wd_scale();
  if (grad_scale != 1.f)
    Scale(grad_scale, param->mutable_grad());
  if (wd > 0)  // L2 regularization, should be done after timing grad_scale
    AXPY(wd, param->data(), param->mutable_grad());
  if (momentum_ > 0) {
    Scale(momentum_, param->mutable_history());
    AXPY(-lr, param->grad(), param->mutable_history());
    AXPY(1.0f, param->history(), param->mutable_data());
  } else {
    AXPY(-lr, param->grad(), param->mutable_data());
  }
  update_time += std::chrono::duration_cast<ms>(get_time::now() - start_tick);
}

/***********************Nesterov******************************/
void NesterovUpdater::Update(int step, Param* param, float grad_scale) {
 if (clip_high_ > clip_low_)
    Clip(clip_low_, clip_high_, param);

  Shape<1> s = Shape1(param->size());
  Tensor<cpu, 1> data(param->mutable_cpu_data(), s);
  Tensor<cpu, 1> grad(param->mutable_cpu_grad(), s);
  Tensor<cpu, 1> history(param->mutable_cpu_history(), s);
  TensorContainer<cpu, 1> tmp(s);
  float lr = lr_gen_->Get(step)*param->lr_scale();
  float wd = weight_decay_*param->wd_scale();
  if (grad_scale != 1.f)
    grad *= grad_scale;
  if (wd > 0)  // L2 regularization, should be done after timing grad_scale
    grad += data * wd;
  Copy(tmp, history);
  history = history * momentum_ + lr * grad;
  tmp = history * (1 + momentum_) - tmp * momentum_;
  data -= tmp;
}
/***********************AdaGrad******************************/
void AdaGradUpdater::Update(int step, Param* param, float grad_scale) {
  if (clip_high_ > clip_low_)
    Clip(clip_low_, clip_high_, param);
  Shape<1> s = Shape1(param->size());
  Tensor<cpu, 1> data(param->mutable_cpu_data(), s);
  Tensor<cpu, 1> grad(param->mutable_cpu_grad(), s);
  Tensor<cpu, 1> history(param->mutable_cpu_history(), s);
  float lr = lr_gen_->Get(step)*param->lr_scale();
  float wd = weight_decay_*param->wd_scale();
  if (grad_scale != 1.f)
    grad *= grad_scale;
  if (wd > 0)  //  L2 regularization, should be done after timing grad_scale
    grad += data * wd;
  history += F<square>(grad);
  data -= lr * grad / (F<sqrtop>(history, proto_.delta()));
}

/***********************RMSProp******************************/
void RMSPropUpdater::Init(const UpdaterProto& proto) {
  Updater::Init(proto);
  rho_ = proto.rmsprop_conf().rho();
}

void RMSPropUpdater::Update(int step, Param* param, float grad_scale) {
 if (clip_high_ > clip_low_)
    Clip(clip_low_, clip_high_, param);

  Shape<1> s=Shape1(param->size());
  Tensor<cpu, 1> data(param->mutable_cpu_data(), s);
  Tensor<cpu, 1> grad(param->mutable_cpu_grad(), s);
  Tensor<cpu, 1> history(param->mutable_cpu_history(), s);
  float lr = lr_gen_->Get(step) * param->lr_scale();
  float wd = weight_decay_ * param->wd_scale();
  if (grad_scale != 1.f)
    grad *= grad_scale;
  if (wd > 0)  //  L2 regularization, should be done after timing grad_scale
    grad += data * wd;
  history = history * rho_ + (1 - rho_) * F<square>(grad);
  data -= lr * grad / (F<sqrtop>(history, proto_.delta()));
}
/***********************AdaDelta******************************
void AdaDeltaUpdater::Init(const UpdaterProto& proto){
  Updater::Init(proto);
  delta_=proto.delta();
  rho_=proto.rho();
  weight_decay_=proto.weight_decay();
}

void AdaDeltaUpdater::Update(int step, Param* param, float grad_scale){
  Shape<1> s=Shape1(param->size());
  Tensor<cpu, 1> data(param->mutable_cpu_data(), s);
  Tensor<cpu, 1> grad(param->mutable_cpu_grad(), s);
  Tensor<cpu, 1> history(param->mutable_cpu_history(), s);
  Tensor<cpu, 1> update(param->mutable_cpu_update(), s);
  TensorContainer<cpu, 1> tmp(s);
  float wd=weight_decay_*param->wd_scale();
  if(wd>0){ // L2 regularization
    grad+=data*wd;
  }
  if(step==0){
    history=0;
    update=0;
  }
  history=history*rho_+(1-rho_)*F<op::square>(grad*grad_scale);
  tmp=grad*F<op::sqrtop>(update, delta_)/F<op::sqrtop>(history, delta_);
  update=rho_*update+(1-rho_)*F<op::square>(tmp);
  data-=tmp;
}
*/

}  // namespace singa
