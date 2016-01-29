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

#include "singa/worker.h"

#include <glog/logging.h>
#include <chrono>
#include <thread>
#include <typeinfo>
#include "singa/utils/cluster.h"
#include "singa/utils/factory.h"
#include "singa/utils/singleton.h"
#include "singa/utils/context.h"
#include "singa/utils/math_blob.h"
#include "singa/server.h"

int kQueueSize=5;
namespace singa {
using ms = std::chrono::microseconds;
using mms = std::chrono::milliseconds;
using get_time = std::chrono::steady_clock;
ms fp_time, bp_time;
using std::string;
Worker* Worker::Create(const AlgProto& conf) {
  auto factory = Singleton<Factory<singa::Worker>>::Instance();
  Worker* worker = nullptr;
  if (conf.has_user_alg())
    worker = factory->Create(conf.user_alg());
  else
    worker = factory->Create(conf.alg());
  return worker;
}
void CUDART_CB Worker::Dev2Host(cudaStream_t stream,
    cudaError_t status, void *userData) {
  auto event = static_cast<CopyEvent*>(userData);
  CHECK(!event->host2dev);
  auto param = event->param;
  param->mutable_grad()->SyncHead();
  auto dealer = Singleton<Context>::Instance()->driver_dealer();
  Worker::SendUpdateMsg(param, event->worker, dealer);
  event->worker->DecToCPU();
  delete event;
  // LOG(ERROR) << "callback d2h";
}
void CUDART_CB Worker::Host2Dev(cudaStream_t stream,
    cudaError_t status, void *userData) {
  auto event = static_cast<CopyEvent*>(userData);
  CHECK(event->host2dev);
  auto *param = event->param;
  // LOG(ERROR) << "callback h2d, param " << param->id();
  param->mutable_data()->SyncHead();
  param->mutable_grad()->SyncHead();
  param->set_version(event->param_version);
  event->worker->DecToGPU();
  delete event;
  // LOG(ERROR) << "callback h2d, param " << param->id();
}

void Worker::Setup(int grp_id, int id, const JobProto& conf,
    NeuralNet* train_net, NeuralNet* val_net, NeuralNet* test_net) {
  grp_id_ = grp_id;
  id_ = id;
  job_conf_ = conf;
  train_net_ = train_net;
  val_net_ = val_net;
  test_net_ = test_net;
  bridge_dealer_ = dealer_ = nullptr;
  // updater_ = Updater::Create(conf.updater());
}

Worker::~Worker() {
  if (updater_) delete updater_;
  if (dealer_) delete dealer_;
  if (bridge_dealer_) delete bridge_dealer_;
}

void Worker::Run() {
  std::this_thread::sleep_for(std::chrono::milliseconds(1000));
  to_gpu=to_cpu=0;
  // setup gpu device
  auto context = Singleton<Context>::Instance();
  int device = context->device_id(std::this_thread::get_id());
  LOG(ERROR) << "Worker (group = " << grp_id_ <<", id = " << id_ << ") "
    << " start on " << (device >= 0 ? "GPU " + std::to_string(device) : "CPU");
  if (device >= 0) {
    context->ActivateDevice(device);
    CHECK_EQ(cudaStreamCreateWithFlags(
            &up_stream_, cudaStreamNonBlocking), cudaSuccess);
     CHECK_EQ(cudaStreamCreateWithFlags(
            &down_stream_, cudaStreamNonBlocking), cudaSuccess);
  }

  auto cluster = Cluster::Get();
  int svr_grp = grp_id_ / cluster->nworker_groups_per_server_group();
  // CHECK(cluster->runtime()->JoinSGroup(grp_id_, id_, svr_grp));
  step_ = job_conf_.step();
  // LOG(ERROR) << "worker initing sockets";
  InitSockets(train_net_);
  // LOG(ERROR) << "worker init sockets";
  InitNetParams(job_conf_, train_net_);
  // LOG(ERROR) << "worker bp";

  auto start_tick = get_time::now();
  LOG(ERROR) << "queue size " << kQueueSize;
  while (!StopNow(step_)) {
    if (ValidateNow(step_) && val_net_ != nullptr) {
      CollectAll(step_, train_net_);
      LOG(ERROR) << "Validation @ step " + std::to_string(step_);
      Test(job_conf_.validate_steps(), kVal, val_net_);
    }
    if (TestNow(step_) && test_net_ != nullptr) {
      CollectAll(step_, train_net_);
      LOG(ERROR) << "Test @ step " + std::to_string(step_);
      Test(job_conf_.test_steps(), kTest, test_net_);
    }
    if (CheckpointNow(step_) && grp_id_ == 0) {
      CollectAll(step_, train_net_);
      Checkpoint(step_, Cluster::Get()->checkpoint_folder(), train_net_);
      job_conf_.set_step(step_);
    }
    if (step_ == 5)
      start_tick = get_time::now();
    TrainOneBatch(step_, train_net_);
    if (DisplayNow(step_) && grp_id_ == 0 && id_ == 0) {
      Display(kTrain | kForward | kBackward,
          "Train @ step " + std::to_string(step_), train_net_);
    }
    step_++;
  }
  int count = step_ - 5;
  LOG(ERROR) << "Time per iteration "
        << std::chrono::duration_cast<ms>(get_time::now() - start_tick).count() / count << " ms";
  LOG(ERROR) << "Time for to cpu "
        << std::chrono::duration_cast<ms>(to_cpu_time).count() / step_ << " ms";
  LOG(ERROR) << "Time for to gpu "
        << std::chrono::duration_cast<ms>(to_gpu_time).count() / step_ << " ms";
  LOG(ERROR) << "Time for upate "
        << std::chrono::duration_cast<ms>(update_time).count() / step_ << " ms";
  LOG(ERROR) << "Time for forward "
        << std::chrono::duration_cast<ms>(fp_time).count() / step_ << " ms";
  LOG(ERROR) << "Time for backward "
        << std::chrono::duration_cast<ms>(bp_time).count() / step_ << " ms";

  // save the model
  if (grp_id_ == 0)
    Checkpoint(step_, Cluster::Get()->checkpoint_folder(), train_net_);
  // clean up
  // cluster->runtime()->LeaveSGroup(grp_id_, id_, svr_grp);
  // notify the stub on worker stop
  Msg* msg = new Msg(Addr(grp_id_, id_, kWorkerParam), Addr(-1, -1, kStub));
  msg->set_type(kStop);
  dealer_->Send(&msg);  // use param dealer to send the stop msg
  LOG(ERROR) << "Worker (group = " <<grp_id_ << ", id = " << id_ << ") stops";
}

void Worker::Test(int steps, Phase phase, NeuralNet* net) {
  for (int step = 0; step < steps; step++)
    TestOneBatch(step, phase, net);
  Display(phase, " ", net);
}

void ConnectStub(int grp, int id, Dealer* dealer, EntityType entity) {
  dealer->Connect(kInprocRouterEndpoint);
  Msg* ping = new Msg(Addr(grp, id, entity), Addr(-1, -1, kStub));
  ping->set_type(kConnect);
  dealer->Send(&ping);
}

void Worker::InitSockets(const NeuralNet* net) {
  // TODO(wangsh): provide a unique sock id from cluster
  dealer_ = new Dealer(0);
  ConnectStub(grp_id_, id_, dealer_, kWorkerParam);
  for (auto layer : net->layers()) {
    if (layer->partition_id() == id_) {
      if (typeid(*layer) == typeid(BridgeDstLayer)
          || typeid(*layer) == typeid(BridgeSrcLayer)) {
        // TODO(wangsh): provide a unique socket id from cluster
        bridge_dealer_ = new Dealer(1);
        ConnectStub(grp_id_, id_, bridge_dealer_, kWorkerLayer);
        break;
      }
    }
  }
  // bind dealer to bridge layers
  if (bridge_dealer_ != nullptr) {
    for (auto dst : net->layers()) {
      if (typeid(*dst) == typeid(BridgeDstLayer)) {
        auto src = net->srclayers(dst)[0];
        name2bridge_[src->name()] = src;
        name2bridge_[dst->name()] = dst;
        if (src->partition_id() == id_) {
          dynamic_cast<BridgeLayer*>(src)->MakePaired(dst, grp_id_,
              bridge_dealer_, &name2bridge_);
        }
        if (dst->partition_id() == id_) {
          dynamic_cast<BridgeLayer*>(dst)->MakePaired(src, grp_id_,
              bridge_dealer_, &name2bridge_);
        }
      }
    }
  }
}

void Worker::InitNetParams(const JobProto& job_conf, NeuralNet* net) {
  // for each server grp, its first subscriber worker grp does the param init
  if (grp_id_ % Cluster::Get()->nworker_groups_per_server_group() == 0) {
    // extract params that should be initialized by this worker
    // must gen a name for each param if the user doesn't config it
    std::unordered_map<string, Param*> name2param;
    for (auto layer : net->layers()) {
      if (layer->partition_id() == id_) {
        for (auto param : layer->GetParams()) {
          // only owners fill the memory of parameter values.
          if (param->owner() == param->id()) {
            CHECK(name2param.find(param->name()) == name2param.end());
            name2param[param->name()] = param;
          }
        }
      }
    }
    vector<string> paths;
    for (const auto& p : job_conf_.checkpoint_path())
      paths.push_back(p);
    net->Load(paths, name2param);
    // init other params who do not have checkpoint version
    for (auto entry : name2param) {
      if (entry.second->version() > 0) {
        //  if load from pre-training params, reset version to start step
        if (job_conf.reset_param_version()) {
          entry.second->set_version(job_conf.step());
        }
      } else {
        entry.second->InitValues(job_conf.step());
        if (!job_conf.reset_param_version())
          LOG(ERROR) << "better reset version of params from checkpoints "
            << "to the same as other newly initialized params!";
      }
    }

    // warmup training before put params to servers
    // for (; step_ < job_conf.warmup_steps(); step_++)
    //  TrainOneBatch(step_, net);
    for (auto layer : net->layers()) {
      if (layer->partition_id() == id_)
        for (auto param : layer->GetParams())
          if (param->owner() == param->id())
            Put(param->version(), param);
    }
  }
  // wait owners in the same procs init params, then no get requests sent
  std::this_thread::sleep_for(std::chrono::milliseconds(1000));
  for (auto layer : net->layers()) {
    if (layer->partition_id() == id_)
      for (auto param : layer->GetParams())
        Get(job_conf.warmup_steps(), param);
  }
}

void Worker::Checkpoint(int step, const std::string& folder, NeuralNet* net) {
  BlobProtos bps;
  for (auto layer : net->layers()) {
    if (layer->partition_id() == id_) {
      for (auto param : layer->GetParams()) {
        // only owners fill the memory of parameter values.
        if (param->owner() == param->id()) {
          auto *blob = bps.add_blob();
          param->ToProto(blob);
          bps.add_version(param->version());
          bps.add_name(param->name());
        }
      }
    }
  }
  char buf[256];
  snprintf(buf, sizeof(buf), "%s/step%d-worker%d", folder.c_str(), step, id_);
  LOG(INFO) << "checkpoint to " << buf;
  WriteProtoToBinaryFile(bps, buf);
}

int Worker::Put(int step, Param* param) {
  if (dealer_ == nullptr) {
    LOG(WARNING) << "Null dealer in worker (" << grp_id_ << ", " << id_ << ")";
    return 1;
  }
  // set Blob head to cpu to avoid calling cudaMemcpy by the stub thread, which
  // would hang on some machines.
  param->data().cpu_data();
  Msg* msg = new Msg(Addr(grp_id_, id_, kWorkerParam), Addr(-1, -1, kStub));
  msg->set_trgt(ParamTrgt(param->owner(), 0), step);
  msg->set_type(kPut);
  dealer_->Send(&msg);
  return 1;
}

int Worker::Get(int step, Param* param) {
  if (param->version() >= step)
    return 1;
  if (dealer_ == nullptr) {
    LOG(WARNING) << "Null dealer in worker (" << grp_id_ << ", " << id_ << ")";
    return 1;
  }
  // set Blob head to cpu to avoid calling cudaMemcpy by the stub thread, which
  // would hang on some machines.
  param->mutable_data()->mutable_cpu_data();

  Msg* msg = new Msg(Addr(grp_id_, id_, kWorkerParam), Addr(-1, -1, kStub));
  msg->set_trgt(ParamTrgt(param->owner(), 0), step);
  msg->set_type(kGet);
  dealer_->Send(&msg);
  return 1;
}

int Worker::Update(int step, Param* param) {
  param->set_last_version(param->version());
  if (param->grad().HeadAtGPU()) {
    // LOG(ERROR) << "update param " << param->id();
    param->mutable_grad()->CopyToCPUAsync(down_stream_, Worker::Dev2Host,
        new CopyEvent(param, this, false));
    IncToCPU();
    return 1;
  } else {
    // head of data Blob (SyncMem) to cpu, because the stub thread may use
    // cudaMemcpy copy gradients into msgs. cudaMemcpy hangs when called by the
    // stub thread on some GPU machines.
    // TODO(wangwei) fix this issue and remove the following line.
    // optimize for training with single worker by removing stub and server, and
    // updating parameters locally inside the worker GPU. Then we do not need to
    // transfer gradients and parameter values between GPU-CPU.
    param->grad().cpu_data();
    // change the head of SyncMem to cpu; otherwise, the updated parameter
    // values would not be synced to gpu (since the head is at gpu).
    param->mutable_data()->mutable_cpu_data();
  }
  if (dealer_ == nullptr) {
    LOG(WARNING) << "Null dealer in worker (" << grp_id_ << ", " << id_ << ")";
    return 1;
  }
  SendUpdateMsg(param, this, dealer_);
  return 1;
}


void Worker::SendUpdateMsg(Param* param, Worker* worker,
    Dealer* dealer) {
  Msg* msg = new Msg(Addr(worker->grp_id(), worker->id(), kWorkerParam),
      Addr(-1, -1, kStub));
  msg->set_trgt(ParamTrgt(param->owner(), 0), worker->step());
  msg->set_type(kUpdate);
  dealer->Send(&msg);
}

int Worker::CollectAll(int step, NeuralNet* net) {
  auto& layers = net->layers();
  for (auto& layer : layers) {
    if (layer->partition_id() == id_) {
      for (Param* p : layer->GetParams()) {
        Collect(step, p);
      }
    }
  }
  return 1;
}

int Worker::Collect(int step, Param* param) {
  CopyToGPUAsync();
  int k = 0;
  while (param->version() <= param->last_version()) {
    std::this_thread::sleep_for(std::chrono::milliseconds(kCollectSleepTime));
    CopyToGPUAsync();
    k++;
    // LOG(ERROR) << "wait  "<< param->id() << " at " << step << " by " <<id_;
  }
  if (k)
    LOG(ERROR) << "param id="<<param->id()<<" wait times " << k;
  return 1;
}

void Worker::Display(int flag, const std::string& prefix, NeuralNet* net) {
  for (auto layer : net->layers()) {
    if (layer->partition_id() == id_) {
      const string& disp = layer->ToString(false, flag);
      if (disp.length())
        LOG(ERROR) << prefix << "  " << disp;
    }
  }
}

void Worker::CopyToGPUAsync() {
  int n = 0;
  while(n < kQueueSize) {
    auto event = new CopyEvent();
    if (!copy_queue_.try_pop(*event))
      break;
    CHECK(event->param != nullptr);
    CHECK(event->param->data().count()>0);
    // LOG(ERROR) << "dequeue param " << event->param->id(); // event->param->version()<< " " << event->param->size()<< " " << event->worker->step() << event->param_version;
    // copy param values
    event->param->mutable_data()->CopyToGPUAsync(up_stream_, Worker::Host2Dev,
        event);
    // LOG(ERROR) << "Host to Dev, param=" << event->param->id() << " to_gpu=" << to_gpu.load() << " to_cpu="<<to_cpu.load();
    IncToGPU();
    n++;
    // TODO copy layer value/gradient
  }
  // if (n > 0) LOG(ERROR) << "Inserted " << n << " fetches";
}

/****************************BPWorker**********************************/
void BPWorker::TrainOneBatch(int step, NeuralNet* net) {
  // LOG(ERROR) << "====================================";
  auto start_tick = get_time::now();
  Forward(step, kTrain, net);
  // LOG(ERROR) << "---------------------------------";
  auto tick = get_time::now();
  fp_time +=std::chrono::duration_cast<ms>(tick - start_tick);
  Backward(step, net);
  // LOG(ERROR) << "bp end";
  bp_time +=std::chrono::duration_cast<ms>(get_time::now() - tick);
}

void BPWorker::TestOneBatch(int step, Phase phase, NeuralNet* net) {
  Forward(step, phase, net);
}

void BPWorker::Forward(int step, Phase phase, NeuralNet* net) {
  map<string, string> label;
  for (auto& layer : net->layers()) {
    CopyToGPUAsync();
    if (layer->partition_id() == id_) {
      if (phase == kTrain && layer->unroll_index() == 0) {
        // wait until param is updated
        for (Param* p : layer->GetParams()) {
          Collect(step, p);
        }
      }
      // DLOG(ERROR) << "Forward " << layer->name();
      layer->ComputeFeature(phase | kForward, net->srclayers(layer));
      if (job_conf_.debug() && DisplayNow(step) && grp_id_ == 0)
        label[layer->name()] = layer->ToString(true, phase | kForward);
    }
  }
  if (label.size()) {
    const string path = Cluster::Get()->vis_folder() + "/fp-step"
      + std::to_string(step) +"-loc" + std::to_string(id_) + ".json";
    WriteStringToTextFile(path, net->ToGraph(false).ToJson(label));
  }
}

void BPWorker::Backward(int step, NeuralNet* net) {
  map<string, string> label;
  auto& layers = net->layers();
  for (auto it = layers.rbegin(); it != layers.rend(); it++) {
    CopyToGPUAsync();
    Layer* layer = *it;
    if (layer->partition_id() == id_) {
      layer->ComputeGradient(kTrain | kBackward, net->srclayers(layer));
      if (job_conf_.debug() && DisplayNow(step) && grp_id_ == 0)
        label[layer->name()] = layer->ToString(true, kTrain | kBackward);
      for (Param* p : layer->GetParams())
        Update(step, p);
    }
  }
  if (label.size()) {
    const string path = Cluster::Get()->vis_folder() + "/bp-step"
      + std::to_string(step) + "-loc" + std::to_string(id_) + ".json";
    WriteStringToTextFile(path, net->ToGraph(false).Reverse().ToJson(label));
  }
}

/***************************BPTTWorker*********************************/
void BPTTWorker::Forward(int step, Phase phase, NeuralNet* net) {
  map<string, string> label;
  for (auto& layer : net->layers()) {
    if (layer->partition_id() == id_) {
      if (phase == kTrain && layer->unroll_index() == 0) {
        // wait until param is updated
        for (Param* p : layer->GetParams()) {
          Collect(step, p);
          Zero(p->mutable_grad());
        }
      }
      vector<Layer*> src = net->srclayers(layer);
      if ((phase & kTest) && typeid(*layer) == typeid(RNNDummyLayer)) {
        CHECK_LE(src.size(), 1);
        auto dummy = dynamic_cast<RNNDummyLayer*>(layer);
        Layer* srclayer = net->name2layer(dummy->srclayer(step));
        if (step > 0)
          CHECK(srclayer != nullptr);
        if (srclayer != nullptr) {
          src.clear();
          src.push_back(srclayer);
        }
      }
      // if full state rnn and not the starting of a new passing of the dataset,
      // feed the hidden state of the last unit to the first unit.
      if (layer->unroll_index() == 0 && full_state_ && !begin_) {
        Layer* last = net->last_unroll_layer(layer);
        CHECK(last != nullptr);
        if (last != layer || (phase & kTest))
          src.push_back(last);
      }
      // LOG(ERROR) << layer->name() << " forward";
      // int ret =
      layer->ComputeFeature(phase | kForward, src);
      /*
      if ((phase & Phase::kTrain) && ret == Status::kEnd)
        begin_ = true;
      */
      if (job_conf_.debug() && DisplayNow(step) && grp_id_ == 0)
        label[layer->name()] = layer->ToString(true, phase | kForward);
    }
  }
  if (label.size()) {
    const string path = Cluster::Get()->vis_folder() + "/fp-step"
      + std::to_string(step) +"-loc" + std::to_string(id_) + ".json";
    WriteStringToTextFile(path, net->ToGraph(false).ToJson(label));
  }
}

void BPTTWorker::Backward(int step, NeuralNet* net) {
  map<string, string> label;
  auto& layers = net->layers();
  for (auto it = layers.rbegin(); it != layers.rend(); it++) {
    Layer* layer = *it;
    if (layer->partition_id() == id_) {
      layer->ComputeGradient(kTrain | kBackward | kAggGrad,
          net->srclayers(layer));
      // LOG(ERROR) << layer->name() << " backward";
      if (job_conf_.debug() && DisplayNow(step) && grp_id_ == 0)
        label[layer->name()] = layer->ToString(true, kTrain | kBackward);
      // unrolled layers share parameter data and grad, just update the 1st one
      if (layer->unroll_index() == 0)
        for (Param* p : layer->GetParams())
          Update(step, p);
    }
  }
  if (label.size()) {
    const string path = Cluster::Get()->vis_folder() + "/bp-step"
      + std::to_string(step) + "-loc" + std::to_string(id_) + ".json";
    WriteStringToTextFile(path, net->ToGraph(false).Reverse().ToJson(label));
  }
}
void BPTTWorker::Display(int flag, const std::string& prefix, NeuralNet* net) {
  std::unordered_map<string, float> perf;
  for (auto layer : net->layers()) {
    if (layer->partition_id() == id_) {
      const string& disp = layer->ToString(false, flag);
      for (const auto& entry : GetMetricFromString(disp))
        perf[entry.first] += entry.second;
    }
  }
  string disp = prefix + " ";
  for (const auto& entry : perf)
    disp += entry.first + " = " + std::to_string(entry.second) + ", ";
  LOG(ERROR) << disp;
}
/****************************CDWorker**********************************/
void CDWorker::TrainOneBatch(int step, NeuralNet* net) {
  const auto& layers = net->layers();
  for (auto* layer : layers) {
    for (Param* p : layer->GetParams())  // wait until param is updated
      Collect(step, p);
    layer->ComputeFeature(kPositive, net->srclayers(layer));
  }
  for (auto* layer : layers)
    if (typeid(*layer) == typeid(RBMVisLayer)
          || typeid(*layer) == typeid(RBMHidLayer))
      layer->ComputeFeature(kNegative | kTest, net->srclayers(layer));
  for (int i = 1; i < job_conf_.train_one_batch().cd_conf().cd_k(); i++) {
    for (auto* layer : layers) {
      if (typeid(*layer) == typeid(RBMVisLayer)
          || typeid(*layer) == typeid(RBMHidLayer))
      layer->ComputeFeature(kNegative, net->srclayers(layer));
    }
  }
  for (auto* layer : layers) {
    if (typeid(*layer) == typeid(RBMVisLayer)
        || typeid(*layer) == typeid(RBMHidLayer)) {
      layer->ComputeGradient(kTrain, net->srclayers(layer));
      for (Param* p : layer->GetParams()) {
        Update(step, p);
      }
    }
  }
}

void CDWorker::TestOneBatch(int step, Phase phase, NeuralNet* net) {
  auto& layers = net->layers();
  for (auto *layer : layers)
    layer->ComputeFeature(kPositive, net->srclayers(layer));
  for (auto *layer : layers)
    if (typeid(*layer) == typeid(RBMVisLayer))
      layer->ComputeFeature(kNegative | kTest, net->srclayers(layer));
}

}  // namespace singa
