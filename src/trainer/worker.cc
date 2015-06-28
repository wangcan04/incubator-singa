#include <glog/logging.h>
#include <thread>
#include <memory>
#include <iostream>
#include <chrono>
#include <thread>
#include "utils/singleton.h"
#include "utils/factory.h"
#include "utils/vis/tsne.h"
#include "trainer/worker.h"
#include "proto/model.pb.h"
#include "proto/common.pb.h"
#include "mshadow/tensor.h"
using namespace mshadow;
namespace singa {
using std::thread;
using mshadow::cpu;

Worker::Worker(int thread_id, int group_id, int worker_id):
  thread_id_(thread_id), group_id_(group_id), worker_id_(worker_id){
}

void Worker::Setup(const ModelProto& model,
    shared_ptr<NeuralNet> train_net){
  train_net_=train_net;
  modelproto_=model;
  auto cluster=Cluster::Get();
  if(!(cluster->nserver_groups()&&cluster->server_update())){
    updater_=shared_ptr<Updater>(Singleton<Factory<Updater>>::Instance()
        ->Create("Updater"));
    updater_->Init(model.updater());
  }
}

void Worker::ConnectStub(shared_ptr<Dealer> dealer, EntityType type){
  if(updater_==nullptr){
    auto cluster=Cluster::Get();
    int sgid=group_id_/cluster->nworker_groups_per_server_group();
    CHECK(cluster->runtime()->JoinSGroup(group_id_, worker_id_, sgid));
  }

  dealer->Connect(kInprocRouterEndpoint);
  Msg* ping=new Msg();
  ping->set_src(group_id_, worker_id_, type);
  ping->set_dst(-1,-1,kStub);
  ping->set_type(kConnect);
  ping->add_frame("PING", 4);
  dealer->Send(&ping);
  ping=dealer->Receive();
  string pong((char*)ping->frame_data(), ping->frame_size());
  CHECK_STREQ("PONG", pong.c_str());
  delete ping;
}

void Worker::Run(){
  LOG(ERROR)<<"Worker (group_id = "<<group_id_
    <<", id = "<<worker_id_<<") starts";
  dealer_=make_shared<Dealer>(2*thread_id_);
  ConnectStub(dealer_, kWorkerParam);
  for(auto layer: train_net_->layers())
    if(layer->partitionid()==worker_id_)
      if(layer->is_bridgedstlayer()||layer->is_bridgesrclayer()){
        layer_dealer_=make_shared<Dealer>(2*thread_id_+1);
        ConnectStub(layer_dealer_, kWorkerLayer);
        break;
      }
  step_=modelproto_.step();
  // init params
  for(auto layer: train_net_->layers()){
    if(layer->partitionid()==worker_id_)
      for(auto param: layer->GetParams()){
        // only owners fill the memory of parameter values.
        // others share the memory with owners hence do not need to put/get.
        if(param->owner() == param->id()){
          if(group_id_%Cluster::Get()->nworker_groups_per_server_group()==0)
            param->InitValues(0);
          else{
            Get(param, modelproto_.warmup_steps());
          }
        }
      }
  }
  Metric perf;
  if(group_id_%Cluster::Get()->nworker_groups_per_server_group()==0){
    for(step_=0;step_<modelproto_.warmup_steps();step_++)
      RunOneBatch(step_, &perf);
    for(auto layer: train_net_->layers()){
      if(layer->partitionid()==worker_id_)
        for(auto param: layer->GetParams())
          if(param->owner()==param->id())
            Put(param, step_);
    }
  }
  while(!StopNow(step_)){
    RunOneBatch(step_, &perf);
    step_++;
  }

  Stop();
  LOG(ERROR)<<"Worker (group_id = "<<group_id_
    <<", id = "<<worker_id_<<") stops";
}

void Worker::Stop(){
  auto cluster=Cluster::Get();
  if(updater_ == nullptr){
    int sgid=group_id_/cluster->nworker_groups_per_server_group();
    cluster->runtime()->LeaveSGroup(group_id_, worker_id_, sgid);
  }
  Msg* msg=new Msg();
  msg->set_src(group_id_, worker_id_, kWorkerParam);
  msg->set_dst(-1,-1, kStub);
  msg->set_type(kStop);
  dealer_->Send(&msg); // use param dealer to send the stop msg
}
int Worker::Put(shared_ptr<Param> param, int step){
  Msg* msg=new Msg();
  msg->set_src(group_id_, worker_id_, kWorkerParam);
  msg->set_dst(-1, -1, kStub);
  msg->set_type(kPut);
  msg->set_trgt(param->owner(), 0, step);
  dealer_->Send(&msg);
  return 1;
}
int Worker::Get(shared_ptr<Param> param, int step){
  Msg* msg=new Msg();
  msg->set_src(group_id_, worker_id_, kWorkerParam);
  msg->set_dst(-1, -1, kStub);
  msg->set_type(kGet);
  msg->set_trgt(param->owner(), 0, step);
  dealer_->Send(&msg);
  return 1;
}
int Worker::Update(shared_ptr<Param> param, int step){
  param->set_local_version(param->version());
  if(updater_){
    updater_->Update(step, param);
    param->set_version(param->version()+1);
  }else{
    Msg* msg=new Msg();
    msg->set_src(group_id_, worker_id_, kWorkerParam);
    msg->set_dst(-1, -1, kStub);
    msg->set_type(kUpdate);
    msg->set_trgt(param->owner(), 0, step);
    dealer_->Send(&msg);
  }
  return 1;
}

int Worker::CollectAll(shared_ptr<NeuralNet> net, int step){
  auto& layers=net->layers();
  for(auto& layer: layers){
    if(layer->partitionid()==worker_id_)
      for(shared_ptr<Param> p: layer->GetParams()){
        Collect(p, step);
      }
  }
  return 1;
}
int Worker::Collect(shared_ptr<Param> param, int step){
  while(param->version()<=param->local_version()){
    std::this_thread::sleep_for(std::chrono::milliseconds(kCollectSleepTime));
  }
  return 1;
}
const void Worker::DisplayPerformance(const Metric & perf, const string& prefix){
  Msg* msg=new Msg();
  msg->set_src(group_id_, worker_id_, kWorkerParam);
  msg->set_dst(-1,-1, kStub);
  msg->set_type(kMetric);
  msg->set_trgt(step_,0,0);
  const string disp=perf.ToString();
  msg->add_frame(prefix.c_str(), prefix.length());
  msg->add_frame(disp.c_str(), disp.length());
  dealer_->Send(&msg);
  //LOG(ERROR)<<prefix<<" "<<perf.ToString();
}

void Worker::RunOneBatch(int step, Metric* perf){
  if(ValidateNow(step)){
    //LOG(ERROR)<<"Validation at step "<<step;
    CollectAll(validation_net_, step);
    Test(modelproto_.validation_steps(),kValidation, validation_net_);
  }
  if(TestNow(step)){
    //LOG(ERROR)<<"Test at step "<<step;
    CollectAll(test_net_, step);
    Test(modelproto_.test_steps(), kTest, test_net_);
    string vispath = Cluster::Get()->vis_folder() + "/param-";
    for(auto param: train_net_->params()){
      if(param->vis()){
        BlobProto bp;
        CHECK_EQ(param->data().shape().size(), 2);
        param->data().ToProto(&bp);
        string fpath = vispath+param->name() + std::to_string(step) + ".dat";
        WriteProtoToBinaryFile(bp, fpath.c_str());
      }
    }
  }
  TrainOneBatch(step, perf);
  //LOG(ERROR)<<"Train "<<step;
  if(perf!=nullptr){
    perf->Inc();
    if(DisplayNow(step)){
      //perf->Avg();
      DisplayPerformance(*perf, "Train");
      perf->Reset();
    }
  }
  /*
  if(CheckpointNow(step)){
    pm_->Checkpoint(cluster_->workspace()+"/snapshot-"+std::to_string(step));
  }
  */
}

void Worker::ReceiveBlobs(shared_ptr<NeuralNet> net){
}

void Worker::SendBlob(){
}

void Worker::Test(int nsteps, Phase phase, shared_ptr<NeuralNet> net){
  Metric perf;
  vector<double*> feature;
  vector<int> dim;
  vector<string> name;
  LabelFeatureProto lf;
  TensorContainer<cpu,1> idx_sample(Shape1(nsteps));
  TSingleton<Random<cpu>>::Instance()->SampleUniform(idx_sample);
  int nbatch=0;
  float sample_prob = 1;
  for(int i=0;i<nsteps;i++)
    if(idx_sample[i]< sample_prob)
      nbatch++;

  for(auto layer: net->layers()){
    if(layer->vis() && !layer->is_labellayer()){
      int count = layer->data(nullptr).count();
      feature.push_back(new double[count * nbatch]);
      dim.push_back(count / layer->data(nullptr).shape()[0]);
      name.push_back(layer->name());
    }
  }
  for(int step=0, step_id=0;step<nsteps;step++){
    TestOneBatch(step, phase, net, &perf);
    if(idx_sample[step]< sample_prob){
      auto it=feature.begin();
      for(auto layer: net->layers()){
        if(layer->vis()){
          int count = layer->data(nullptr).count();
          const float *dat = layer->data(nullptr).cpu_data();
          if(layer->is_labellayer())
            for(int i=0; i < count; i++)
              lf.add_label(static_cast<int>(dat[i]));
          else{
            auto* dst= (*it) + count * step_id;
            for(int i=0;i<count;i++)
              dst[i]=static_cast<double>(dat[i]);
            it++;
          }
        }
      }
      step_id++;
    }
  }
  string path = Cluster::Get()->vis_folder()+"/step-" + std::to_string(step_);
  TSNE tsne;
  int npoints = lf.label_size();
  double* points = new double[npoints * 2];
  for(size_t i = 0; i < feature.size(); i++){
    tsne.run(feature.at(i),npoints, dim.at(i), points);
    delete feature.at(i);
    for(int j=0; j<npoints;j++){
      lf.add_x(points[2*j]);
      lf.add_y(points[2*j+1]);
    }
    string fpath = path + "-layer-"+ name.at(i) + ".dat";
    WriteProtoToBinaryFile(lf, fpath.c_str());
    lf.clear_x();
    lf.clear_y();
  }
  delete points;
  //perf.Avg();
  if(phase==kValidation)
    DisplayPerformance(perf, "Validation");
  else if (phase==kTest)
    DisplayPerformance(perf, "Test");
}

/****************************BPWorker**********************************/

BPWorker::BPWorker(int thread_id, int group_id, int worker_id):
  Worker(thread_id, group_id, worker_id){
}

void BPWorker::Forward(int step, Phase phase, shared_ptr<NeuralNet> net){
  auto& layers=net->layers();
  for(auto& layer: layers){
    if(layer->partitionid()==worker_id_){
      if(layer->is_bridgedstlayer()){
        auto* dst=static_cast<BridgeDstLayer*>(layer.get());
        while(!dst->ready()){
          auto msg=layer_dealer_->Receive();
          CHECK_EQ(msg->src_first(), group_id_);
          string name((char*)msg->frame_data(), msg->frame_size());
          auto tmp=net->name2layer(name);
          CHECK(tmp->is_bridgedstlayer());
          auto* dstlayer=static_cast<BridgeDstLayer*>(tmp.get());
          auto data=dstlayer->mutable_data(nullptr);
          msg->next_frame();
          memcpy(data->mutable_cpu_data(), msg->frame_data(), msg->frame_size());
          dstlayer->set_ready(true);
          delete msg;
        }
      }
      if(phase==kTrain){
        for(shared_ptr<Param> p: layer->GetParams()){
          Collect(p, step);
        }
      }
      //clock_t s=clock();
      layer->ComputeFeature(phase);
      //LOG(ERROR)<<layer->name()<<":"<<(clock()-s)*1.0/CLOCKS_PER_SEC;
      if(layer->is_bridgesrclayer()){
        auto dst=layer->dstlayers().at(0);
        Msg *msg=new Msg();
        msg->set_src(group_id_, worker_id_, kWorkerLayer);
        msg->set_dst(group_id_, dst->partitionid(), kWorkerLayer);
        msg->add_frame(dst->name().c_str(), dst->name().length());
        auto const & blob=layer->data(nullptr);
        msg->add_frame(blob.cpu_data(), blob.count()*sizeof(float));
        layer_dealer_->Send(&msg);
      }
      if(phase==kTrain&&DisplayDebugInfo(step)
          &&layer->mutable_data(nullptr)!=nullptr){
        LOG(INFO)<<StringPrintf("Forward layer  %10s data norm1 %13.9f",
            layer->name().c_str(), layer->data(nullptr).asum_data());
      }
    }
  }
}

void BPWorker::Backward(int step, shared_ptr<NeuralNet> net){
  auto& layers=net->layers();
  for (auto it = layers.rbegin(); it != layers.rend(); it++){
    shared_ptr<Layer> layer=*it;
    if(layer->partitionid()==worker_id_){
      if(layer->is_bridgesrclayer()){
        //auto* src=static_cast<BridgeSrcLayer*>(layer.get());
        // receive grad blobs
      }
      layer->ComputeGradient();
      if(layer->mutable_grad(nullptr)!=nullptr&&DisplayDebugInfo(step)){
        LOG(INFO)<<StringPrintf("Backward layer %10s grad norm1 %13.9f\t",
            layer->name().c_str(), layer->grad(nullptr).asum_data());
        for(shared_ptr<Param> p: layer->GetParams())
          LOG(INFO)<<StringPrintf("param id %2d, name %10s,\
              value norm1 %13.9f, grad norm1 %13.9f",
              p->id(), p->name().c_str(),
              p->data().asum_data(), p->grad().asum_data());
      }
      for(shared_ptr<Param> p: layer->GetParams()){
        Update(p, step);
      }
      if(layer->is_bridgedstlayer()){
        // send grad blobs
      }
    }
  }
}

void BPWorker::TrainOneBatch(int step, Metric* perf){
  Forward(step, kTrain, train_net_);
  Backward(step, train_net_);
  auto losslayers=train_net_->losslayers();
  for(auto layer: losslayers){
      if(layer->partitionid()==worker_id_){
        const float * ptr=layer->metric().cpu_data();
        /*
        for(int j=0;j<layer->metric().count();j++)
          perf->AddMetric(std::to_string(j)+"#"+layer->name(), ptr[j]);
        */
        // hard code display info
        perf->AddMetric(std::to_string(0)+"#loss", ptr[0]);
        perf->AddMetric(std::to_string(1)+"#accuracy", ptr[1]);
      }
    }
}

void BPWorker::TestOneBatch(int step, Phase phase, shared_ptr<NeuralNet> net, Metric* perf){
  Forward(step, phase, net);
  const auto& losslayers=net->losslayers();
  for(auto layer: losslayers){
    if(layer->partitionid()==worker_id_){
      const float * ptr=layer->metric().cpu_data();
      /*
         for(int j=0;j<layer->metric().count();j++)
         perf.AddMetric(std::to_string(j)+"#"+layer->name(), ptr[j]);
         */
      // hard code display info
      perf->AddMetric(std::to_string(0)+"#loss", ptr[0]);
      perf->AddMetric(std::to_string(1)+"#accuracy", ptr[1]);
    }
  }
  perf->Inc();
}

}  // namespace singa
