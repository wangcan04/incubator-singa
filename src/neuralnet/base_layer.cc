#include "neuralnet/base_layer.h"

#include <cblas.h>
#include <glog/logging.h>
#include <math.h>
#include <cfloat>
#include "utils/factory.h"
#include "utils/singleton.h"

namespace singa {

using std::string;
using std::vector;

Layer* Layer::Create(const LayerProto& proto) {
  auto* factory = Singleton<Factory<Layer>>::Instance();
  Layer* layer = nullptr;
  if (proto.has_user_type())
    layer = factory->Create(proto.user_type());
  else
    layer = factory->Create(proto.type());
  return layer;
}

const string Layer::DebugString(int step, int flag) {
  string ret = StringPrintf("Layer %10s ", name().c_str());
  if ((flag & kForward) == kForward && data_.count() !=0) {
    ret += StringPrintf("data norm1 %13.9f", data_.asum_data());
  } else if ((flag & kBackward) == kBackward) {
    if (grad_.count() != 0)
      ret += StringPrintf("grad norm1 %13.9f\n", grad_.asum_data());
  }
  if ((flag & kTrain) == kTrain) {
    for (Param* p : GetParams()) {
      ret += StringPrintf(
          "param id %2d, name %10s, value norm1 %13.9f, grad norm1 %13.9f\n",
          p->id(), p->name().c_str(), p->data().asum_data(),
          p->grad().asum_data());
    }
  }
  return ret;
}

/************* Implementation for ParserLayer ***********/
void ParserLayer::ComputeFeature(int flag, Metric *perf) {
  CHECK_EQ(srclayers_.size(), 1);
  auto datalayer = static_cast<DataLayer*>(*srclayers_.begin());
  ParseRecords(flag, datalayer->records(), &data_);
}

/************* Implementation for PrefetchLayer ***********/
PrefetchLayer::~PrefetchLayer() {
  if (thread_.joinable())
    thread_.join();
}


void PrefetchLayer::ComputeFeature(int flag, Metric* perf) {
  LOG(FATAL) << "Not implemented";
}

}  // namespace singa
