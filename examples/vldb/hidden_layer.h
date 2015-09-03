#include "neuralnet/layer.h"
#include "myproto.pb.h"

class HiddenLayer : public singa::NeuronLayer {
 public:
  ~HiddenLayer();
  void Setup(const singa::LayerProto& proto, int npartitions) override;
  void ComputeFeature(int flag, singa::Metric* perf) override;
  void ComputeGradient(int flag, singa::Metric* perf) override;
  const std::vector<singa::Param*> GetParams() const override {
    std::vector<singa::Param*> params{weight_, bias_};
    return params;
  }

 private:
  int batchsize_;
  int vdim_, hdim_;
  bool transpose_;
  singa::Param *weight_, *bias_;
};
