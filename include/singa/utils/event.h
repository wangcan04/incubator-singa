#ifndef SINGA_UTILS_EVENT_H_
#define SINGA_UTILS_EVENT_H_

class CopyEvent {
 public:
  static void CUDART_CB Host2Dev(cudaStream_t stream, cudaError_t status, void *userData);
  static void CUDART_CB Dev2Host(cudaStream_t stream, cudaError_t status, void *userData);

  CopyEvent(Param* p, Worker* w, bool h2d):param(p), worker(w), host2dev(h2d){}

 Param* param;
 Worker* worker;
 int param_version = -1;
 bool host2dev;
};

#endif  // SINGA_UTILS_EVENT_H_

