#include "singa/utils/event.h"


void CUDART_CB CopyEvent::Host2Dev(cudaStream_t stream,
    cudaError_t status, void *userData) {
  auto event = static_cast<CopyEvent*>(userData);
  CHECK(event->host2dev);
  auto *param = event->param;
  param->mutable_data()->SyncHead();
  param->set_version(event->param_version);
  delete event;
}

void CUDART_CB CopyEvent::Dev2Host(cudaStream_t stream,
    cudaError_t status, void *userData) {
  auto event = static_cast<CopyEvent*>(userData);
  CHECK(!event->host2dev);
  param->mutable_grad()->SyncHead();
  auto dealer = Singleton<Context>::Get()->driver_dealer();
  if (event->update)
    Worker::SendUpdateMsg(param, event->worker, dealer);
  delete event;
}
