#ifndef SINGA_UTILS_EVENT_H_
#define SINGA_UTILS_EVENT_H_
namespace singa {
class Param;
class Worker;
class CopyEvent {
 public:
  CopyEvent() {}
  CopyEvent(Param *p, Worker* w, bool h2d):param(p), worker(w), host2dev(h2d){}

 Param* param = nullptr;
 Worker* worker = nullptr;
 bool host2dev;
 int param_version = -1;
};
}
#endif  // SINGA_UTILS_EVENT_H_

