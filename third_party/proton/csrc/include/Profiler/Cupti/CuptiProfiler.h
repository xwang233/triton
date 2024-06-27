#ifndef PROTON_PROFILER_CUPTI_PROFILER_H_
#define PROTON_PROFILER_CUPTI_PROFILER_H_

#include "../GPUProfiler.h"

namespace proton {

class CuptiProfiler : public GPUProfiler<CuptiProfiler> {
public:
  CuptiProfiler();
  virtual ~CuptiProfiler();

  void enablePCSampling() { pcSamplingEnabled = true; }

  void disablePCSampling() { pcSamplingEnabled = false; }

  bool isPCSamplingEnabled() const { return pcSamplingEnabled; }

private:
  struct CuptiProfilerPimpl;
  bool pcSamplingEnabled{false};
};

} // namespace proton

#endif // PROTON_PROFILER_CUPTI_PROFILER_H_
