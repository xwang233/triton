#ifndef PROTON_PROFILER_CUPTI_PC_SAMPLING_H_
#define PROTON_PROFILER_CUPTI_PC_SAMPLING_H_

#include "CuptiProfiler.h"
#include "Utility/Singleton.h"

namespace proton {

class CuptiPCSampling : public Singleton<CuptiPCSampling> {

public:
  CuptiPCSampling() = default;
  virtual ~CuptiPCSampling() = default;

  void initialize(void *context, uint32_t frequency);

  void start(void *context);

  void stop(void *context);

  void finalize(void *context);

private:
  struct CuptiPCSamplingPimpl; // forward declaration of the implementation
  std::unique_ptr<CuptiPCSamplingPimpl>
      pImpl; // unique pointer to the implementation
};

} // namespace proton

#endif // PROTON_PROFILER_CUPTI_PC_SAMPLING_H_
