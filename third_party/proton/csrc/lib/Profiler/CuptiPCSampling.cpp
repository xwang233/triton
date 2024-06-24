#include "Profiler/Cupti/CuptiPCSampling.h"
#include "Data/Metric.h"
#include "Driver/GPU/CudaApi.h"
#include "Driver/GPU/CuptiApi.h"
#include "Utility/Map.h"
#include "Utility/String.h"
#include <memory>
#include <tuple>

namespace proton {

namespace {

size_t getNumStallReasons(CUcontext context) {
  size_t numStallReasons = 0;
  CUpti_PCSamplingGetNumStallReasonsParams numStallReasonsParams = {
      .size = CUpti_PCSamplingGetNumStallReasonsParamsSize,
      .pPriv = NULL,
      .ctx = context,
      .numStallReasons = &numStallReasons};
  cupti::pcSamplingGetNumStallReasons<true>(&numStallReasonsParams);
  return numStallReasons;
}

std::pair<char **, uint32_t *>
getStallReasonNamesAndIndices(CUcontext context, size_t numStallReasons) {
  char **stallReasonNames =
      static_cast<char **>(std::calloc(numStallReasons, sizeof(char *)));
  for (size_t i = 0; i < numStallReasons; i++) {
    stallReasonNames[i] = static_cast<char *>(
        std::calloc(CUPTI_STALL_REASON_STRING_SIZE, sizeof(char)));
  }
  uint32_t *stallReasonIndices =
      static_cast<uint32_t *>(std::calloc(numStallReasons, sizeof(uint32_t)));
  // Initialize the names with 128 characters to avoid buffer overflow
  CUpti_PCSamplingGetStallReasonsParams stallReasonsParams = {
      .size = CUpti_PCSamplingGetStallReasonsParamsSize,
      .pPriv = NULL,
      .ctx = context,
      .numStallReasons = numStallReasons,
      .stallReasonIndex = stallReasonIndices,
      .stallReasons = stallReasonNames,
  };
  cupti::pcSamplingGetStallReasons<true>(&stallReasonsParams);
  return std::make_pair(stallReasonNames, stallReasonIndices);
}

size_t matchStallReasonsToIndices(
    size_t numStallReasons, char **stallReasonNames,
    uint32_t *stallReasonIndices,
    std::map<size_t, size_t> &stallReasonIndexToMetricIndex) {
  // In case there's any invalid stall reasons, we only collect valid ones.
  // Invalid ones are swapped to the end of the list
  std::vector<bool> validIndex(numStallReasons, false);
  size_t numValidStalls = 0;
  for (size_t i = 0; i < numStallReasons; i++) {
    auto cuptiStallName = replace(std::string(stallReasonNames[i]), "_", "");
    for (size_t j = 0; j < PCSamplingMetric::PCSamplingMetricKind::Count; j++) {
      auto metricName = toLower(PCSamplingMetric().getValueName(j));
      if (metricName.find(cuptiStallName) != std::string::npos) {
        validIndex[i] = true;
        stallReasonIndexToMetricIndex[stallReasonIndices[i]] = j;
        numValidStalls++;
        break;
      }
    }
  }
  int inValidIndex = -1;
  for (size_t i = 0; i < numStallReasons; i++) {
    if (inValidIndex == -1 && !validIndex[i]) {
      inValidIndex = i;
    } else if (inValidIndex != -1 && validIndex[i]) {
      std::swap(stallReasonIndices[inValidIndex], stallReasonIndices[i]);
      validIndex[inValidIndex] = true;
      inValidIndex = i;
    }
  }
  return numValidStalls;
}

void enablePCSampling(CUcontext context) {
  CUpti_PCSamplingEnableParams params = {
      .size = CUpti_PCSamplingEnableParamsSize,
      .pPriv = NULL,
      .ctx = context,
  };
  cupti::pcSamplingEnable<true>(&params);
}

void disablePCSampling(CUcontext context) {
  CUpti_PCSamplingDisableParams params = {
      .size = CUpti_PCSamplingDisableParamsSize,
      .pPriv = NULL,
      .ctx = context,
  };
  cupti::pcSamplingDisable<true>(&params);
}

void startPCSampling(CUcontext context) {
  CUpti_PCSamplingStartParams params = {
      .size = CUpti_PCSamplingStartParamsSize,
      .pPriv = NULL,
      .ctx = context,
  };
  cupti::pcSamplingStart<true>(&params);
}

void stopPCSampling(CUcontext context) {
  CUpti_PCSamplingStopParams params = {
      .size = CUpti_PCSamplingStopParamsSize,
      .pPriv = NULL,
      .ctx = context,
  };
  cupti::pcSamplingStop<true>(&params);
}

void getPCSamplingData(CUcontext context,
                       CUpti_PCSamplingData *pcSamplingData) {
  CUpti_PCSamplingGetDataParams params = {
      .size = CUpti_PCSamplingGetDataParamsSize,
      .pPriv = NULL,
      .ctx = context,
      .pcSamplingData = pcSamplingData,
  };
  cupti::pcSamplingGetData<true>(&params);
}

class ConfigureData {
public:
  ConfigureData() = default;

  void initialize(CUcontext context, uint32_t frequency);

  CUpti_PCSamplingData *getPCSamplingData() { return &pcSamplingData; }

  ~ConfigureData() {
    for (size_t i = 0; i < numStallReasons; i++)
      std::free(stallReasonNames[i]);
    if (stallReasonNames)
      std::free(stallReasonNames);
    if (stallReasonIndices)
      std::free(stallReasonIndices);
  }

private:
  void configureStallReasons();
  void configureSamplingPeriod(uint32_t frequency);
  void configureSamplingBuffer();
  void configureScratchBuffer();
  void configureHardwareBufferSize();
  void configureStartStopControl();

  // The amount of data reserved on the GPU
  static constexpr uint32_t HardwareBufferSize = 512 * 1024 * 1024;
  // The amount of data copied from the hardware buffer each time
  static constexpr uint32_t ScratchBufferSize = 16 * 1024 * 1024;
  // The number of PCs copied from the scratch buffer each time
  static constexpr uint32_t ScratchBufferPCCount = 4096;

  bool initialized{false};
  CUcontext context{};
  uint32_t frequency{};
  uint32_t numStallReasons{};
  uint32_t validStallReasons{};
  char **stallReasonNames{};
  uint32_t *stallReasonIndices{};
  std::map<size_t, size_t> stallReasonIndexToMetricIndex{};
  CUpti_PCSamplingData pcSamplingData{};
  CUpti_PCSamplingConfigurationInfo stallReasonInfo{};
  CUpti_PCSamplingConfigurationInfo samplingPeriodInfo{};
};

void ConfigureData::configureSamplingPeriod(uint32_t frequency) {
  this->frequency = frequency;
  samplingPeriodInfo.attributeType =
      CUPTI_PC_SAMPLING_CONFIGURATION_ATTR_TYPE_SAMPLING_PERIOD;
  samplingPeriodInfo.attributeData.samplingPeriodData.samplingPeriod =
      frequency;
}

void ConfigureData::configureSamplingBuffer() {
  samplingPeriodInfo.attributeType =
      CUPTI_PC_SAMPLING_CONFIGURATION_ATTR_TYPE_SAMPLING_DATA_BUFFER;
  samplingPeriodInfo.attributeData.samplingDataBufferData.samplingDataBuffer =
      0;
}

void ConfigureData::configureScratchBuffer() {
  samplingPeriodInfo.attributeType =
      CUPTI_PC_SAMPLING_CONFIGURATION_ATTR_TYPE_SCRATCH_BUFFER_SIZE;
  samplingPeriodInfo.attributeData.scratchBufferSizeData.scratchBufferSize =
      ScratchBufferSize;
}

void ConfigureData::configureHardwareBufferSize() {
  samplingPeriodInfo.attributeType =
      CUPTI_PC_SAMPLING_CONFIGURATION_ATTR_TYPE_HARDWARE_BUFFER_SIZE;
  samplingPeriodInfo.attributeData.hardwareBufferSizeData.hardwareBufferSize =
      HardwareBufferSize;
  0;
}

void ConfigureData::configureStartStopControl() {
  samplingPeriodInfo.attributeType =
      CUPTI_PC_SAMPLING_CONFIGURATION_ATTR_TYPE_ENABLE_START_STOP_CONTROL;
  samplingPeriodInfo.attributeData.enableStartStopControlData
      .enableStartStopControl = true;
}

void ConfigureData::configureStallReasons() {
  numStallReasons = getNumStallReasons(context);
  std::tie(this->stallReasonNames, this->stallReasonIndices) =
      getStallReasonNamesAndIndices(context, numStallReasons);
  validStallReasons = matchStallReasonsToIndices(
      numStallReasons, stallReasonNames, stallReasonIndices,
      stallReasonIndexToMetricIndex);
  stallReasonInfo.attributeType =
      CUPTI_PC_SAMPLING_CONFIGURATION_ATTR_TYPE_STALL_REASON;
  stallReasonInfo.attributeData.stallReasonData.stallReasonCount =
      validStallReasons;
  stallReasonInfo.attributeData.stallReasonData.pStallReasonIndex =
      stallReasonIndices;
}

void ConfigureData::initialize(CUcontext context, uint32_t frequency) {
  if (this->initialized)
    return;
  this->initialized = true;
  this->context = context;
  this->frequency = frequency;
  configureSamplingPeriod(frequency);
  configureStallReasons();
  configureHardwareBufferSize();
  configureScratchBuffer();
  configureSamplingBuffer();
  configureStartStopControl();
}

} // namespace

struct CuptiPCSampling::CuptiPCSamplingPimpl {
  void initialize(void *context, uint32_t frequency);

  void start(void *context);

  void stop(void *context);

  void finalize(void *context);

  ConfigureData *getConfigureData(CUcontext context) {
    uint32_t contextId;
    cupti::getContextId<true>(context, &contextId);
    return &contextIdToConfigureData[contextId];
  }

  ThreadSafeMap<size_t, ConfigureData> contextIdToConfigureData;
};

void CuptiPCSampling::CuptiPCSamplingPimpl::initialize(void *context,
                                                       uint32_t frequency) {
  auto cuContext = static_cast<CUcontext>(context);
  auto *configureData = getConfigureData(cuContext);
  configureData->initialize(cuContext, frequency);
  enablePCSampling(cuContext);
}

void CuptiPCSampling::CuptiPCSamplingPimpl::start(void *context) {
  auto cuContext = static_cast<CUcontext>(context);
  auto *configureData = getConfigureData(cuContext);
  startPCSampling(cuContext);
}

void CuptiPCSampling::CuptiPCSamplingPimpl::stop(void *context) {
  auto cuContext = static_cast<CUcontext>(context);
  auto *configureData = getConfigureData(cuContext);
  stopPCSampling(cuContext);
  auto *pcSamplingData = configureData->getPCSamplingData();
  while ((pcSamplingData->totalNumPcs > 0 ||
          pcSamplingData->remainingNumPcs > 0)) {
    // Handle data
    for (size_t i = 0; i < pcSamplingData->totalNumPcs; ++i) {
      auto *pcData = pcSamplingData->pPcData;
    }
    // Get next data
    getPCSamplingData(cuContext, pcSamplingData);
  }
}

void CuptiPCSampling::CuptiPCSamplingPimpl::finalize(void *context) {
  disablePCSampling(static_cast<CUcontext>(context));
}

void CuptiPCSampling::initialize(void *context, uint32_t frequency) {
  pImpl->initialize(context, frequency);
}

void CuptiPCSampling::start(void *context) { pImpl->start(context); }

void CuptiPCSampling::stop(void *context) { pImpl->stop(context); }

void CuptiPCSampling::finalize(void *context) { pImpl->finalize(context); }

} // namespace proton
