#pragma once
#include "sys/sysinfo.h"
#include "sys/types.h"

#include <ceres/ceres.h>

namespace pixsfm {

inline long long TotalPhysicalMemory() {
#if IS_LINUX
  struct sysinfo memInfo;
  sysinfo(&memInfo);
  long long totalPhysMem = memInfo.totalram;
  // Multiply in next statement to avoid int overflow on right hand side...
  totalPhysMem *= memInfo.mem_unit;
  return totalPhysMem;
#endif
  return std::numeric_limits<long long>::max();
}

inline long long UsedPhysicalMemory() {
#if IS_LINUX
  struct sysinfo memInfo;
  sysinfo(&memInfo);
  long long physMemUsed = memInfo.totalram - memInfo.freeram;
  // Multiply in next statement to avoid int overflow on right hand side...
  physMemUsed *= memInfo.mem_unit;
  return physMemUsed;
#endif
  return 0;
}

inline long long FreePhysicalMemory() {
#if IS_LINUX
  struct sysinfo memInfo;
  sysinfo(&memInfo);
  long long physMemFree = memInfo.freeram;
  // Multiply in next statement to avoid int overflow on right hand side...
  physMemFree *= memInfo.mem_unit;
  return physMemFree;
#endif
  return std::numeric_limits<long long>::max();
}

inline long long NumNonZerosJacobian(const ceres::Problem* problem) {
  std::vector<ceres::ResidualBlockId> residual_blocks;
  problem->GetResidualBlocks(&residual_blocks);

  long long num_nonzeros = 0;

  for (ceres::ResidualBlockId res : residual_blocks) {
    std::vector<double*> parameter_blocks;
    int num_residuals =
        problem->GetCostFunctionForResidualBlock(res)->num_residuals();
    problem->GetParameterBlocksForResidualBlock(res, &parameter_blocks);
    int num_active_params = 0;
    for (double* param_block : parameter_blocks) {
      if (!problem->IsParameterBlockConstant(param_block)) {
        num_active_params += problem->ParameterBlockLocalSize(param_block);
      }
    }
    num_nonzeros += num_active_params * num_residuals;
  }
  return num_nonzeros;
}

inline std::string MemoryString(int bytes, std::string unit = "KB") {
  std::stringstream ss;
  if (unit == "KB") {
    ss << bytes / 1024. << unit;
  } else if (unit == "MB") {
    ss << bytes / 1024. / 1024. << unit;
  } else if (unit == "GB") {
    ss << bytes / 1024. / 1024. / 1024. << unit;
  }
  return ss.str();
}

}  // namespace pixsfm