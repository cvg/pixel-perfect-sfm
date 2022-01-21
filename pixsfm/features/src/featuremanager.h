#pragma once
#include "_pixsfm/src/helpers.h"
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

#include <colmap/util/logging.h>
#include <colmap/util/types.h>

#include <boost/serialization/vector.hpp>
#include <utility>
#include <vector>

#include "util/src/log_exceptions.h"
#include "util/src/types.h"

#include "features/src/featureset.h"

namespace pixsfm {

template <typename dtype>
using VecFSet = std::vector<FeatureSet<dtype>>;

template <typename dtype>
class FeatureManager {
 public:
  FeatureManager(std::vector<int> channels_per_level, py::array_t<dtype> dummy);

  FeatureManager(std::string& h5_path, bool fill, std::string level_prefix);

  class FeatureSet<dtype>& FeatureSet(int level_index);
  VecFSet<dtype>& FeatureSets();
  int NumLevels();

  // Locks features, i.e. they cannot be unloaded anymore
  void Lock();

  size_t CurrentMemory() const;
  size_t NumBytes() const;  // Total memory required if all is loaded

 private:
  VecFSet<dtype> feature_sets_;
  int num_levels_;
  std::shared_ptr<std::mutex> mutex_ = std::make_shared<std::mutex>();
};

}  // namespace pixsfm