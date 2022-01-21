
#include <highfive/H5DataSet.hpp>
#include <highfive/H5DataSpace.hpp>
#include <highfive/H5File.hpp>
#include <highfive/H5Group.hpp>

#include "features/src/featuremanager.h"

namespace pixsfm {

template <typename dtype>
FeatureManager<dtype>::FeatureManager(std::vector<int> channels_per_level,
                                      py::array_t<dtype> dummy)
    : num_levels_(channels_per_level.size()) {
  for (int level_idx = 0; level_idx < num_levels_; ++level_idx) {
    feature_sets_.emplace_back(channels_per_level[level_idx]);
  }
}

template <typename dtype>
FeatureManager<dtype>::FeatureManager(std::string& h5_path, bool fill,
                                      std::string level_prefix)
    : num_levels_(-1) {
  HighFive::File file(h5_path, HighFive::File::ReadOnly);
  HighFive::Attribute chann_attr = file.getAttribute("channels_per_level");
  std::vector<int> channels_per_level;
  chann_attr.read(channels_per_level);

  num_levels_ = channels_per_level.size();

  for (int level_idx = 0; level_idx < num_levels_; ++level_idx) {
    std::string h5_key = level_prefix + std::to_string(level_idx);
    feature_sets_.emplace_back(h5_path, h5_key, channels_per_level[level_idx],
                               fill, mutex_);
  }

  if (fill) {
    Lock();
  }
}

template <typename dtype>
class FeatureSet<dtype>& FeatureManager<dtype>::FeatureSet(int level_index) {
  return feature_sets_.at(level_index);
}

template <typename dtype>
VecFSet<dtype>& FeatureManager<dtype>::FeatureSets() {
  return feature_sets_;
}

template <typename dtype>
int FeatureManager<dtype>::NumLevels() {
  return num_levels_;
}

template <typename dtype>
void FeatureManager<dtype>::Lock() {
  for (auto& fset : feature_sets_) {
    fset.Lock();
  }
}

template <typename dtype>
size_t FeatureManager<dtype>::CurrentMemory() const {
  size_t num_bytes = 0;
  for (const auto& fset : feature_sets_) {
    num_bytes += fset.CurrentMemory();
  }
  return num_bytes;
}

template <typename dtype>
size_t FeatureManager<dtype>::NumBytes() const {
  size_t num_bytes = 0;
  for (const auto& fset : feature_sets_) {
    num_bytes += fset.NumBytes();
  }
  return num_bytes;
}

template class FeatureManager<double>;
template class FeatureManager<float>;
template class FeatureManager<float16>;

}  // namespace pixsfm