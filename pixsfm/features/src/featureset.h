#pragma once

#include "_pixsfm/src/helpers.h"
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

#include <highfive/H5DataSet.hpp>
#include <highfive/H5DataSpace.hpp>
#include <highfive/H5File.hpp>
#include <highfive/H5Group.hpp>

#include <colmap/scene/reconstruction.h>
#include <colmap/util/logging.h>
#include <colmap/util/types.h>

#include <boost/serialization/vector.hpp>
#include <utility>
#include <vector>

#include "util/src/log_exceptions.h"
#include "util/src/simple_logger.h"
#include "util/src/types.h"

#include "features/src/featuremap.h"
#include "features/src/featurepatch.h"
#include <mutex>

#include <algorithm>

namespace pixsfm {

template <typename dtype>
using MapStringFMap = std::unordered_map<std::string, FeatureMap<dtype>>;

// Holds Features of one level, no view. Either H5 or Holds direct
template <typename dtype>
class FeatureSet {
 public:
  FeatureSet(MapStringFMap<dtype>& feature_dict, int channels);
  FeatureSet(
      std::string& h5_path, std::string& level_key, int channels, bool fill,
      std::shared_ptr<std::mutex> parent_mutex = std::shared_ptr<std::mutex>());
  FeatureSet(int channels);
  FeatureSet(dtype type_identifier, int channels);
  FeatureSet();

  FeatureSet(const FeatureSet<dtype>& other);  // required for mutex
  FeatureSet(FeatureSet<dtype>&& other);       // required for mutex

  void Swap(FeatureSet<dtype>&);
  FeatureSet<dtype>& operator=(FeatureSet<dtype> other);

  inline MapStringFMap<dtype>& FeatureMaps();
  inline FeatureMap<dtype>& GetFeatureMap(const std::string& name);
  inline void AddFeatureMap(const std::string& name, FeatureMap<dtype>& fmap);
  inline void Emplace(const std::string& name, FeatureMap<dtype>& fmap);

  // Loads and increments counter
  size_t Load(std::unordered_set<std::string>& required_maps, bool fill);
  size_t Load(std::unordered_map<std::string, std::vector<colmap::point2D_t>>&
                  required_patches,
              bool fill);

  // Decrements counter, but does NOT remove data (This is done by a subsequent
  // flush call)
  void Unload(std::unordered_set<std::string>& required_maps);
  void Unload(std::unordered_map<std::string, std::vector<colmap::point2D_t>>&
                  required_patches);

  // Deletes only data if stemmed from H5 and reference_count == 0
  size_t Flush();

  // Locks features, i.e. they cannot be unloaded anymore
  void Lock();

  inline int Channels() const;

  inline void FlushEveryN(int n);

  size_t CurrentMemory() const;
  size_t NumBytes() const;  // Total memory required if all is loaded

  inline bool HasFeatureMap(const std::string& name);

  inline std::vector<std::string> Keys() const;

  inline void DisconnectH5();

  std::shared_ptr<FeatureSet<dtype>> GetPtr();

  inline void UseParallelIO(bool do_parallel);

 private:
  MapStringFMap<dtype> feature_maps_;
  int channels_;
  int flush_every_n_ = 1;
  bool has_h5_connection_ = false;  // Only if true we flush!
  bool parallel_io_ = false;

  std::string h5_path_;
  std::string h5_key_;
  std::shared_ptr<std::mutex> mutex_ = std::make_shared<std::mutex>();

  std::atomic<int> flush_count{1};
};

template <typename dtype>
MapStringFMap<dtype>& FeatureSet<dtype>::FeatureMaps() {
  return feature_maps_;
}

template <typename dtype>
int FeatureSet<dtype>::Channels() const {
  return channels_;
}

template <typename dtype>
void FeatureSet<dtype>::FlushEveryN(int n) {
  flush_every_n_ = std::max(n, 16);
}

template <typename dtype>
void FeatureSet<dtype>::UseParallelIO(bool do_parallel) {
  if (do_parallel) {
    hbool_t is_threadsafe;
    H5is_library_threadsafe(&is_threadsafe);
    THROW_CHECK(is_threadsafe);
  }
  parallel_io_ = do_parallel;
}

template <typename dtype>
FeatureMap<dtype>& FeatureSet<dtype>::GetFeatureMap(const std::string& name) {
  return feature_maps_.at(name);
}

template <typename dtype>
bool FeatureSet<dtype>::HasFeatureMap(const std::string& name) {
  return (feature_maps_.find(name) != feature_maps_.end());
}

template <typename dtype>
void FeatureSet<dtype>::AddFeatureMap(const std::string& name,
                                      FeatureMap<dtype>& fmap) {
  feature_maps_[name] = fmap;
}

template <typename dtype>
void FeatureSet<dtype>::DisconnectH5() {
  has_h5_connection_ = false;
}

template <typename dtype>
void FeatureSet<dtype>::Emplace(const std::string& name,
                                FeatureMap<dtype>& fmap) {
  feature_maps_.emplace(name, fmap);
}

template <typename dtype>
std::vector<std::string> FeatureSet<dtype>::Keys() const {
  std::vector<std::string> keys;
  for (const auto& el : feature_maps_) {
    keys.push_back(el.first);
  }
  return keys;
}

}  // namespace pixsfm
