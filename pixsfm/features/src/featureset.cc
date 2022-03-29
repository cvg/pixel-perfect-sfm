#include "features/src/featureset.h"

#include <third-party/progressbar.h>

#include "util/src/misc.h"
#include "util/src/simple_logger.h"

#include <H5Ipublic.h>
#include <H5Opublic.h>

namespace pixsfm {

template <typename dtype>
FeatureSet<dtype>::FeatureSet(MapStringFMap<dtype>& feature_dict, int channels)
    : channels_(channels), feature_maps_(feature_dict) {}

template <typename dtype>
FeatureSet<dtype>::FeatureSet(int channels) : channels_(channels) {}

template <typename dtype>
FeatureSet<dtype>::FeatureSet(dtype type_identifier, int channels)
    : channels_(channels) {}

template <typename dtype>
FeatureSet<dtype>::FeatureSet(std::string& h5_path, std::string& level_key,
                              int channels, bool fill,
                              std::shared_ptr<std::mutex> parent_mutex)
    : channels_(channels),
      has_h5_connection_(true),
      h5_path_(h5_path),
      h5_key_(level_key) {
  // @TODO
  // For some reason parallel read/write from H5 is slower than locking it
  // But it works
  // H5is_library_threadsafe(&parallel_io_);

  // if (!parallel_io_) {
  //     STDLOG(WARN)<<"Detected HDF5 library not built threadsafe. "<<
  //     "Consider recompiling for increased performance."<<std::endl;
  // }
  if (parent_mutex) {
    mutex_ = parent_mutex;
  }
  HighFive::File file(h5_path_, HighFive::File::ReadOnly);
  HighFive::Group level_group = file.getGroup(h5_key_);
  std::vector<std::string> imnames = GetImageKeys(level_group);  //@TODO
  std::unordered_set<std::string> imnames_set(imnames.begin(), imnames.end());
  Load(imnames_set, fill);
}

template <typename dtype>
FeatureSet<dtype>::FeatureSet() : channels_(-1) {}

// Loads and increments counter
template <typename dtype>
size_t FeatureSet<dtype>::Load(std::unordered_set<std::string>& required_maps,
                               bool fill) {
  if (!has_h5_connection_) {
    return 0;
  }

  if (!parallel_io_) {
    mutex_->lock();
  }

  size_t num_bytes = 0;
  {
    HighFive::File file(h5_path_, HighFive::File::ReadOnly);
    HighFive::Group level_group = file.getGroup(h5_key_);
    STDLOG(INFO) << "Loading featuremaps from H5 File." << std::endl;
    LogProgressbar bar(required_maps.size());
    for (auto& image_name : required_maps) {
      HighFive::Group map_group = level_group.getGroup(image_name);

      if (HasFeatureMap(image_name)) {
        num_bytes += feature_maps_[image_name].LoadFromH5Group(map_group, fill);
      } else {
        feature_maps_.emplace(image_name,
                              std::move(FeatureMap<dtype>(map_group, fill)));
        num_bytes += (fill ? feature_maps_[image_name].CurrentMemory() : 0);
      }
      bar.update();
    }
  }

  if (!parallel_io_) {
    mutex_->unlock();
  }

  return num_bytes;
}

template <typename dtype>
size_t FeatureSet<dtype>::Load(
    std::unordered_map<std::string, std::vector<colmap::point2D_t>>&
        required_patches,
    bool fill) {
  if (!has_h5_connection_) {
    return 0;
  }

  if (!parallel_io_) {
    mutex_->lock();
  }

  size_t num_bytes = 0;
  {
    HighFive::File file(h5_path_, HighFive::File::ReadOnly);
    HighFive::Group level_group = file.getGroup(h5_key_);
    STDLOG(INFO) << "Loading patches from H5 File." << std::endl;
    size_t num_patches = 0;
    for (auto& map_data : required_patches) {
      num_patches += map_data.second.size();
    }
    LogProgressbar bar(num_patches);
    for (auto& map_data : required_patches) {
      std::string image_name = map_data.first;
      std::vector<colmap::point2D_t>& required_patch_ids = map_data.second;

      if (required_patch_ids.size() == 0) {
        continue;
      }

      HighFive::Group map_group = level_group.getGroup(image_name);

      if (HasFeatureMap(image_name)) {
        num_bytes += feature_maps_.at(image_name)
                        .LoadFromH5Group(map_group, fill, &required_patch_ids);
      } else {
        feature_maps_.emplace(
            image_name,
            std::move(FeatureMap<dtype>(map_group, fill, &required_patch_ids)));
        num_bytes += (fill ? feature_maps_[image_name].CurrentMemory() : 0);
      }
      bar.update(map_data.second.size());
    }
  }

  if (!parallel_io_) {
    mutex_->unlock();
  }
  return num_bytes;
}

template <typename dtype>
void FeatureSet<dtype>::Unload(std::unordered_set<std::string>& required_maps) {
  if (feature_maps_.empty()) {
    return;
  }
  for (auto& image_name : required_maps) {
    feature_maps_.at(image_name).Unload();
  }
}

template <typename dtype>
void FeatureSet<dtype>::Unload(
    std::unordered_map<std::string, std::vector<colmap::point2D_t>>&
        required_patches) {
  if (feature_maps_.empty()) {
    return;
  }

  if (!has_h5_connection_) {
    return;
  }

  for (auto& map_data : required_patches) {
    std::string image_name = map_data.first;
    std::vector<colmap::point2D_t>& required_patch_ids = map_data.second;

    if (required_patch_ids.size() == 0) {
      continue;
    }

    feature_maps_.at(image_name).Unload(&required_patch_ids);
  }
}

template <typename dtype>
size_t FeatureSet<dtype>::Flush() {
  --flush_count;  // Atomic
  if (flush_count > 0) {
    return 0;
  }

  if (!has_h5_connection_) {
    return 0;
  }

  size_t num_bytes = 0;
  for (auto& map_pair : feature_maps_) {
    num_bytes += map_pair.second.Flush();
  }
  flush_count = flush_every_n_;
  return num_bytes;
}

// Locks features, i.e. they cannot be unloaded anymore
template <typename dtype>
void FeatureSet<dtype>::Lock() {
  for (auto& fmap_pair : feature_maps_) {
    fmap_pair.second.Lock();
  }
}

template <typename dtype>
FeatureSet<dtype>::FeatureSet(const FeatureSet<dtype>& other)
    : channels_(other.channels_) {
  // std::lock_guard<std::mutex> lock(*other.mutex_);
  feature_maps_ = other.feature_maps_;
  has_h5_connection_ = other.has_h5_connection_;
  h5_path_ = other.h5_path_;
  h5_key_ = other.h5_key_;
}

template <typename dtype>
FeatureSet<dtype>::FeatureSet(FeatureSet<dtype>&& other)
    : channels_(other.channels_) {
  std::lock_guard<std::mutex> lock(*other.mutex_);
  feature_maps_ = std::move(other.feature_maps_);
  has_h5_connection_ = std::move(other.has_h5_connection_);
  h5_path_ = std::move(other.h5_path_);
  h5_key_ = std::move(other.h5_key_);
}

template <typename dtype>
void FeatureSet<dtype>::Swap(FeatureSet<dtype>& second) {
  using std::swap;
  if (mutex_ != second.mutex_) {
    std::lock(*mutex_, *second.mutex_);
    std::lock_guard<std::mutex> l1(*mutex_, std::adopt_lock);
    std::lock_guard<std::mutex> l2(*second.mutex_, std::adopt_lock);
    swap(channels_, second.channels_);
    swap(feature_maps_, second.feature_maps_);
    swap(has_h5_connection_, second.has_h5_connection_);
    swap(h5_path_, second.h5_path_);
    swap(h5_key_, second.h5_key_);
  } else {
    std::lock_guard<std::mutex> l1(*mutex_);
    swap(channels_, second.channels_);
    swap(feature_maps_, second.feature_maps_);
    swap(has_h5_connection_, second.has_h5_connection_);
    swap(h5_path_, second.h5_path_);
    swap(h5_key_, second.h5_key_);
  }
}

template <typename dtype>
FeatureSet<dtype>& FeatureSet<dtype>::operator=(FeatureSet<dtype> other) {
  Swap(other);
  return *this;
}

template <typename dtype>
size_t FeatureSet<dtype>::CurrentMemory() const {
  size_t num_bytes = 0;
  for (const auto& fmap_pair : feature_maps_) {
    num_bytes += fmap_pair.second.CurrentMemory();
  }
  return num_bytes;
}

template <typename dtype>
size_t FeatureSet<dtype>::NumBytes() const {
  size_t num_bytes = 0;
  for (const auto& fmap_pair : feature_maps_) {
    num_bytes += fmap_pair.second.NumBytes();
  }
  return num_bytes;
}

template class FeatureSet<double>;
template class FeatureSet<float>;
template class FeatureSet<float16>;

}  // namespace pixsfm