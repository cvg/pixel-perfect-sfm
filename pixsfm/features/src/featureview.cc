#include "features/src/featureview.h"

#include "util/src/memory.h"

namespace pixsfm {

template <typename dtype>
FeatureView<dtype>::FeatureView(
    FeatureSet<dtype>* feature_set,
    std::unordered_map<colmap::image_t, std::string>& image_id_to_name)
    : feature_set_(feature_set), image_id_to_name_(image_id_to_name) {
  LoadAllPatches();
}

template <typename dtype>
FeatureView<dtype>::FeatureView(FeatureSet<dtype>* feature_set,
                                std::unordered_set<std::string>& image_names)
    : feature_set_(feature_set) {
  colmap::image_t cnt = 1;  // colmap image_ids always start at 1 for some
                            // reason
  for (const std::string& image_name : image_names) {
    image_id_to_name_.emplace(cnt, image_name);
    ++cnt;
  }
  LoadAllPatches();
}
template <typename dtype>
FeatureView<dtype>::FeatureView(
    FeatureSet<dtype>* feature_set,
    std::unordered_map<std::string, std::vector<colmap::point2D_t>>&
        req_patches)
    : req_patches_(req_patches), feature_set_(feature_set) {
  int cnt = 1;
  for (auto& pair : req_patches_) {
    if (pair.second.size() == 0) {
      continue;
    }
    image_id_to_name_[cnt] = pair.first;
    ++cnt;
  }
  LoadRequiredPatches();
}

template <typename dtype>
FeatureView<dtype>::FeatureView(
    FeatureSet<dtype>* feature_set, const Graph* graph,
    const std::unordered_set<size_t>& required_nodes)
    : feature_set_(feature_set), image_id_to_name_(graph->image_id_to_name) {
  for (size_t node_idx : required_nodes) {
    const FeatureNode* node = graph->nodes[node_idx];
    const std::string& image_name = image_id_to_name_[node->image_id];
    req_patches_[image_name].push_back(node->feature_idx);
  }
  LoadRequiredPatches();
}

template <typename dtype>
FeatureView<dtype>::FeatureView(FeatureSet<dtype>* feature_set,
                                const Graph* graph)
    : feature_set_(feature_set), image_id_to_name_(graph->image_id_to_name) {
  for (const FeatureNode* node : graph->nodes) {
    const std::string& image_name = image_id_to_name_[node->image_id];
    req_patches_[image_name].push_back(node->feature_idx);
  }

  LoadRequiredPatches();
}

// BA Constructor all
template <typename dtype>
FeatureView<dtype>::FeatureView(FeatureSet<dtype>* feature_set,
                                const colmap::Reconstruction* reconstruction)
    : feature_set_(feature_set) {
  for (const colmap::image_t& image_id : reconstruction->RegImageIds()) {
    if (reconstruction->Image(image_id).NumPoints3D() == 0) {
      continue;
    }
    std::string image_name = reconstruction->Image(image_id).Name();
    image_id_to_name_.emplace(image_id, image_name);
    req_patches_.emplace(image_name, std::vector<colmap::point2D_t>(0));
    colmap::point2D_t p2D_id = 0;
    for (auto& point2D : reconstruction->Image(image_id).Points2D()) {
      if (point2D.HasPoint3D()) {
        req_patches_[image_name].push_back(p2D_id);
      }
      ++p2D_id;
    }
  }
  LoadRequiredPatches();
}

template <typename dtype>
FeatureView<dtype>::FeatureView(
    FeatureSet<dtype>* feature_set,
    const colmap::Reconstruction* reconstruction,
    const std::unordered_set<colmap::point3D_t>& point3D_ids)
    : feature_set_(feature_set) {
  for (const colmap::image_t& image_id : reconstruction->RegImageIds()) {
    if (reconstruction->Image(image_id).NumPoints3D() == 0) {
      continue;
    }
    std::string image_name = reconstruction->Image(image_id).Name();
    image_id_to_name_.emplace(image_id, image_name);
    req_patches_.emplace(image_name, std::vector<colmap::point2D_t>(0));
  }
  for (const colmap::point3D_t& point3D_id : point3D_ids) {
    const colmap::Track& track = reconstruction->Point3D(point3D_id).Track();
    for (auto& track_el : track.Elements()) {
      colmap::image_t image_id = track_el.image_id;
      req_patches_[image_id_to_name_[image_id]].push_back(track_el.point2D_idx);
    }
  }
  LoadRequiredPatches();
}

template <typename dtype>
void FeatureView<dtype>::LoadRequiredPatches() {
  size_t num_bytes_loaded = feature_set_->Load(req_patches_, true);
  size_t num_bytes_freed = feature_set_->Flush();

  STDLOG(DEBUG) << "Loaded: " << MemoryString(num_bytes_loaded) << std::endl;
  STDLOG(DEBUG) << "Freed: " << MemoryString(num_bytes_freed) << std::endl;
  STDLOG(DEBUG) << "Net Memory Change:"
                << MemoryString(num_bytes_loaded - num_bytes_freed)
                << std::endl;
}

template <typename dtype>
void FeatureView<dtype>::LoadAllPatches() {
  std::unordered_set<std::string> image_names;
  for (auto& pair : image_id_to_name_) {
    image_names.emplace(pair.second);
  }
  size_t num_bytes_loaded = feature_set_->Load(image_names, true);
  size_t num_bytes_freed = feature_set_->Flush();

  STDLOG(DEBUG) << "Loaded: " << MemoryString(num_bytes_loaded) << std::endl;
  STDLOG(DEBUG) << "Freed: " << MemoryString(num_bytes_freed) << std::endl;
  STDLOG(DEBUG) << "Net Memory Change:"
                << MemoryString(num_bytes_loaded - num_bytes_freed)
                << std::endl;
}

template <typename dtype>
FeatureView<dtype>::~FeatureView() {
  Clear();
}

template <typename dtype>
void FeatureView<dtype>::Clear() {
  if (feature_set_) {  // not NULL we try unload
    if (req_patches_.size() > 0) {
      int num_patches = 0;
      for (auto& pair : req_patches_) {
        num_patches += pair.second.size();
      }
      STDLOG(DEBUG) << "Unloading " << num_patches << " patches..."
                    << std::endl;
      feature_set_->Unload(req_patches_);
    } else {
      std::unordered_set<std::string> imnames;
      for (auto& pair : image_id_to_name_) {
        imnames.emplace(pair.second);
      }
      STDLOG(DEBUG) << "Unloading " << imnames.size() << " maps..."
                    << std::endl;
      feature_set_->Unload(imnames);
    }
  }
  req_patches_.clear();
  image_id_to_name_.clear();
  feature_set_ = NULL;
}

template <typename dtype>
int FeatureView<dtype>::Channels() const {
  return feature_set_->Channels();
}

template <typename dtype>
FeatureMap<dtype>& FeatureView<dtype>::GetFeatureMap(colmap::image_t image_id) {
  return feature_set_->GetFeatureMap(image_id_to_name_.at(image_id));
}

template <typename dtype>
FeatureMap<dtype>& FeatureView<dtype>::GetFeatureMap(
    const std::string& image_name) {
  return feature_set_->GetFeatureMap(image_name);
}

template <typename dtype>
FeaturePatch<dtype>& FeatureView<dtype>::GetFeaturePatch(
    colmap::image_t image_id, colmap::point2D_t point2D_idx) {
  return GetFeatureMap(image_id).GetFeaturePatch(point2D_idx);
}

template <typename dtype>
FeaturePatch<dtype>& FeatureView<dtype>::GetFeaturePatch(
    const std::string& image_name, colmap::point2D_t point2D_idx) {
  return GetFeatureMap(image_name).GetFeaturePatch(point2D_idx);
}

template <typename dtype>
bool FeatureView<dtype>::HasFeaturePatch(colmap::image_t image_id,
                                         colmap::point2D_t point2D_idx) {
  bool has_im_id = image_id_to_name_.find(image_id) != image_id_to_name_.end();
  if (has_im_id) {
    bool contains_fmap =
        feature_set_->HasFeatureMap(image_id_to_name_[image_id]);
    if (contains_fmap) {
      return GetFeatureMap(image_id).HasFeaturePatch(point2D_idx);
    }
  }
  return false;
}

template <typename dtype>
bool FeatureView<dtype>::HasFeaturePatch(const std::string& image_name,
                                         colmap::point2D_t point2D_idx) {
  // We do not check if the string exists in image_id_to_name
  // (performance), since we assume that fmap.HasFeaturePatch checks if the
  // patch actually holds data.
  bool contains_fmap = feature_set_->HasFeatureMap(image_name);
  if (contains_fmap) {
    return GetFeatureMap(image_name).HasFeaturePatch(point2D_idx);
  } else {
    return false;
  }
}

template <typename dtype>
std::unordered_map<colmap::image_t, std::string>&
FeatureView<dtype>::Mapping() {
  return image_id_to_name_;
}

template <typename dtype>
FeatureView<dtype>::FeatureView(const FeatureView<dtype>& other)
    : feature_set_(other.feature_set_),
      image_id_to_name_(other.image_id_to_name_),
      req_patches_(other.req_patches_) {
  // Increment reference counts
  if (req_patches_.size() > 0) {
    feature_set_->Load(req_patches_, true);
  } else {
    std::unordered_set<std::string> imnames;
    for (auto& pair : image_id_to_name_) {
      imnames.emplace(pair.second);
    }
    feature_set_->Load(imnames, true);
  }
}

template <typename dtype>
FeatureView<dtype>::FeatureView(FeatureView<dtype>&& other)
    : feature_set_(std::move(other.feature_set_)),
      image_id_to_name_(std::move(other.image_id_to_name_)),
      req_patches_(std::move(other.req_patches_)) {
  other.feature_set_ = NULL;
  // NO Increment of reference counts. We tell the other feature view to not
  // unload, so we just transfer the ownership of data.
}

template <typename dtype>
FeatureView<dtype>& FeatureView<dtype>::operator=(
    const FeatureView<dtype>& other) {
  Clear();
  feature_set_ = other.feature_set_;
  image_id_to_name_ = other.image_id_to_name_;
  req_patches_ = other.req_patches_;
  if (req_patches_.size() > 0) {
    feature_set_->Load(req_patches_, true);
  } else {
    std::unordered_set<std::string> imnames;
    for (auto& pair : image_id_to_name_) {
      imnames.emplace(pair.second);
    }
    feature_set_->Load(imnames, true);
  }
  return *this;
}

template <typename dtype>
FeatureView<dtype>& FeatureView<dtype>::operator=(FeatureView<dtype>&& other) {
  Clear();
  feature_set_ = std::move(other.feature_set_);
  image_id_to_name_ = std::move(other.image_id_to_name_);
  req_patches_ = std::move(other.req_patches_);
  other.feature_set_ = NULL;
  return *this;
}

template <typename dtype>
size_t FeatureView<dtype>::ReservedMemory() const {
  size_t num_bytes = 0;
  if (req_patches_.size() > 0) {
    for (auto& req_pair : req_patches_) {
      FeatureMap<dtype>& fmap = feature_set_->GetFeatureMap(req_pair.first);
      if (fmap.IsSparse()) {
        for (const auto& point2D_idx : req_pair.second) {
          num_bytes += fmap.GetFeaturePatch(point2D_idx).NumBytes();
        }
      } else {
        num_bytes += fmap.NumBytes();
      }
    }
  } else {
    for (auto& pair : image_id_to_name_) {
      num_bytes += feature_set_->GetFeatureMap(pair.second).NumBytes();
    }
  }
  return num_bytes;
}

template <typename dtype>
colmap::image_t FeatureView<dtype>::FindImageId(const std::string& image_name) {
  for (auto& pair : image_id_to_name_) {
    if (pair.second == image_name) {
      return pair.first;
    }
  }
  THROW_CUSTOM_CHECK_MSG(false, std::invalid_argument, "image_name not found");
}

template class FeatureView<double>;
template class FeatureView<float>;
template class FeatureView<float16>;

}  // namespace pixsfm