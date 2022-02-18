#include "features/src/featuremap.h"
#include "util/src/simple_logger.h"
#include <assert.h>
#include <pybind11/eigen.h>

namespace pixsfm {

template <typename dtype>
FeatureMap<dtype>::FeatureMap(py::array_t<dtype, py::array::c_style> patches,
                              std::vector<int>& point2D_ids,
                              std::vector<Eigen::Vector2i>& corners,
                              py::dict metadata)
    : is_sparse_(metadata["is_sparse"].cast<bool>()), channels_(-1) {
  py::buffer_info buffer_info = patches.request();

  // extract data an shape of input array
  dtype* data_ptr = static_cast<dtype*>(buffer_info.ptr);
  std::vector<ssize_t> shape;
  shape = buffer_info.shape;
  channels_ = static_cast<int>(shape[3]);
  THROW_CHECK_EQ(shape.size(), 4);

  int height = shape[1];
  int width = shape[2];
  int channels = shape[3];

  int patch_size = height * width * channels;

  std::array<int, 3> patch_shape = {height, width, channels};
  size_t n_patches = shape[0];

  THROW_CHECK(is_sparse_ || n_patches == 1);

  patches_.reserve(n_patches);

  Eigen::Vector2d scale = metadata["scale"].cast<Eigen::Vector2d>();

  for (int i = 0; i < n_patches; i++) {
    patches_.emplace((is_sparse_ ? point2D_ids[i] : kDensePatchId),
                     FeaturePatch<dtype>(&data_ptr[i * patch_size], patch_shape,
                                         corners[i], scale, false));
  }

  ref_array_ptr.reset(new py::array_t<dtype, py::array::c_style>(patches));
}

template <typename dtype>
FeatureMap<dtype>::FeatureMap(HighFive::Group& group, bool fill,
                              std::vector<colmap::point2D_t>* point2D_ids)
    : is_sparse_(false), channels_(-1) {
  InitFromH5Group(group, fill, point2D_ids);
}

template <typename dtype>
FeatureMap<dtype>::FeatureMap() : is_sparse_(false), channels_(-1) {}

template <typename dtype>
FeatureMap<dtype>::FeatureMap(int channels, bool is_sparse)
    : is_sparse_(is_sparse), channels_(channels) {}

// Storage Format 1
template <typename dtype>
size_t FeatureMap<dtype>::InitFromH5Group(
    HighFive::Group& group, bool fill,
    std::vector<colmap::point2D_t>* point2D_ids) {
  HighFive::Attribute format_attr = group.getAttribute("format");
  format_attr.read(h5_format_);

  if (h5_format_ == 1) {
    return LoadFromH5Grouped(group, fill, point2D_ids);
  } else if (h5_format_ == 2) {
    return InitFromH5GroupChunked(group, fill, point2D_ids);
  } else {
    throw std::runtime_error("Unknown featuremap format.");
  }
}

template <typename dtype>
size_t FeatureMap<dtype>::LoadFromH5Group(
    HighFive::Group& group, bool fill,
    std::vector<colmap::point2D_t>* point2D_ids) {
  if (h5_format_ == 1) {
    return LoadFromH5Grouped(group, fill, point2D_ids);
  } else if (h5_format_ == 2) {
    return LoadFromH5GroupChunked(group, fill, point2D_ids);
  } else {
    throw std::runtime_error("Unknown featuremap format.");
  }
}

// Storage Format 1
template <typename dtype>
size_t FeatureMap<dtype>::LoadFromH5Grouped(
    HighFive::Group& group, bool fill,
    std::vector<colmap::point2D_t>* point2D_ids) {
  size_t num_bytes = 0;

  // HighFive::Attribute scale_attr = group.getAttribute("scale");
  // scale_attr.read(scale_);
  HighFive::Attribute shape_attr = group.getAttribute("shape");
  std::vector<ssize_t> shape(4);
  shape_attr.read(shape);

  channels_ = static_cast<int>(shape[3]);

  int sparse;
  HighFive::Attribute sparse_attr = group.getAttribute("is_sparse");
  sparse_attr.read(sparse);
  is_sparse_ = bool(sparse);

  std::vector<std::string> keys;

  if ((!point2D_ids) || (!is_sparse_)) {
    keys = group.listObjectNames();
  } else {
    for (int i = 0; i < point2D_ids->size(); i++) {
      keys.push_back(std::to_string((*point2D_ids)[i]));
    }
  }

  for (std::string& key : keys) {
    colmap::point2D_t patch_id =
        is_sparse_ ? static_cast<colmap::point2D_t>(std::stoi(key))
                   : kDensePatchId;
    HighFive::DataSet dataset = group.getDataSet(key);

    if (HasFeaturePatch(patch_id)) {
      num_bytes += patches_[patch_id].LoadFromH5Dataset(dataset, fill);
    } else {
      patches_.emplace(patch_id, std::move(FeaturePatch<dtype>(dataset, fill)));
      num_bytes += GetFeaturePatch(patch_id).CurrentMemory();
    }
  }

  return num_bytes;
}

template <typename dtype>
size_t FeatureMap<dtype>::InitFromH5GroupChunked(
    HighFive::Group& group, bool fill,
    std::vector<colmap::point2D_t>* point2D_ids) {
  HighFive::DataSet patches_dataset = group.getDataSet("patches");
  std::vector<size_t> shape = patches_dataset.getDimensions();
  channels_ = static_cast<int>(shape[3]);
  int sparse;
  HighFive::Attribute sparse_attr = group.getAttribute("is_sparse");
  sparse_attr.read(sparse);
  is_sparse_ = bool(sparse);

  std::vector<std::string> keys;

  HighFive::DataSet kp_id_dataset = group.getDataSet("keypoint_ids");
  std::vector<int> stored_keypoint_ids;
  kp_id_dataset.read(stored_keypoint_ids);

  THROW_CHECK_EQ(shape[0], stored_keypoint_ids.size());

  p2D_idx_to_h5id_.clear();
  for (int& point2D_idx : stored_keypoint_ids) {
    auto it = find(stored_keypoint_ids.begin(), stored_keypoint_ids.end(),
                   point2D_idx);
    size_t idx = it - stored_keypoint_ids.begin();
    p2D_idx_to_h5id_.emplace(static_cast<colmap::point2D_t>(point2D_idx), idx);
  }

  std::vector<colmap::point2D_t> required_p2D_ids;
  if (!point2D_ids) {
    for (int& point2D_id : stored_keypoint_ids) {
      required_p2D_ids.push_back(static_cast<colmap::point2D_t>(point2D_id));
    }
  } else {
    required_p2D_ids = *point2D_ids;
  }

  std::unordered_set<colmap::point2D_t> missing_p2D_ids;
  for (colmap::point2D_t& point2D_idx : required_p2D_ids) {
    colmap::point2D_t patch_id = is_sparse_ ? point2D_idx : kDensePatchId;
    if (!HasFeaturePatch(patch_id)) {
      missing_p2D_ids.emplace(point2D_idx);
    }
  }

  // Initialize missing patches (usually only called on init)
  if (!missing_p2D_ids.empty()) {
    HighFive::DataSet scale_dataset = group.getDataSet("scales");
    std::vector<double> scales(scale_dataset.getElementCount());
    scale_dataset.read(scales.data());

    THROW_CHECK_EQ(stored_keypoint_ids.size(), scales.size() / 2);

    HighFive::DataSet corner_dataset = group.getDataSet("corners");
    std::vector<int> corners(corner_dataset.getElementCount());
    corner_dataset.read(corners.data());

    std::array<int, 3> patch_shape = {static_cast<int>(shape[1]),
                                      static_cast<int>(shape[2]),
                                      static_cast<int>(shape[3])};

    THROW_CHECK_EQ(corners.size(), scales.size());
    for (const colmap::point2D_t& point2D_idx : missing_p2D_ids) {
      // If element was found
      colmap::point2D_t patch_id = is_sparse_ ? point2D_idx : kDensePatchId;
      size_t idx = p2D_idx_to_h5id_[point2D_idx];
      patches_.emplace(
          patch_id, std::move(FeaturePatch<dtype>(
                        nullptr, patch_shape,
                        Eigen::Map<Eigen::Vector2i>(&corners[2 * idx]),
                        Eigen::Map<Eigen::Vector2d>(&scales[2 * idx]), false)));
    }
  }

  return LoadFromH5GroupChunked(group, fill, &required_p2D_ids);
}

template <typename dtype>
size_t FeatureMap<dtype>::LoadFromH5GroupChunked(
    HighFive::Group& group, bool fill,
    std::vector<colmap::point2D_t>* point2D_ids) {
  size_t num_bytes = 0;
  if (fill) {
    THROW_CHECK(!p2D_idx_to_h5id_.empty())
    HighFive::DataSet patches_dataset = group.getDataSet("patches");
    std::vector<size_t> shape = patches_dataset.getDimensions();

    std::vector<colmap::point2D_t> required_p2D_ids;
    if (!point2D_ids) {
      for (auto& pair : p2D_idx_to_h5id_) {
        required_p2D_ids.push_back(pair.first);
      }
    } else {
      required_p2D_ids = *point2D_ids;
    }

    for (colmap::point2D_t& point2D_idx : required_p2D_ids) {
      size_t idx = p2D_idx_to_h5id_[point2D_idx];
      HighFive::Selection chunk = patches_dataset.select(
          {idx, 0, 0, 0}, {1, shape[1], shape[2], shape[3]});

      colmap::point2D_t patch_id = is_sparse_ ? point2D_idx : kDensePatchId;
      num_bytes += patches_[patch_id].LoadDataFromH5Chunk(chunk, fill);
    }
  }

  return num_bytes;
}

// Storage Format 2
template <typename dtype>
void FeatureMap<dtype>::Unload(std::vector<colmap::point2D_t>* point2D_ids) {
  if (point2D_ids) {
    int cnt = 0;
    for (colmap::point2D_t point2D_idx : (*point2D_ids)) {
      // We cannot throw here since this is called inside the constructor
      // of FeatureView -> noexcept by default.
      if (!HasFeaturePatch(point2D_idx)) {
        STDLOG(WARN) << "WARNING FATAL: point2D_idx=" << point2D_idx
                     << " not found during unloading. "
                     << "Continuing with other patches." << std::endl;
        STDLOG(WARN) << cnt << " / " << point2D_ids->size() << std::endl;
      }
      if (is_sparse_) {
        // Since we already checked above we do not do the same below.
        patches_[point2D_idx].Unload();
      } else {
        patches_[kDensePatchId].Unload();
        return;  // We do not want to double unload dense maps!
      }
    }
  } else {
    for (auto& patch_pair : patches_) {
      patch_pair.second.Unload();
    }
  }
}

template <typename dtype>
size_t FeatureMap<dtype>::Flush() {
  size_t bytes_freed = 0;
  for (const auto& patch_pair : patches_) {
    bytes_freed += GetFeaturePatch(patch_pair.first).Flush();
  }
  return bytes_freed;
}

template <typename dtype>
void FeatureMap<dtype>::Lock() {
  for (auto& patch_pair : patches_) {
    patch_pair.second.Lock();
  }
}

template <typename dtype>
size_t FeatureMap<dtype>::CurrentMemory() const {
  size_t num_bytes = 0;
  for (const auto& patch_pair : patches_) {
    num_bytes += patch_pair.second.CurrentMemory();
  }
  return num_bytes;
}

template <typename dtype>
int FeatureMap<dtype>::Size() const {
  size_t size = 0;
  for (const auto& patch_pair : patches_) {
    size += patch_pair.second.Size();
  }
  return size;
}

template <typename dtype>
size_t FeatureMap<dtype>::NumBytes() const {
  size_t num_bytes = 0;
  for (const auto& patch_pair : patches_) {
    num_bytes += patch_pair.second.NumBytes();
  }
  return num_bytes;
}

template class FeatureMap<double>;
template class FeatureMap<float>;
template class FeatureMap<float16>;
}  // namespace pixsfm