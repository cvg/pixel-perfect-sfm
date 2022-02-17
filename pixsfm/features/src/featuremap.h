#pragma once
#include "_pixsfm/src/helpers.h"
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

#include <highfive/H5DataSet.hpp>
#include <highfive/H5DataSpace.hpp>
#include <highfive/H5File.hpp>
#include <highfive/H5Group.hpp>

#include <colmap/util/types.h>

#include <boost/serialization/vector.hpp>
#include <utility>
#include <vector>

#include <colmap/util/logging.h>

#include "util/src/log_exceptions.h"
#include "util/src/types.h"

#include "features/src/featurepatch.h"

namespace pixsfm {

template <typename dtype>
using MapIdFPatch = std::unordered_map<colmap::point2D_t, FeaturePatch<dtype>>;

template <typename dtype>
class FeatureMap {
 public:
  FeatureMap(py::array_t<dtype, py::array::c_style> patches,
             std::vector<int>& point2D_ids,
             std::vector<Eigen::Vector2i>& corners, py::dict metadata);

  FeatureMap(py::array_t<dtype, py::array::c_style> patches,
             std::vector<int>& point2D_ids,
             std::vector<Eigen::Vector2i>& corners,
             std::vector<Eigen::Vector2d>& scales, py::dict metadata);

  FeatureMap(HighFive::Group& group, bool fill = false,
             std::vector<colmap::point2D_t>* point2D_ids = NULL);

  FeatureMap();

  FeatureMap(int channels, bool is_sparse);

  size_t InitFromH5Group(HighFive::Group& group, bool fill,
                         std::vector<colmap::point2D_t>* point2D_ids = NULL);
  size_t LoadFromH5Group(HighFive::Group& group, bool fill,
                         std::vector<colmap::point2D_t>* point2D_ids = NULL);

  // Storage Format 1
  size_t LoadFromH5Grouped(HighFive::Group& group, bool fill,
                           std::vector<colmap::point2D_t>* point2D_ids = NULL);

  // Storage Format 2
  size_t InitFromH5GroupChunked(
      HighFive::Group& group, bool fill,
      std::vector<colmap::point2D_t>* point2D_ids = NULL);
  size_t LoadFromH5GroupChunked(
      HighFive::Group& group, bool fill,
      std::vector<colmap::point2D_t>* point2D_ids = NULL);

  void Unload(std::vector<colmap::point2D_t>* point2D_ids = NULL);
  size_t Flush();

  // Locks features, i.e. they cannot be unloaded anymore
  void Lock();

  inline FeaturePatch<dtype>& GetFeaturePatch(colmap::point2D_t point2D_id);
  inline bool HasFeaturePatch(colmap::point2D_t point2D_idx) const;

  MapIdFPatch<dtype>& Patches();

  inline bool IsSparse() const;

  inline int Channels() const;
  inline std::vector<ssize_t> Shape() const;
  inline int NumPatches() const;
  int Size() const;

  size_t CurrentMemory() const;
  size_t NumBytes() const;  // Total memory required if all is loaded

  inline void AddFeaturePatch(colmap::point2D_t point2D_idx,
                              const FeaturePatch<dtype>& patch);

  inline std::vector<colmap::point2D_t> Keys() const;

 private:
  MapIdFPatch<dtype> patches_;
  int channels_;
  bool is_sparse_;
  std::unordered_map<colmap::point2D_t, size_t> p2D_idx_to_h5id_;
  int h5_format_ = -1;
  // Use a pointer to avoid unintended calls to the python API
  std::shared_ptr<py::array_t<dtype, py::array::c_style>> ref_array_ptr;
};

template <typename dtype>
FeaturePatch<dtype>& FeatureMap<dtype>::GetFeaturePatch(
    colmap::point2D_t point2D_id) {
  if (is_sparse_) {
    return patches_.at(point2D_id);
  } else {
    return patches_.at(kDensePatchId);
  }
}

template <typename dtype>
bool FeatureMap<dtype>::HasFeaturePatch(colmap::point2D_t point2D_idx) const {
  return (is_sparse_ ? (patches_.find(point2D_idx) != patches_.end())
                     : (patches_.size() == 1 &&
                        patches_.begin()->first == kDensePatchId));
}

template <typename dtype>
void FeatureMap<dtype>::AddFeaturePatch(colmap::point2D_t point2D_idx,
                                        const FeaturePatch<dtype>& patch) {
  if (!is_sparse_) {
    THROW_CHECK_EQ(point2D_idx, kDensePatchId);
  };
  THROW_CHECK_EQ(channels_, patch.Channels());
  patches_[point2D_idx] = patch;
}

template <typename dtype>
int FeatureMap<dtype>::NumPatches() const {
  // THROW_CHECK_EQ(shape_[0], patches_.size());
  return patches_.size();
}

template <typename dtype>
std::vector<ssize_t> FeatureMap<dtype>::Shape() const {
  std::vector<ssize_t> shape(4, 0);
  shape[0] = patches_.size();
  shape[3] = channels_;
  for (const auto& patch : patches_) {
    if (shape[1] == 0) {
      shape[1] = patch.second.Shape()[0];
    }
    if (shape[1] != patch.second.Shape()[0]) {
      shape[1] == -1;
    }
    if (shape[2] == 0) {
      shape[2] = patch.second.Shape()[1];
    }
    if (shape[2] != patch.second.Shape()[1]) {
      shape[2] == -1;
    }
  }
  return shape;
}

template <typename dtype>
int FeatureMap<dtype>::Channels() const {
  return channels_;
}

template <typename dtype>
MapIdFPatch<dtype>& FeatureMap<dtype>::Patches() {
  return patches_;
}

template <typename dtype>
bool FeatureMap<dtype>::IsSparse() const {
  return is_sparse_;
}

template <typename dtype>
std::vector<colmap::point2D_t> FeatureMap<dtype>::Keys() const {
  std::vector<colmap::point2D_t> keys;
  for (const auto& el : patches_) {
    keys.push_back(el.first);
  }
  return keys;
}

}  // namespace pixsfm
