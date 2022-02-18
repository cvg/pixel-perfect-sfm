#pragma once

#include "_pixsfm/src/helpers.h"
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

#include <highfive/H5DataSet.hpp>
#include <highfive/H5DataSpace.hpp>
#include <highfive/H5File.hpp>
#include <highfive/H5Group.hpp>

#include <boost/serialization/vector.hpp>
#include <utility>
#include <vector>

#include <colmap/util/logging.h>

#include "base/src/interpolation.h"

#include "util/src/log_exceptions.h"
#include "util/src/math.h"
#include "util/src/types.h"

#include <atomic>
#include <mutex>

namespace pixsfm {

struct PatchStatus {
  int reference_count = 0;
  bool is_locked = false;
};

/*******************************************************************************
FeaturePatch:
*******************************************************************************/

template <typename dtype>
class FeaturePatch {
 public:
  FeaturePatch();

  // Python loaders
  FeaturePatch(py::array_t<dtype, py::array::c_style> pyarray,
               Eigen::Vector2i& corner, Eigen::Vector2d& scale,
               bool do_copy = false);

  FeaturePatch(dtype* data_ptr, std::array<int, 3> shape,
               Eigen::Ref<const Eigen::Vector2i> corner,
               Eigen::Ref<const Eigen::Vector2d> scale, bool do_copy = false);

  FeaturePatch(HighFive::DataSet& dataset, bool fill = true);

  FeaturePatch(const FeaturePatch<dtype>& other);  // Copy
  FeaturePatch(FeaturePatch<dtype>&& other);       // Move

  FeaturePatch<dtype>& operator=(const FeaturePatch<dtype>& other);  // Copy
  FeaturePatch<dtype>& operator=(FeaturePatch<dtype>&& other);       // Move

  // Storage Format 1
  size_t LoadFromH5Dataset(HighFive::DataSet& dataset, bool fill = true);

  // Storage Format 2
  size_t LoadDataFromH5Chunk(HighFive::Selection& chunk, bool fill);

  // Decrements Reference Count, but does NOT delete Data
  void Unload();

  // Deletes Data if reference_count==0
  size_t Flush();

  // Locks features, i.e. they cannot be unloaded anymore
  void Lock();

  // Allocates memory for patch data
  void Allocate();

  inline const dtype* Data() const;
  inline dtype* Data();

  py::array_t<dtype, py::array::c_style> AsNumpyArray();

  size_t CurrentMemory() const;
  // inline const std::vector<T>& Vector() const;

  // T GetEntry(const int idx);
  dtype GetEntry(const int y, const int x, const int c);
  dtype* GetEntryPtr(int y, int x, int c);

  template <typename T>
  inline void SetEntry(int y, int x, int c, T value);

  // const dtype* GetEntry(const int x, const int y);
  VectorX<dtype> GetDescriptor(const int x, const int y);

  std::vector<VectorX<dtype>> GetDescriptorPatch(const int x, const int y,
                                                 int patch_size);

  inline int Width() const;
  inline int Height() const;
  inline int Channels() const;
  inline const std::array<int, 3>& Shape() const;

  inline int X0() const;
  inline int Y0() const;

  inline const Eigen::Vector2i& Corner() const;
  inline const Eigen::Vector2d& Scale() const;

  inline double UpsamplingFactor() const;
  inline void SetUpsamplingFactor(double s);

  inline int Size() const;
  inline size_t NumBytes() const;

  inline bool IsReference() const;
  inline bool HasData() const;

  inline PatchStatus Status() const;

  template <typename T>
  Eigen::Matrix<T, 2, 1> GetPixelCoordinates(const T* xy) const;

  template <typename T>
  void ToPixelCoordinates(const T* xy, T* uv) const;

  Eigen::Vector2d GetPixelCoordinatesVec(const Eigen::Vector2d& xy);

 protected:
  std::vector<dtype>
      data_;         // If we do not own any data, leave this uninitialized
  dtype* data_ptr_;  // Ptr to data. Used for functionalities

  Eigen::Vector2i corner_;
  Eigen::Vector2d scale_;

  double upsampling_factor_ =
      1.0;  // Default no upsamping, really only required in costmaps

  std::array<int, 3> shape_;
  std::shared_ptr<py::array_t<dtype, py::array::c_style>> ref_array_ptr_;

  PatchStatus status_;
  std::mutex mutex_;
};

template <typename dtype>
int FeaturePatch<dtype>::Width() const {
  return shape_[1];
}

template <typename dtype>
int FeaturePatch<dtype>::Height() const {
  return shape_[0];
}

template <typename dtype>
int FeaturePatch<dtype>::Channels() const {
  return shape_[2];
}

template <typename dtype>
int FeaturePatch<dtype>::X0() const {
  return corner_[0];
}

template <typename dtype>
int FeaturePatch<dtype>::Y0() const {
  return corner_[1];
}

template <typename dtype>
int FeaturePatch<dtype>::Size() const {
  return shape_[0] * shape_[1] * shape_[2];
}

template <typename dtype>
size_t FeaturePatch<dtype>::NumBytes() const {
  return Size() * sizeof(dtype);
}

template <typename dtype>
const std::array<int, 3>& FeaturePatch<dtype>::Shape() const {
  return shape_;
}

template <typename dtype>
const Eigen::Vector2i& FeaturePatch<dtype>::Corner() const {
  return corner_;
}

template <typename dtype>
const Eigen::Vector2d& FeaturePatch<dtype>::Scale() const {
  return scale_;
}

template <typename dtype>
double FeaturePatch<dtype>::UpsamplingFactor() const {
  return upsampling_factor_;
}

template <typename dtype>
void FeaturePatch<dtype>::SetUpsamplingFactor(double s) {
  upsampling_factor_ = s;
}

template <typename dtype>
PatchStatus FeaturePatch<dtype>::Status() const {
  return status_;
}

template <typename dtype>
bool FeaturePatch<dtype>::IsReference() const {
  return ((data_ptr_ != data_.data()) && HasData());
}

template <typename dtype>
bool FeaturePatch<dtype>::HasData() const {
  return (
      data_ptr_);  // If data_ptr_==nullptr, the patch does not hold any data.
}

template <typename dtype>
const dtype* FeaturePatch<dtype>::Data() const {
  return data_ptr_;
}

template <typename dtype>
dtype* FeaturePatch<dtype>::Data() {
  return data_ptr_;
}

template <typename dtype>
template <typename T>
void FeaturePatch<dtype>::SetEntry(int y, int x, int c, T value) {
  data_ptr_[(y * Width() + x) * Channels() + c] = dtype(value);
}

template <typename dtype>
template <typename T>
void FeaturePatch<dtype>::ToPixelCoordinates(const T* xy, T* uv) const {
  uv[0] = (xy[0] * scale_[0] - 0.5 - double(corner_[0])) * upsampling_factor_;
  uv[1] = (xy[1] * scale_[1] - 0.5 - double(corner_[1])) * upsampling_factor_;
}

template <typename dtype>
template <typename T>
Eigen::Matrix<T, 2, 1> FeaturePatch<dtype>::GetPixelCoordinates(
    const T* xy) const {
  Eigen::Matrix<T, 2, 1> uv;
  ToPixelCoordinates(xy, uv.data());
  return uv;
}

template <typename dtype>
Eigen::Vector2d FeaturePatch<dtype>::GetPixelCoordinatesVec(
    const Eigen::Vector2d& xy) {
  return GetPixelCoordinates(xy.data());
}

}  // namespace pixsfm