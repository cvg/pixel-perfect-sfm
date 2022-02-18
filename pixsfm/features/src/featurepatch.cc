#include "features/src/featurepatch.h"
#include "util/src/simple_logger.h"
#include <assert.h>
#include <atomic>
#include <mutex>
namespace pixsfm {

template <typename dtype>
FeaturePatch<dtype>::FeaturePatch()
    : data_(0),
      data_ptr_(nullptr),
      corner_({0, 0}),
      scale_({1.0, 1.0}),
      shape_({0, 0, 0}) {}

template <typename dtype>
FeaturePatch<dtype>::FeaturePatch(
    py::array_t<dtype, py::array::c_style> pyarray, Eigen::Vector2i& corner,
    Eigen::Vector2d& scale, bool do_copy)
    : data_(0),
      data_ptr_(nullptr),
      corner_(corner),
      scale_(scale),
      shape_({0, 0, 0}) {
  // request a buffer descriptor from Python
  py::buffer_info buffer_info = pyarray.request();

  // extract data an shape of input array
  data_ptr_ = static_cast<dtype*>(buffer_info.ptr);
  std::vector<ssize_t> shape = buffer_info.shape;
  shape_[0] = shape[0];
  shape_[1] = shape[1];
  shape_[2] = shape[2];

  if (do_copy) {
    ssize_t size = Size();
    THROW_CHECK_EQ(buffer_info.size, size);
    data_.assign(data_ptr_, data_ptr_ + size);
    data_ptr_ = data_.data();
  }

  status_.is_locked = true;
  status_.reference_count = 1;

  ref_array_ptr_.reset(new py::array_t<dtype, py::array::c_style>(pyarray));
}

template <typename dtype>
FeaturePatch<dtype>::FeaturePatch(dtype* data_ptr, std::array<int, 3> shape,
                                  Eigen::Ref<const Eigen::Vector2i> corner,
                                  Eigen::Ref<const Eigen::Vector2d> scale,
                                  bool do_copy)
    : data_ptr_(data_ptr), corner_(corner), scale_(scale), shape_(shape) {
  if (do_copy) {
    ssize_t size = Size();
    data_.assign(data_ptr, data_ptr + size);
    data_ptr_ = data_.data();
  }

  status_.is_locked = (data_ptr_) ? true : false;
  status_.reference_count = (data_ptr_) ? 1 : 0;
}

template <typename dtype>
FeaturePatch<dtype>::FeaturePatch(HighFive::DataSet& dataset, bool fill)
    : FeaturePatch() {
  LoadFromH5Dataset(dataset, fill);
  status_.is_locked = false;
}

template <typename dtype>
FeaturePatch<dtype>::FeaturePatch(const FeaturePatch<dtype>& other)
    : data_(other.data_),
      data_ptr_(data_.data()),
      shape_(other.shape_),
      corner_(other.corner_),
      scale_(other.scale_),
      status_(other.status_),
      upsampling_factor_(other.upsampling_factor_),
      ref_array_ptr_(other.ref_array_ptr_) {
  if (other.IsReference()) {
    data_ptr_ = other.data_ptr_;
  } else {
    data_ptr_ = data_.data();
  }
}

template <typename dtype>
FeaturePatch<dtype>::FeaturePatch(FeaturePatch<dtype>&& other)
    : FeaturePatch() {
  std::lock_guard<std::mutex> lock(other.mutex_);
  shape_ = std::move(other.shape_);
  corner_ = std::move(other.corner_);
  scale_ = std::move(other.scale_);
  status_ = std::move(other.status_);
  upsampling_factor_ = std::move(other.upsampling_factor_);
  ref_array_ptr_ = std::move(other.ref_array_ptr_);
  if (other.IsReference()) {
    data_ptr_ = std::move(other.data_ptr_);
  } else {
    data_ = std::move(other.data_);
    data_ptr_ = data_.data();
  }
  other.data_ptr_ = nullptr;
}

template <typename dtype>
FeaturePatch<dtype>& FeaturePatch<dtype>::operator=(
    const FeaturePatch<dtype>& other) {
  std::lock_guard<std::mutex> lock(mutex_);
  shape_ = other.shape_;
  corner_ = other.corner_;
  scale_ = other.scale_;
  ref_array_ptr_ = other.ref_array_ptr_;
  status_ = other.status_;
  upsampling_factor_ = other.upsampling_factor_;
  if (other.IsReference()) {
    data_ptr_ = other.data_ptr_;
    data_.clear();
    data_.shrink_to_fit();
  } else {
    data_ = other.data_;
    data_ptr_ = data_.data();
  }

  return *this;
}

template <typename dtype>
FeaturePatch<dtype>& FeaturePatch<dtype>::operator=(
    FeaturePatch<dtype>&& other) {
  std::lock(mutex_, other.mutex_);
  std::lock_guard<std::mutex> lock(mutex_, std::adopt_lock);
  std::lock_guard<std::mutex> lock_other(other.mutex_, std::adopt_lock);
  // Swap of data_ might be the better option
  shape_ = std::move(other.shape_);
  corner_ = std::move(other.corner_);
  scale_ = std::move(other.scale_);
  ref_array_ptr_ = std::move(other.ref_array_ptr_);
  status_ = std::move(other.status_);
  upsampling_factor_ = std::move(other.upsampling_factor_);

  if (other.IsReference()) {
    data_ptr_ = std::move(other.data_ptr_);
    data_.clear();
    data_.shrink_to_fit();
  } else {
    data_ = std::move(other.data_);
    data_ptr_ = data_.data();
  }
  other.data_ptr_ = nullptr;
  return *this;
}

template <typename dtype>
dtype FeaturePatch<dtype>::GetEntry(const int y, const int x, const int c) {
  return *GetEntryPtr(y, x, c);
}

template <typename dtype>
dtype* FeaturePatch<dtype>::GetEntryPtr(int y, int x, int c) {
  return &data_ptr_[(y * Width() + x) * Channels() + c];
}

template <typename dtype>
py::array_t<dtype, py::array::c_style> FeaturePatch<dtype>::AsNumpyArray() {
  py::str dummy_data_owner;  // Workaround, we need a valid pybind object as
                             // base
  auto out_array = py::array_t<dtype, py::array::c_style>(
      {Height(), Width(), Channels()},  // shape
      {Channels() * Width() * sizeof(dtype), Channels() * sizeof(dtype),
       sizeof(dtype)},
      // C-style contiguous strides for double
      data_ptr_,
      dummy_data_owner);  // numpy array references this parent
  return out_array;
}

// Storage Format 1
template <typename dtype>
size_t FeaturePatch<dtype>::LoadFromH5Dataset(HighFive::DataSet& dataset,
                                              bool fill) {
  // mutex_.lock();
  std::lock_guard<std::mutex> lock(mutex_);
  if (IsReference() && HasData()) {
    // mutex_.unlock();
    throw(std::runtime_error(
        "Reference Patch can not be overwritten by H5 Data."));
  }

  if (HasData()) {
    ++status_.reference_count;
    if (status_.reference_count < 1) {
      status_.reference_count = 1;
    }
    return 0;  // Nothing to load
  }

  size_t memory_before = CurrentMemory();

  // Load Shapes -- SAFETY CHECK
  std::vector<size_t> shape = dataset.getSpace().getDimensions();

  for (int i = 0; i < 3; i++) {
    shape_[i] = static_cast<int>(shape[i]);
  }

  // SAFETY OVERRIDE
  dataset.getAttribute("corner").read(corner_.data());
  dataset.getAttribute("scale").read(scale_.data());

  // Load Data
  if (fill) {
    data_.resize(Height() * Width() * Channels());
    dataset.read(data_.data());

    data_ptr_ = data_.data();  // Make Real Patch
    status_.reference_count =
        1;  // Note that this patch is referenced by loading.
    return NumBytes();
  } else {
    status_.reference_count =
        0;  // if no data and no fill, we set the ref count to 0.
    return 0;
  }
}

// Storage Format 1
template <typename dtype>
size_t FeaturePatch<dtype>::LoadDataFromH5Chunk(HighFive::Selection& chunk,
                                                bool fill) {
  std::lock_guard<std::mutex> lock(mutex_);
  if (IsReference() && HasData()) {
    throw(std::runtime_error(
        "Reference Patch can not be overwritten by H5 Data."));
  }

  if (HasData()) {
    ++status_.reference_count;
    if (status_.reference_count < 1) {
      status_.reference_count = 1;
    }
    return 0;  // Nothing to load
  }

  size_t memory_before = CurrentMemory();

  // Load Shapes -- SAFETY CHECK
  std::vector<size_t> shape = chunk.getMemSpace().getDimensions();
  for (int i = 0; i < 3; i++) {
    shape_[i] = static_cast<int>(shape[i + 1]);
  }

  // Load Data
  if (fill) {
    data_.resize(Height() * Width() * Channels());
    chunk.read(data_.data());
    data_ptr_ = data_.data();  // Make Real Patch
    status_.reference_count =
        1;  // Note that this patch is referenced by loading.
    return NumBytes();
  } else {
    status_.reference_count =
        0;  // if no data and no fill, we set the ref count to 0.
    return 0;
  }
  // mutex_.unlock();
}

// Decrements Reference Count, but does NOT delete Data
// should become critical section in parallel
template <typename dtype>
void FeaturePatch<dtype>::Unload() {
  std::lock_guard<std::mutex> lock(mutex_);
  if (!status_.is_locked && HasData() && !IsReference()) {
    --status_.reference_count;
  }
}

// Deletes Data if reference_count==0
// Should become critical section in parallel
template <typename dtype>
size_t FeaturePatch<dtype>::Flush() {
  std::lock_guard<std::mutex> lock(mutex_);
  if (!status_.is_locked && status_.reference_count <= 0 && HasData() &&
      !IsReference()) {
    data_.clear();
    data_.shrink_to_fit();
    data_ptr_ = nullptr;
    status_.reference_count = 0;  // Reset
    return NumBytes();
  }
  return 0;
}

template <typename dtype>
size_t FeaturePatch<dtype>::CurrentMemory() const {
  // We only count floating point data.
  size_t num_bytes = 0;
  if (HasData()) {
    num_bytes += NumBytes();  // Bytes of data -> main contribution
  }
  return num_bytes;
}

template <typename dtype>
void FeaturePatch<dtype>::Lock() {
  status_.is_locked = true;
}

template <typename dtype>
void FeaturePatch<dtype>::Allocate() {
  std::lock_guard<std::mutex> lock(mutex_);
  if (IsReference() || HasData() || status_.is_locked) {
    THROW_EXCEPTION(std::runtime_error, "Allocation invalid!");
  }
  data_.resize(Size());
  data_ptr_ = data_.data();
  status_.is_locked = true;
  status_.reference_count = 1;
}

/*******************************************************************************
FeaturePatch: Template Specializations
*******************************************************************************/

template class FeaturePatch<double>;
template class FeaturePatch<float>;
template class FeaturePatch<float16>;

}  // namespace pixsfm