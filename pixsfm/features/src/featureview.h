#pragma once
#include "_pixsfm/src/helpers.h"
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

#include <colmap/base/reconstruction.h>
#include <colmap/util/logging.h>
#include <colmap/util/types.h>

#include <boost/serialization/vector.hpp>
#include <utility>
#include <vector>

#include "base/src/graph.h"

#include "util/src/log_exceptions.h"
#include "util/src/misc.h"
#include "util/src/simple_logger.h"
#include "util/src/types.h"

#include "features/src/featuremap.h"
#include "features/src/featurepatch.h"
#include "features/src/featureset.h"

namespace pixsfm {

// After initialization always in loaded state
template <typename dtype>
class FeatureView {
 public:
  // KA Constructor
  FeatureView(
      FeatureSet<dtype>* feature_set,
      std::unordered_map<colmap::image_t, std::string>& image_id_to_name);

  FeatureView(FeatureSet<dtype>* feature_set,
              std::unordered_set<std::string>& image_names);

  FeatureView(FeatureSet<dtype>* feature_set,
              std::unordered_map<std::string, std::vector<colmap::point2D_t>>&
                  req_patches);

  FeatureView(FeatureSet<dtype>* feature_set, const Graph* graph,
              const std::unordered_set<size_t>& required_nodes);

  FeatureView(FeatureSet<dtype>* feature_set, const Graph* graph);

  // BA Constructor reduced
  FeatureView(FeatureSet<dtype>* feature_set,
              const colmap::Reconstruction* reconstruction,
              std::unordered_set<colmap::image_t> required_image_ids);

  // BA Constructor reduced
  FeatureView(FeatureSet<dtype>* feature_set,
              const colmap::Reconstruction* reconstruction,
              const std::unordered_set<colmap::point3D_t>& point3D_ids);

  // BA Constructor all
  FeatureView(FeatureSet<dtype>* feature_set,
              const colmap::Reconstruction* reconstruction);

  FeatureView(const FeatureView<dtype>& other);
  FeatureView(FeatureView<dtype>&& other);

  FeatureView<dtype>& operator=(const FeatureView<dtype>& other);
  FeatureView<dtype>& operator=(FeatureView<dtype>&& other);

  // ~FeatureView(); Unload H5 FeatureMaps
  int Channels() const;
  FeatureMap<dtype>& GetFeatureMap(colmap::image_t image_id);
  FeatureMap<dtype>& GetFeatureMap(const std::string& image_name);

  FeaturePatch<dtype>& GetFeaturePatch(colmap::image_t image_id,
                                       colmap::point2D_t point2D_idx);

  FeaturePatch<dtype>& GetFeaturePatch(const std::string& image_name,
                                       colmap::point2D_t point2D_idx);

  bool HasFeaturePatch(colmap::image_t image_id, colmap::point2D_t point2D_idx);

  bool HasFeaturePatch(const std::string& image_name,
                       colmap::point2D_t point2D_idx);

  std::unordered_map<colmap::image_t, std::string>& Mapping();

  colmap::image_t FindImageId(const std::string& image_name);

  size_t ReservedMemory() const;

  void Clear();

  ~FeatureView();

 private:
  void LoadRequiredPatches();
  void LoadAllPatches();

  FeatureSet<dtype>* feature_set_;
  std::unordered_map<colmap::image_t, std::string> image_id_to_name_;
  std::unordered_map<std::string, std::vector<colmap::point2D_t>> req_patches_;
};

}  // namespace pixsfm