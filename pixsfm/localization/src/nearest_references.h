#pragma once

#include <ceres/ceres.h>
#include <colmap/base/reconstruction.h>
#include <colmap/optim/bundle_adjustment.h>
#include <colmap/util/types.h>

#include <vector>

#include "features/src/dynamic_patch_interpolator.h"
#include "features/src/featuremap.h"
#include "features/src/featurepatch.h"
#include "features/src/references.h"

#include "base/src/interpolation.h"
#include "util/src/types.h"

namespace pixsfm {

template <typename dtype>
std::vector<DescriptorMatrixXd> FindNearestReferences(
    FeatureMap<dtype>& query_fmap,
    std::unordered_map<colmap::point3D_t, Reference>& references,
    Eigen::Ref<Eigen::Matrix<double, -1, 2, Eigen::RowMajor>>& keypoints,
    std::vector<colmap::point3D_t> point3D_ids,
    InterpolationConfig& interpolation_config,
    std::vector<colmap::point3D_t>* patch_idxs) {
  THROW_CHECK_EQ(interpolation_config.nodes.size(), 1);
  DynamicPatchInterpolator interpolator(interpolation_config);
  std::vector<DescriptorMatrixXd> closest_references;
  for (int i = 0; i < keypoints.rows(); i++) {
    Reference& reference = references.at(point3D_ids[i]);
    THROW_CHECK_MSG(reference.HasObservations(),
                    "Missing observations in references. Extract references "
                    "with option keep_observations=True.");
    Eigen::Vector2d xy = keypoints.row(i).transpose();
    colmap::point2D_t patch_idx = (patch_idxs) ? patch_idxs->at(i) : i;
    DescriptorMatrixXd query_desc = interpolator.InterpolateNodes<-1, -1>(
        query_fmap.GetFeaturePatch(patch_idx), xy);
    double min_distance = std::numeric_limits<double>::max();
    DescriptorMatrixXd nearest;
    for (DescriptorMatrixXd& descriptor : reference.observations) {
      double d = (descriptor - query_desc).squaredNorm();
      if (d < min_distance) {
        min_distance = d;
        nearest = descriptor;
      }
    }
    closest_references.push_back(nearest);
  }
  return closest_references;
}

}  // namespace pixsfm