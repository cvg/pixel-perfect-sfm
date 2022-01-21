#pragma once

#pragma once

#include <ceres/ceres.h>
#include <colmap/base/reconstruction.h>
#include <colmap/optim/bundle_adjustment.h>
#include <colmap/util/types.h>

#include "base/src/interpolation.h"
#include "features/src/featuremap.h"
#include "features/src/featurepatch.h"
#include "features/src/references.h"
#include "util/src/log_exceptions.h"
#include "util/src/types.h"

#include "localization/src/query_keypoint_optimizer.h"

namespace pixsfm {

#define FEATURE_REFERENCE_CASES          \
  REGISTER_METHOD(128, 1)                \
  REGISTER_METHOD(1, 16)                 \
  THROW_EXCEPTION(std::invalid_argument, \
                  "Unsupported dimensions (CHANNELS,N_NODES).");

class SingleQueryKeypointOptimizer : public QueryKeypointOptimizer {
  using QueryKO = QueryKeypointOptimizer;
  using QueryKO::interpolation_config_;
  using QueryKO::options_;

 public:
  SingleQueryKeypointOptimizer(const QueryKeypointOptimizerOptions& options,
                               const InterpolationConfig& interpolation_config)
      : QueryKO::QueryKeypointOptimizer(options, interpolation_config) {}

  template <typename dtype, typename RefType>
  bool RunQuery(Eigen::Ref<KeypointMatrixd> keypoints, FeatureMap<dtype>& fmap,
                RefType& references,
                std::vector<colmap::point2D_t>* patch_idxs = NULL,
                std::vector<bool>* inliers = NULL);

 protected:
  template <int CHANNELS, int N_NODES, typename dtype>
  bool RunQuery(Eigen::Ref<KeypointMatrixd> keypoints, FeatureMap<dtype>& fmap,
                std::vector<Eigen::Ref<DescriptorMatrixXd>>& references,
                std::vector<colmap::point2D_t>* patch_idxs = NULL,
                std::vector<bool>* inliers = NULL);

  template <int CHANNELS, int N_NODES, typename dtype>
  bool RunQuery(Eigen::Ref<KeypointMatrixd> keypoints, FeatureMap<dtype>& fmap,
                std::vector<std::vector<DescriptorMatrixXd>>& references_per_kp,
                std::vector<colmap::point2D_t>* patch_idxs = NULL,
                std::vector<bool>* inliers = NULL);

  template <int CHANNELS, int N_NODES, typename dtype>
  bool RunQuery(Eigen::Ref<KeypointMatrixd> keypoints, FeatureMap<dtype>& fmap,
                std::vector<Reference>& references,
                std::vector<colmap::point2D_t>* patch_idxs = NULL,
                std::vector<bool>* inliers = NULL);
};

template <typename dtype, typename RefType>
bool SingleQueryKeypointOptimizer::RunQuery(
    Eigen::Ref<KeypointMatrixd> keypoints, FeatureMap<dtype>& fmap,
    RefType& references, std::vector<colmap::point2D_t>* patch_idxs,
    std::vector<bool>* inliers) {
  if (patch_idxs) {
    THROW_CHECK_EQ(patch_idxs->size(), keypoints.rows())
  }
  size_t channels = fmap.Channels();
  size_t n_nodes = interpolation_config_.nodes.size();
#define REGISTER_METHOD(CHANNELS, N_NODES)                          \
  if (channels == CHANNELS && n_nodes == N_NODES) {                 \
    return RunQuery<CHANNELS, N_NODES>(keypoints, fmap, references, \
                                       patch_idxs, inliers);        \
  }
  FEATURE_REFERENCE_CASES
#undef REGISTER_METHOD
}

template <int CHANNELS, int N_NODES, typename dtype>
bool SingleQueryKeypointOptimizer::RunQuery(
    Eigen::Ref<KeypointMatrixd> keypoints, FeatureMap<dtype>& fmap,
    std::vector<Eigen::Ref<DescriptorMatrixXd>>& references,
    std::vector<colmap::point2D_t>* patch_idxs, std::vector<bool>* inliers) {
  ceres::LossFunction* loss_function = options_.loss.get();

  THROW_CHECK_EQ(references.size(), keypoints.rows());

  ceres::Problem problem(options_.problem_options);

  for (int idx = 0; idx < keypoints.rows(); idx++) {
    if (inliers) {
      if (!(*inliers)[idx]) {
        continue;
      }
    }

    colmap::point2D_t patch_idx = idx;
    if (patch_idxs) {
      patch_idx = patch_idxs->at(idx);
    }

    QueryKO::AddFeatureReferenceResidual<CHANNELS, N_NODES>(
        &problem, keypoints.data() + 2 * idx, references[idx].data(),
        fmap.GetFeaturePatch(patch_idx), loss_function);

    QueryKO::ParameterizeKeypoint(&problem, keypoints.data() + 2 * idx,
                                  fmap.GetFeaturePatch(patch_idx),
                                  fmap.IsSparse());
  }

  return QueryKO::SolveProblem(&problem);
}

template <int CHANNELS, int N_NODES, typename dtype>
bool SingleQueryKeypointOptimizer::RunQuery(
    Eigen::Ref<KeypointMatrixd> keypoints, FeatureMap<dtype>& fmap,
    std::vector<std::vector<DescriptorMatrixXd>>& references_per_kp,
    std::vector<colmap::point2D_t>* patch_idxs, std::vector<bool>* inliers) {
  ceres::LossFunction* loss_function = options_.loss.get();

  THROW_CHECK_EQ(references_per_kp.size(), keypoints.rows());

  ceres::Problem problem(options_.problem_options);

  for (int idx = 0; idx < keypoints.rows(); idx++) {
    if (inliers) {
      if (!(*inliers)[idx]) {
        continue;
      }
    }

    colmap::point2D_t patch_idx = idx;
    if (patch_idxs) {
      patch_idx = patch_idxs->at(idx);
    }

    bool added_to_problem = false;
    for (int k = 0; k < references_per_kp[idx].size(); k++) {
      QueryKO::AddFeatureReferenceResidual<CHANNELS, N_NODES>(
          &problem, keypoints.data() + 2 * idx,
          references_per_kp[idx][k].data(), fmap.GetFeaturePatch(patch_idx),
          loss_function);
      added_to_problem = true;
    }

    if (added_to_problem) {
      QueryKO::ParameterizeKeypoint(&problem, keypoints.data() + 2 * idx,
                                    fmap.GetFeaturePatch(patch_idx),
                                    fmap.IsSparse());
    }
  }

  return QueryKO::SolveProblem(&problem);
}

template <int CHANNELS, int N_NODES, typename dtype>
bool SingleQueryKeypointOptimizer::RunQuery(
    Eigen::Ref<KeypointMatrixd> keypoints, FeatureMap<dtype>& fmap,
    std::vector<Reference>& references,
    std::vector<colmap::point2D_t>* patch_idxs, std::vector<bool>* inliers) {
  ceres::LossFunction* loss_function = options_.loss.get();

  THROW_CHECK_EQ(references.size(), keypoints.rows());

  ceres::Problem problem(options_.problem_options);

  for (int idx = 0; idx < keypoints.rows(); idx++) {
    if (inliers) {
      if (!(*inliers)[idx]) {
        continue;
      }
    }

    colmap::point2D_t patch_idx = idx;
    if (patch_idxs) {
      patch_idx = patch_idxs->at(idx);
    }

    if (!references[idx].HasObservations()) {
      QueryKO::AddFeatureReferenceResidual<CHANNELS, N_NODES>(
          &problem, keypoints.data() + 2 * idx,
          references[idx].DescriptorData(), fmap.GetFeaturePatch(patch_idx),
          loss_function);
    } else {
      for (auto& descr : references[idx].observations) {
        QueryKO::AddFeatureReferenceResidual<CHANNELS, N_NODES>(
            &problem, keypoints.data() + 2 * idx, descr.data(),
            fmap.GetFeaturePatch(patch_idx), loss_function);
      }
    }

    QueryKO::ParameterizeKeypoint(&problem, keypoints.data() + 2 * idx,
                                  fmap.GetFeaturePatch(patch_idx),
                                  fmap.IsSparse());
  }

  return QueryKO::SolveProblem(&problem);
}

#undef FEATURE_REFERENCE_CASES
}  // namespace pixsfm