
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

#include "localization/src/query_bundle_optimizer.h"

namespace pixsfm {

#define FEATURE_REFERENCE_CASES          \
  REGISTER_METHOD(128, 1)                \
  REGISTER_METHOD(1, 16)                 \
  REGISTER_METHOD(3, 16)                 \
  THROW_EXCEPTION(std::invalid_argument, \
                  "Unsupported dimensions (CHANNELS,N_NODES).");

class SingleQueryBundleOptimizer : public QueryBundleOptimizer {
  using QueryBO = QueryBundleOptimizer;
  using QueryBO::interpolation_config_;
  using QueryBO::options_;

 public:
  SingleQueryBundleOptimizer(const QueryBundleOptimizerOptions& options,
                             const InterpolationConfig& interpolation_config)
      : QueryBO::QueryBundleOptimizer(options, interpolation_config) {}

  template <typename dtype, typename RefType>
  bool RunQuery(Eigen::Ref<Eigen::Vector4d>& qvec,
                Eigen::Ref<Eigen::Vector3d>& tvec, colmap::Camera& camera,
                std::vector<Eigen::Vector3d>& points3D, FeatureMap<dtype>& fmap,
                RefType& references,
                std::vector<colmap::point2D_t>* patch_idxs = NULL,
                std::vector<bool>* inliers = NULL);

  template <int CHANNELS, int N_NODES, typename dtype>
  bool RunQuery(Eigen::Ref<Eigen::Vector4d>& qvec,
                Eigen::Ref<Eigen::Vector3d>& tvec, colmap::Camera& camera,
                std::vector<Eigen::Vector3d>& points3D, FeatureMap<dtype>& fmap,
                std::vector<Eigen::Ref<DescriptorMatrixXd>>& references,
                std::vector<colmap::point2D_t>* patch_idxs = NULL,
                std::vector<bool>* inliers = NULL);

  template <int CHANNELS, int N_NODES, typename dtype>
  bool RunQuery(Eigen::Ref<Eigen::Vector4d>& qvec,
                Eigen::Ref<Eigen::Vector3d>& tvec, colmap::Camera& camera,
                std::vector<Eigen::Vector3d>& points3D, FeatureMap<dtype>& fmap,
                std::vector<Reference>& references,
                std::vector<colmap::point2D_t>* patch_idxs = NULL,
                std::vector<bool>* inliers = NULL);

  template <int CHANNELS, int N_NODES, typename dtype>
  bool RunQuery(Eigen::Ref<Eigen::Vector4d>& qvec,
                Eigen::Ref<Eigen::Vector3d>& tvec, colmap::Camera& camera,
                std::vector<Eigen::Vector3d>& points3D, FeatureMap<dtype>& fmap,
                std::vector<std::vector<DescriptorMatrixXd>>& references,
                std::vector<colmap::point2D_t>* patch_idxs = NULL,
                std::vector<bool>* inliers = NULL);
};

template <typename dtype, typename RefType>
bool SingleQueryBundleOptimizer::RunQuery(
    Eigen::Ref<Eigen::Vector4d>& qvec, Eigen::Ref<Eigen::Vector3d>& tvec,
    colmap::Camera& camera, std::vector<Eigen::Vector3d>& points3D,
    FeatureMap<dtype>& fmap, RefType& references,
    std::vector<colmap::point2D_t>* patch_idxs, std::vector<bool>* inliers) {
  size_t channels = fmap.Channels();
  size_t n_nodes = interpolation_config_.nodes.size();
  if (patch_idxs) {
    THROW_CHECK_EQ(patch_idxs->size(), points3D.size())
  }
#define REGISTER_METHOD(CHANNELS, N_NODES)                                 \
  if (channels == CHANNELS && n_nodes == N_NODES) {                        \
    return RunQuery<CHANNELS, N_NODES>(qvec, tvec, camera, points3D, fmap, \
                                       references, patch_idxs, inliers);   \
  }
  FEATURE_REFERENCE_CASES
#undef REGISTER_METHOD
}

template <int CHANNELS, int N_NODES, typename dtype>
bool SingleQueryBundleOptimizer::RunQuery(
    Eigen::Ref<Eigen::Vector4d>& qvec, Eigen::Ref<Eigen::Vector3d>& tvec,
    colmap::Camera& camera, std::vector<Eigen::Vector3d>& points3D,
    FeatureMap<dtype>& fmap,
    std::vector<Eigen::Ref<DescriptorMatrixXd>>& references,
    std::vector<colmap::point2D_t>* patch_idxs, std::vector<bool>* inliers) {
  ceres::LossFunction* loss_function = options_.loss.get();

  THROW_CHECK_EQ(references.size(), points3D.size());
  THROW_CHECK_EQ(interpolation_config_.nodes.size(), 1);

  ceres::Problem problem(options_.problem_options);

  int n_corrs = points3D.size();

  for (int idx = 0; idx < n_corrs; idx++) {
    if (inliers) {
      if (!(*inliers)[idx]) {
        continue;
      }
    }

    colmap::point2D_t patch_idx = idx;
    if (patch_idxs) {
      patch_idx = patch_idxs->at(idx);
    }

    QueryBO::AddFeatureReferenceResidual<CHANNELS, N_NODES>(
        &problem, camera.ModelId(), camera.ParamsData(), qvec.data(),
        tvec.data(), points3D[idx].data(), references[idx].data(), NULL,
        fmap.GetFeaturePatch(patch_idx), loss_function);
    problem.SetParameterBlockConstant(points3D[idx].data());
  }

  QueryBO::ParameterizeQuery(&problem, camera, qvec.data(), tvec.data());

  return QueryBO::SolveProblem(&problem);
}

template <int CHANNELS, int N_NODES, typename dtype>
bool SingleQueryBundleOptimizer::RunQuery(
    Eigen::Ref<Eigen::Vector4d>& qvec, Eigen::Ref<Eigen::Vector3d>& tvec,
    colmap::Camera& camera, std::vector<Eigen::Vector3d>& points3D,
    FeatureMap<dtype>& fmap,
    std::vector<std::vector<DescriptorMatrixXd>>& references,
    std::vector<colmap::point2D_t>* patch_idxs, std::vector<bool>* inliers) {
  ceres::LossFunction* loss_function = options_.loss.get();

  THROW_CHECK_EQ(references.size(), points3D.size());
  THROW_CHECK_EQ(interpolation_config_.nodes.size(), 1);

  ceres::Problem problem(options_.problem_options);

  int n_corrs = points3D.size();

  for (int idx = 0; idx < n_corrs; idx++) {
    if (inliers) {
      if (!(*inliers)[idx]) {
        continue;
      }
    }

    colmap::point2D_t patch_idx = idx;
    if (patch_idxs) {
      patch_idx = patch_idxs->at(idx);
    }

    for (auto& descr : references[idx]) {
      QueryBO::AddFeatureReferenceResidual<CHANNELS, N_NODES>(
          &problem, camera.ModelId(), camera.ParamsData(), qvec.data(),
          tvec.data(), points3D[idx].data(), descr.data(), NULL,
          fmap.GetFeaturePatch(patch_idx), loss_function);
    }
    problem.SetParameterBlockConstant(points3D[idx].data());
  }

  QueryBO::ParameterizeQuery(&problem, camera, qvec.data(), tvec.data());

  return QueryBO::SolveProblem(&problem);
}

template <int CHANNELS, int N_NODES, typename dtype>
bool SingleQueryBundleOptimizer::RunQuery(
    Eigen::Ref<Eigen::Vector4d>& qvec, Eigen::Ref<Eigen::Vector3d>& tvec,
    colmap::Camera& camera, std::vector<Eigen::Vector3d>& points3D,
    FeatureMap<dtype>& fmap, std::vector<Reference>& references,
    std::vector<colmap::point2D_t>* patch_idxs, std::vector<bool>* inliers) {
  ceres::LossFunction* loss_function = options_.loss.get();

  THROW_CHECK_EQ(references.size(), points3D.size());

  ceres::Problem problem(options_.problem_options);

  int n_corrs = points3D.size();

  for (int idx = 0; idx < n_corrs; idx++) {
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
      QueryBO::AddFeatureReferenceResidual<CHANNELS, N_NODES>(
          &problem, camera.ModelId(), camera.ParamsData(), qvec.data(),
          tvec.data(), points3D[idx].data(), references[idx].DescriptorData(),
          references[idx].NodeOffsets3DData(), fmap.GetFeaturePatch(patch_idx),
          loss_function);
    } else {
      for (auto& descr : references[idx].observations) {
        QueryBO::AddFeatureReferenceResidual<CHANNELS, N_NODES>(
            &problem, camera.ModelId(), camera.ParamsData(), qvec.data(),
            tvec.data(), points3D[idx].data(), descr.data(), NULL,
            fmap.GetFeaturePatch(patch_idx), loss_function);
      }
    }

    problem.SetParameterBlockConstant(points3D[idx].data());
  }

  QueryBO::ParameterizeQuery(&problem, camera, qvec.data(), tvec.data());

  return QueryBO::SolveProblem(&problem);
}

#undef FEATURE_REFERENCE_CASES
}  // namespace pixsfm