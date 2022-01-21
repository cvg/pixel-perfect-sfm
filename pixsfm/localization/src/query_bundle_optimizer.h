

#pragma once
#include "_pixsfm/src/helpers.h"
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

#include <ceres/ceres.h>
#include <colmap/base/reconstruction.h>
#include <colmap/optim/bundle_adjustment.h>
#include <colmap/util/logging.h>
#include <colmap/util/misc.h>
#include <colmap/util/threading.h>
#include <colmap/util/types.h>

#include "features/src/featuremap.h"
#include "features/src/featurepatch.h"

#include "base/src/interpolation.h"
#include "util/src/log_exceptions.h"
#include "util/src/statistics.h"
#include "util/src/types.h"

#include "localization/src/query_refinement_options.h"
#include "residuals/src/feature_reference.h"

namespace pixsfm {
// Interface
#define FEATURE_REFERENCE_CASES          \
  REGISTER_METHOD(128, 1)                \
  REGISTER_METHOD(1, 1)                  \
  THROW_EXCEPTION(std::invalid_argument, \
                  "Unsupported dimensions (CHANNELS,N_NODES).");

class QueryBundleOptimizer {
 public:
  QueryBundleOptimizer(const QueryBundleOptimizerOptions& options,
                       const InterpolationConfig& interpolation_config)
      : options_(options), interpolation_config_(interpolation_config) {
    //@TODO
  }

  bool SolveProblem(ceres::Problem* problem);

  template <int CHANNELS, int N_NODES, typename dtype>
  inline ceres::ResidualBlockId AddFeatureReferenceResidual(
      ceres::Problem* problem, int camera_model_id, double* camera_params_data,
      double* qvec_data, double* tvec_data, double* xyz_data,
      const double* reference_descriptor_data,
      const double* node_offsets3D,  // CAN BE NULL if N_NODES==1
      const FeaturePatch<dtype>& patch, ceres::LossFunction* loss_function);

  template <typename dtype>
  inline ceres::ResidualBlockId AddFeatureReferenceResidual(
      ceres::Problem* problem, int camera_model_id, double* camera_params_data,
      double* qvec_data, double* tvec_data, double* xyz_data,
      const double* reference_descriptor_data,
      const double* node_offsets3D,  // CAN BE NULL if N_NODES==1
      const FeaturePatch<dtype>& patch, ceres::LossFunction* loss_function);

  void ParameterizeQuery(ceres::Problem* problem, colmap::Camera& camera,
                         double* qvec, double* tvec);

 protected:
  QueryBundleOptimizerOptions options_;
  InterpolationConfig interpolation_config_;
};

bool QueryBundleOptimizer::SolveProblem(ceres::Problem* problem) {
  if (problem->NumResiduals() == 0) {
    return false;
  }

  ceres::Solver::Options solver_options = options_.solver_options;

  solver_options.num_threads =
      colmap::GetEffectiveNumThreads(solver_options.num_threads);
#if CERES_VERSION_MAJOR < 2
  solver_options.num_linear_solver_threads =
      colmap::GetEffectiveNumThreads(solver_options.num_linear_solver_threads);
#endif  // CERES_VERSION_MAJOR

  std::string solver_error;
  THROW_CUSTOM_CHECK_MSG(solver_options.IsValid(&solver_error),
                         std::invalid_argument, solver_error.c_str());

  ceres::Solver::Summary summary;
  ceres::Solve(solver_options, problem, &summary);
  if (solver_options.minimizer_progress_to_stdout) {
    std::cout << std::endl;
  }

  if (options_.print_summary) {
    colmap::PrintHeading2("Query Bundle adjustment report");
    PrintSolverSummary(
        summary);  // We need to replace this with our own Printer!!!
  }

  STDLOG(INFO) << "QBA Time: " << summary.total_time_in_seconds
               << "s, cost change: "
               << std::sqrt(summary.initial_cost /
                            summary.num_residuals_reduced)
               << " --> "
               << std::sqrt(summary.final_cost / summary.num_residuals_reduced)
               << std::endl;

  // TearDown(reconstruction);
  return true;
}

void QueryBundleOptimizer::ParameterizeQuery(ceres::Problem* problem,
                                             colmap::Camera& camera,
                                             double* qvec, double* tvec) {
  // Parametrize Extrinsics
  ceres::LocalParameterization* quaternion_parameterization =
      new ceres::QuaternionParameterization;
  problem->SetParameterization(qvec, quaternion_parameterization);

  // Parametrize Intrinsics
  const bool constant_camera = !options_.refine_focal_length &&
                               !options_.refine_principal_point &&
                               !options_.refine_extra_params;
  if (constant_camera) {
    problem->SetParameterBlockConstant(camera.ParamsData());
  } else {
    std::vector<int> const_camera_params;
    if (!options_.refine_focal_length) {
      const std::vector<size_t>& params_idxs = camera.FocalLengthIdxs();
      const_camera_params.insert(const_camera_params.end(), params_idxs.begin(),
                                 params_idxs.end());
    }
    if (!options_.refine_principal_point) {
      const std::vector<size_t>& params_idxs = camera.PrincipalPointIdxs();
      const_camera_params.insert(const_camera_params.end(), params_idxs.begin(),
                                 params_idxs.end());
    }
    if (!options_.refine_extra_params) {
      const std::vector<size_t>& params_idxs = camera.ExtraParamsIdxs();
      const_camera_params.insert(const_camera_params.end(), params_idxs.begin(),
                                 params_idxs.end());
    }

    if (const_camera_params.size() > 0) {
      ceres::SubsetParameterization* camera_params_parameterization =
          new ceres::SubsetParameterization(
              static_cast<int>(camera.NumParams()), const_camera_params);
      problem->SetParameterization(camera.ParamsData(),
                                   camera_params_parameterization);
    }
  }
}

template <typename dtype>
ceres::ResidualBlockId QueryBundleOptimizer::AddFeatureReferenceResidual(
    ceres::Problem* problem, int camera_model_id, double* camera_params_data,
    double* qvec_data, double* tvec_data, double* xyz_data,
    const double* reference_descriptor_data,
    const double* node_offsets3D,  // CAN BE NULL if N_NODES==1
    const FeaturePatch<dtype>& patch, ceres::LossFunction* loss_function) {
  size_t channels = patch.Channels();
  size_t n_nodes = interpolation_config_.nodes.size();
#define REGISTER_METHOD(CHANNELS, N_NODES)                                  \
  if (channels == CHANNELS && n_nodes == N_NODES) {                         \
    return AddFeatureReferenceResidual<CHANNELS, N_NODES>(                  \
        problem, camera_model_id, camera_params_data, qvec_data, tvec_data, \
        xyz_data, reference_descriptor_data, node_offsets3D, patch,         \
        loss_function);                                                     \
  }
  FEATURE_REFERENCE_CASES
#undef REGISTER_METHOD
}

template <int CHANNELS, int N_NODES, typename dtype>
ceres::ResidualBlockId QueryBundleOptimizer::AddFeatureReferenceResidual(
    ceres::Problem* problem, int camera_model_id, double* camera_params_data,
    double* qvec_data, double* tvec_data, double* xyz_data,
    const double* reference_descriptor_data,
    const double* node_offsets3D,  // CAN BE NULL if N_NODES==1
    const FeaturePatch<dtype>& patch, ceres::LossFunction* loss_function) {
  // CREATE DEFAULT COST FUNCTION //
  ceres::CostFunction* feature_cost_function = nullptr;

  feature_cost_function =
      CreateFeatureReferenceCostFunctor<CHANNELS, N_NODES, -1>(
          camera_model_id, patch, reference_descriptor_data, node_offsets3D,
          interpolation_config_);

  // ADD COST FUNCTION TO PROBLEM //
  ceres::ResidualBlockId block_id =
      problem->AddResidualBlock(feature_cost_function, loss_function, qvec_data,
                                tvec_data, xyz_data, camera_params_data);

  return block_id;
}

#undef FEATURE_REFERENCE_CASES

}  // namespace pixsfm