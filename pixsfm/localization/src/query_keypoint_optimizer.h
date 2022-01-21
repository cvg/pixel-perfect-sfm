#pragma once

#include "_pixsfm/src/helpers.h"
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

#include <ceres/ceres.h>
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

#define FEATURE_REFERENCE_CASES          \
  REGISTER_METHOD(128, 1)                \
  REGISTER_METHOD(1, 1)                  \
  THROW_EXCEPTION(std::invalid_argument, \
                  "Unsupported dimensions (CHANNELS,N_NODES).");

class QueryKeypointOptimizer {
 public:
  QueryKeypointOptimizer(const QueryKeypointOptimizerOptions& options,
                         const InterpolationConfig& interpolation_config)
      : options_(options), interpolation_config_(interpolation_config) {
    //@TODO
  }

  bool SolveProblem(ceres::Problem* problem);

  template <int CHANNELS, int N_NODES, typename dtype>
  inline ceres::ResidualBlockId AddFeatureReferenceResidual(
      ceres::Problem* problem, double* xy,
      const double* reference_descriptor_data, const FeaturePatch<dtype>& patch,
      ceres::LossFunction* loss_function);

  template <typename dtype>
  inline ceres::ResidualBlockId AddFeatureReferenceResidual(
      ceres::Problem* problem, double* xy,
      const double* reference_descriptor_data, const FeaturePatch<dtype>& patch,
      ceres::LossFunction* loss_function);

  template <typename dtype>
  inline void ParameterizeKeypoint(ceres::Problem* problem, double* xy,
                                   const FeaturePatch<dtype>& fpatch,
                                   bool is_sparse = true);

 protected:
  QueryKeypointOptimizerOptions options_;
  InterpolationConfig interpolation_config_;
};

bool QueryKeypointOptimizer::SolveProblem(ceres::Problem* problem) {
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
    colmap::PrintHeading2("Query Keypoint Adjustment Report");
    PrintSolverSummary(
        summary);  // We need to replace this with our own Printer!!!
  }

  STDLOG(INFO) << "QKA Time: " << summary.total_time_in_seconds
               << "s, cost change: "
               << std::sqrt(summary.initial_cost /
                            summary.num_residuals_reduced)
               << " --> "
               << std::sqrt(summary.final_cost / summary.num_residuals_reduced)
               << std::endl;

  return true;
}

template <typename dtype>
ceres::ResidualBlockId QueryKeypointOptimizer::AddFeatureReferenceResidual(
    ceres::Problem* problem, double* xy,
    const double* reference_descriptor_data, const FeaturePatch<dtype>& patch,
    ceres::LossFunction* loss_function) {
  size_t channels = patch.Channels();
  size_t n_nodes = interpolation_config_.nodes.size();
#define REGISTER_METHOD(CHANNELS, N_NODES)                             \
  if (channels == CHANNELS && n_nodes == N_NODES) {                    \
    return AddFeatureReferenceResidual<CHANNELS, N_NODES>(             \
        problem, xy, reference_descriptor_data, patch, loss_function); \
  }
  FEATURE_REFERENCE_CASES
#undef REGISTER_METHOD
}

template <int CHANNELS, int N_NODES, typename dtype>
ceres::ResidualBlockId QueryKeypointOptimizer::AddFeatureReferenceResidual(
    ceres::Problem* problem, double* xy,
    const double* reference_descriptor_data, const FeaturePatch<dtype>& patch,
    ceres::LossFunction* loss_function) {
  // CREATE DEFAULT COST FUNCTION //
  ceres::CostFunction* feature_cost_function =
      FeatureReference2DCostFunctor<dtype, CHANNELS, N_NODES>::Create(
          patch, interpolation_config_, reference_descriptor_data);

  // ADD COST FUNCTION TO PROBLEM //
  ceres::ResidualBlockId block_id =
      problem->AddResidualBlock(feature_cost_function, loss_function, xy);

  return block_id;
}

template <typename dtype>
void QueryKeypointOptimizer::ParameterizeKeypoint(
    ceres::Problem* problem, double* xy, const FeaturePatch<dtype>& fpatch,
    bool is_sparse) {
  if (options_.bound > 0.0 || is_sparse) {
    const double* scale = fpatch.Scale().data();
    double dx = fpatch.Width() / scale[0];   // In Pixel!
    double dy = fpatch.Height() / scale[1];  // In Pixel!
    double lowerx = (fpatch.X0() + 0.5) / scale[0];
    double lowery = (fpatch.Y0() + 0.5) / scale[1];

    double upperx = lowerx + dx;
    double uppery = lowery + dy;

    if (options_.bound > 0.0) {
      // const double* initial_keypoint =
      // initial_keypoints[node->image_id][node->feature_idx].data();
      const double* ref_xy =
          xy;  // could set to other (initial) xy in multiscale optim.
      upperx = std::min(ref_xy[0] + options_.bound / scale[0], upperx);
      uppery = std::min(ref_xy[1] + options_.bound / scale[1], uppery);
      lowerx = std::max(ref_xy[0] - options_.bound / scale[0], lowerx);
      lowery = std::max(ref_xy[1] - options_.bound / scale[1], lowery);
    }

    problem->SetParameterLowerBound(xy, 0, lowerx);
    problem->SetParameterLowerBound(xy, 1, lowery);
    problem->SetParameterUpperBound(xy, 0, upperx);
    problem->SetParameterUpperBound(xy, 1, uppery);
  }
}

#undef FEATURE_REFERENCE_CASES

}  // namespace pixsfm