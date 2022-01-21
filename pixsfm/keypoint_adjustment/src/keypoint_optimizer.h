#pragma once
#include "_pixsfm/src/helpers.h"
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

#include <highfive/H5DataSet.hpp>
#include <highfive/H5DataSpace.hpp>
#include <highfive/H5File.hpp>
#include <highfive/H5Group.hpp>

#include <ceres/ceres.h>
#include <colmap/base/projection.h>
#include <colmap/base/reconstruction.h>
#include <colmap/optim/bundle_adjustment.h>
#include <colmap/util/alignment.h>
#include <colmap/util/logging.h>
#include <colmap/util/misc.h>
#include <colmap/util/threading.h>
#include <colmap/util/timer.h>
#include <colmap/util/types.h>

#include "features/src/featuremap.h"
#include "features/src/featurepatch.h"
#include "features/src/featureset.h"
#include "features/src/featureview.h"
#include "features/src/references.h"

#include "base/src/interpolation.h"
#include "util/src/log_exceptions.h"
#include "util/src/statistics.h"
#include "util/src/stringprintf.h"
#include "util/src/types.h"

#include "residuals/src/feature_reference.h"
#include "residuals/src/featuremetric.h"

#include "base/src/graph.h"
#include "keypoint_adjustment/src/keypoint_adjustment_options.h"

namespace pixsfm {

class KeypointOptimizerBase {
 public:
  KeypointOptimizerBase(const KeypointOptimizerOptions& options,
                        std::shared_ptr<KeypointAdjustmentSetup> setup)
      : options_(options), problem_setup_(setup) {}
  // Helper
  inline void ParameterizeKeypoints(
      const std::unordered_set<size_t>& nodes_in_problem,
      MapNameKeypoints* keypoints, const Graph* graph);

  // Helper
  template <typename dtype>
  inline void ParameterizeKeypoints(
      const std::unordered_set<size_t>& nodes_in_problem,
      MapNameKeypoints* keypoints, const Graph* graph,
      FeatureView<dtype>& feature_view);

  inline bool SolveProblem();

  inline const ceres::Solver::Summary& Summary() const;

 protected:
  KeypointOptimizerOptions options_;
  std::shared_ptr<KeypointAdjustmentSetup> problem_setup_;
  const InterpolationConfig interpolation_config_;

  // Ceres
  std::unique_ptr<ceres::Problem> problem_;
  ceres::Solver::Summary summary_;

  std::vector<size_t> will_be_optimized_;
};

bool KeypointOptimizerBase::SolveProblem() {
  ceres::Solver::Options solver_options = options_.solver_options;

  solver_options.num_threads =
      colmap::GetEffectiveNumThreads(solver_options.num_threads);
#if CERES_VERSION_MAJOR < 2
  solver_options.num_linear_solver_threads =
      colmap::GetEffectiveNumThreads(solver_options.num_linear_solver_threads);
#endif  // CERES_VERSION_MAJOR

  std::string solver_error;
  // CHECK(solver_options.IsValid(&solver_error)) << solver_error;
  THROW_CUSTOM_CHECK_MSG(solver_options.IsValid(&solver_error),
                         std::invalid_argument, solver_error.c_str());

  ceres::Solve(solver_options, problem_.get(), &summary_);

  if (solver_options.minimizer_progress_to_stdout) {
    STDLOG(COUT) << std::endl;
  }

  if (options_.print_summary) {
    // colmap::PrintHeading2("Keypoint adjustment report");
    PrintSolverSummary(summary_);
  }

  return true;
}

const ceres::Solver::Summary& KeypointOptimizerBase::Summary() const {
  return summary_;
}

template <typename dtype>
void KeypointOptimizerBase::ParameterizeKeypoints(
    const std::unordered_set<size_t>& nodes_in_problem,
    MapNameKeypoints* keypoints, const Graph* graph,
    FeatureView<dtype>& feature_view) {
  //@TODO: Add option to set bounds for keypoints from a fixed initial position
  for (size_t node_idx : nodes_in_problem) {
    if (will_be_optimized_[node_idx]) {
      const FeatureNode* node = graph->nodes[node_idx];
      const std::string& image_name =
          graph->image_id_to_name.at(node->image_id);
      double* keypoint =
          keypoints->at(image_name).row(node->feature_idx).data();

      FeatureMap<dtype>& fmap = feature_view.GetFeatureMap(node->image_id);
      if (problem_setup_->IsNodeConstant(node)) {
        problem_->SetParameterBlockConstant(keypoint);
      } else if (options_.bound > 0.0 || fmap.IsSparse()) {
        const FeaturePatch<dtype>& fpatch =
            fmap.GetFeaturePatch(node->feature_idx);

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
          upperx = std::min(keypoint[0] + options_.bound / scale[0], upperx);
          uppery = std::min(keypoint[1] + options_.bound / scale[1], uppery);
          lowerx = std::max(keypoint[0] - options_.bound / scale[0], lowerx);
          lowery = std::max(keypoint[1] - options_.bound / scale[1], lowery);
        }

        problem_->SetParameterLowerBound(keypoint, 0, lowerx);
        problem_->SetParameterLowerBound(keypoint, 1, lowery);
        problem_->SetParameterUpperBound(keypoint, 0, upperx);
        problem_->SetParameterUpperBound(keypoint, 1, uppery);
      }
    }
  }
  //@TODO: Loop over keypoints in config that should be set constant
}

void PrintParallelKeypointOptimizerIteration(ceres::IterationSummary& summary) {
  const char* kReportRowFormat =
      "% 4d: f:% 8e d:% 3.2e g:% 3.2e h:% 3.2e "
      "rho:% 3.2e mu:% 3.2e eta:% 3.2e li:% 3d";

  std::string output = "";
  // clang-format off
    if (summary.iteration == 0) {
      output += "iter      cost      cost_change  |gradient|   |step|    tr_radius  ls_iter  iter_time  total_time\n";  // NOLINT
    }

    output += StringPrintf(
        "% 4d % 8e   % 3.2e   % 3.2e  % 3.2e  % 3.2e     % 4d   % 3.2e   % 3.2e",  // NOLINT
                   // clang-format on
          summary.iteration, summary.cost, summary.cost_change,
          summary.gradient_max_norm, summary.step_norm,
          summary.trust_region_radius, summary.linear_solver_iterations,
          summary.iteration_time_in_seconds,
          summary.cumulative_time_in_seconds);
  STDLOG(COUT) << output << std::endl;
}

}  // namespace pixsfm