#pragma once

#include <ceres/ceres.h>

#include <iomanip>
#include <unordered_map>

#include <math.h>

#include "util/src/simple_logger.h"

namespace pixsfm {

inline void MergeIterationSummary(const ceres::IterationSummary& other,
                                  ceres::IterationSummary& target,
                                  bool valid_other_iter) {
  // Value of the objective function.
  target.cost += other.cost;

  if (valid_other_iter) {
    // Not sure if this is correct
    if ((target.cost_change + other.cost_change) > 0.0) {
      target.relative_decrease =
          (target.relative_decrease * target.cost_change +
           other.relative_decrease * other.cost_change) /
          (target.cost_change + other.cost_change);
    }

    target.cost_change += other.cost_change;

    target.gradient_max_norm =
        std::max(target.gradient_max_norm, other.gradient_max_norm);

    target.gradient_norm = std::sqrt(std::pow(target.gradient_norm, 2) +
                                     std::pow(other.gradient_norm, 2));

    target.step_norm =
        std::sqrt(std::pow(target.step_norm, 2) + std::pow(other.step_norm, 2));

    target.trust_region_radius =
        std::min(target.trust_region_radius, other.trust_region_radius);

    target.step_size += other.step_size;
    target.iteration_time_in_seconds += other.iteration_time_in_seconds;
    target.step_solver_time_in_seconds += other.step_solver_time_in_seconds;
  }

  target.step_is_valid = (target.cost_change >= 0.0);

  // Time (in seconds) since the user called Solve().
  target.cumulative_time_in_seconds += other.cumulative_time_in_seconds;
}

inline void MergeSolverSummary(const ceres::Solver::Summary& other,
                               ceres::Solver::Summary& target) {
  target.initial_cost += other.initial_cost;

  target.final_cost += other.final_cost;

  target.fixed_cost += other.fixed_cost;

  target.preprocessor_time_in_seconds += other.preprocessor_time_in_seconds;

  target.minimizer_time_in_seconds += other.minimizer_time_in_seconds;

  target.postprocessor_time_in_seconds += other.postprocessor_time_in_seconds;

  target.total_time_in_seconds += other.total_time_in_seconds;

  target.linear_solver_time_in_seconds += other.linear_solver_time_in_seconds;

  target.residual_evaluation_time_in_seconds +=
      other.residual_evaluation_time_in_seconds;

  target.jacobian_evaluation_time_in_seconds +=
      other.jacobian_evaluation_time_in_seconds;

  target.inner_iteration_time_in_seconds +=
      other.inner_iteration_time_in_seconds;

  target.line_search_cost_evaluation_time_in_seconds +=
      other.line_search_cost_evaluation_time_in_seconds;

  target.line_search_gradient_evaluation_time_in_seconds +=
      other.line_search_gradient_evaluation_time_in_seconds;

  target.num_parameter_blocks += other.num_parameter_blocks;

  target.num_parameters += other.num_parameters;

  target.num_effective_parameters += other.num_effective_parameters;

  target.num_residual_blocks += other.num_residual_blocks;

  target.num_residuals += other.num_residuals;

  target.num_parameter_blocks_reduced += other.num_parameter_blocks_reduced;

  target.num_parameters_reduced += other.num_parameters_reduced;

  target.num_effective_parameters_reduced +=
      other.num_effective_parameters_reduced;

  target.num_residual_blocks_reduced += other.num_residual_blocks_reduced;

  target.num_residuals_reduced += other.num_residuals_reduced;

  target.num_successful_steps = 0;

  target.num_unsuccessful_steps = 0;

  // First extend iterations
  int n_initial_its = target.iterations.size();
  for (int i = n_initial_its; i < other.iterations.size(); i++) {
    target.iterations.push_back(other.iterations[i]);
    if (i > 0) {
      MergeIterationSummary(target.iterations[n_initial_its - 1],
                            target.iterations[i], false);
    }
  }

  // Now merge overlapping
  int n_other_iterations = other.iterations.size();
  for (int i = 0; i < n_initial_its && n_other_iterations > 0; i++) {
    auto& other_its = other.iterations[std::min(n_other_iterations - 1, i)];
    MergeIterationSummary(other_its, target.iterations[i],
                          (i < n_other_iterations));
  }
}

template <typename idx_t>
inline ceres::Solver::Summary AccumulateSummaries(
    std::unordered_map<idx_t, ceres::Solver::Summary>& summaries) {
  ceres::Solver::Summary summary;
  bool first = true;
  int i = 0;
  for (auto& pair : summaries) {
    if (first) {
      summary = pair.second;  // copy metavalues
      first = false;
    } else {
      MergeSolverSummary(pair.second, summary);
    }
  }
  // Fix valid iterations (since we do not have access to all values, these
  // might be too conservative)
  summary.num_successful_steps = summary.num_unsuccessful_steps = 0;
  for (int i = 1; i < summary.iterations.size(); i++) {
    bool success = summary.iterations[i].cost_change > 0.0;
    summary.iterations[i].step_is_valid = success;
    summary.iterations[i].step_is_successful = success;
    summary.iterations[i].step_is_nonmonotonic = success;
    if (success) {
      summary.num_successful_steps += 1;
    } else {
      summary.num_unsuccessful_steps += 1;
    }
  }
  return summary;
}

inline void PrintSolverSummary(const ceres::Solver::Summary& summary) {
  STDLOG(COUT) << std::right << std::setw(16) << "Residuals : ";
  STDLOG(COUT) << std::left << summary.num_residuals_reduced << std::endl;

  STDLOG(COUT) << std::right << std::setw(16) << "Parameters : ";
  STDLOG(COUT) << std::left << summary.num_effective_parameters_reduced
               << std::endl;

  STDLOG(COUT) << std::right << std::setw(16) << "Iterations : ";
  STDLOG(COUT) << std::left
               << summary.num_successful_steps + summary.num_unsuccessful_steps
               << std::endl;

  STDLOG(COUT) << std::right << std::setw(16) << "Time : ";
  STDLOG(COUT) << std::left << summary.total_time_in_seconds << " [s]"
               << std::endl;

  STDLOG(COUT) << std::right << std::setw(16) << "Initial cost : ";
  STDLOG(COUT) << std::right << std::setprecision(6)
               << std::sqrt(summary.initial_cost /
                            summary.num_residuals_reduced)
               << "" << std::endl;

  STDLOG(COUT) << std::right << std::setw(16) << "Final cost : ";
  STDLOG(COUT) << std::right << std::setprecision(6)
               << std::sqrt(summary.final_cost / summary.num_residuals_reduced)
               << "" << std::endl;

  STDLOG(COUT) << std::right << std::setw(16) << "Termination : ";

  std::string termination = "";

  switch (summary.termination_type) {
    case ceres::CONVERGENCE:
      termination = "Convergence";
      break;
    case ceres::NO_CONVERGENCE:
      termination = "No convergence";
      break;
    case ceres::FAILURE:
      termination = "Failure";
      break;
    case ceres::USER_SUCCESS:
      termination = "User success";
      break;
    case ceres::USER_FAILURE:
      termination = "User failure";
      break;
    default:
      termination = "Unknown";
      break;
  }

  STDLOG(COUT) << std::right << termination << std::endl;
  STDLOG(COUT) << std::endl;
}

}  // namespace pixsfm