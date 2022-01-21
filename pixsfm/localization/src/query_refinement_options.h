#pragma once
#include "util/src/types.h"
#include <ceres/ceres.h>

namespace pixsfm {

struct QueryBundleOptimizerOptions {
  // Ceres-Solver options.
  ceres::Solver::Options solver_options;

  // Python only options
  ceres::Problem::Options problem_options;

  QueryBundleOptimizerOptions() {
    loss.reset(new ceres::CauchyLoss(0.25));
    solver_options.function_tolerance = 0.0;
    solver_options.gradient_tolerance = 0.0;
    solver_options.parameter_tolerance = 1.0e-5;
    solver_options.minimizer_progress_to_stdout = true;
    solver_options.max_num_iterations = 100;
    solver_options.max_linear_solver_iterations = 200;
    solver_options.max_num_consecutive_invalid_steps = 10;
    solver_options.max_consecutive_nonmonotonic_steps = 10;
    solver_options.linear_solver_type = ceres::DENSE_SCHUR;
    solver_options.num_threads = -1;
#if CERES_VERSION_MAJOR < 2
    solver_options.num_linear_solver_threads = -1;
#endif  // CERES_VERSION_MAJOR
    problem_options.loss_function_ownership = ceres::DO_NOT_TAKE_OWNERSHIP;
  }

  QueryBundleOptimizerOptions(std::shared_ptr<ceres::LossFunction> loss_)
      : QueryBundleOptimizerOptions() {
    loss = loss_;
  }

  // Whether to refine the focal length parameter group.
  bool refine_focal_length = false;

  // Whether to refine the principal point parameter group.
  bool refine_principal_point = false;

  // Whether to refine the extra parameter group.
  bool refine_extra_params = false;

  // Whether to print a final summary.
  bool print_summary = true;

  bool Check() const;

  std::shared_ptr<ceres::LossFunction> loss;
};

struct QueryKeypointOptimizerOptions {
  // Ceres-Solver options.
  ceres::Solver::Options solver_options;

  QueryKeypointOptimizerOptions() {
    // CERES options.
    loss.reset(new ceres::TrivialLoss());
    solver_options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
    solver_options.max_num_iterations = 100;
    solver_options.minimizer_progress_to_stdout = true;
    solver_options.max_num_consecutive_invalid_steps = 10;
    solver_options.function_tolerance = 0.0;
    solver_options.gradient_tolerance = 0.0;
    solver_options.parameter_tolerance = 1.0e-4;
    solver_options.num_threads = -1;
#if CERES_VERSION_MAJOR < 2
    solver_options.num_linear_solver_threads = -1;
#endif  // CERES_VERSION_MAJOR
    problem_options.loss_function_ownership = ceres::DO_NOT_TAKE_OWNERSHIP;
  }

  QueryKeypointOptimizerOptions(std::shared_ptr<ceres::LossFunction> loss_)
      : QueryKeypointOptimizerOptions() {
    loss = loss_;
  }

  // Whether to print a final summary.
  bool print_summary = true;

  bool Check() const;

  // How much pixels in image space the keypoints are allowed to move during
  // optimization.
  double bound = -1.0;

  // Python only options
  ceres::Problem::Options problem_options;

  std::shared_ptr<ceres::LossFunction> loss;
};

struct GeometricPoseEstimationOptions {
  int implemented = 0;
};

}  // namespace pixsfm