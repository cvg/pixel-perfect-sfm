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
#include <colmap/scene/projection.h>
#include <colmap/scene/reconstruction.h>
#include <colmap/estimators/bundle_adjustment.h>
#include <colmap/util/types.h>

#include "features/src/featuremap.h"
#include "features/src/featurepatch.h"
#include "features/src/featureset.h"

#include "util/src/types.h"

namespace pixsfm {

// Configuration container to setup bundle adjustment problems.
class BundleAdjustmentSetup : public colmap::BundleAdjustmentConfig {
 public:
  BundleAdjustmentSetup() : colmap::BundleAdjustmentConfig() {}
  BundleAdjustmentSetup(const colmap::BundleAdjustmentConfig& config)
      : colmap::BundleAdjustmentConfig(config) {}

  // Overwrite methods which trigger glog
  void SetConstantPose(const colmap::image_t image_id);
  void SetConstantTvec(const colmap::image_t image_id,
                       const std::vector<int>& idxs);
  void AddVariablePoint(const colmap::point3D_t point3D_id);
  void AddConstantPoint(const colmap::point3D_t point3D_id);

  inline void SetAllConstant(colmap::Reconstruction* reconstruction);
};

struct BundleOptimizerOptions {
  // Ceres-Solver options.
  ceres::Solver::Options solver_options;

  BundleOptimizerOptions() {
    loss.reset(new ceres::CauchyLoss(0.25));
    solver_options.function_tolerance = 0.0;
    solver_options.gradient_tolerance = 0.0;
    solver_options.parameter_tolerance = 0.0;
    solver_options.minimizer_progress_to_stdout = true;
    solver_options.max_num_iterations = 100;
    solver_options.max_linear_solver_iterations = 200;
    solver_options.max_num_consecutive_invalid_steps = 10;
    solver_options.max_consecutive_nonmonotonic_steps = 10;
    solver_options.num_threads = -1;
#if CERES_VERSION_MAJOR < 2
    solver_options.num_linear_solver_threads = -1;
#endif  // CERES_VERSION_MAJOR
    // Since we use smart pointers to manage loss functions
    problem_options.loss_function_ownership = ceres::DO_NOT_TAKE_OWNERSHIP;
  }

  // Whether to refine the focal length parameter group.
  bool refine_focal_length = true;

  // Whether to refine the principal point parameter group.
  bool refine_principal_point = false;

  // Whether to refine the extra parameter group.
  bool refine_extra_params = true;

  // Whether to refine the extrinsic parameter group.
  bool refine_extrinsics = true;

  // Whether to print a final summary.
  bool print_summary = true;

  // Minimum number of residuals to enable multi-threading. Note that
  // single-threaded is typically better for small bundle adjustment problems
  // due to the overhead of threading.
  int min_num_residuals_for_multi_threading = 50000;

  bool Check() const;

  std::shared_ptr<ceres::LossFunction> loss;

  int min_track_length =
      -1;  // Use this to control which points should be optimized

  // Python only options
  ceres::Problem::Options problem_options;

  // Whether the solver should be interruptable from python
  bool register_pyinterrupt_callback = false;
};

}  // namespace pixsfm
