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
#include <colmap/util/types.h>

#include "base/src/graph.h"
#include "base/src/interpolation.h"

#include "util/src/types.h"

namespace pixsfm {

// Configuration container to setup keypoint adjustment problems.
class KeypointAdjustmentSetup {
 public:
  KeypointAdjustmentSetup() {}

  void SetImageConstant(colmap::image_t image_id);
  void SetNodeConstant(const FeatureNode* node);
  void SetKeypointConstant(colmap::image_t image_id,
                           colmap::point2D_t feature_idx);
  void SetKeypointsConstant(colmap::image_t image_id,
                            std::vector<colmap::point2D_t> feature_idxs);
  bool IsNodeConstant(const FeatureNode* node) const;
  bool IsKeypointConstant(colmap::image_t image_id,
                          colmap::point2D_t feature_idx) const;

  void SetMaskedNodesConstant(const Graph* graph,
                              const std::vector<bool>& mask);

 private:
  std::unordered_set<colmap::image_t> constant_images;
  std::unordered_map<colmap::image_t, std::unordered_set<colmap::point2D_t>>
      constant_keypoints;
};

struct KeypointOptimizerOptions {
  // Ceres-Solver options.
  ceres::Solver::Options solver_options;

  KeypointOptimizerOptions() {
    // CERES options.
    loss.reset(new ceres::CauchyLoss(0.25));
    solver_options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
    solver_options.max_num_iterations = 100;
    solver_options.minimizer_progress_to_stdout = true;
    solver_options.max_num_consecutive_invalid_steps = 10;
    solver_options.function_tolerance = 0.0;
    solver_options.gradient_tolerance = 0.0;
    solver_options.parameter_tolerance = 1.0e-4;
    solver_options.num_threads = 1;
#if CERES_VERSION_MAJOR < 2
    solver_options.num_linear_solver_threads = 1;
#endif  // CERES_VERSION_MAJOR
    problem_options.loss_function_ownership = ceres::TAKE_OWNERSHIP;
  }

  KeypointOptimizerOptions(std::shared_ptr<ceres::LossFunction> loss_)
      : KeypointOptimizerOptions() {
    // Python takes ownership of the loss function
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

}  // namespace pixsfm
