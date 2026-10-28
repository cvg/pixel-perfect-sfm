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
#include <colmap/scene/reconstruction.h>
#include <colmap/util/types.h>

#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "features/src/featuremap.h"
#include "features/src/featurepatch.h"
#include "features/src/featureset.h"

#include "util/src/types.h"

namespace pixsfm {

// Configuration container to setup bundle adjustment problems. Mirrors the
// image-based colmap::BundleAdjustmentConfig of COLMAP 3.8 (COLMAP 4.x
// configures poses per frame instead).
class BundleAdjustmentSetup {
 public:
  BundleAdjustmentSetup() = default;

  size_t NumImages() const;
  size_t NumPoints() const;
  size_t NumConstantCameras() const;
  size_t NumConstantPoses() const;
  size_t NumConstantTvecs() const;
  size_t NumVariablePoints() const;
  size_t NumConstantPoints() const;

  // Add / remove images from the configuration.
  void AddImage(const colmap::image_t image_id);
  bool HasImage(const colmap::image_t image_id) const;
  void RemoveImage(const colmap::image_t image_id);

  // Set cameras of added images as constant or variable. By default all
  // cameras of added images are variable.
  void SetConstantCamera(const colmap::camera_t camera_id);
  void SetVariableCamera(const colmap::camera_t camera_id);
  bool IsConstantCamera(const colmap::camera_t camera_id) const;

  // Set the pose of added images as constant. The pose is defined as the
  // rotational and translational part of the projection matrix.
  void SetConstantPose(const colmap::image_t image_id);
  void SetVariablePose(const colmap::image_t image_id);
  bool HasConstantPose(const colmap::image_t image_id) const;

  // Set the translational part of the pose, hence the constant pose
  // indices may be in [0, 1, 2] and must be unique.
  void SetConstantTvec(const colmap::image_t image_id,
                       const std::vector<int>& idxs);
  void RemoveConstantTvec(const colmap::image_t image_id);
  bool HasConstantTvec(const colmap::image_t image_id) const;

  // Add / remove points from the configuration. Note that points can either
  // be variable or constant but not both at the same time.
  void AddVariablePoint(const colmap::point3D_t point3D_id);
  void AddConstantPoint(const colmap::point3D_t point3D_id);
  bool HasPoint(const colmap::point3D_t point3D_id) const;
  bool HasVariablePoint(const colmap::point3D_t point3D_id) const;
  bool HasConstantPoint(const colmap::point3D_t point3D_id) const;
  void RemoveVariablePoint(const colmap::point3D_t point3D_id);
  void RemoveConstantPoint(const colmap::point3D_t point3D_id);

  // Access configuration data.
  const std::unordered_set<colmap::image_t>& Images() const;
  const std::unordered_set<colmap::point3D_t>& VariablePoints() const;
  const std::unordered_set<colmap::point3D_t>& ConstantPoints() const;
  const std::vector<int>& ConstantTvec(const colmap::image_t image_id) const;

 private:
  std::unordered_set<colmap::camera_t> constant_camera_ids_;
  std::unordered_set<colmap::image_t> image_ids_;
  std::unordered_set<colmap::point3D_t> variable_point3D_ids_;
  std::unordered_set<colmap::point3D_t> constant_point3D_ids_;
  std::unordered_set<colmap::image_t> constant_poses_;
  std::unordered_map<colmap::image_t, std::vector<int>> constant_tvecs_;
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
