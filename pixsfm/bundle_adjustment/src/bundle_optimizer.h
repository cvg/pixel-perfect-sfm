#pragma once

#include <ceres/ceres.h>

#include <colmap/base/camera_models.h>
#include <colmap/base/camera_rig.h>
#include <colmap/base/cost_functions.h>
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
#include "features/src/references.h"

#include "base/src/callbacks.h"

#include "util/src/log_exceptions.h"
#include "util/src/statistics.h"
#include "util/src/types.h"

#include "bundle_adjustment/src/bundle_adjustment_options.h"

#include "util/src/memory.h"
#include "util/src/simple_logger.h"

namespace pixsfm {

/*******************************************************************************
BundleOptimizer (CTRP Interface):
*******************************************************************************/

template <typename Derived>
class BundleOptimizer {
 public:
  BundleOptimizer(const BundleOptimizerOptions& options,
                  const BundleAdjustmentSetup& setup)
      : options_(options), setup_(setup) {
    // @TODO: Verify options
  }

  template <int... Ns, typename... Params>
  bool Run(colmap::Reconstruction* reconstruction, Params&... parameters);

  bool SolveProblem(colmap::Reconstruction* reconstruction);

  // Get the Ceres solver summary for the last call to `Solve`.
  const ceres::Solver::Summary& Summary() const;

  template <int... Ns, typename... Params>
  void SetUp(colmap::Reconstruction* reconstruction,
             ceres::LossFunction* loss_function, Params&... parameters);

  void TearDown(colmap::Reconstruction* reconstruction);

  void Clear();

  void Reset();

  ceres::Problem* Problem();

  // Top-Level Functions, for automatic building
  template <int... Ns, typename... Params>
  void AddImageToProblem(const colmap::image_t image_id,
                         colmap::Reconstruction* reconstruction,
                         ceres::LossFunction* loss_function,
                         Params&... parameters);

  // Add Track to Problem, alternative to image-based add
  template <int... Ns, typename... Params>
  void AddPointToProblem(const colmap::point3D_t point3D_id,
                         colmap::Reconstruction* reconstruction,
                         ceres::LossFunction* loss_function,
                         Params&... parameters);

  void Parameterize(colmap::Reconstruction* reconstruction);

 protected:
  void ParameterizePoints(colmap::Reconstruction* reconstruction);
  void ParameterizeImages(colmap::Reconstruction* reconstruction);
  void ParameterizeCameras(colmap::Reconstruction* reconstruction);

  bool RegisterPoint3DObservation(colmap::point3D_t point3D_id,
                                  colmap::image_t image_id,
                                  colmap::point2D_t point2D_idx,
                                  colmap::Reconstruction* reconstruction);

  BundleOptimizerOptions options_;
  BundleAdjustmentSetup setup_;

  // Only for top-level functions!!! Otherwise user has to control this variable
  std::unordered_map<colmap::point3D_t, std::unordered_set<size_t>>
      point3D_reg_track_idx_;

  std::unordered_map<colmap::point3D_t, size_t> image_num_residuals_;
  std::unordered_map<colmap::image_t, size_t> point3D_num_residuals_;
  std::unordered_map<colmap::camera_t, size_t> camera_num_residuals_;

  // Ceres
  std::unique_ptr<ceres::Problem> problem_;
  ceres::Solver::Summary summary_;
};

/*******************************************************************************
BundleOptimizer Methods:
*******************************************************************************/

template <typename Derived>
template <int... Ns, typename... Params>
bool BundleOptimizer<Derived>::Run(colmap::Reconstruction* reconstruction,
                                   Params&... parameters) {
  THROW_CUSTOM_CHECK_MSG(reconstruction, std::invalid_argument,
                         "reconstruction cannot be NULL.");

  THROW_CUSTOM_CHECK_MSG(!problem_, std::invalid_argument,
                         "Cannot use the same BundleOptimizer multiple times");

  Clear();
  problem_.reset(new ceres::Problem(options_.problem_options));

  ceres::LossFunction* loss_function = options_.loss.get();
  THROW_CUSTOM_CHECK_MSG(loss_function, std::invalid_argument,
                         "loss_function cannot be NULL.");

  if (options_.solver_options.use_inner_iterations) {
    options_.solver_options.inner_iteration_ordering.reset(
        new ceres::ParameterBlockOrdering);
  }
  SetUp<Ns...>(reconstruction, loss_function, parameters...);
  return SolveProblem(reconstruction);
}

template <typename Derived>
template <int... Ns, typename... Params>
void BundleOptimizer<Derived>::SetUp(colmap::Reconstruction* reconstruction,
                                     ceres::LossFunction* loss_function,
                                     Params&... parameters) {
  if (!problem_) {
    STDLOG(WARN) << "Resetting problem..." << std::endl;
    problem_.reset(new ceres::Problem(options_.problem_options));
  }

  STDLOG(DEBUG) << "Building bundle adjustment problem..." << std::endl;
  for (const colmap::image_t image_id : setup_.Images()) {
    AddImageToProblem<Ns...>(image_id, reconstruction, loss_function,
                             parameters...);
  }
  for (const auto point3D_id : setup_.VariablePoints()) {
    AddPointToProblem<Ns...>(point3D_id, reconstruction, loss_function,
                             parameters...);
  }
  for (const auto point3D_id : setup_.ConstantPoints()) {
    AddPointToProblem<Ns...>(point3D_id, reconstruction, loss_function,
                             parameters...);
  }
  STDLOG(DEBUG) << "Parameterize reconstruction..." << std::endl;
  Parameterize(reconstruction);
  STDLOG(DEBUG) << "Problem built successfully." << std::endl;
}

template <typename Derived>
const ceres::Solver::Summary& BundleOptimizer<Derived>::Summary() const {
  return summary_;
}

template <typename Derived>
bool BundleOptimizer<Derived>::SolveProblem(
    colmap::Reconstruction* reconstruction) {
  if (problem_->NumResiduals() == 0) {
    return false;
  }
  ceres::Solver::Options solver_options = options_.solver_options;

  // Empirical choice.
  const size_t kMaxNumImagesDirectDenseSolver = 50;
  const size_t kMaxNumImagesDirectSparseSolver = 1000;
  const size_t num_images = setup_.NumImages();
  if (num_images <= kMaxNumImagesDirectDenseSolver) {
    solver_options.linear_solver_type = ceres::DENSE_SCHUR;
  } else if (num_images <= kMaxNumImagesDirectSparseSolver) {
    solver_options.linear_solver_type = ceres::SPARSE_SCHUR;
  } else {  // Indirect sparse (preconditioned CG) solver.
    solver_options.linear_solver_type = ceres::ITERATIVE_SCHUR;
    solver_options.preconditioner_type = ceres::SCHUR_JACOBI;
  }

  solver_options.num_threads =
      colmap::GetEffectiveNumThreads(solver_options.num_threads);
#if CERES_VERSION_MAJOR < 2
  solver_options.num_linear_solver_threads =
      colmap::GetEffectiveNumThreads(solver_options.num_linear_solver_threads);
#endif  // CERES_VERSION_MAJOR

  long long expected_num_nonzeros = NumNonZerosJacobian(problem_.get());
  STDLOG(DEBUG) << "Num Nonzeros: " << expected_num_nonzeros << std::endl;
  if (expected_num_nonzeros * 8 /*Bytes*/ > FreePhysicalMemory()) {
    int mem_required = expected_num_nonzeros * 8 / 1024 / 1024;
    STDLOG(WARN) << "Required (est.) memory for jacobian [" << mem_required
                 << MemoryString(mem_required, "MB") << "] exceeds free RAM ["
                 << MemoryString(FreePhysicalMemory(), "MB") << "]."
                 << std::endl;
  }
  PyInterruptCallback py_interrupt_callback;
  if (options_.register_pyinterrupt_callback) {
    solver_options.callbacks.push_back(&py_interrupt_callback);
  }

  ProgressBarIterationCallback progressbar_callback(
      solver_options.max_num_iterations);
  if (!solver_options.minimizer_progress_to_stdout) {
    progressbar_callback.ProgressBar().update(0);
    solver_options.callbacks.push_back(&progressbar_callback);
  }

  std::string solver_error;
  CHECK(solver_options.IsValid(&solver_error)) << solver_error;
  STDLOG(DEBUG) << "Start solver." << std::endl;
  ceres::Solve(solver_options, problem_.get(), &summary_);
  if (solver_options.minimizer_progress_to_stdout) {
    std::cout << std::endl;
  } else {
    progressbar_callback.ProgressBar().finish();
  }

  if (options_.print_summary) {
    colmap::PrintHeading2("Bundle adjustment report");
    PrintSolverSummary(summary_);
    // We need to replace this with our own Printer!!!
  }
  STDLOG(INFO)
      << "BA Time: " << summary_.total_time_in_seconds << "s, cost change: "
      << std::sqrt(summary_.initial_cost / summary_.num_residuals_reduced)
      << " --> "
      << std::sqrt(summary_.final_cost / summary_.num_residuals_reduced)
      << std::endl;

  // TearDown(reconstruction);
  return true;
}

template <typename Derived>
template <int... Ns, typename... Params>
void BundleOptimizer<Derived>::AddImageToProblem(
    const colmap::image_t image_id, colmap::Reconstruction* reconstruction,
    ceres::LossFunction* loss_function, Params&... parameters) {
  colmap::Image& image = reconstruction->Image(image_id);
  colmap::Camera& camera = reconstruction->Camera(image.CameraId());

  image.NormalizeQvec();

  // Add residuals to bundle adjustment problem.
  for (colmap::point2D_t point2D_idx = 0; point2D_idx < image.NumPoints2D();
       point2D_idx++) {
    const colmap::Point2D& point2D = image.Point2D(point2D_idx);

    if (!point2D.HasPoint3D()) {
      continue;
    }

    colmap::Point3D& point3D = reconstruction->Point3D(point2D.Point3DId());

    if (static_cast<int>(point3D.Track().Length()) <
        options_.min_track_length) {
      continue;
    }
    int num_blocks = static_cast<Derived*>(this)->template AddResiduals<Ns...>(
        image_id, point2D_idx, reconstruction, loss_function, parameters...);
  }
}

// Add Track to Problem, alternative to image-based add
template <typename Derived>
template <int... Ns, typename... Params>
void BundleOptimizer<Derived>::AddPointToProblem(
    const colmap::point3D_t point3D_id, colmap::Reconstruction* reconstruction,
    ceres::LossFunction* loss_function, Params&... parameters) {
  // Points added in this functions should add residuals of to observations
  // where the image_id is not contained in setup_.Images()!

  colmap::Point3D& point3D = reconstruction->Point3D(point3D_id);

  // Is 3D point already fully contained in the problem? If so we skip.
  if (point3D_reg_track_idx_[point3D_id].size() == point3D.Track().Length()) {
    return;
  }

  for (const auto& track_el : point3D.Track().Elements()) {
    // Skip observations that were already added in `FillImages`.
    if (setup_.HasImage(track_el.image_id)) {
      continue;
    }

    colmap::Image& image = reconstruction->Image(track_el.image_id);
    colmap::Camera& camera = reconstruction->Camera(image.CameraId());
    const colmap::Point2D& point2D = image.Point2D(track_el.point2D_idx);

    // We do not want to refine the camera of images that are not
    // part of `constant_image_ids_`, `constant_image_ids_`,
    // `constant_x_image_ids_`.
    if (camera_num_residuals_[image.CameraId()] == 0) {
      setup_.SetConstantCamera(image.CameraId());
    }
    int num_blocks = static_cast<Derived*>(this)->template AddResiduals<Ns...>(
        track_el.image_id, track_el.point2D_idx, reconstruction, loss_function,
        parameters...);
  }
}

template <typename Derived>
bool BundleOptimizer<Derived>::RegisterPoint3DObservation(
    colmap::point3D_t point3D_id, colmap::image_t image_id,
    colmap::point2D_t point2D_idx, colmap::Reconstruction* reconstruction) {
  colmap::Point3D& point3D = reconstruction->Point3D(point3D_id);

  for (int track_idx = 0; track_idx < point3D.Track().Length(); ++track_idx) {
    auto& track_el = point3D.Track().Element(track_idx);
    if (track_el.image_id == image_id && track_el.point2D_idx == point2D_idx) {
      point3D_reg_track_idx_[point3D_id].insert(track_idx);
      return true;
    }
  }

  throw(std::runtime_error("Failed to register track element."));
  return false;
}

// @TODO: For clusters, we might want to have variable points which are not
// fully contained!
template <typename Derived>
void BundleOptimizer<Derived>::ParameterizePoints(
    colmap::Reconstruction* reconstruction) {
  int cnt = 0;  // Number of automatically constant points

  for (const auto& elem : point3D_reg_track_idx_) {
    colmap::Point3D& point3D = reconstruction->Point3D(elem.first);
    int min_track_length =
        options_.min_track_length > 0
            ? std::min(options_.min_track_length,
                       static_cast<int>(point3D.Track().Length()))
            : point3D.Track().Length();
    if (min_track_length > elem.second.size()) {
      problem_->SetParameterBlockConstant(point3D.XYZ().data());
      ++cnt;
    } else if (options_.solver_options.use_inner_iterations) {
      // Ruhe-Weldin 2 approximation to the VarPro Algorithm:
      // Use EPI (RCS is solved outside, i.e. only camera parameters)
      options_.solver_options.inner_iteration_ordering->AddElementToGroup(
          point3D.XYZ().data(), 0);
    }
  }
  STDLOG(DEBUG) << "Num Constant points3D: " << cnt << std::endl;
  // This could be dangerous for us! -> Need to check whether given point has
  // been added to problem.
  for (const colmap::point3D_t point3D_id : setup_.ConstantPoints()) {
    colmap::Point3D& point3D = reconstruction->Point3D(point3D_id);
    problem_->SetParameterBlockConstant(point3D.XYZ().data());
  }
}

template <typename Derived>
void BundleOptimizer<Derived>::ParameterizeImages(
    colmap::Reconstruction* reconstruction) {
  for (const auto& element : image_num_residuals_) {
    colmap::image_t image_id = element.first;
    if (element.second > 0) {
      colmap::Image& image = reconstruction->Image(image_id);
      colmap::Camera& camera = reconstruction->Camera(image.CameraId());

      double* qvec_data = image.Qvec().data();
      double* tvec_data = image.Tvec().data();

      const bool constant_pose = !options_.refine_extrinsics ||
                                 setup_.HasConstantPose(image_id) ||
                                 !setup_.HasImage(image_id);

      // Set pose parameterization.
      if (!constant_pose) {
        ceres::LocalParameterization* quaternion_parameterization =
            new ceres::QuaternionParameterization;
        problem_->SetParameterization(qvec_data, quaternion_parameterization);
        if (setup_.HasConstantTvec(image_id)) {
          const std::vector<int>& constant_tvec_idxs =
              setup_.ConstantTvec(image_id);
          ceres::SubsetParameterization* tvec_parameterization =
              new ceres::SubsetParameterization(3, constant_tvec_idxs);
          problem_->SetParameterization(tvec_data, tvec_parameterization);
        }
      } else {
        problem_->SetParameterBlockConstant(qvec_data);
        problem_->SetParameterBlockConstant(tvec_data);
      }
    }
  }
}

template <typename Derived>
void BundleOptimizer<Derived>::ParameterizeCameras(
    colmap::Reconstruction* reconstruction) {
  const bool constant_camera = !options_.refine_focal_length &&
                               !options_.refine_principal_point &&
                               !options_.refine_extra_params;
  for (const auto& element : camera_num_residuals_) {
    colmap::camera_t camera_id = element.first;
    if (element.second <= 0) {
      std::cerr << "Found camera without any residuals?" << std::endl;
      continue;
    }
    colmap::Camera& camera = reconstruction->Camera(camera_id);

    if (constant_camera || setup_.IsConstantCamera(camera_id)) {
      problem_->SetParameterBlockConstant(camera.ParamsData());
      continue;
    } else {
      std::vector<int> const_camera_params;

      if (!options_.refine_focal_length) {
        const std::vector<size_t>& params_idxs = camera.FocalLengthIdxs();
        const_camera_params.insert(const_camera_params.end(),
                                   params_idxs.begin(), params_idxs.end());
      }
      if (!options_.refine_principal_point) {
        const std::vector<size_t>& params_idxs = camera.PrincipalPointIdxs();
        const_camera_params.insert(const_camera_params.end(),
                                   params_idxs.begin(), params_idxs.end());
      }
      if (!options_.refine_extra_params) {
        const std::vector<size_t>& params_idxs = camera.ExtraParamsIdxs();
        const_camera_params.insert(const_camera_params.end(),
                                   params_idxs.begin(), params_idxs.end());
      }

      if (const_camera_params.size() > 0) {
        ceres::SubsetParameterization* camera_params_parameterization =
            new ceres::SubsetParameterization(
                static_cast<int>(camera.NumParams()), const_camera_params);
        problem_->SetParameterization(camera.ParamsData(),
                                      camera_params_parameterization);
      }
    }
  }
}

template <typename Derived>
void BundleOptimizer<Derived>::Parameterize(
    colmap::Reconstruction* reconstruction) {
  ParameterizePoints(reconstruction);
  STDLOG(DEBUG) << "Points successfully parameterized." << std::endl;
  ParameterizeImages(reconstruction);
  STDLOG(DEBUG) << "Poses successfully parameterized." << std::endl;
  ParameterizeCameras(reconstruction);
  STDLOG(DEBUG) << "Cameras successfully parameterized." << std::endl;
}

template <typename Derived>
void BundleOptimizer<Derived>::Clear() {
  point3D_reg_track_idx_.clear();
  image_num_residuals_.clear();
  point3D_num_residuals_.clear();
  camera_num_residuals_.clear();
}

template <typename Derived>
void BundleOptimizer<Derived>::Reset() {
  Clear();
  problem_.reset(new ceres::Problem(options_.problem_options));
}

template <typename Derived>
ceres::Problem* BundleOptimizer<Derived>::Problem() {
  return problem_.get();
}

}  // namespace pixsfm
