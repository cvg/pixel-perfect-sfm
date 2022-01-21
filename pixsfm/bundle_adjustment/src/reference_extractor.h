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
#include <colmap/util/timer.h>
#include <colmap/util/types.h>

#include "features/src/dynamic_patch_interpolator.h"
#include "features/src/featuremap.h"
#include "features/src/featurepatch.h"
#include "features/src/featureset.h"
#include "features/src/featureview.h"
#include "features/src/patch_interpolator.h"
#include "features/src/references.h"

#include "base/src/interpolation.h"
#include "base/src/irls_optim.h"

#include "util/src/log_exceptions.h"
#include "util/src/misc.h"
#include "util/src/simple_logger.h"
#include "util/src/types.h"

#include "base/src/parallel_optimizer.h"
#include "base/src/projection.h"

#include <third-party/progressbar.h>

#include "base/src/graph.h"

#include <unordered_map>

namespace pixsfm {
using Refs = std::unordered_map<colmap::point3D_t, Reference>;

// Only register templates which benefit from static sizes (dynamic also supp.)
#define REFERENCE_CASES  \
  REGISTER_METHOD(3, 16) \
  REGISTER_METHOD(3, 4)

template <int N_NODES = Eigen::Dynamic>
inline OffsetMatrix3d<N_NODES> NodeOffsets3D(
    const colmap::Image& image, const colmap::Camera& camera,
    const Eigen::Vector3d& xyz, InterpolationConfig& interpolation_config);

struct ReferenceConfig {
  ReferenceConfig() { loss.reset(new ceres::CauchyLoss(0.25)); }
  ReferenceConfig(std::shared_ptr<ceres::LossFunction> loss_) { loss = loss_; }
  bool keep_observations = false;
  int iters = 10;
  bool compute_offsets3D = false;
  bool closest_to_robust_mean = true;
  int num_threads = -1;

  std::shared_ptr<ceres::LossFunction> loss;
};

class ReferenceExtractor
    : public ParallelOptimizer<ReferenceExtractor, colmap::point3D_t> {
  using Parallel = ParallelOptimizer<ReferenceExtractor, colmap::point3D_t>;

 public:
  ReferenceExtractor(const ReferenceConfig& config,
                     const InterpolationConfig& interpolation_config);

  // Good Header
  template <int CHANNELS, int N_NODES, typename dtype>
  inline Reference ComputeReference(
      const colmap::point3D_t point3D_id, const colmap::Track& track,
      const std::vector<Eigen::Vector2d>& xys,
      const colmap::Reconstruction* reconstruction, FeatureView<dtype>& f_view);

  template <typename dtype>
  inline Refs Run(const std::vector<int>& problem_labels,
                  const colmap::Reconstruction* reconstruction,
                  FeatureSet<dtype>& fset);

  template <int CHANNELS, int N_NODES, typename dtype>
  inline double RunSubset(
      const std::unordered_set<colmap::point3D_t>& point3D_ids,
      Refs& references, const colmap::Reconstruction* reconstruction,
      FeatureSet<dtype>& f_set);

  // Interface for CostMapExtractor
  template <int CHANNELS, int N_NODES, typename dtype>
  inline double RunSubset(
      const std::unordered_set<colmap::point3D_t>& point3D_ids,
      Refs& references, const colmap::Reconstruction* reconstruction,
      FeatureView<dtype>& f_view);

  inline std::unordered_map<colmap::point3D_t, Reference> InitReferences(
      const std::vector<int>& problem_labels);

  static ReferenceExtractor Create(ReferenceExtractor* other) {
    return ReferenceExtractor(other->config_, other->interpolation_config_);
  }

 private:
  template <typename dtype>
  std::pair<colmap::Track, std::vector<Eigen::Vector2d>> GetVisibleObservations(
      const colmap::point3D_t point3D_id,
      const colmap::Reconstruction* reconstruction, FeatureView<dtype>& f_view);

  template <int CHANNELS, int N_NODES, typename T, typename dtype>
  inline void FillDescriptorTrack(T& refdata, const colmap::Track& track,
                                  const std::vector<Eigen::Vector2d>& xys,
                                  FeatureView<dtype>& f_view);

  ReferenceConfig config_;
  InterpolationConfig interpolation_config_;
  colmap::Timer timer_;
};

template <typename dtype>
Refs ReferenceExtractor::Run(const std::vector<int>& problem_labels,
                             const colmap::Reconstruction* reconstruction,
                             FeatureSet<dtype>& fset) {
  int n_nodes = interpolation_config_.nodes.size();
  int channels = fset.Channels();
  STDLOG(INFO) << "Extracting references." << std::endl;
  // Create empty references
  auto references = InitReferences(problem_labels);
  fset.FlushEveryN(colmap::GetEffectiveNumThreads(n_threads_));
  bool found = false;
  std::unordered_map<size_t, double> sec;
#define REGISTER_METHOD(CHANNELS, N_NODES)                                     \
  if (channels == CHANNELS && n_nodes == N_NODES) {                            \
    sec = Parallel::RunParallel<CHANNELS, N_NODES>(problem_labels, references, \
                                                   reconstruction, fset);      \
    found = true;                                                              \
  }
  REFERENCE_CASES
#undef REGISTER_METHOD
  if (!found) {
    STDLOG(DEBUG) << "Unsupported template params in references. "
                  << "Running with dynamic templates." << std::endl;
    sec = Parallel::RunParallel<-1, -1>(problem_labels, references,
                                        reconstruction, fset);
  }

  fset.FlushEveryN(1);
  fset.Flush();

  STDLOG(INFO) << "Reference Extraction Time: " << parallel_solver_time_ << "s"
               << std::endl;

  STDLOG(DEBUG) << "Optimizer CPU Time: " << AccumulateValues(sec) << "s"
                << std::endl;

  return references;
}

ReferenceExtractor::ReferenceExtractor(
    const ReferenceConfig& config,
    const InterpolationConfig& interpolation_config)
    : config_(config),
      interpolation_config_(interpolation_config),
      ParallelOptimizer(config.num_threads) {}

template <typename dtype>
std::pair<colmap::Track, std::vector<Eigen::Vector2d>>
ReferenceExtractor::GetVisibleObservations(
    const colmap::point3D_t point3D_id,
    const colmap::Reconstruction* reconstruction, FeatureView<dtype>& f_view) {
  const colmap::Point3D& point3D = reconstruction->Point3D(point3D_id);

  // Construct visible track
  colmap::Track visible_track;
  std::vector<Eigen::Vector2d> xys;
  // xys.reserve(point3D.Track().Length());
  for (auto& track_el : point3D.Track().Elements()) {
    colmap::image_t image_id = track_el.image_id;
    colmap::point2D_t point2D_idx = track_el.point2D_idx;

    // Check if subset contains this patch
    if (!f_view.HasFeaturePatch(image_id, point2D_idx)) {
      STDLOG(WARN) << "Warning: Patch at (" << image_id << ", " << point2D_idx
                   << ") does not exist." << std::endl;
      continue;
    }

    visible_track.AddElement(image_id, point2D_idx);

    // Project 3D point to 2D
    const colmap::Image& image = reconstruction->Image(image_id);
    const colmap::Camera& camera = reconstruction->Camera(image.CameraId());

    Eigen::Vector2d projected;
    WorldToPixel(camera, image.Qvec(), image.Tvec(), point3D.XYZ(),
                 projected.data());
    xys.push_back(projected);
  }
  return std::make_pair(visible_track, xys);
}

template <int CHANNELS, int N_NODES, typename dtype>
double ReferenceExtractor::RunSubset(
    const std::unordered_set<colmap::point3D_t>& point3D_ids, Refs& references,
    const colmap::Reconstruction* reconstruction, FeatureSet<dtype>& fset) {
  FeatureView<dtype> fview(&fset, reconstruction, point3D_ids);
  return RunSubset<CHANNELS, N_NODES>(point3D_ids, references, reconstruction,
                                      fview);
}

template <int CHANNELS, int N_NODES, typename dtype>
double ReferenceExtractor::RunSubset(
    const std::unordered_set<colmap::point3D_t>& point3D_ids, Refs& references,
    const colmap::Reconstruction* reconstruction, FeatureView<dtype>& f_view) {
  // Loop over all point3Ds and construct visible tracks
  LogProgressbar bar(point3D_ids.size());
  timer_.Start();
  for (colmap::point3D_t point3D_id : point3D_ids) {
    std::pair<colmap::Track, std::vector<Eigen::Vector2d>> visible_obs =
        GetVisibleObservations(point3D_id, reconstruction, f_view);
    // Compute Reference Data
    if (visible_obs.first.Length() > 0) {
      // Allows concurrent writes!
      references[point3D_id] = ComputeReference<CHANNELS, N_NODES>(
          point3D_id, visible_obs.first, visible_obs.second, reconstruction,
          f_view);
    }
    bar.update();
  }
  timer_.Pause();
  return timer_.ElapsedSeconds();
}

template <int CHANNELS, int N_NODES, typename dtype>
Reference ReferenceExtractor::ComputeReference(
    const colmap::point3D_t point3D_id, const colmap::Track& track,
    const std::vector<Eigen::Vector2d>& xys,
    const colmap::Reconstruction* reconstruction, FeatureView<dtype>& f_view) {
  std::vector<DescriptorMatrixd<N_NODES, CHANNELS>> descriptors(track.Length());
  FillDescriptorTrack<CHANNELS, N_NODES>(descriptors, track, xys, f_view);
  DescriptorMatrixd<N_NODES, CHANNELS> robust_mean =
      RobustMeanIRLS<CHANNELS, N_NODES>(descriptors, config_.loss.get(),
                                        config_.iters, interpolation_config_);
  Eigen::RowVectorXd distances(track.Length());
  for (int i = 0; i < track.Length(); i++) {
    distances(i) = (descriptors[i] - robust_mean).squaredNorm();
  }

  int ref_idx;
  float min_coeff = distances.minCoeff(&ref_idx);

  ReferenceData refdata;

  if (config_.keep_observations) {
    refdata.track = track;
    for (int i = 0; i < track.Length(); i++) {
      refdata.observations.push_back(descriptors[i]);
      refdata.costs.push_back(distances(i));
    }
  }
  int n_nodes = interpolation_config_.nodes.size();
  int channels = f_view.Channels();
  if (!config_.compute_offsets3D) {
    return Reference(
        track.Element(ref_idx),
        config_.closest_to_robust_mean ? descriptors[ref_idx] : robust_mean,
        &refdata);
  } else {
    if (n_nodes > 1) {
      THROW_CHECK(reconstruction);
      const colmap::Image& image =
          reconstruction->Image(track.Element(ref_idx).image_id);
      const colmap::Camera& camera = reconstruction->Camera(image.CameraId());
      auto node_offsets = NodeOffsets3D<N_NODES>(
          image, camera, reconstruction->Point3D(point3D_id).XYZ(),
          interpolation_config_);

      return Reference(
          track.Element(ref_idx),
          config_.closest_to_robust_mean ? descriptors[ref_idx] : robust_mean,
          &refdata, node_offsets);

    } else {
      OffsetMatrix3d<1> node_offsets;
      node_offsets << 0.0, 0.0, 0.0;

      return Reference(
          track.Element(ref_idx),
          config_.closest_to_robust_mean ? descriptors[ref_idx] : robust_mean,
          &refdata, node_offsets);
    }
  }
}

template <int CHANNELS, int N_NODES, typename T, typename dtype>
void ReferenceExtractor::FillDescriptorTrack(
    T& refdata, const colmap::Track& track,
    const std::vector<Eigen::Vector2d>& xys, FeatureView<dtype>& f_view) {
  static constexpr bool DYNAMIC = (N_NODES == -1 || CHANNELS == -1);
  DynamicPatchInterpolator interpolator(interpolation_config_);
  for (int idx = 0; idx < track.Length(); idx++) {
    const colmap::TrackElement& track_el = track.Element(idx);
    colmap::image_t image_id = track_el.image_id;
    colmap::point2D_t point2D_idx = track_el.point2D_idx;
    if (DYNAMIC) {
      refdata[idx].resize(interpolation_config_.nodes.size(),
                          f_view.Channels());
    }
    interpolator.EvaluateNodes<CHANNELS>(
        f_view.GetFeaturePatch(image_id, point2D_idx), xys[idx].data(),
        refdata[idx].data());
  }
}

std::unordered_map<colmap::point3D_t, Reference>
ReferenceExtractor::InitReferences(const std::vector<int>& problem_labels) {
  std::unordered_map<colmap::point3D_t, Reference> references;
  for (colmap::point3D_t p3Did = 0; p3Did < problem_labels.size(); p3Did++) {
    if (problem_labels[p3Did] >= 0) {
      references.emplace(p3Did, Reference());
    }
  }
  return references;
}

template <int N_NODES = Eigen::Dynamic>
OffsetMatrix3d<N_NODES> NodeOffsets3D(
    const colmap::Image& image, const colmap::Camera& camera,
    const Eigen::Vector3d& xyz, InterpolationConfig& interpolation_config) {
  OffsetMatrix3d<N_NODES> offsets;
  if (N_NODES == Eigen::Dynamic) {
    offsets.resize(interpolation_config.nodes.size(), 3);
  } else {
    assert(N_NODES == interpolation_config.nodes.size());
  }
  const auto& Tmat = image.ProjectionMatrix();

  Eigen::Vector2d projected = colmap::ProjectPointToImage(xyz, Tmat, camera);

  double depth;
  CalculateDepth(image.Qvec().data(), image.Tvec().data(), xyz.data(), &depth);

  for (int i = 0; i < interpolation_config.nodes.size(); i++) {
    Eigen::Vector2d projected_node = projected;
    projected_node(0) += interpolation_config.nodes[i][0];
    projected_node(1) += interpolation_config.nodes[i][1];

    Eigen::Vector2d xy = camera.ImageToWorld(projected_node);
    Eigen::Vector3d xyz_node{xy(0), xy(1), 1.0};

    xyz_node = xyz_node * depth;
    Eigen::Vector3d res =
        image.RotationMatrix().transpose() * (xyz_node - image.Tvec()) - xyz;

    offsets.row(i) = res.transpose();
  }
  return offsets;
}

}  // namespace pixsfm
