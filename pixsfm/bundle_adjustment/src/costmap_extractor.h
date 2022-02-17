#pragma once

#include <ceres/ceres.h>
#include <colmap/base/projection.h>
#include <colmap/util/types.h>

#include "features/src/featuremap.h"
#include "features/src/featurepatch.h"
#include "features/src/featureset.h"
#include "features/src/featureview.h"
#include "features/src/patch_interpolator.h"
#include "features/src/references.h"

#include "base/src/interpolation.h"
#include "base/src/irls_optim.h"
#include "base/src/parallel_optimizer.h"

#include "bundle_adjustment/src/reference_extractor.h"

#include "util/src/log_exceptions.h"
#include "util/src/misc.h"
#include "util/src/simple_logger.h"
#include "util/src/types.h"

#include "base/src/projection.h"

#include <third-party/progressbar.h>

#include "base/src/graph.h"

#include <unordered_map>

namespace pixsfm {

#define COSTMAP_CASES  \
  REGISTER_METHOD(128) \
  REGISTER_METHOD(3)

struct CostMapConfig {
  CostMapConfig() { loss.reset(new ceres::TrivialLoss()); }

  double upsampling_factor = 1.0;
  bool as_gradientfield = true;

  bool compute_cross_derivative = false;
  bool apply_sqrt = false;

  int num_threads = -1;

  inline int GetEffectiveChannels() {
    if (as_gradientfield) {
      if (compute_cross_derivative) {
        return 4;
      } else {
        return 3;
      }
    }
    return 1;
  }

  std::shared_ptr<ceres::LossFunction> loss;
};

class CostMapExtractor
    : public ParallelOptimizer<CostMapExtractor, colmap::point3D_t> {
  using Parallel = ParallelOptimizer<CostMapExtractor, colmap::point3D_t>;
  using Refs = std::unordered_map<colmap::point3D_t, Reference>;

 public:
  CostMapExtractor(CostMapConfig& config,
                   InterpolationConfig& interpolation_config);

  // Better performance
  // problem_labels: problem_labels[p3D_id] == problem_idx
  template <typename dtype_o, typename dtype>
  std::pair<FeatureSet<dtype_o>, Refs> Run(
      std::vector<int>& problem_labels, colmap::Reconstruction& reconstruction,
      FeatureSet<dtype>& fset, ReferenceExtractor* ref_extractor);

  // Easier to use and customize
  template <int CHANNELS, typename dtype_o, typename dtype>
  double RunSubset(std::unordered_set<colmap::point3D_t>& point3D_ids,
                   FeatureSet<dtype_o>& costmaps,
                   colmap::Reconstruction& reconstruction,
                   FeatureSet<dtype>& fset, Refs& references,
                   ReferenceExtractor* ref_extractor);

  template <typename dtype_o, typename dtype>
  FeatureSet<dtype_o> CreateShallowCostmapFSet(FeatureSet<dtype>& fset,
                                               int out_channels,
                                               double upsampling_factor);

  template <int CHANNELS, typename dtype_o, typename dtype>
  void FillPointCostmap(FeaturePatch<dtype>& fpatch, Reference& reference,
                        FeaturePatch<dtype_o>& cost_patch);

  static CostMapExtractor Create(CostMapExtractor* other);

 protected:
  CostMapConfig config_;
  InterpolationConfig interpolation_config_;
  colmap::Timer timer_;
};

CostMapExtractor::CostMapExtractor(CostMapConfig& config,
                                   InterpolationConfig& interpolation_config)
    : config_(config),
      interpolation_config_(interpolation_config),
      ParallelOptimizer(config.num_threads) {}

CostMapExtractor CostMapExtractor::Create(CostMapExtractor* other) {
  return CostMapExtractor(other->config_, other->interpolation_config_);
}

template <typename dtype_o, typename dtype>
std::pair<FeatureSet<dtype_o>, Refs> CostMapExtractor::Run(
    std::vector<int>& problem_labels, colmap::Reconstruction& reconstruction,
    FeatureSet<dtype>& fset, ReferenceExtractor* ref_extractor) {
  STDLOG(INFO) << "Extracting references and costmaps." << std::endl;
  int n_nodes = interpolation_config_.nodes.size();
  THROW_CHECK_EQ(n_nodes, 1);

  int channels = fset.Channels();

  FeatureSet<dtype_o> costmaps = CreateShallowCostmapFSet<dtype_o>(
      fset, config_.GetEffectiveChannels(), config_.upsampling_factor);

  Refs references = ref_extractor->InitReferences(problem_labels);

  fset.FlushEveryN(colmap::GetEffectiveNumThreads(config_.num_threads));

  bool found = false;
  std::unordered_map<size_t, double> sec;
#define REGISTER_METHOD(CHANNELS)                                           \
  if (channels == CHANNELS) {                                               \
    sec = Parallel::RunParallel<CHANNELS>(problem_labels, costmaps,         \
                                          reconstruction, fset, references, \
                                          ref_extractor);                   \
    found = true;                                                           \
  }

  COSTMAP_CASES
#undef REGISTER_METHOD

  if (!found) {
    THROW_EXCEPTION(std::invalid_argument,
                    "CostMap extract: Unknown dimensions.");
  }

  fset.FlushEveryN(1);
  fset.Flush();

  STDLOG(DEBUG) << "Costmap Extraction Time: " << parallel_solver_time_ << "s"
                << std::endl;

  STDLOG(DEBUG) << "Optimizer CPU Time: " << AccumulateValues(sec) << "s"
                << std::endl;

  return std::make_pair(costmaps, references);
}

// Easier to use and customize
template <int CHANNELS, typename dtype_o, typename dtype>
double CostMapExtractor::RunSubset(
    std::unordered_set<colmap::point3D_t>& point3D_ids,
    FeatureSet<dtype_o>& costmaps, colmap::Reconstruction& reconstruction,
    FeatureSet<dtype>& fset, Refs& references,
    ReferenceExtractor* ref_extractor) {
  FeatureView<dtype> fview(&fset, &reconstruction, point3D_ids);
  double seconds = 0.0;

  if (ref_extractor) {
    seconds += ref_extractor->RunSubset<CHANNELS, 1>(point3D_ids, references,
                                                     &reconstruction, fview);
  }

  timer_.Start();
  for (colmap::point3D_t point3D_id : point3D_ids) {
    Reference& reference = references.at(point3D_id);
    auto& point3D = reconstruction.Point3D(point3D_id);
    for (auto& track_el : point3D.Track().Elements()) {
      colmap::image_t image_id = track_el.image_id;
      colmap::point2D_t point2D_idx = track_el.point2D_idx;
      FeatureMap<dtype>& fmap = fview.GetFeatureMap(image_id);
      THROW_CHECK(fmap.IsSparse());
      FeaturePatch<dtype>& fpatch = fmap.GetFeaturePatch(point2D_idx);

      std::string& image_name = fview.Mapping().at(image_id);
      auto& cost_fmap = costmaps.GetFeatureMap(image_name);
      FeaturePatch<dtype_o>& cost_patch =
          cost_fmap.GetFeaturePatch(point2D_idx);

      cost_patch.Allocate();
      FillPointCostmap<CHANNELS, dtype_o>(fpatch, reference, cost_patch);
    }
  };
  timer_.Pause();
  return seconds + timer_.ElapsedSeconds();
}

template <int CHANNELS, typename dtype_o, typename dtype>
void CostMapExtractor::FillPointCostmap(FeaturePatch<dtype>& fpatch,
                                        Reference& reference,
                                        FeaturePatch<dtype_o>& cost_patch) {
  THROW_CHECK_EQ(fpatch.Channels(), CHANNELS);
  THROW_CHECK_EQ(cost_patch.Channels(), config_.GetEffectiveChannels())

  PatchInterpolator<dtype, CHANNELS, 1> patch_interpolator(
      interpolation_config_, fpatch);

  double scale_factor = 1.0 / config_.upsampling_factor;

  for (int y = 0; y < cost_patch.Height(); y++) {
    for (int x = 0; x < cost_patch.Width(); x++) {
      Eigen::Vector2d xy{x * scale_factor, y * scale_factor};
      Eigen::Matrix<double, CHANNELS, 1> f;

      if (config_.as_gradientfield) {
        Eigen::Matrix<double, CHANNELS, 1> dfdr;
        Eigen::Matrix<double, CHANNELS, 1> dfdc;
        Eigen::Matrix<double, CHANNELS, 1> dfdrc;

        // Easy compute when scale factor == 1
        if (cost_patch.Width() == fpatch.Width() &&
            cost_patch.Height() == fpatch.Height() &&
            !config_.compute_cross_derivative) {
          // Do not perform interpolation when not necessary
          f = Eigen::Map<Eigen::Matrix<dtype, CHANNELS, 1>>(
                  fpatch.Data() + (y * fpatch.Width() + x) * CHANNELS)
                  .template cast<double>();
          int top = std::min(fpatch.Height() - 1, y + 1);
          int bottom = std::max(0, y - 1);
          int right = std::min(fpatch.Width() - 1, x + 1);
          int left = std::max(0, x - 1);

          // Central difference
          dfdr = (Eigen::Map<Eigen::Matrix<dtype, CHANNELS, 1>>(
                      fpatch.Data() + (top * fpatch.Width() + x) * CHANNELS) -
                  Eigen::Map<Eigen::Matrix<dtype, CHANNELS, 1>>(
                      fpatch.Data() + (bottom * fpatch.Width() + x) * CHANNELS))
                     .template cast<double>();

          dfdc = (Eigen::Map<Eigen::Matrix<dtype, CHANNELS, 1>>(
                      fpatch.Data() + (y * fpatch.Width() + right) * CHANNELS) -
                  Eigen::Map<Eigen::Matrix<dtype, CHANNELS, 1>>(
                      fpatch.Data() + (y * fpatch.Width() + left) * CHANNELS))
                     .template cast<double>();

          dfdr *= 0.5;
          dfdc *= 0.5;
        } else {
          patch_interpolator.EvaluateLocal(
              xy.data(), f.data(), dfdr.data(), dfdc.data(),
              config_.compute_cross_derivative ? dfdrc.data() : NULL);
        }

        Eigen::Matrix<double, CHANNELS, 1> residuals =
            f - Eigen::Map<Eigen::Matrix<double, CHANNELS, 1>>(
                    reference.DescriptorData());

        double cost = residuals.squaredNorm();

        double rho[3];
        config_.loss.get()->Evaluate(cost, rho);
        cost = rho[0] * 0.5;

        double dcostdr = 0.0;
        double dcostdc = 0.0;
        double dcostdrc = 0.0;

        if (cost > 1.0e-8) {
          dcostdr = rho[1] * residuals.dot(dfdr);
          dcostdc = rho[1] * residuals.dot(dfdc);

          if (config_.compute_cross_derivative) {
            dcostdrc = rho[2] * 2.0 * residuals.dot(dfdr) * residuals.dot(dfdc);
            dcostdrc += rho[1] * (dfdr.dot(dfdc) + dfdrc.dot(residuals));
          }

          if (config_.apply_sqrt) {
            cost = std::sqrt(cost);
            if (config_.compute_cross_derivative) {
              dcostdrc *= 0.5 / cost;
              dcostdrc += -0.5 * 0.5 / (cost * cost * cost) * dcostdr * dcostdc;
            }
            dcostdr *= 0.5 / cost;
            dcostdc *= 0.5 / cost;
          }
        }

        cost_patch.SetEntry(y, x, 0, cost);
        cost_patch.SetEntry(y, x, 1, dcostdr);
        cost_patch.SetEntry(y, x, 2, dcostdc);

        if (config_.compute_cross_derivative) {
          cost_patch.SetEntry(y, x, 3, dcostdrc);
        }

      } else {
        double cost;
        if (cost_patch.Width() == fpatch.Width() &&
            cost_patch.Height() == fpatch.Height()) {
          // Do not perform interpolation when not necessary
          Eigen::Matrix<dtype, CHANNELS, 1> descr(
              Eigen::Map<Eigen::Matrix<dtype, CHANNELS, 1>>(
                  fpatch.Data() + (y * fpatch.Width() + x) * CHANNELS));

          cost = (descr.template cast<double>() -
                  Eigen::Map<Eigen::Matrix<double, CHANNELS, 1>>(
                      reference.DescriptorData()))
                     .squaredNorm();
        } else {
          patch_interpolator.EvaluateLocal(xy.data(), f.data(), NULL, NULL);
          cost = (f - Eigen::Map<Eigen::Matrix<double, CHANNELS, 1>>(
                          reference.DescriptorData()))
                     .squaredNorm();
        }
        double rho[3];
        config_.loss->Evaluate(cost, rho);
        cost = rho[0] * 0.5;

        if (config_.apply_sqrt) {
          cost = std::sqrt(cost);
        }
        cost_patch.SetEntry(y, x, 0, cost);
      }
    }
  }
}

template <typename dtype_o, typename dtype>
FeatureSet<dtype_o> CostMapExtractor::CreateShallowCostmapFSet(
    FeatureSet<dtype>& fset, int out_channels, double upsampling_factor) {
  FeatureSet<dtype_o> cost_fset(out_channels);
  for (auto& fmap_pair : fset.FeatureMaps()) {
    std::vector<ssize_t> shape = fmap_pair.second.Shape();
    FeatureMap<dtype_o> cost_fmap(out_channels, fmap_pair.second.IsSparse());
    for (auto& patch_pair : fmap_pair.second.Patches()) {
      std::array<int, 3> patch_shape = {
          static_cast<int>(patch_pair.second.Height() *
                           (upsampling_factor + 1.0e-6)),
          static_cast<int>(patch_pair.second.Width() *
                           (upsampling_factor + 1.0e-6)),
          out_channels};
      cost_fmap.Patches().emplace(
          patch_pair.first,
          FeaturePatch<dtype_o>(
              NULL, patch_shape, patch_pair.second.Corner(),
              patch_pair.second.Scale()));  // No Fill, no allocation

      cost_fmap.GetFeaturePatch(patch_pair.first)
          .SetUpsamplingFactor(upsampling_factor);
    }
    cost_fset.Emplace(fmap_pair.first, cost_fmap);
  }
  return cost_fset;
}

}  // namespace pixsfm