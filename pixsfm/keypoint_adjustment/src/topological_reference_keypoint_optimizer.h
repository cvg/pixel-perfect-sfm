#include "keypoint_adjustment/src/featuremetric_keypoint_optimizer.h"

namespace pixsfm {

class TopologicalReferenceKeypointOptimizer
    : public FeatureMetricKeypointOptimizer {
 public:
  struct Options : public FeatureMetricKeypointOptimizer::Options {
    Options() : FeatureMetricKeypointOptimizer::Options() {
      // overwrite options
      weight_by_sim = false;
      root_regularize_weight = 1.0;
      root_edges_only = true;
    }
  };
  TopologicalReferenceKeypointOptimizer(
      const Options& options, std::shared_ptr<KeypointAdjustmentSetup> setup,
      const InterpolationConfig& interpolation_config)
      : FeatureMetricKeypointOptimizer(options, setup, interpolation_config) {
    STDLOG(INFO) << "Start topological-reference keypoint adjustment."
                 << std::endl;
  }

  using FeatureMetricKeypointOptimizer::FeatureMetricKeypointOptimizer;
  using FeatureMetricKeypointOptimizer::Run;
  using FeatureMetricKeypointOptimizer::RunParallel;
  using FeatureMetricKeypointOptimizer::RunSubset;
};

}  // namespace pixsfm