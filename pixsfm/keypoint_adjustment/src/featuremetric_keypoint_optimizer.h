#pragma once
#include "keypoint_adjustment/src/topological_keypoint_optimizer.h"

#include "features/src/featureview.h"
#include "residuals/src/featuremetric.h"

#include "base/src/callbacks.h"
#include "base/src/parallel_optimizer.h"

#include "util/src/statistics.h"

namespace pixsfm {

#define FEATUREMETRIC_CASES              \
  REGISTER_METHOD(128, 1)                \
  REGISTER_METHOD(1, 1)                  \
  THROW_EXCEPTION(std::invalid_argument, \
                  "Unsupported dimensions (CHANNELS,N_NODES).");

// Inherits SetUp and Parameterize from Topological,
// Inherits RunParallel from ParallelOptimizer
class FeatureMetricKeypointOptimizer
    : public TopologicalKeypointOptimizer<FeatureMetricKeypointOptimizer>,
      public ParallelOptimizer<FeatureMetricKeypointOptimizer, size_t> {
  // Abbreviations
  using TKOptim = TopologicalKeypointOptimizer<FeatureMetricKeypointOptimizer>;
  using Parallel = ParallelOptimizer<FeatureMetricKeypointOptimizer, size_t>;

 public:
  struct Options : public TKOptim::Options {
    Options() : TKOptim::Options() {}
    // Only required for parallel optimizer as long as we do not store
    // these options as class variable
    Options(const TKOptim::Options& options) : TKOptim::Options(options) {}
    int num_threads = -1;  // For parallel scheduler
  };
  // We do not need to store above options as a variable again since we
  // only access new values (num_threads) on constructor,
  // and just use the rest from the
  // inherited options
  FeatureMetricKeypointOptimizer(
      const Options& options, std::shared_ptr<KeypointAdjustmentSetup> setup,
      const InterpolationConfig& interpolation_config)
      : TopologicalKeypointOptimizer(options, setup),
        ParallelOptimizer(options.num_threads),
        interpolation_config_(interpolation_config) {
    STDLOG(INFO) << "Start feature-metric keypoint adjustment." << std::endl;
  }

  template <typename dtype>
  bool Run(MapNameKeypoints* keypoints, const Graph* graph,
           const std::vector<size_t>& track_labels,
           const std::vector<bool>& root_labels,
           FeatureSet<dtype>& feature_set) {
    std::unordered_set<size_t> nodes_in_problem;
    for (const auto& node : graph->nodes) {
      nodes_in_problem.insert(node->node_idx);
    }
    // Add callback to allow keyboard interruption
    PyInterruptCallback py_interrupt_callback;
    options_.solver_options.callbacks.push_back(&py_interrupt_callback);

    RunSubset(nodes_in_problem, keypoints, graph, track_labels, root_labels,
              feature_set);
    return true;
  }

  template <typename dtype>
  bool RunParallel(const std::vector<int>& problem_labels,
                   MapNameKeypoints* keypoints, const Graph* graph,
                   const std::vector<size_t>& track_labels,
                   const std::vector<bool>& root_labels,
                   FeatureSet<dtype>& feature_set) {
    // Reduce flushing frequency for better speed

    options_.solver_options.logging_type = ceres::SILENT;

    const size_t n_nodes = graph->nodes.size();
    THROW_CHECK_EQ(track_labels.size(), n_nodes);
    THROW_CHECK_EQ(root_labels.size(), n_nodes);
    THROW_CHECK_EQ(problem_labels.size(), n_nodes);

    feature_set.FlushEveryN(colmap::GetEffectiveNumThreads(n_threads_));
    auto summaries =
        Parallel::RunParallel(problem_labels, keypoints, graph, track_labels,
                              root_labels, feature_set);
    feature_set.FlushEveryN(1);
    feature_set.Flush();
    summary_ = AccumulateSummaries(summaries);
    if (options_.solver_options.minimizer_progress_to_stdout) {
      for (auto& it_summary : summary_.iterations) {
        PrintParallelKeypointOptimizerIteration(it_summary);
      }
      STDLOG(COUT) << std::endl;
    }
    if (options_.print_summary) {
      PrintSolverSummary(summary_);
    }

    STDLOG(INFO) << "KA Time: " << parallel_solver_time_
                 << "s"
                    ", cost change: "
                 << std::sqrt(summary_.initial_cost /
                              summary_.num_residuals_reduced)
                 << " --> "
                 << std::sqrt(summary_.final_cost /
                              summary_.num_residuals_reduced)
                 << std::endl;

    STDLOG(DEBUG) << "Optimizer CPU Time: " << summary_.total_time_in_seconds
                  << "s" << std::endl;
    return true;
  }

  // Required interface for parallel computation
  // Called inside parallel solve -> should optimize nodes_in_problem only
  template <typename dtype>
  ceres::Solver::Summary RunSubset(
      const std::unordered_set<size_t>& nodes_in_problem,
      MapNameKeypoints* keypoints, const Graph* graph,
      const std::vector<size_t>& track_labels,
      const std::vector<bool>& root_labels, FeatureSet<dtype>& feature_set) {
    size_t channels = feature_set.Channels();
    size_t n_nodes = interpolation_config_.nodes.size();
    // Load Data (Thread-Safe). Auto-Clear on destruction
    FeatureView<dtype> feature_view(&feature_set, graph, nodes_in_problem);
#define REGISTER_METHOD(CHANNELS, N_NODES)                                    \
  if (channels == CHANNELS && n_nodes == N_NODES) {                           \
    TKOptim::Run<CHANNELS, N_NODES>(nodes_in_problem, keypoints, graph,       \
                                    track_labels, root_labels, feature_view); \
    return summary_;                                                          \
  }
    FEATUREMETRIC_CASES
#undef REGISTER_METHOD

    return summary_;
  }

  // Parallel interface: Given another adjuster, create a similar new one
  static FeatureMetricKeypointOptimizer Create(
      FeatureMetricKeypointOptimizer* other) {
    return FeatureMetricKeypointOptimizer(
        other->options_, other->problem_setup_, other->interpolation_config_);
  }

  template <int CHANNELS, int N_NODES, typename dtype>
  int AddIntraResiduals(MapNameKeypoints* keypoints, const Graph* graph,
                        const std::vector<size_t>& track_labels,
                        const std::vector<bool>& root_labels,
                        const size_t node_src_idx, const size_t node_dst_idx,
                        double weight, ceres::LossFunction* loss_function,
                        FeatureView<dtype>& feature_view);

 protected:
  InterpolationConfig interpolation_config_;
};

template <int CHANNELS, int N_NODES, typename dtype>
int FeatureMetricKeypointOptimizer::AddIntraResiduals(
    MapNameKeypoints* keypoints, const Graph* graph,
    const std::vector<size_t>& track_labels,
    const std::vector<bool>& root_labels, const size_t node_src_idx,
    const size_t node_dst_idx, double weight,
    ceres::LossFunction* loss_function, FeatureView<dtype>& feature_view) {
  if (track_labels[node_src_idx] != track_labels[node_dst_idx]) {
    return 0;
  }

  if (options_.root_edges_only && !root_labels[node_src_idx] &&
      !root_labels[node_dst_idx]) {
    return 0;
  }

  const FeatureNode* node_src = graph->nodes[node_src_idx];
  const FeatureNode* node_dst = graph->nodes[node_dst_idx];

  const std::string& image_name_src =
      graph->image_id_to_name.at(node_src->image_id);
  const std::string& image_name_dst =
      graph->image_id_to_name.at(node_dst->image_id);
  //@TODO: Potentially wrap this similar to bundle adjuster
  // AddFeatureMetricResidual(patch1, kp1, patch2, kp2, weight, loss)

  ceres::CostFunction* cost_function = nullptr;

  cost_function = FeatureMetric2DCostFunctor<dtype, CHANNELS, N_NODES>::Create(
      feature_view.GetFeaturePatch(node_src->image_id, node_src->feature_idx),
      feature_view.GetFeaturePatch(node_dst->image_id, node_dst->feature_idx),
      interpolation_config_);

  ceres::ResidualBlockId block_id = problem_->AddResidualBlock(
      cost_function,
      new ceres::ScaledLoss(loss_function, weight,
                            ceres::DO_NOT_TAKE_OWNERSHIP),
      keypoints->at(image_name_src).row(node_src->feature_idx).data(),
      keypoints->at(image_name_dst).row(node_dst->feature_idx).data());

  will_be_optimized_[node_src->node_idx] = true;
  will_be_optimized_[node_dst->node_idx] = true;

  return 1;  // Numbers of cost functions added
}

#undef FEATURE_METRIC_CASES
}  // namespace pixsfm