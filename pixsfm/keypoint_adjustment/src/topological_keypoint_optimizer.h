
#pragma once
#include "_pixsfm/src/helpers.h"

#include <ceres/ceres.h>
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
#include "features/src/featureset.h"
#include "features/src/featureview.h"
#include "features/src/references.h"

#include "base/src/interpolation.h"
#include "util/src/log_exceptions.h"
#include "util/src/types.h"

#include "residuals/src/featuremetric.h"

#include "base/src/graph.h"
#include "keypoint_adjustment/src/keypoint_adjustment_options.h"
#include "keypoint_adjustment/src/keypoint_optimizer.h"
namespace pixsfm {

// Low level solver: Given component indices
template <typename Derived>
class TopologicalKeypointOptimizer : public KeypointOptimizerBase {
 public:
  struct Options : KeypointOptimizerOptions {
    using KeypointOptimizerOptions::KeypointOptimizerOptions;
    bool weight_by_sim = true;
    double root_regularize_weight = -1.0;
    bool root_edges_only = false;
  };
  TopologicalKeypointOptimizer(const Options& options,
                               std::shared_ptr<KeypointAdjustmentSetup> setup)
      : KeypointOptimizerBase(options, setup), options_(options) {}

  template <int... Ns, typename... Params>
  inline bool Run(const std::unordered_set<size_t>& nodes_in_problem,
                  MapNameKeypoints* keypoints, const Graph* graph,
                  const std::vector<size_t>& track_labels,
                  const std::vector<bool>& root_labels, Params&... parameters);

  template <int... Ns, typename... Params>
  inline void SetUp(const std::unordered_set<size_t>& nodes_in_problem,
                    MapNameKeypoints* keypoints, const Graph* graph,
                    const std::vector<size_t>& track_labels,
                    const std::vector<bool>& root_labels,
                    ceres::LossFunction* loss_function, Params&... parameters);

 protected:
  Derived* GetDerived() { return static_cast<Derived*>(this); }
  size_t n_intra_edges = 0;
  size_t n_inter_edges = 0;
  Options options_;
};

template <typename Derived>
template <int... Ns, typename... Params>
bool TopologicalKeypointOptimizer<Derived>::Run(
    const std::unordered_set<size_t>& nodes_in_problem,
    MapNameKeypoints* keypoints, const Graph* graph,
    const std::vector<size_t>& track_labels,
    const std::vector<bool>& root_labels, Params&... parameters) {
  THROW_CUSTOM_CHECK_MSG(keypoints, std::invalid_argument,
                         "keypoints cannot be NULL.");
  THROW_CUSTOM_CHECK_MSG(
      !problem_, std::invalid_argument,
      "Cannot use the same KeypointOptimizer multiple times");

  problem_.reset(new ceres::Problem(options_.problem_options));
  will_be_optimized_.resize(graph->nodes.size(), false);
  ceres::LossFunction* loss_function = options_.loss.get();
  // CHECK_NOTNULL(loss_function);
  THROW_CUSTOM_CHECK_MSG(loss_function, std::invalid_argument,
                         "loss function cannot be NULL.");
  // if (options_.solver_options.use_inner_iterations) {
  //     options_.solver_options.inner_iteration_ordering.reset(new
  //     ceres::ParameterBlockOrdering);
  // }
  SetUp<Ns...>(nodes_in_problem, keypoints, graph, track_labels, root_labels,
               loss_function, parameters...);

  return SolveProblem();
}
template <typename Derived>
template <int... Ns, typename... Params>
void TopologicalKeypointOptimizer<Derived>::SetUp(
    const std::unordered_set<size_t>& nodes_in_problem,
    MapNameKeypoints* keypoints, const Graph* graph,
    const std::vector<size_t>& track_labels,
    const std::vector<bool>& root_labels, ceres::LossFunction* loss_function,
    Params&... parameters) {
  std::vector<size_t> connected_to_root(graph->nodes.size(), false);
  std::unordered_map<size_t, size_t> track_root_node_idx;

  bool regularize_to_root = options_.root_regularize_weight > 0.0;

  std::vector<std::pair<size_t, Match>> edges_in_problem;
  for (size_t node_idx : nodes_in_problem) {
    const FeatureNode* node = graph->nodes[node_idx];
    for (auto& match : node->out_matches) {
      size_t dst_idx = match.node_idx;
      if (track_labels[node_idx] == track_labels[dst_idx]) {
        edges_in_problem.push_back(std::make_pair(node_idx, match));
        ++n_intra_edges;

        if (regularize_to_root) {
          if (root_labels[node_idx]) {
            track_root_node_idx[track_labels[node_idx]] = node_idx;
            connected_to_root[node_idx] = true;
            connected_to_root[dst_idx] = true;
          }
          if (root_labels[dst_idx]) {
            track_root_node_idx[track_labels[dst_idx]] = dst_idx;
            connected_to_root[node_idx] = true;
            connected_to_root[dst_idx] = true;
          }
        }
      } else {
        // TODO: AddInterResiduals()
      }
    }
  }
  int n_new_edges = 0;
  for (const auto& edge : edges_in_problem) {
    size_t node_idx1 = edge.first;
    size_t node_idx2 = edge.second.node_idx;
    double sim = edge.second.sim;
    FeatureNode* node1 = graph->nodes[node_idx1];
    FeatureNode* node2 = graph->nodes[node_idx2];
    // Avoid optimizing a keypoint to itself
    const std::string& image_name1 =
        graph->image_id_to_name.at(node1->image_id);
    const std::string& image_name2 =
        graph->image_id_to_name.at(node2->image_id);

    if (keypoints->at(image_name1).row(node1->feature_idx).data() ==
        keypoints->at(image_name2).row(node2->feature_idx).data()) {
      continue;
    }

    int new_residual_blocks = GetDerived()->template AddIntraResiduals<Ns...>(
        keypoints, graph, track_labels, root_labels, node_idx1, node_idx2,
        options_.weight_by_sim ? sim : 1.0, loss_function, parameters...);

    if (regularize_to_root) {
      std::vector<size_t> idxs = {node_idx1, node_idx2};
      for (size_t node_idx : idxs) {
        if (!connected_to_root[node_idx]) {
          int new_root_residual_blocks =
              GetDerived()->template AddIntraResiduals<Ns...>(
                  keypoints, graph, track_labels, root_labels, node_idx,
                  track_root_node_idx[track_labels[node_idx]],
                  options_.root_regularize_weight, loss_function,
                  parameters...);
          n_new_edges++;
          connected_to_root[node_idx] = true;
        }
      }
    }
  }

  GetDerived()->ParameterizeKeypoints(nodes_in_problem, keypoints, graph,
                                      parameters...);
}

}  // namespace pixsfm