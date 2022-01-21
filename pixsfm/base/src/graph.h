#pragma once
#include <cmath>
#include <map>
#include <set>
#include <string>
#include <vector>

#include <algorithm>

#include <chrono>

#include <limits>

#include <fcntl.h>
#include <unistd.h>

#include <iostream>

#include <colmap/util/types.h>

#include "util/src/log_exceptions.h"
#include "util/src/simple_logger.h"

// Copyright (c) 2020, ETH Zurich, CVG, Mihai Dusmanu
// (mihai.dusmanu@inf.ethz.ch)

// Adapted by Philipp Lindenberger for this project.

namespace pixsfm {

typedef std::tuple<size_t, size_t, double> Edge;

struct Match {
  size_t node_idx;
  double sim;
};

class FeatureNode {
 public:
  FeatureNode(colmap::image_t, colmap::point2D_t);
  colmap::image_t image_id = -1;
  colmap::point2D_t feature_idx = -1;
  size_t node_idx = -1;
  std::vector<Match> out_matches;  // outward matches
};

class Graph {
 public:
  size_t AddNode(FeatureNode*);
  size_t AddNode(std::string, colmap::point2D_t);
  size_t AddNode(colmap::image_t, colmap::point2D_t);
  FeatureNode* FindOrCreateNode(std::string, colmap::point2D_t);
  std::vector<size_t> GetDegrees();
  std::vector<double> GetScores();
  std::vector<Edge> GetEdges();
  void AddEdge(FeatureNode* node1, FeatureNode* node2, double sim);
  void RegisterMatches(std::string imname1, std::string imname2,
                       size_t* matches,       // Nx2
                       double* similarities,  // Nx1
                       size_t n_matches);
  ~Graph();
  std::vector<FeatureNode*> nodes;
  std::unordered_map<std::string, colmap::image_t> image_name_to_id;
  std::unordered_map<colmap::image_t, std::string> image_id_to_name;
  std::map<std::pair<colmap::image_t, colmap::point2D_t>, size_t> node_map;
};

size_t union_find_get_root(const size_t node_idx,
                           std::vector<int>& parent_nodes);

std::vector<size_t> ComputeTrackLabels(Graph& graph);

std::vector<double> ComputeScoreLabels(Graph& graph,
                                       std::vector<size_t>& track_labels);

std::vector<bool> ComputeRootLabels(Graph& graph,
                                    std::vector<size_t> track_labels,
                                    std::vector<double> score_labels);

std::vector<std::pair<size_t, size_t>> CountEdgesAB(
    Graph& graph, const std::vector<size_t>& track_labels,
    const std::vector<bool> is_root);

std::vector<size_t> CountTrackEdges(Graph& graph,
                                    const std::vector<size_t>& track_labels);

}  // namespace pixsfm