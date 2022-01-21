#include "graph.h"

namespace pixsfm {

std::vector<size_t> Graph::GetDegrees() {
  std::vector<size_t> output_degrees(nodes.size(), 0);
  for (auto& node : nodes) {
    output_degrees[node->node_idx] += node->out_matches.size();
    for (auto& match : node->out_matches) {
      output_degrees[match.node_idx]++;
    }
  }
  return output_degrees;
}

std::vector<double> Graph::GetScores() {
  std::vector<double> scores(nodes.size(), 0.0);
  for (auto& node : nodes) {
    for (auto& match : node->out_matches) {
      scores[match.node_idx] += match.sim;
      scores[node->node_idx] += match.sim;
    }
  }
  return scores;
}

std::vector<Edge> Graph::GetEdges() {
  std::vector<Edge> edges;
  for (auto& node : nodes) {
    for (auto& match : node->out_matches) {
      edges.push_back(
          std::make_tuple(node->node_idx, match.node_idx, match.sim));
    }
  }
  return edges;
}

FeatureNode* Graph::FindOrCreateNode(std::string image_name,
                                     colmap::point2D_t feature_idx) {
  // First register image_name to ID mappings
  auto ret = image_name_to_id.insert(
      std::make_pair(image_name, image_name_to_id.size()));
  size_t image_id = ret.first->second;
  if (ret.second) {  // Newly inserted
    image_id_to_name.insert(std::make_pair(image_id, image_name));
  }
  auto it = node_map.find(std::make_pair(image_id, feature_idx));
  if (it != node_map.end()) {
    return nodes[it->second];
  } else {
    FeatureNode* node = new FeatureNode(image_id, feature_idx);
    size_t node_idx = AddNode(node);
    node_map.insert(
        std::make_pair(std::make_pair(image_id, feature_idx), node_idx));
    return node;
  }
}

void Graph::AddEdge(FeatureNode* node1, FeatureNode* node2, double sim) {
  Match match;
  match.node_idx = node2->node_idx;
  match.sim = sim;
  node1->out_matches.push_back(match);
}

void Graph::RegisterMatches(std::string imname1, std::string imname2,
                            size_t* matches,       // Nx2
                            double* similarities,  // Nx1
                            size_t n_matches) {
  for (size_t match_idx = 0; match_idx < n_matches; ++match_idx) {
    colmap::point2D_t feature_idx1 = matches[2 * match_idx];
    colmap::point2D_t feature_idx2 = matches[2 * match_idx + 1];
    double similarity = similarities ? similarities[match_idx] : 1.0;

    FeatureNode* node1 = FindOrCreateNode(imname1, feature_idx1);
    FeatureNode* node2 = FindOrCreateNode(imname2, feature_idx2);

    AddEdge(node1, node2, similarity);
  }
}

Graph::~Graph() {
  for (FeatureNode* node_ptr : nodes) {
    delete node_ptr;
  }
}

FeatureNode::FeatureNode(colmap::image_t image_id_,
                         colmap::point2D_t feature_idx_)
    : image_id(image_id_), feature_idx(feature_idx_) {}

size_t Graph::AddNode(FeatureNode* node) {
  nodes.push_back(node);
  node->node_idx = nodes.size() - 1;
  return node->node_idx;
}

size_t Graph::AddNode(colmap::image_t image_id, colmap::point2D_t feature_idx) {
  FeatureNode* node = new FeatureNode(image_id, feature_idx);
  nodes.push_back(node);
  node->node_idx = nodes.size() - 1;
  return node->node_idx;
}

size_t Graph::AddNode(std::string imname, colmap::point2D_t feature_idx) {
  auto ret =
      image_name_to_id.insert(std::make_pair(imname, image_name_to_id.size()));
  size_t image_id = ret.first->second;
  if (ret.second) {  // Newly inserted
    image_id_to_name.insert(std::make_pair(image_id, imname));
  }
  return AddNode(image_id, feature_idx);
}

size_t union_find_get_root(const size_t node_idx,
                           std::vector<int>& parent_nodes) {
  if (parent_nodes[node_idx] == -1) {
    return node_idx;
  }
  // Union-find path compression heuristic.
  parent_nodes[node_idx] =
      union_find_get_root(parent_nodes[node_idx], parent_nodes);
  return parent_nodes[node_idx];
}

std::vector<size_t> ComputeTrackLabels(Graph& graph) {
  typedef std::tuple<double, size_t, size_t> edge_tuple;
  STDLOG(INFO) << "Computing tracks..." << std::endl;
  const size_t n_nodes = graph.nodes.size();
  STDLOG(INFO) << "# graph nodes:"
               << " " << n_nodes << std::endl;
  std::vector<edge_tuple> edges;
  for (FeatureNode* node : graph.nodes) {
    for (auto& match : node->out_matches) {
      edges.push_back(
          std::make_tuple(match.sim, node->node_idx, match.node_idx));
    }
  }
  STDLOG(INFO) << "# graph edges:"
               << " " << edges.size() << std::endl;

  // auto start = std::chrono::high_resolution_clock::now();
  // Build the MSF.
  std::sort(edges.begin(), edges.end());
  std::reverse(edges.begin(), edges.end());

  std::vector<int> parent_nodes(n_nodes, -1);
  std::vector<std::set<colmap::image_t>> images_in_track(n_nodes);

  for (size_t node_idx = 0; node_idx < n_nodes; ++node_idx) {
    images_in_track[node_idx].insert(graph.nodes[node_idx]->image_id);
  }

  for (auto it : edges) {
    size_t node_idx1 = std::get<1>(it);
    size_t node_idx2 = std::get<2>(it);

    size_t root1 = union_find_get_root(node_idx1, parent_nodes);
    size_t root2 = union_find_get_root(node_idx2, parent_nodes);

    if (root1 != root2) {
      std::set<colmap::image_t> intersection;
      std::set_intersection(
          images_in_track[root1].begin(), images_in_track[root1].end(),
          images_in_track[root2].begin(), images_in_track[root2].end(),
          std::inserter(intersection, intersection.begin()));
      if (intersection.size() != 0) {
        continue;
      }
      // Union-find merging heuristic.
      if (images_in_track[root1].size() < images_in_track[root2].size()) {
        parent_nodes[root1] = root2;
        images_in_track[root2].insert(images_in_track[root1].begin(),
                                      images_in_track[root1].end());
        images_in_track[root1].clear();
      } else {
        parent_nodes[root2] = root1;
        images_in_track[root1].insert(images_in_track[root2].begin(),
                                      images_in_track[root2].end());
        images_in_track[root2].clear();
      }
    }
  }

  // Compute the tracks.
  std::vector<size_t> track_labels(n_nodes, -1);

  size_t n_tracks = 0;
  for (size_t node_idx = 0; node_idx < n_nodes; ++node_idx) {
    if (parent_nodes[node_idx] == -1) {
      track_labels[node_idx] = n_tracks++;
    }
  }
  STDLOG(INFO) << "# tracks:"
               << " " << n_tracks << std::endl;

  for (size_t node_idx = 0; node_idx < n_nodes; ++node_idx) {
    if (track_labels[node_idx] != -1) {
      continue;
    }
    track_labels[node_idx] =
        track_labels[union_find_get_root(node_idx, parent_nodes)];
  }

  return track_labels;
}

std::vector<double> ComputeScoreLabels(Graph& graph,
                                       std::vector<size_t>& track_labels) {
  const size_t n_nodes = graph.nodes.size();
  std::vector<double> score_labels(n_nodes, 0.0);
  for (size_t node_idx = 0; node_idx < n_nodes; ++node_idx) {
    FeatureNode* node = graph.nodes[node_idx];

    for (auto& match : node->out_matches) {
      if (track_labels[node_idx] == track_labels[match.node_idx]) {
        score_labels[node_idx] += match.sim;
        score_labels[match.node_idx] += match.sim;
      }
    }
  }
  return score_labels;
}

std::vector<bool> ComputeRootLabels(Graph& graph,
                                    std::vector<size_t> track_labels,
                                    std::vector<double> score_labels) {
  // Find the root nodes.
  const size_t n_nodes = graph.nodes.size();
  const size_t n_tracks =
      (*std::max_element(track_labels.begin(), track_labels.end())) + 1;
  std::vector<std::pair<double, size_t>> scores;

  for (size_t node_idx = 0; node_idx < n_nodes; ++node_idx) {
    scores.push_back(std::make_pair(score_labels[node_idx], node_idx));
  }

  std::sort(scores.begin(), scores.end());
  std::reverse(scores.begin(), scores.end());

  std::vector<bool> is_root(n_nodes, false);
  std::vector<bool> has_root(n_tracks, false);

  for (auto it : scores) {
    size_t node_idx = it.second;

    if (has_root[track_labels[node_idx]]) {
      continue;
    }

    is_root[node_idx] = true;
    has_root[track_labels[node_idx]] = true;
  }

  return is_root;
}

std::vector<std::pair<size_t, size_t>> CountEdgesAB(
    Graph& graph, const std::vector<size_t>& track_labels,
    const std::vector<bool> is_root) {
  // first holds edges A-B, second holds edges b-b
  const size_t n_nodes = graph.nodes.size();
  std::vector<std::pair<size_t, size_t>> track_edge_counts(n_nodes, {0, 0});
  for (FeatureNode* node : graph.nodes) {
    size_t node_idx1 = node->node_idx;
    for (Match& match : node->out_matches) {
      size_t node_idx2 = match.node_idx;
      size_t track_idx1 = track_labels[node_idx1];
      size_t track_idx2 = track_labels[node_idx2];
      if (track_idx1 == track_idx2) {
        if (is_root[node_idx1] || is_root[node_idx2]) {
          track_edge_counts[track_idx1].first += 1;
        } else {
          track_edge_counts[track_idx1].second += 1;
        }
      }
    }
  }

  return track_edge_counts;
}

std::vector<size_t> CountTrackEdges(Graph& graph,
                                    const std::vector<size_t>& track_labels) {
  const size_t n_tracks =
      std::unordered_set<size_t>(track_labels.begin(), track_labels.end())
          .size();
  std::vector<size_t> track_edge_counts(n_tracks, 0);
  for (FeatureNode* node : graph.nodes) {
    size_t node_idx1 = node->node_idx;
    for (Match& match : node->out_matches) {
      size_t node_idx2 = match.node_idx;
      size_t track_idx1 = track_labels[node_idx1];
      size_t track_idx2 = track_labels[node_idx2];
      if (track_idx1 == track_idx2) {
        track_edge_counts[track_idx1]++;
      }
    }
  }

  return track_edge_counts;
}

}  // namespace pixsfm