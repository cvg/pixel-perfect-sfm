#include "keypoint_adjustment/src/keypoint_adjustment_options.h"

namespace pixsfm {

bool KeypointAdjustmentSetup::IsNodeConstant(const FeatureNode* node) const {
  return IsKeypointConstant(node->image_id, node->feature_idx);
}

bool KeypointAdjustmentSetup::IsKeypointConstant(
    colmap::image_t image_id, colmap::point2D_t feature_idx) const {
  auto image_it = constant_images.find(image_id);
  bool constant_image = image_it != constant_images.end();
  if (!constant_image) {
    try {
      auto& const_kps = constant_keypoints.at(image_id);
      return const_kps.find(feature_idx) != const_kps.end();
    } catch (std::out_of_range& e) {
      return false;
    }
  }
  return true;
}

void KeypointAdjustmentSetup::SetMaskedNodesConstant(
    const Graph* graph, const std::vector<bool>& mask) {
  THROW_CHECK_EQ(mask.size(), graph->nodes.size());
  for (int i = 0; i < mask.size(); i++) {
    if (mask[i]) {
      SetNodeConstant(graph->nodes[i]);
    }
  }
}

void KeypointAdjustmentSetup::SetNodeConstant(const FeatureNode* node) {
  SetKeypointConstant(node->image_id, node->feature_idx);
}

void KeypointAdjustmentSetup::SetKeypointConstant(
    colmap::image_t image_id, colmap::point2D_t feature_idx) {
  constant_keypoints[image_id].insert(feature_idx);
}

void KeypointAdjustmentSetup::SetImageConstant(colmap::image_t image_id) {
  constant_images.insert(image_id);
}

}  // namespace pixsfm