#include "bundle_adjustment/src/bundle_adjustment_options.h"

#include <colmap/util/misc.h>

namespace pixsfm {

void BundleAdjustmentSetup::SetConstantPose(const colmap::image_t image_id) {
  THROW_CHECK(HasImage(image_id));
  THROW_CHECK(!HasConstantTvec(image_id));
  colmap::BundleAdjustmentConfig::SetConstantPose(image_id);
}

void BundleAdjustmentSetup::SetConstantTvec(const colmap::image_t image_id,
                                            const std::vector<int>& idxs) {
  THROW_CHECK_GT(idxs.size(), 0);
  THROW_CHECK_LE(idxs.size(), 3);
  THROW_CHECK(HasImage(image_id));
  THROW_CHECK(!HasConstantPose(image_id));
  THROW_CHECK(!colmap::VectorContainsDuplicateValues(idxs));
  colmap::BundleAdjustmentConfig::SetConstantTvec(image_id, idxs);
}
void BundleAdjustmentSetup::AddVariablePoint(
    const colmap::point3D_t point3D_id) {
  THROW_CHECK(!HasConstantPoint(point3D_id));
  colmap::BundleAdjustmentConfig::AddVariablePoint(point3D_id);
}
void BundleAdjustmentSetup::AddConstantPoint(
    const colmap::point3D_t point3D_id) {
  THROW_CHECK(!HasVariablePoint(point3D_id));
  colmap::BundleAdjustmentConfig::AddConstantPoint(point3D_id);
}
}  // namespace pixsfm
