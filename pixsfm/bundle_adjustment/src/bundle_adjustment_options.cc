#include "bundle_adjustment/src/bundle_adjustment_options.h"

#include <colmap/util/misc.h>

#include "util/src/log_exceptions.h"

namespace pixsfm {

size_t BundleAdjustmentSetup::NumImages() const { return image_ids_.size(); }

size_t BundleAdjustmentSetup::NumPoints() const {
  return variable_point3D_ids_.size() + constant_point3D_ids_.size();
}

size_t BundleAdjustmentSetup::NumConstantCameras() const {
  return constant_camera_ids_.size();
}

size_t BundleAdjustmentSetup::NumConstantPoses() const {
  return constant_poses_.size();
}

size_t BundleAdjustmentSetup::NumConstantTvecs() const {
  return constant_tvecs_.size();
}

size_t BundleAdjustmentSetup::NumVariablePoints() const {
  return variable_point3D_ids_.size();
}

size_t BundleAdjustmentSetup::NumConstantPoints() const {
  return constant_point3D_ids_.size();
}

void BundleAdjustmentSetup::AddImage(const colmap::image_t image_id) {
  image_ids_.insert(image_id);
}

bool BundleAdjustmentSetup::HasImage(const colmap::image_t image_id) const {
  return image_ids_.find(image_id) != image_ids_.end();
}

void BundleAdjustmentSetup::RemoveImage(const colmap::image_t image_id) {
  image_ids_.erase(image_id);
}

void BundleAdjustmentSetup::SetConstantCamera(
    const colmap::camera_t camera_id) {
  constant_camera_ids_.insert(camera_id);
}

void BundleAdjustmentSetup::SetVariableCamera(
    const colmap::camera_t camera_id) {
  constant_camera_ids_.erase(camera_id);
}

bool BundleAdjustmentSetup::IsConstantCamera(
    const colmap::camera_t camera_id) const {
  return constant_camera_ids_.find(camera_id) != constant_camera_ids_.end();
}

void BundleAdjustmentSetup::SetConstantPose(const colmap::image_t image_id) {
  THROW_CHECK(HasImage(image_id));
  THROW_CHECK(!HasConstantTvec(image_id));
  constant_poses_.insert(image_id);
}

void BundleAdjustmentSetup::SetVariablePose(const colmap::image_t image_id) {
  constant_poses_.erase(image_id);
}

bool BundleAdjustmentSetup::HasConstantPose(
    const colmap::image_t image_id) const {
  return constant_poses_.find(image_id) != constant_poses_.end();
}

void BundleAdjustmentSetup::SetConstantTvec(const colmap::image_t image_id,
                                            const std::vector<int>& idxs) {
  THROW_CHECK_GT(idxs.size(), 0);
  THROW_CHECK_LE(idxs.size(), 3);
  THROW_CHECK(HasImage(image_id));
  THROW_CHECK(!HasConstantPose(image_id));
  THROW_CHECK(!colmap::VectorContainsDuplicateValues(idxs));
  constant_tvecs_.emplace(image_id, idxs);
}

void BundleAdjustmentSetup::RemoveConstantTvec(
    const colmap::image_t image_id) {
  constant_tvecs_.erase(image_id);
}

bool BundleAdjustmentSetup::HasConstantTvec(
    const colmap::image_t image_id) const {
  return constant_tvecs_.find(image_id) != constant_tvecs_.end();
}

void BundleAdjustmentSetup::AddVariablePoint(
    const colmap::point3D_t point3D_id) {
  THROW_CHECK(!HasConstantPoint(point3D_id));
  variable_point3D_ids_.insert(point3D_id);
}

void BundleAdjustmentSetup::AddConstantPoint(
    const colmap::point3D_t point3D_id) {
  THROW_CHECK(!HasVariablePoint(point3D_id));
  constant_point3D_ids_.insert(point3D_id);
}

bool BundleAdjustmentSetup::HasPoint(
    const colmap::point3D_t point3D_id) const {
  return HasVariablePoint(point3D_id) || HasConstantPoint(point3D_id);
}

bool BundleAdjustmentSetup::HasVariablePoint(
    const colmap::point3D_t point3D_id) const {
  return variable_point3D_ids_.find(point3D_id) !=
         variable_point3D_ids_.end();
}

bool BundleAdjustmentSetup::HasConstantPoint(
    const colmap::point3D_t point3D_id) const {
  return constant_point3D_ids_.find(point3D_id) !=
         constant_point3D_ids_.end();
}

void BundleAdjustmentSetup::RemoveVariablePoint(
    const colmap::point3D_t point3D_id) {
  variable_point3D_ids_.erase(point3D_id);
}

void BundleAdjustmentSetup::RemoveConstantPoint(
    const colmap::point3D_t point3D_id) {
  constant_point3D_ids_.erase(point3D_id);
}

const std::unordered_set<colmap::image_t>& BundleAdjustmentSetup::Images()
    const {
  return image_ids_;
}

const std::unordered_set<colmap::point3D_t>&
BundleAdjustmentSetup::VariablePoints() const {
  return variable_point3D_ids_;
}

const std::unordered_set<colmap::point3D_t>&
BundleAdjustmentSetup::ConstantPoints() const {
  return constant_point3D_ids_;
}

const std::vector<int>& BundleAdjustmentSetup::ConstantTvec(
    const colmap::image_t image_id) const {
  return constant_tvecs_.at(image_id);
}

}  // namespace pixsfm
