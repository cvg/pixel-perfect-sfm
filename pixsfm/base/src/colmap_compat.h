#pragma once

// Helpers bridging the COLMAP 3.8 image pose API (qvec/tvec stored on the
// image) that pixsfm was written against and the COLMAP 4.x API, where poses
// are stored as rig_from_world on frames.
//
// pixsfm optimizes poses as two parameter blocks: the rotation quaternion
// (4 values, Eigen order x, y, z, w) and the translation (3 values). Both
// blocks point into the frame's Rigid3d::params. This only equals the image's
// cam_from_world if the image is the reference sensor of its frame, which is
// always the case for single-camera reconstructions.

#include <colmap/geometry/rigid3.h>
#include <colmap/scene/frame.h>
#include <colmap/scene/image.h>
#include <colmap/scene/reconstruction.h>
#include <colmap/sensor/models.h>

#include <stdexcept>
#include <string>

namespace pixsfm {

// Camera models supported by pixsfm's templated residuals (all perspective
// models; spherical models are not supported).
#define PIXSFM_CAMERA_MODEL_SWITCH_CASES  \
  PERSPECTIVE_CAMERA_MODEL_CASES          \
  default:                                \
    CAMERA_MODEL_DOES_NOT_EXIST_EXCEPTION \
    break;

inline colmap::Rigid3d& MutableCamFromWorld(colmap::Image& image) {
  if (!image.HasFramePtr()) {
    throw std::invalid_argument("Image " + std::to_string(image.ImageId()) +
                                " is not associated with a frame.");
  }
  if (!image.IsRefInFrame()) {
    throw std::invalid_argument(
        "Image " + std::to_string(image.ImageId()) +
        " is not the reference sensor of its rig. pixsfm only supports "
        "trivial rigs (one camera per frame).");
  }
  return image.FramePtr()->RigFromWorld();
}

inline double* QvecData(colmap::Image& image) {
  return MutableCamFromWorld(image).params.data();
}

inline double* TvecData(colmap::Image& image) {
  return MutableCamFromWorld(image).params.data() + 4;
}

inline void NormalizeQvec(colmap::Image& image) {
  MutableCamFromWorld(image).rotation().normalize();
}

}  // namespace pixsfm
