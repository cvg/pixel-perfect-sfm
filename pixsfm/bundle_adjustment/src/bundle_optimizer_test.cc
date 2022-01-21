// Copyright (c) 2018, ETH Zurich and UNC Chapel Hill.
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
//     * Redistributions of source code must retain the above copyright
//       notice, this list of conditions and the following disclaimer.
//
//     * Redistributions in binary form must reproduce the above copyright
//       notice, this list of conditions and the following disclaimer in the
//       documentation and/or other materials provided with the distribution.
//
//     * Neither the name of ETH Zurich and UNC Chapel Hill nor the names of
//       its contributors may be used to endorse or promote products derived
//       from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//
// Author: Johannes L. Schoenberger (jsch-at-demuc-dot-de)
// Edited by: Philipp Lindenberger

#define TEST_NAME "bundle_adjustment/bundle_optimizer"
#include <colmap/util/testing.h>

#include <colmap/base/camera_models.h>
#include <colmap/base/correspondence_graph.h>
#include <colmap/base/projection.h>
#include <colmap/optim/bundle_adjustment.h>
#include <colmap/util/random.h>

#include "bundle_adjustment/src/bundle_adjustment_options.h"
#include "bundle_adjustment/src/geometric_bundle_optimizer.h"

// This project defines a general BundleAdjuster interface where only the
// AddResiduals function needs to be overwritten. Here we test if the setup
// is the same as in COLMAP by comparing all reconstruction params with colmap,
// over all tests specified in COLMAP using the geometric reprojection error.

// We have some differences to COLMAP in how we solve the problem, thus
// a rather big tolerance
#define TOL 1.0e-4

#define BOOST_CHECK_ALL_CLOSE(vec1, vec2)             \
  for (int i = 0; i < vec1.size(); i++) {             \
    BOOST_CHECK_LT(std::abs(vec1[i] - vec2[i]), TOL); \
  }

namespace colmap {

void GeneratePointCloud(const size_t num_points, const Eigen::Vector3d& min,
                        const Eigen::Vector3d& max,
                        Reconstruction* reconstruction) {
  for (size_t i = 0; i < num_points; ++i) {
    Eigen::Vector3d xyz;
    xyz.x() = RandomReal(min.x(), max.x());
    xyz.y() = RandomReal(min.y(), max.y());
    xyz.z() = RandomReal(min.z(), max.z());
    reconstruction->AddPoint3D(xyz, Track());
  }
}

void GenerateReconstruction(const size_t num_images, const size_t num_points,
                            Reconstruction* reconstruction,
                            CorrespondenceGraph* correspondence_graph) {
  SetPRNGSeed(0);

  GeneratePointCloud(num_points, Eigen::Vector3d(-1, -1, -1),
                     Eigen::Vector3d(1, 1, 1), reconstruction);

  const double kFocalLengthFactor = 1.2;
  const size_t kImageSize = 1000;

  for (size_t i = 0; i < num_images; ++i) {
    const camera_t camera_id = static_cast<camera_t>(i);
    const image_t image_id = static_cast<image_t>(i);

    Camera camera;
    camera.InitializeWithId(SimpleRadialCameraModel::model_id,
                            kFocalLengthFactor * kImageSize, kImageSize,
                            kImageSize);
    camera.SetCameraId(camera_id);
    reconstruction->AddCamera(camera);

    Image image;
    image.SetImageId(image_id);
    image.SetCameraId(camera_id);
    image.SetName(std::to_string(i));
    image.Qvec() = ComposeIdentityQuaternion();
    image.Tvec() =
        Eigen::Vector3d(RandomReal(-1.0, 1.0), RandomReal(-1.0, 1.0), 10);
    image.SetRegistered(true);
    reconstruction->AddImage(image);

    const Eigen::Matrix3x4d proj_matrix = image.ProjectionMatrix();

    std::vector<Eigen::Vector2d> points2D;
    for (const auto& point3D : reconstruction->Points3D()) {
      BOOST_CHECK(HasPointPositiveDepth(proj_matrix, point3D.second.XYZ()));
      // Get exact projection of 3D point.
      Eigen::Vector2d point2D =
          ProjectPointToImage(point3D.second.XYZ(), proj_matrix, camera);
      // Add some uniform noise.
      point2D += Eigen::Vector2d(RandomReal(-2.0, 2.0), RandomReal(-2.0, 2.0));
      points2D.push_back(point2D);
    }

    correspondence_graph->AddImage(image_id, num_points);
    reconstruction->Image(image_id).SetPoints2D(points2D);
  }

  reconstruction->SetUp(correspondence_graph);

  for (size_t i = 0; i < num_images; ++i) {
    const image_t image_id = static_cast<image_t>(i);
    TrackElement track_el;
    track_el.image_id = image_id;
    track_el.point2D_idx = 0;
    for (const auto& point3D : reconstruction->Points3D()) {
      reconstruction->AddObservation(point3D.first, track_el);
      track_el.point2D_idx += 1;
    }
  }
}

}  // namespace colmap

// We always test for similarity with colmap bundle adjuster
namespace pixsfm {

void CompareReconstructions(colmap::Reconstruction* reconstruction1,
                            colmap::Reconstruction* reconstruction2) {
  for (auto& image_pair : reconstruction1->Images()) {
    auto& image1 = image_pair.second;
    auto& image2 = reconstruction2->Image(image_pair.first);
    BOOST_CHECK_ALL_CLOSE(image1.Qvec(), image2.Qvec());
    BOOST_CHECK_ALL_CLOSE(image1.Tvec(), image2.Tvec());
  }

  for (auto& camera_pair : reconstruction1->Cameras()) {
    auto& camera1 = camera_pair.second;
    auto& camera2 = reconstruction2->Camera(camera_pair.first);
    BOOST_CHECK_ALL_CLOSE(camera1.Params(), camera2.Params());
  }

  for (auto& point_pair : reconstruction1->Points3D()) {
    auto& point3D1 = point_pair.second;
    auto& point3D2 = reconstruction2->Point3D(point_pair.first);
    BOOST_CHECK_ALL_CLOSE(point3D1.XYZ(), point3D2.XYZ());
  }
}

void TestBA(colmap::Reconstruction& reconstruction,
            colmap::BundleAdjustmentOptions& colmap_options,
            colmap::BundleAdjustmentConfig& config) {
  colmap::Reconstruction reconstruction2 = reconstruction;
  BundleAdjustmentSetup setup(config);  // copy config

  BundleOptimizerOptions options = BundleOptimizerOptions();

  options.loss = std::shared_ptr<ceres::LossFunction>(new ceres::TrivialLoss());

  options.refine_extrinsics = colmap_options.refine_extrinsics;
  options.refine_focal_length = colmap_options.refine_focal_length;
  options.refine_principal_point = colmap_options.refine_principal_point;
  options.refine_extra_params = colmap_options.refine_extra_params;
  options.solver_options = colmap_options.solver_options;
  options.solver_options.minimizer_progress_to_stdout = false;
  options.print_summary = false;
  colmap_options.print_summary = false;
  colmap::BundleAdjuster bundle_adjuster(colmap_options, config);
  BOOST_REQUIRE(bundle_adjuster.Solve(&reconstruction));
  const auto summary = bundle_adjuster.Summary();

  GeometricBundleOptimizer geom_bundle_adjuster(options, setup);
  BOOST_REQUIRE(geom_bundle_adjuster.Run(&reconstruction2));
  CompareReconstructions(&reconstruction, &reconstruction2);
}
BOOST_AUTO_TEST_CASE(TestTwoView) {
  colmap::Reconstruction reconstruction;
  colmap::CorrespondenceGraph correspondence_graph;
  colmap::GenerateReconstruction(2, 100, &reconstruction,
                                 &correspondence_graph);

  colmap::BundleAdjustmentConfig config;
  config.AddImage(0);
  config.AddImage(1);
  config.SetConstantPose(0);
  config.SetConstantTvec(1, {0});

  colmap::BundleAdjustmentOptions options;
  TestBA(reconstruction, options, config);
}

BOOST_AUTO_TEST_CASE(TestTwoViewConstantCamera) {
  colmap::Reconstruction reconstruction;
  colmap::CorrespondenceGraph correspondence_graph;
  colmap::GenerateReconstruction(2, 100, &reconstruction,
                                 &correspondence_graph);

  colmap::BundleAdjustmentConfig config;
  config.AddImage(0);
  config.AddImage(1);
  config.SetConstantPose(0);
  config.SetConstantPose(1);
  config.SetConstantCamera(0);

  colmap::BundleAdjustmentOptions options;
  TestBA(reconstruction, options, config);
}

BOOST_AUTO_TEST_CASE(TestPartiallyContainedTracks) {
  colmap::Reconstruction reconstruction;
  colmap::CorrespondenceGraph correspondence_graph;
  colmap::GenerateReconstruction(3, 100, &reconstruction,
                                 &correspondence_graph);
  const auto variable_point3D_id =
      reconstruction.Image(2).Point2D(0).Point3DId();
  reconstruction.DeleteObservation(2, 0);

  colmap::BundleAdjustmentConfig config;
  config.AddImage(0);
  config.AddImage(1);
  config.SetConstantPose(0);
  config.SetConstantPose(1);

  colmap::BundleAdjustmentOptions options;
  TestBA(reconstruction, options, config);
}

BOOST_AUTO_TEST_CASE(TestPartiallyContainedTracksForceToOptimizePoint) {
  colmap::Reconstruction reconstruction;
  colmap::CorrespondenceGraph correspondence_graph;
  colmap::GenerateReconstruction(3, 100, &reconstruction,
                                 &correspondence_graph);
  const colmap::point3D_t variable_point3D_id =
      reconstruction.Image(2).Point2D(0).Point3DId();
  const colmap::point3D_t add_variable_point3D_id =
      reconstruction.Image(2).Point2D(1).Point3DId();
  const colmap::point3D_t add_constant_point3D_id =
      reconstruction.Image(2).Point2D(2).Point3DId();
  reconstruction.DeleteObservation(2, 0);

  colmap::BundleAdjustmentConfig config;
  config.AddImage(0);
  config.AddImage(1);
  config.SetConstantPose(0);
  config.SetConstantPose(1);
  config.AddVariablePoint(add_variable_point3D_id);
  config.AddConstantPoint(add_constant_point3D_id);

  colmap::BundleAdjustmentOptions options;
  TestBA(reconstruction, options, config);
}

BOOST_AUTO_TEST_CASE(TestConstantPoints) {
  colmap::Reconstruction reconstruction;
  colmap::CorrespondenceGraph correspondence_graph;
  colmap::GenerateReconstruction(2, 100, &reconstruction,
                                 &correspondence_graph);

  const colmap::point3D_t constant_point3D_id1 = 1;
  const colmap::point3D_t constant_point3D_id2 = 2;

  colmap::BundleAdjustmentConfig config;
  config.AddImage(0);
  config.AddImage(1);
  config.SetConstantPose(0);
  config.SetConstantPose(1);
  config.AddConstantPoint(constant_point3D_id1);
  config.AddConstantPoint(constant_point3D_id2);

  colmap::BundleAdjustmentOptions options;
  TestBA(reconstruction, options, config);
}

BOOST_AUTO_TEST_CASE(TestVariableImage) {
  colmap::Reconstruction reconstruction;
  colmap::CorrespondenceGraph correspondence_graph;
  colmap::GenerateReconstruction(3, 100, &reconstruction,
                                 &correspondence_graph);

  colmap::BundleAdjustmentConfig config;
  config.AddImage(0);
  config.AddImage(1);
  config.AddImage(2);
  config.SetConstantPose(0);
  config.SetConstantTvec(1, {0});

  colmap::BundleAdjustmentOptions options;
  TestBA(reconstruction, options, config);
}

BOOST_AUTO_TEST_CASE(TestConstantFocalLength) {
  colmap::Reconstruction reconstruction;
  colmap::CorrespondenceGraph correspondence_graph;
  colmap::GenerateReconstruction(2, 100, &reconstruction,
                                 &correspondence_graph);

  colmap::BundleAdjustmentConfig config;
  config.AddImage(0);
  config.AddImage(1);
  config.SetConstantPose(0);
  config.SetConstantTvec(1, {0});

  colmap::BundleAdjustmentOptions options;
  options.refine_focal_length = false;
  TestBA(reconstruction, options, config);
}

BOOST_AUTO_TEST_CASE(TestVariablePrincipalPoint) {
  colmap::Reconstruction reconstruction;
  colmap::CorrespondenceGraph correspondence_graph;
  colmap::GenerateReconstruction(2, 100, &reconstruction,
                                 &correspondence_graph);

  colmap::BundleAdjustmentConfig config;
  config.AddImage(0);
  config.AddImage(1);
  config.SetConstantPose(0);
  config.SetConstantTvec(1, {0});

  colmap::BundleAdjustmentOptions options;
  options.refine_principal_point = true;
  TestBA(reconstruction, options, config);
}

BOOST_AUTO_TEST_CASE(TestConstantExtraParam) {
  colmap::Reconstruction reconstruction;
  colmap::CorrespondenceGraph correspondence_graph;
  colmap::GenerateReconstruction(2, 100, &reconstruction,
                                 &correspondence_graph);

  colmap::BundleAdjustmentConfig config;
  config.AddImage(0);
  config.AddImage(1);
  config.SetConstantPose(0);
  config.SetConstantTvec(1, {0});

  colmap::BundleAdjustmentOptions options;
  options.refine_extra_params = false;
  TestBA(reconstruction, options, config);
}
}  // namespace pixsfm