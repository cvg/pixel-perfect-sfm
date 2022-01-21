set(FOLDER_NAME "bundle_adjustment")

PIXSFM_ADD_SOURCES(
    src/bundle_adjustment_options.h
    src/bundle_adjustment_options.cc
    src/bundle_optimizer.h
    src/costmap_bundle_optimizer.h
    src/feature_reference_bundle_optimizer.h
    src/patch_warp_bundle_optimizer.h
    src/geometric_bundle_optimizer.h
    src/reference_extractor.h
    src/costmap_extractor.h
)

PYPIXSFM_ADD_SOURCES(bindings.cc)

PIXSFM_ADD_TEST(bundle_optimizer_test src/bundle_optimizer_test.cc)