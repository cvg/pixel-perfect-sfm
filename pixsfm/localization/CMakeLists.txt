set(FOLDER_NAME "localization")

PIXSFM_ADD_SOURCES(
    src/query_bundle_optimizer.h
    src/query_keypoint_optimizer.h
    src/single_query_keypoint_optimizer.h
    src/single_query_bundle_optimizer.h
    src/nearest_references.h
    src/query_refinement_options.h
)

PYPIXSFM_ADD_SOURCES(bindings.cc)