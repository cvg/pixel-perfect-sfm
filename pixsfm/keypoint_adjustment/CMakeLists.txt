set(FOLDER_NAME "keypoint_adjustment")

PIXSFM_ADD_SOURCES(
    src/keypoint_optimizer.h
    src/topological_keypoint_optimizer.h
    src/featuremetric_keypoint_optimizer.h
    src/keypoint_adjustment_options.h src/keypoint_adjustment_options.cc
)

PYPIXSFM_ADD_SOURCES(bindings.cc)