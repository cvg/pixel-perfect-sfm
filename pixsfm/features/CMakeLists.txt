set(FOLDER_NAME "features")

PIXSFM_ADD_SOURCES(
    src/featurepatch.h
    src/featurepatch.cc
    src/patch_interpolator.h
    src/dynamic_patch_interpolator.h
    src/featuremap.h
    src/featuremap.cc
    src/featuremanager.h
    src/featuremanager.cc
    src/featureview.h
    src/featureview.cc
    src/featureset.h
    src/featureset.cc
    src/references.h
    src/references.cc
)

PYPIXSFM_ADD_SOURCES(bindings.cc)