set(FOLDER_NAME "residuals")

PIXSFM_ADD_SOURCES(
    src/feature_reference.h
    src/featuremetric.h
    src/geometric.h
)

PYPIXSFM_ADD_SOURCES(bindings.cc)