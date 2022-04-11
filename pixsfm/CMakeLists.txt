# Here we basically just load the subdirectories

add_subdirectory(base)
add_subdirectory(util)
add_subdirectory(features)
add_subdirectory(keypoint_adjustment)
add_subdirectory(bundle_adjustment)
add_subdirectory(localization)
# add_subdirectory(exe)
add_subdirectory(residuals)

add_subdirectory(pyceres)
add_subdirectory(_pixsfm)

PIXSFM_ADD_STATIC_LIBRARY(pixsfm ${PIXSFM_SOURCES})

target_link_libraries(pixsfm
 ${PIXSFM_INTERNAL_LIBRARIES} ${PIXSFM_EXTERNAL_LIBRARIES})

add_library(pypixsfm ${PYPIXSFM_SOURCES})
target_link_libraries(pypixsfm PRIVATE pixsfm)

PIXSFM_ADD_PYMODULE(_pixsfm
    _pixsfm/bindings.cc
)
target_link_libraries(_pixsfm PRIVATE
 pypixsfm  ${PIXSFM_INTERNAL_LIBRARIES} ${PIXSFM_EXTERNAL_LIBRARIES})
