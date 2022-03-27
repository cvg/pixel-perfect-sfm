# cmake version to be used
cmake_minimum_required( VERSION 3.0 )

# project name
project(PIXSFM)

################################################################################
# Include CMake dependencies
################################################################################
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)
set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fvisibility=hidden -fPIC -std=c++14")


# Include helper macros and commands, and allow the included file to override
# the CMake policies in this file
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/CMakeHelper.cmake NO_POLICY_SCOPE)

################################################################################
# Options
################################################################################
option(TESTS_ENABLED "Whether to build test binaries" ON)
option(AVX2_ENABLED "Whether to compile with AVX2 support" ON)

if(TESTS_ENABLED)
    enable_testing()
endif()

################################################################################
# Find packages
################################################################################
# find_package(CUDA ${CUDA_MIN_VERSION} REQUIRED) #not-required
find_package(Eigen3 3.3 REQUIRED)
find_package(COLMAP REQUIRED)

################################################################################
# Compiler specific configuration
################################################################################

if (AVX2_ENABLED)
  add_definitions("-DAVX2_ENABLED")
  message("Compiling with AVX2 support.")
  # @TODO investigate linking with non-avx code, it works on some machines without
  # -DEIGEN_DONT_VECTORIZE
  # We assume here colmap was compiled with default flags, thus we need to set
  # -DEIGEN_MAX_ALIGN_BYTES=16 to avoid segfaults (otherwise eigen aligns
  # on 32 bytes if avx2 enabled)
  set(EIGEN_DISABLE_ALIGN_FLAGS
    "-DEIGEN_MAX_ALIGN_BYTES=16 -DEIGEN_MAX_STATIC_ALIGN_BYTES=16 -DEIGEN_DONT_VECTORIZE")
  set (AVX2_CXX_FLAGS "-mavx2 -mf16c -mfma")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${AVX2_CXX_FLAGS} ${EIGEN_DISABLE_ALIGN_FLAGS}")
endif()

if (NOT EXISTS ${CMAKE_BINARY_DIR}/CMakeCache.txt)
  if (NOT CMAKE_BUILD_TYPE)
    message(STATUS "Setting build type to 'Release' as none was specified.")
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "" FORCE)
  endif()
endif()

add_definitions("-DBOOST_TEST_DYN_LINK")

################################################################################
# Add sources
################################################################################

# set(TEMP_INCLUDE_DIR ${CMAKE_BINARY_DIR}/include)
# file(MAKE_DIRECTORY ${TEMP_INCLUDE_DIR})

set(PIXSFM_INCLUDE_DIRS
  ${HDF5_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIR}
  ${PROJECT_SOURCE_DIR}
  ${FREEIMAGE_INCLUDE_DIRS}
  ${CERES_INCLUDE_DIRS}
  ${COLMAP_INCLUDE_DIRS}
)

set(PIXSFM_LINK_DIRS
  ${COLMAP_LINK_DIRS}
)

set(PIXSFM_EXTERNAL_LIBRARIES
  ${CERES_LIBRARIES}
  ${FREEIMAGE_LIBRARIES}
  ${COLMAP_LIBRARIES}
  ${HDF5_C_LIBRARIES}
)

set(PIXSFM_INTERNAL_LIBRARIES
  HighFive
  pybind11::module
  pybind11::embed
)

include_directories(
    third-party
    pixsfm
    ${PIXSFM_INCLUDE_DIRS}
)

link_directories(${PIXSFM_LINK_DIRS})

set(PYBIND11_INSTALL ON)
add_subdirectory(third-party)
add_subdirectory(pixsfm)


################################################################################
# Generate source groups for Visual Studio, XCode, etc.
################################################################################

PIXSFM_ADD_SOURCE_DIR(pixsfm/base/src BASE_SRCS *.h *.cc)
# PIXSFM_ADD_SOURCE_DIR(pixsfm/exe EXE_SRCS *.h *.cc)
PIXSFM_ADD_SOURCE_DIR(pixsfm/util UTIL_SRCS *.h *.cc)
PIXSFM_ADD_SOURCE_DIR(pixsfm/bundle_adjustment BA_SRCS *.h *.cc)
PIXSFM_ADD_SOURCE_DIR(pixsfm/keypoint_adjustment KA_SRCS *.h *.cc)
PIXSFM_ADD_SOURCE_DIR(pixsfm/localization LOC_SRCS *.h *.cc)
PIXSFM_ADD_SOURCE_DIR(pixsfm/features FEAT_SRCS *.h *.cc)
PIXSFM_ADD_SOURCE_DIR(pixsfm/residuals RESID_SRCS *.h *.cc)


# Add all of the source files to a regular library target, as using a custom
# target does not allow us to set its C++ include directories (and thus
# intellisense can't find any of the included files).
add_library(
    ${PIXSFM_SRC_ROOT_FOLDER}
    ${BASE_SRCS}
    # ${EXE_SRCS}
    ${UTIL_SRCS}
    ${BA_SRCS}
    ${KA_SRCS}
    ${LOC_SRCS}
    ${FEAT_SRCS}
    ${RESID_SRCS}
)

# Prevent the library from being compiled automatically.
set_target_properties(
    ${PIXSFM_SRC_ROOT_FOLDER} PROPERTIES
    EXCLUDE_FROM_ALL 1
    EXCLUDE_FROM_DEFAULT_BUILD 1)


################################################################################
# Install and uninstall scripts
################################################################################

# Install header files.
install(DIRECTORY pixsfm/
        DESTINATION include/pixsfm
        FILES_MATCHING PATTERN "*.h")
install(DIRECTORY third-party/
        DESTINATION include/pixsfm/third-party
        FILES_MATCHING
        PATTERN "*.h"
        PATTERN "HighFive" EXCLUDE
        PATTERN "pybind11" EXCLUDE)

# Generate and install CMake configuration.
configure_file("${CMAKE_CURRENT_SOURCE_DIR}/cmake/CMakeConfig.cmake.in"
               "${CMAKE_CURRENT_BINARY_DIR}/PIXSFMConfig.cmake" @ONLY)
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/PIXSFMConfig.cmake"
        DESTINATION "share/pixsfm")

# Install find_package scripts for dependencies.
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/cmake
        DESTINATION share/pixsfm
        FILES_MATCHING PATTERN "Find*.cmake")

# Configure the uninstallation script.
configure_file("${CMAKE_CURRENT_SOURCE_DIR}/cmake/CMakeUninstall.cmake.in"
               "${CMAKE_CURRENT_BINARY_DIR}/CMakeUninstall.cmake"
               IMMEDIATE @ONLY)
add_custom_target(uninstall COMMAND ${CMAKE_COMMAND} -P ${CMAKE_CURRENT_BINARY_DIR}/CMakeUninstall.cmake)
set_target_properties(uninstall PROPERTIES FOLDER ${CMAKE_TARGETS_ROOT_FOLDER})

