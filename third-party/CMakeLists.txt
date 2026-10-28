include(GNUInstallDirs)

set(CMAKE_INSTALL_INCLUDEDIR ${CMAKE_INSTALL_INCLUDEDIR}/pixsfm/third-party)
set(CMAKE_INSTALL_DATAROOTDIR ${CMAKE_INSTALL_DATAROOTDIR}/pixsfm/third-party)

if(NOT HighFive_FOUND)
  message(STATUS "Using the bundled HighFive")
  # Add current path to include --> required for HighFive to find half.hpp
  set(CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH};${CMAKE_CURRENT_SOURCE_DIR})
  set(HIGHFIVE_USE_HALF_FLOAT ON)

  # Avoid building docs and examples
  set(HIGHFIVE_BUILD_DOCS OFF)
  set(HIGHFIVE_EXAMPLES OFF)
  set(HIGHFIVE_UNIT_TESTS OFF)
  # pixsfm does not use HighFive's Boost support, which requires the Boost
  # system library (removed in Boost 1.89).
  set(HIGHFIVE_USE_BOOST OFF)

  add_subdirectory(HighFive)
endif()

if(NOT pybind11_FOUND)
  message(STATUS "Using the bundled pybind11")
  add_subdirectory(pybind11)
endif()
