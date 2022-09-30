include(GNUInstallDirs)

set(CMAKE_INSTALL_INCLUDEDIR ${CMAKE_INSTALL_INCLUDEDIR}/pixsfm/third-party)
set(CMAKE_INSTALL_DATAROOTDIR ${CMAKE_INSTALL_DATAROOTDIR}/pixsfm/third-party)

# Add current path to include --> required for HighFive to find half.hpp
set(CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH};${CMAKE_CURRENT_SOURCE_DIR})
set(HIGHFIVE_USE_HALF_FLOAT ON)

# Avoid building docs and examples
set(HIGHFIVE_BUILD_DOCS OFF)
set(HIGHFIVE_EXAMPLES OFF)

add_subdirectory(HighFive)
add_subdirectory(pybind11)