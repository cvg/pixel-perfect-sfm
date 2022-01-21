include(GNUInstallDirs)
set(CMAKE_INSTALL_INCLUDEDIR ${CMAKE_INSTALL_INCLUDEDIR}/pixsfm/third-party)
set(CMAKE_INSTALL_DATAROOTDIR ${CMAKE_INSTALL_DATAROOTDIR}/pixsfm/third-party)

add_subdirectory(HighFive)
add_subdirectory(pybind11)