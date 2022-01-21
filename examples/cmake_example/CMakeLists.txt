cmake_minimum_required(VERSION 3.0)

project(CMakeExample)

find_package(PIXSFM REQUIRED)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

include_directories(${PIXSFM_INCLUDE_DIRS})
link_directories(${PIXSFM_LINK_DIRS})

add_executable(example example.cc)
target_link_libraries(example ${PIXSFM_LIBRARIES})