## Adapted from COLMAP

# Determine project compiler.
if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    set(IS_MSVC TRUE)
endif()
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    set(IS_GNU TRUE)
endif()
if(CMAKE_CXX_COMPILER_ID MATCHES ".*Clang")
    set(IS_CLANG TRUE)
endif()

# Determine project architecture.
if(CMAKE_SYSTEM_PROCESSOR MATCHES "[ix].?86|amd64|AMD64")
    set(IS_X86 TRUE)
endif()

# Determine project operating system.
string(REGEX MATCH "Linux" IS_LINUX ${CMAKE_SYSTEM_NAME})
string(REGEX MATCH "DragonFly|BSD" IS_BSD ${CMAKE_SYSTEM_NAME})
string(REGEX MATCH "SunOS" IS_SOLARIS ${CMAKE_SYSTEM_NAME})
if(WIN32)
    SET(IS_WINDOWS TRUE BOOL INTERNAL)
endif()
if(APPLE)
    SET(IS_MACOS TRUE BOOL INTERNAL)
endif()

# Enable solution folders.
set_property(GLOBAL PROPERTY USE_FOLDERS ON)
set(CMAKE_TARGETS_ROOT_FOLDER "cmake")
set_property(GLOBAL PROPERTY PREDEFINED_TARGETS_FOLDER
             ${CMAKE_TARGETS_ROOT_FOLDER})
set(PIXSFM_TARGETS_ROOT_FOLDER "pixsfm_targets")
set(PIXSFM_SRC_ROOT_FOLDER "pixsfm_sources")

# This macro will search for source files in a given directory, will add them
# to a source group (folder within a project), and will then return paths to
# each of the found files. The usage of the macro is as follows:
# PIXSFM_ADD_SOURCE_DIR(
#     <source directory to search>
#     <output variable with found source files>
#     <search expressions such as *.h *.cc>)
macro(PIXSFM_ADD_SOURCE_DIR SRC_DIR SRC_VAR)
    # Create the list of expressions to be used in the search.
    set(GLOB_EXPRESSIONS "")
    foreach(ARG ${ARGN})
        list(APPEND GLOB_EXPRESSIONS ${SRC_DIR}/${ARG})
        list(APPEND GLOB_EXPRESSIONS ${SRC_DIR}/src/${ARG})
    endforeach()
    # Perform the search for the source files.
    file(GLOB ${SRC_VAR} RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
         ${GLOB_EXPRESSIONS})
    # Create the source group.
    string(REPLACE "/" "\\" GROUP_NAME ${SRC_DIR})
    source_group(${GROUP_NAME} FILES ${${SRC_VAR}})
    # Clean-up.
    unset(GLOB_EXPRESSIONS)
    unset(ARG)
    unset(GROUP_NAME)
endmacro(PIXSFM_ADD_SOURCE_DIR)

# Macro to add source files to COLMAP library.
macro(PIXSFM_ADD_SOURCES)
    set(SOURCE_FILES "")
    set(HEADER_FILES)
    foreach(SOURCE_FILE ${ARGN})
        if(SOURCE_FILE MATCHES "^/.*")
            set(ABS_FILE_PATH ${SOURCE_FILE})
        else()
            set(ABS_FILE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/${SOURCE_FILE}")
        endif()
        list(APPEND SOURCE_FILES ${ABS_FILE_PATH})
        if(${ABS_FILE_PATH} MATCHES "^.*\\.(h)$")
            set(HEADER_FILES ${HEADER_FILES} ${ABS_FILE_PATH})
        endif()
    endforeach()
    set(PIXSFM_SOURCES ${PIXSFM_SOURCES} ${SOURCE_FILES} PARENT_SCOPE)
    # file(COPY ${HEADER_FILES} DESTINATION ${TEMP_INCLUDE_DIR}/${FOLDER_NAME}/)
endmacro(PIXSFM_ADD_SOURCES)


# Macro to add source files to COLMAP library.
macro(PYPIXSFM_ADD_SOURCES)
    set(SOURCE_FILES "")
    set(HEADER_FILES)
    foreach(SOURCE_FILE ${ARGN})
        if(SOURCE_FILE MATCHES "^/.*")
            set(ABS_FILE_PATH ${SOURCE_FILE})
        else()
            set(ABS_FILE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/${SOURCE_FILE}")
        endif()
        list(APPEND SOURCE_FILES ${ABS_FILE_PATH})
        if(${ABS_FILE_PATH} MATCHES "^.*\\.(h)$")
            set(HEADER_FILES ${HEADER_FILES} ${ABS_FILE_PATH})
        endif()
    endforeach()
    set(PYPIXSFM_SOURCES ${PYPIXSFM_SOURCES} ${SOURCE_FILES} PARENT_SCOPE)
endmacro(PYPIXSFM_ADD_SOURCES)

# Replacement for the normal add_library() command. The syntax remains the same
# in that the first argument is the target name, and the following arguments
# are the source files to use when building the target.
macro(PIXSFM_ADD_LIBRARY TARGET_NAME)
    # ${ARGN} will store the list of source files passed to this function.
    add_library(${TARGET_NAME} ${ARGN})
    set_target_properties(${TARGET_NAME} PROPERTIES FOLDER
        ${PIXSFM_TARGETS_ROOT_FOLDER}/${FOLDER_NAME})
    install(TARGETS ${TARGET_NAME} DESTINATION lib/pixsfm/)
endmacro(PIXSFM_ADD_LIBRARY)

macro(PIXSFM_ADD_STATIC_LIBRARY TARGET_NAME)
    # ${ARGN} will store the list of source files passed to this function.
    add_library(${TARGET_NAME} STATIC ${ARGN})
    set_target_properties(${TARGET_NAME} PROPERTIES FOLDER
        ${PIXSFM_TARGETS_ROOT_FOLDER}/${FOLDER_NAME})
    install(TARGETS ${TARGET_NAME} DESTINATION lib/pixsfm)
endmacro(PIXSFM_ADD_STATIC_LIBRARY)

# Replacement for the normal add_executable() command. The syntax remains the
# same in that the first argument is the target name, and the following
# arguments are the source files to use when building the target.
macro(PIXSFM_ADD_EXECUTABLE TARGET_NAME)
    # ${ARGN} will store the list of source files passed to this function.
    add_executable(${TARGET_NAME} ${ARGN})
    set_target_properties(${TARGET_NAME} PROPERTIES FOLDER
        ${PIXSFM_TARGETS_ROOT_FOLDER}/${FOLDER_NAME})
    target_link_libraries(${TARGET_NAME} pixsfm)
    install(TARGETS ${TARGET_NAME} DESTINATION bin/)
endmacro(PIXSFM_ADD_EXECUTABLE)


macro(PIXSFM_ADD_PYMODULE TARGET_NAME)
    # ${ARGN} will store the list of source files passed to this function.
    pybind11_add_module(${TARGET_NAME} ${ARGN})
    # set_target_properties(${TARGET_NAME} PROPERTIES FOLDER
    #     ${PIXSFM_TARGETS_ROOT_FOLDER}/${FOLDER_NAME})
    target_link_libraries(${TARGET_NAME} PRIVATE pixsfm)
    install(TARGETS ${TARGET_NAME} DESTINATION bin/)
endmacro(PIXSFM_ADD_PYMODULE)

# Wrapper for test executables.
macro(PIXSFM_ADD_TEST TARGET_NAME)
    if(TESTS_ENABLED)
        # ${ARGN} will store the list of source files passed to this function.
        add_executable(${TARGET_NAME} ${ARGN})
        set_target_properties(${TARGET_NAME} PROPERTIES FOLDER
            ${PIXSFM_TARGETS_ROOT_FOLDER}/${FOLDER_NAME})
        target_link_libraries(${TARGET_NAME} pixsfm
                              ${PIXSFM_EXTERNAL_LIBRARIES}
                              ${Boost_UNIT_TEST_FRAMEWORK_LIBRARY})
        add_test("${FOLDER_NAME}/${TARGET_NAME}" ${TARGET_NAME})
    endif()
endmacro(PIXSFM_ADD_TEST)
