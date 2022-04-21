# Taken from https://github.com/pybind/pybind11/issues/3081#issuecomment-935902332
# Find informations about the current python environment.
# by melMass
#
# Finds the following:
#
# - PYTHON_EXECUTABLE
# - PYTHON_INCLUDE_DIR
# - PYTHON_LIBRARY
# - PYTHON_SITE
# - PYTHON_NUMPY_INCLUDE_DIR
#
# - PYTHONLIBS_VERSION_STRING (The full version id. ie "3.7.4")
# - PYTHON_VERSION_MAJOR
# - PYTHON_VERSION_MINOR
# - PYTHON_VERSION_PATCH
#
#

function(debug_message messages)
  # message(STATUS "")
  message(STATUS "🐍 ${messages}")
  message(STATUS "\n")
endfunction()

if (NOT DEFINED PYTHON_EXECUTABLE)
  execute_process(
    COMMAND which python
    OUTPUT_VARIABLE PYTHON_EXECUTABLE OUTPUT_STRIP_TRAILING_WHITESPACE
  )
endif()

execute_process(
  COMMAND ${PYTHON_EXECUTABLE} -c "from __future__ import print_function; from distutils.sysconfig import get_python_inc; print(get_python_inc())"
  OUTPUT_VARIABLE PYTHON_INCLUDE_DIR OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET
)

if (NOT EXISTS ${PYTHON_INCLUDE_DIR})
  message(FATAL "Python include directory not found.")
endif()

execute_process(
  COMMAND ${PYTHON_EXECUTABLE} -c "from __future__ import print_function; import os, numpy.distutils; print(os.pathsep.join(numpy.distutils.misc_util.get_numpy_include_dirs()))"
  OUTPUT_VARIABLE PYTHON_NUMPY_INCLUDE_DIR OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET
)

execute_process(
  COMMAND ${PYTHON_EXECUTABLE} -c "from __future__ import print_function; import distutils.sysconfig as sysconfig; print('-L' + sysconfig.get_config_var('LIBDIR') + '/' + sysconfig.get_config_var('LDLIBRARY'))"
  OUTPUT_VARIABLE PYTHON_LIBRARY OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET
)

execute_process(
  COMMAND ${PYTHON_EXECUTABLE} -c "from __future__ import print_function; import platform; print(platform.python_version())"
  OUTPUT_VARIABLE PYTHONLIBS_VERSION_STRING OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET
)

execute_process(
  COMMAND ${PYTHON_EXECUTABLE} -c "from __future__ import print_function; from distutils.sysconfig import get_python_lib; print(get_python_lib())"
  OUTPUT_VARIABLE PYTHON_SITE OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET
)

set(PYTHON_VIRTUAL_ENV $ENV{VIRTUAL_ENV})
string(REPLACE "." ";" _VERSION_LIST ${PYTHONLIBS_VERSION_STRING})

list(GET _VERSION_LIST 0 PYTHON_VERSION_MAJOR)
list(GET _VERSION_LIST 1 PYTHON_VERSION_MINOR)
list(GET _VERSION_LIST 2 PYTHON_VERSION_PATCH)



debug_message("Found Python ${PYTHON_VERSION_MAJOR} (${PYTHONLIBS_VERSION_STRING})")
debug_message("PYTHON_EXECUTABLE: ${PYTHON_EXECUTABLE}")
debug_message("PYTHON_INCLUDE_DIR: ${PYTHON_INCLUDE_DIR}")
debug_message("PYTHON_LIBRARY: ${PYTHON_LIBRARY}")
debug_message("PYTHON_NUMPY_INCLUDE_DIR: ${PYTHON_NUMPY_INCLUDE_DIR}")
