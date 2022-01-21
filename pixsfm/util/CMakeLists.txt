set(FOLDER_NAME "util")

PIXSFM_ADD_SOURCES(
    src/misc.h
    src/statistics.h
    src/types.h
    src/math.h
    src/simple_logger.h
    src/simple_logger.cc
    src/log_exceptions.h
    src/memory.h
    src/threading.h src/threading.cc
    src/stringprintf.h src/stringprintf.cc
    src/py_interrupt.h
    src/glog.cc
)

PYPIXSFM_ADD_SOURCES(bindings.cc)