set(FOLDER_NAME "base")

PIXSFM_ADD_SOURCES(
    src/irls_optim.h
    src/interpolation.h
    src/undistortion.h
    src/projection.h
    src/graph.h 
    src/graph.cc
    src/parallel_optimizer.h
    src/grid2d.h
    src/cubic_hermite_spline_simd.h
    src/callbacks.h
)

PYPIXSFM_ADD_SOURCES(bindings.cc)

PIXSFM_ADD_TEST(undistortion_test src/undistortion_test.cc)
PIXSFM_ADD_TEST(projection_test src/projection_test.cc)
PIXSFM_ADD_TEST(interpolation_test src/interpolation_test.cc)
PIXSFM_ADD_TEST(irls_optim_test src/irls_optim_test.cc)