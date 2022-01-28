## Q: My code crashed or my Jupyter notebook does not respond.

If you are running a Jupyter notebook, you might need to check the terminal where you spawned the notebook to see the stack trace and the error message.

### COLMAP check failed

If the error message looks like
```
Check failed: some_condition
*** Check failure stack trace: ***
...
Aborted (core dumped)
```
then this is likely an exception thrown by COLMAP. COLMAP and Ceres use GLOG and throw SIGTERM whenever an error occurs, thereby killing the program. Have a close look at the the condition of the failed check and try to figure out what happened.

### Unsupported number of channels

If the exception mentions `Unknown Dimensions (CHANNELS, N_NODES)`, then you are likely trying to use a different number of channels or interpolation nodes as the code was compiled for. The code is heavily templated, and requires explicit template instantiation. Therefore, certain number of channels and number of interpolation nodes are (by default) not supported to limit build time. If you want to use different channels and interpolation grids, we provide simple macros in the C++ solvers, usually on top of files such as `pixsfm/bundle_adjustment/src/feature_reference_bundle_optimizer.h`.

### Out of memory

If your program is killed during a memory-intensive operation, such as the extraction of dense features or either optimizations, you might be running out of memory. Such error message can confirm it:
```
terminate called after throwing an instance of 'std::bad_alloc'
  what():  std::bad_alloc
Abort
```
We generally warn the user when the code estimates *a priori* that more memory is required. Look for such warnings and try out the low-memory configuration.

### Issues with vectorization

`pixsfm` builds with AVX2 vectorization enabled by default. This might induce some problems on some compute architectures. In case of doubt, we can build with vectorization disabled:

```bash
AVX2_ENABLED=OFF pip install -e .
```

We are aware of this problem and are actively investigating it. Please open an issue and tell us about your build environment.

## Q: How to use pixsfm as a C++ library?

We use CMake to handle the compilation. We can build and install the library as follow:
```
mkdir build && cd build
cmake ..
make -j
sudo make install
```

It is also possible to install it via `pip`:
```
sudo -s  # if root rights are required in the install path
pip install -e . --global-option="build_ext" --global-option="--install-prefix=/usr/local/"
```

To link against the library, we show an example in `examples/cmake_example/CMakeLists.txt`.

## Q: Can I install pycolmap from PyPI?

We currently require that pycolmap is compiled from source and therefore cannot simply be installed with `pip install pycolmap`. Be aware that hloc, if installed first, will pull pycolmap from PyPI. We are looking for solutions to package binaries for pycolmap and pixsfm - any help is welcome!

## Q: How can I analyze intermediate results during an optimization?

To inspect intermediate results during a ceres optimization (BA, KA), you can register an `pyceres.IterationCallback` to the optimizer. Here is a small example that recomputes the reprojection error in every iteration during Bundle Adjustment:

```python
from pixsfm.bundle_adjustment.main import GeometricBundleAdjuster
from pixsfm import pyceres
import pycolmap

class ReconstructionCallback(pyceres.IterationCallback):
    def __init__(self, reconstruction: pycolmap.Reconstruction):
        pyceres.IterationCallback.__init__(self)
        self.reconstruction = reconstruction
    def __call__(self, summary: pyceres.IterationSummary):
        # small hack to update all reprojection errors in reconstruction
        self.reconstruction.filter_all_points3D(float("inf"), 0.0)
        print(self.reconstruction.compute_mean_reprojection_error())
        return pyceres.CallbackReturnType.SOLVER_CONTINUE

conf = {
    "optimizer": {
        "solver": {
            # Ensure reconstruction parameters are updated in-place
            "update_state_every_iteration": True
        }
    }
}

optim = GeometricBundleAdjuster(conf)
# Register the callback
optim.callbacks.append(ReconstructionCallback(reconstruction))
# run the refinement
optim.refine(reconstruction)
```
