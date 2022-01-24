## Design Principles
The core of the code-base is implemented in C++ and we provide bindings with high granularity. The binding code is auto-imported into the specific folders. The `pixsfm` library can be linked to C++ projects but is designed to be primarily a Python package.

### Assumptions

- Keypoints are always assumed to be in COLMAP coordinates: the point with coordinate $(0, 0)$ is the top left corner of the top left pixel. When reading or writing keypoints in hloc coordinates, they should be transformed on loading/writing.

- Optimizations involving the Ceres-Solver are always performed in-place for both KA and BA, and use automatic differentiation with Jets.

- Poses are expressed in the COLMAP convention, from world to camera, as quaternion and translation vectors.

- The C++ code is heavily templated (on `dtype`, `CHANNELS`, `N_NODES` (= patch size) and `CameraModel`) and requires explicit template instantiations. For the two parameter templates `CHANNELS` and `N_NODES` we thus only support a set of common values by default to limit the build time. If other dimensions are required, we provide macros that can usually be found on top of the respective C++ header files (see `bundle_adjustment/src/feature_reference_bundle_optimizer.h` for an example.). It is necessary to rebuild the library after changing these values.

### Granularity & Configuration

The codebase can be called and configured at different levels from Python:
1. from the command line interface: you can override options with dotlists, use existing configuration files, or write your own.

2. call the individual objects for KA or BA and use the problem_setup to decide on which parameters are optimized or set constant.

3. build a custom featuremetric optimization: manually add the different residuals and parameters and setup the Ceres problem and solver - see [the documentation of PyCeres](https://github.com/Edwinem/ceres_python_bindings), our own port in `pixsfm/pyceres`, and the different residuals in `pixsfm/residuals`

### Managing Options
We use [OmegaConf](https://omegaconf.readthedocs.io/) for merging configs in Python and accept both YAML files and command line dotlists as input.

The C++ back-end code is configured via multipled C++ config-like structures, such as `InterpolationConfig`. These objects can be intialized from a Python dictionary and behave similarly to Python dataclasses (see `pixsfm/_pixsfm/src/helpers.h` --> `void make_dataclass`). This constructor recursively merges options defined in the dict to the C++-defined defaults. Note that this merging is strict, i.e. unknown options or incorrect types will raise exceptions.

When calling C++ function from Python, any argument that expects a C++ config structure can accept either an instance of the structure or a Python dictionary. In the latter case, casting-by-merging, as described above, is performed automatically.

## Sparse vs. Dense Mode
Since high-dimensional deep features require a significant amount of memory, we provide a "sparse" mode which extracts a local patch around each given keypoint. Such a local feature patch is a tensor of dimension `[patch_size,patch_size,C]` and is extracted per keypoint per image. In addition to the data, it requires the following metadata:
- `corner`: The indices of the lower left pixel in the original featuremap
- `scale`: The scale of the featuremap w.r.t. the image size

Sparse mode generally incurs no accuracy loss if the patch size is larger than 10 pixels. See [Features](./features.md) for a detailed description of the data structure.

## Extracting Features
For KA, we only extract features which are required in the matches graph, i.e. all nodes registered (a node is one keypoint) will be processed, and unused keypoints will be left-out (see sparse-vs-dense mode in the features documentation).

For BA, we need features for each triangulated observation. Note that this is guaranteed to be a subset of the features in KA, thus features are NOT required to be extracted again after KA.

For information on available options have a look into `pixsfm/features/feature_extractor.py`. The loading functions from matches graph or reconstruction can be found in `pixsfm/extract.py`.

The system supports multi-level features, e.g. if the selected model outputs two featuremaps per input image, all first and all second featuremaps are stored in separate `FeatureSet`'s. By default, every optimization starts with the last feature set in the list, then repeats optimizations with all other feature sets in reverse order. This behaviour can be edited by overriding the `level_indices` option (with an index list, e.g. `[0,1]` starts with the feature set at index `0` and then repeats with the second.).

Features can be extracted as `half` (16-bit), `float` (32-bit) or `double` (64-bit), and all subsequent optimizations also support these datatypes. The preferred datatype is `half`, since it requires the least amount of memory and is sufficiently accurate in most cases.

If L2 normalization should be used during interpolation (`interpolation.l2_normalize`), we assume the data stored in feature_maps to be already l2_normalized (can be enforced by `dense_features.l2_normalize=True`).

## High Level Solvers
In `pixsfm/keypoint_adjustment/main.py` and `pixsfm/bundle_adjustment/main.py` we provide high-level solver classes that perform all major steps of KA/BA in python. The C++ solver's exposed through pybind operate on a lower level.

## Parallelization
Parallelization in C++ requires releasing the GIL. Since operating with pybind objects in C++ runs Python code (e.g. the buffer protocol), which requires the GIL, we avoid them in most parts of the C++ code.

KA and reference or costmap extraction support parallel optimization of independent subproblems. To configure what is optimized together, `problem_labels: List[int]` is used. A problem, i.e. all elements with the same problem label, are optimized jointly by a single thread. Elements where `problem_labels[i] == -1` are not added to the problem.

See `base/src/parallel_optimizer.h` for details.

## Interpolation
By default we use "BICUBIC" interpolation with SIMD instructions for increased performance (see the benchmark in `examples/benchmark`).

The variable `interpolation.nodes` takes a list of `[x,y]` offsets (in featuremap-coords, so after resizing) which define a patch (a pixel in a patch is called a interpolation node).

For example, a 2x2 patch could be defined in the yaml file as:
```YAML
interpolation.nodes: [[-0.5, -0.5], [0.5, -0.5], [-0.5, 0.5], [0.5, 0.5]]
```
The output of the interpolation is of shape `np.ndarray[nodes.size(), patch.channels]`.

The typical pipeline of interpolation is:

1. Convert from COLMAP-coordinates to featuremap-coordinates (using `FeaturePatch.scale` and `FeaturePatch.corner`)
2. Extract a descriptor for each node in `interpolation.nodes`
3. Optional: Apply L2-normalization
4. Optional: Apply NCC per channel

The whole pipeline is autodiff-compatible. Note that using NCC-normalization with only a single interpolation node will result in all-1-descriptors and should be avoided (no gradients). For photometric BA, L2-normalization should be deactivated. For more details we refer to `base/src/interpolation.h` and `features/src/patch_interpolator.h`.

## Keypoint Adjustment
The input format is a `Map_NameKeypoints` which is similar to a Python `Dict[str,np.ndarray[2,N]]`, where the keys are image names. To configure the optimization problem, i.e. setting individual keypoints constant, use the `KeypointAdjustmentSetup`.

```python
from pixsfm.keypoint_adjustment import KeypointAdjuster, KeypointAdjustmentSetup

# Parameters:
# - graph: ka.Graph; matches-graph
# - keypoints: MapNamePyArrayDouble; keypoints dictionary
# - feature_manager: dense features

config = {"strategy": "featuremetric"}
solver = KeypointAdjuster.create(config)  # return specific solver

# By default all keypoints are variable
setup = KeypointAdjustmentSetup()
# Setting all keypoints of an image constant
setup.set_image_constant(graph.image_name_to_id["image1.jpg"])
# Setting individual keypoints constant
setup.set_keypoint_constant(graph.image_name_to_id["image2.jpg"], 5)

solver.refine_multilevel(
    keypoints, feature_manager, graph, problem_setup=setup)
```

The  available optimization strategies are:
- `featuremetric` (default): minimize the featuremetric error along all (intra) graph edges.
- `topological_reference`: minimize the featuremetric error between the root node and all nodes within the track towards root. Much faster but slightly less accurate.

For parallelism, we assign a problem label for each node in the graph (see function `find_problem_labels` in `pixsfm/keypoint_adjustment/main.py`). Options related to parallelism / scheduling:
-   `optimizer.num_threads`: number of outer threads (-1 = all threads)
-   `optimizer.solver.num_threads`: Number of inner threads used in solving each independent KA problem.
-   `max_kps_per_problem`: Defines how the problem is split. Lower values result in better balancing and reduced memory consumption, but larger threading overhead.
-   `split_in_subproblems`: Whether the problem should be split in independent subproblems or optimized globally.

## Bundle Adjustment
The input format is a `pycolmap.Reconstruction` object, a wrapper of the `Reconstruction` class of COLMAP. See the documentation in the respective repo for details. The Bundle Adjuster is designed to be configured similarly to COLMAP, see `pixsfm/bundle_adjustment/src/bundle_optimizer_test.cc`. To configure the optimization problem, i.e. adding images/points, setting images/cameras/points constant, use the `BundleAdjustmentSetup`.

```python
from pixsfm.bundle_adjustment import BundleAdjuster, BundleAdjustmentSetup

# Parameters:
# - reconstruction: pycolmap.Reconstruction; COLMAP reconstruction
# - feature_manager: dense features

config = {"strategy": "feature_reference"}
solver = BundleAdjuster.create(config)  # return specific solver

# By default all keypoints are variable
setup = BundleAdjustmentSetup()
# Mandatory: Add the image_ids which we want to optimize
reg_image_ids = reconstruction.reg_image_ids()
setup.add_images(set(reg_image_ids))
# Mandatory: Fix 7 DoF
# Set both rotation and translation vector of an image constant
setup.set_constant_pose(reg_image_ids[0])
# Set "x" entry of tvec = [x y z] constant
setup.set_constant_tvec(reg_image_ids[1], [0])

# optional: configure problem
setup.set_camera_constant(reconstruction.images[reg_image_ids[0]].camera_id)

solver.refine_multilevel(
    reconstruction, feature_manager, problem_setup=setup)
```

For common use cases, we provide the following options to configure the problem:
-   `optimizer.refine_extrinsics` (poses, rotation+translation)
-   `optimizer.refine_focal_length`
-   `optimizer.refine_principal_point`
-   `optimizer.refine_extra_params` (= other camera parameters)

If these options are set to false, they are applied regardless of the config in `setup`.

The  available optimization strategies are:
-   `feature_reference` (default): minimize the featuremetric error between the robust reference in each track and its reprojected locations.
-   `patch_warp`: minimize the featuremetric error between a patch around the robust reference in each track and patches around its reprojected locations, using fronto-parallel warping.
-   `costmaps`: similar to `feature_reference`, but pre-compute the featuremetric error to the reference to greatly reduce memory consumption and the jacobian size. This is an approximate solver. Only works in sparse-mode and does not work with >1 interpolation nodes.
-   `geometric`: classic bundle adjustment minimizing the geometric reprojection error.

Right now only the following camera models are supported for patch warping: `SIMPLE_PINHOLE`, `PINHOLE`, `SIMPLE_RADIAL`, `RADIAL`, `OPENCV`. Note that all cameras in the reconstruction are required to have the same camera model (different camera param values are allowed). Mixed camera models will be supported in a future release.

For parallelism in reference/costmap extraction, we assign a problem label for each point3D-id (see function `find_problem_labels` in `bundle_adjustment/main.py`). Options related to parallelism / scheduling:
-   `references.num_threads` / `costmaps.num_threads`: number of outer threads in reference/costmap extraction (-1 = all threads).
-   `optimizer.solver.num_threads`: Number of threads used in solving the global BA problem.
-   `max_tracks_per_problem`: Defines how the problem is split. Lower values result in better balancing and reduced memory consumption, but larger threading overhead.

## Localization
This project provides two refinements for camera localization: Query Keypoint Adjustment refines the keypoints before PnP and Query Bundle Adjustment refines the pose after PnP. For the refinement, we rely on reference feature descriptors of the 3D model, which can be obtained using the `FeatureExtractor` from bundle adjustment.

Localization can be set up like this:

```python
from pixsfm.localization import QueryLocalizer

# localizer computes references for the entire reconstruction at init
# Parameters:
# - config: Union[dict,DictConfig]: config of localization (use {} for default)
# - reconstruction: pycolmap.Reconstruction; reference COLMAP reconstruction
# - feature_manager: features of reference reconstruction
localizer = QueryLocalizer(
    reconstruction,
    conf=config,
    dense_features=feature_manager
)

# Parameters:
# - keypoints: np.ndarray[N,2]: keypoints
# - pnp_point2D_idxs: List[int]: keypoint ids of correspondences
# - pnp_point3D_ids: List[int]: ids of corresponding 3D point in the reconstruction
# - query_camera: pycolmap.Camera: camera of query image
# - image_path: Path: path to query image
pose_dict = localizer.localize(
    keypoints,
    pnp_point2D_idxs,
    pnp_point3D_ids,
    query_camera,
    image_path=image_path,
)

qvec, tvec, success = pose_dict["qvec"], pose_dict["tvec"], pose_dict["success"]
```