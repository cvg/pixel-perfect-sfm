# Features

## C++ data structure
We provide a data structure that can handle two different inputs:

1. NumPy arrays: the data is converted without copy at initialization by storing references to numpy data;
2. Path to a HDF5 cache file: the data is loaded on demand via `FeatureView`.

The following Hierarchy is used (from fine to coarse):
1. `FeaturePatch`: Stores the underlying data and metadata.
2. `FeatureMap`: Stores a map {`keypoint_id`->`FeaturePatch`}. If the data is stored in dense mode, i.e. one large featuremap, the dense patch is stored at `features.kDenseId`. You can query the data associated to a keypoint via `FeatureMap.fpatch(keypoint_id)`, and it will either return the patch associated to this keypoint or the dense patch.
3. `FeatureSet`: A map {`imagename`->`FeatureMap`} of similar data (i.e. the same layer in the deep features).
4. `FeatureManager`: List of feature sets.

## Accessing Data
Since the low-memory cache mode loads data on demand, i.e. the `FeatureManager` has only loaded metadata, we provide a simple thread-safe interface to load features from file if necessary, the `FeatureView`.

This class has two main functionalities:
1.  On construction, it loads data (if necessary) and assures that the data remains available until it is deleted.
2.  It provides a way of mapping data, i.e. accessing a `FeatureMap` through the
    `image_id` of a COLMAP reconstruction rather than the image name.

By accessing data through the `FeatureView` it is guaranteed that the `FeaturePatch` has actually loaded the corresponding data by maintaining a reference count. It allows fine-grain access of data in a `FeatureSet`:
```python
from pixsfm.features import FeatureView
from pixsfm.extract import load_features_from_cache
fmanager = load_features_from_cache(path_to_cache_file)  # only loads metadata in H5
feature_set = fmanager.fsets[0]  # level 0, first layer

required_patches = {"image1.jpg": [1,5,6]}
fview = FeatureView(feature_set, required_patches)

# Preferred: Checks if this fview references this data point
fview.fpatch("image1.jpg", 5).data

fview.fmap("image1.jpg").patches[5].data  # OK!
fview.fmap("image1.jpg").patches[10].data  # NOT OK! (i.e. might return None)
del fview  # unloads the data

feature_set.fmaps["image1.jpg"].patches[5].data  # NOT OK!

 # Loading all patches of an image
fview = FeatureView(feature_set, ["image1.jpg"])  # pass a list, not a dict
fview.fmap("image1.jpg").patches[10].data  # OK!
```

We provide some easy overloads, e.g. to load the associated patches of all valid observations in a reconstruction:

```python
# (image_id, point2D_idx) = valid observation, i.e. triangulated
fview = FeatureView(feature_set, reconstruction)
fview.fmap(image_id).patches[point2D_idx].data  # OK!
```

Note that multiple `FeatureView`'s can be used to keep individual references to certain parts of the data. Creating a `FeatureView` is thread-safe both on the same and over multiple `FeatureSet`'s as long as they belong to the same `FeatureManager`.

## Feature-Reference
During bundle adjustment, we extract robust references, which are defined by a source-observation `(image_id, point2D_idx)` and a reference descriptor `np.ndarray[n_nodes,channels]`. See `pixsfm/features/src/references.h` for details. For each point3D which is optimized, one reference is extracted and stored in a
map {`point3D_id`->`features.Reference`}.

## Using your own features
Put your model into `pixsfm/features/models/<model_name>.py`. To be compatible with the pipeline, it has to inherit from `BaseModel`, and define the following methods:
- `_forward(self, tensor: torch.Tensor)`: extracts a list of featuremaps
from a preprocessed image.
- `_preprocess(self,image: PIL.Image)`: (optional) prepares the image for inference.

The class should also define the following variables:
-   `.output_dims: List[int]`: dimensionality (channels) of each returned featuremap
-   `.scales: List[float]`: scale of each returned featuremap (i.e. downsampling factor)

You can then directly access your model through the configuration:
```python
"dense_features": {
    "model": {
        "name": <model_name>  # =your file name in features/models/
        <your configs go here>
    }
}
```



