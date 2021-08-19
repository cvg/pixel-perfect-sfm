# Pixel-Perfect Structure-from-Motion (ICCV 2021 Oral)

We introduce a framework that improves the accuracy of Structure-from-Motion by refining keypoints, camera poses, and 3D points using the direct alignment of deep features. It is presented in our paper:
- [Pixel-Perfect Structure-from-Motion with Featuremetric Refinement](https://arxiv.org/abs/2108.08291)
- to appear at [ICCV 2021](http://iccv2021.thecvf.com/)
- Authors: [Philipp Lindenberger](https://scholar.google.com/citations?user=FMVAi2YAAAAJ&hl=en)\*, [Paul-Edouard Sarlin](psarlin.com/)\*, [Viktor Larsson](http://people.inf.ethz.ch/vlarsson/), and [Marc Pollefeys](http://people.inf.ethz.ch/pomarc/)
- Website: [psarlin.com/pixsfm](https://psarlin.com/pixsfm/)

 This repository will host the code to run and evaluate our refinement. Please subscribe to [this issue](https://github.com/cvg/pixel-perfect-sfm/issues/1) if you wish to be notified of the code release.

<p align="center">
  <a href="https://arxiv.org/abs/2108.08291"><img src="doc/pipeline.svg" width="90%"/></a>
</p>


## Abstract

Finding local features that are repeatable across multiple views is a cornerstone of sparse 3D reconstruction. The classical image matching paradigm detects keypoints per-image once and for all, which can yield poorly-localized features and propagate large errors to the final geometry. In this paper, we refine two key steps of structure-from-motion by a direct alignment of low-level image information from multiple views: we first adjust the initial keypoint locations prior to any geometric estimation, and subsequently refine points and camera poses as a post-processing. This refinement is robust to large detection noise and appearance changes, as it optimizes a _featuremetric_ error based on dense features predicted by a neural network. This significantly improves the accuracy of camera poses and scene geometry for a wide range of keypoint detectors, challenging viewing conditions, and off-the-shelf deep features. Our system easily scales to large image collections, enabling pixel-perfect crowd-sourced localization at scale. Our code will be publicly available at as an add-on to the popular SfM software COLMAP.

## BibTex Citation

Please consider citing our work if you use any code from this repo or ideas presented in the paper:

```
@inproceedings{lindenberger2021pixsfm,
  author    = {Philipp Lindenberger and
               Paul-Edouard Sarlin and
               Viktor Larsson and
               Marc Pollefeys},
  title     = {{Pixel-Perfect Structure-from-Motion with Featuremetric Refinement}},
  booktitle = {ICCV},
  year      = {2021},
}
```
