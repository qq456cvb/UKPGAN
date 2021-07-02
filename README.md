# UKPGAN: Unsupervised KeyPoint GANeration.

This repo is a TensorFlow implementation of our work UKPGAN. UKPGAN is an **unsupervised** 3D keypoint detector where keypoints are detected so that they could reconstruct the original object shape. Note that our keypoint detector solely depends on local features and is both translational and rotational invariant.

![intro](images/intro.jpg?raw=true)
## Quick Start
1. Install [Pybind11](https://pybind11.readthedocs.io/en/latest/) and [PCL](https://github.com/PointCloudLibrary/pcl) C++ dependencies.
2. Create env from environment.yml.
```
conda env create -f environment.yml
```
3. Download ShapeNet point clouds from [KeypointNet](https://github.com/qq456cvb/KeypointNet) and unzip **pcds** folder to the root.

4. Compile smoothed density value (SDV) source files.
```
cd sdv_src
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make
cd ../..
```
5. Open a separate terminal to monitor training process
```
visdom -port 1080
```
6. Run 
```
python train.py
```

## Evaluation
Once trained, to evaluate the IoU with human annotations, first download [KeypointNet](https://github.com/qq456cvb/KeypointNet) data (you may only download the category that you wish to evaluate), then run
```
python eval_iou.py
```
and modify ``kpnet_root`` and ``cat_name`` variables when necessary.

## Related Projects
- [KeypointNet: A Large-scale 3D Keypoint Dataset Aggregated from Numerous Human Annotations](https://github.com/qq456cvb/KeypointNet)
- [TopNet: Structural Point Cloud Decoder](https://github.com/lynetcha/completion3d)
- [The Perfect Match: 3D Point Cloud Matching with Smoothed Densities](https://github.com/zgojcic/3DSmoothNet)
## Citation
If you find our algorithm useful in your research, please consider citing:
```
@article{you2020ukpgan,
  title={UKPGAN: Unsupervised KeyPoint GANeration},
  author={You, Yang and Liu, Wenhai and Li, Yong-Lu and Wang, Weiming and Lu, Cewu},
  journal={arXiv preprint arXiv:2011.11974},
  year={2020}
}
```