<h1 align="center">
UKPGAN: A General Self-Supervised Keypoint Detector
</h1>

<p align='center'>
<img align="center" src='images/teaser.jpg' width='70%'> </img>
</p>

<div align="center">
<h3>
<a href="https://qq456cvb.github.io">Yang You</a>, Wenhai Liu, Yanjie Ze, Yong-Lu Li, Weiming Wang, Cewu Lu
<br>
<br>
CVPR 2022
<br>
<br>
<a href='https://arxiv.org/pdf/2011.11974.pdf'>
  <img src='https://img.shields.io/badge/Paper-PDF-orange?style=flat&logo=arxiv&logoColor=orange' alt='Paper PDF'>
</a>
<a href='https://qq456cvb.github.io/projects/ukpgan'>
  <img src='https://img.shields.io/badge/Project-Page-green?style=flat&logo=googlechrome&logoColor=green' alt='Project Page'>
</a>
  <!-- <a href='https://colab.research.google.com/'>
    <img src='https://colab.research.google.com/assets/colab-badge.svg' alt='Google Colab'>
  </a> -->
<br>
</h3>
</div>
 
UKPGAN is a **self-supervised** 3D keypoint detector on both rigid/non-rigid objects and real scenes. Note that our keypoint detector solely depends on local features and is both translational and rotational invariant.
  
# Contents
- [Overview](#overview)
- [Installation](#installation)
- [Train on ShapeNet Models](#train-on-shapenet-models)
- [Test on ShapeNet Models](#test-on-shapenet-models)
- [Train on SMPL Models](#train-on-smpl-models)
- [Test on SMPL Models](#test-on-smpl-models)
- [Test on Real-world Scenes](#test-on-real-world-scenes)
- [Pretrained Models](#pretrained-models)
- [Related Projects](#related-projects)
- [Citation](#citation)
# Overview
This repo is a TensorFlow implementation of our work UKPGAN. 
  
# Installation
<details>
<summary><b>Create Conda Environments</b></summary>

```
conda env create -f environment.yml
```
</details>
<details>
<summary><b>Compile smoothed density value (SDV) source files</b></summary>

First install [Pybind11](https://pybind11.readthedocs.io/en/latest/) and [PCL](https://github.com/PointCloudLibrary/pcl) C++ dependencies. Then run the following command to build the SDV feature extractor:

```
cd sdv_src
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make
cd ../..
```
</details>

If you want to visualize keypoint results, you will need to install ``open3d``.
# Train on ShapeNet Models
<details>
<summary><b>Prepare Data</b></summary>

Download ShapeNet point clouds from [KeypointNet](https://github.com/qq456cvb/KeypointNet) and unzip **pcds** folder to the root.
</details>
<details>
<summary><b>Category Configuration</b></summary>

Change the category name ``cat_name`` to what you want in ``config/config.yaml``.
</details>
<details>
<summary><b>Start Training</b></summary>

Open a separate terminal to monitor training process:

```
visdom -port 1080
```
Then run (e.g., chair):
```
python train.py cat_name=chair
```
</details>

# Test on ShapeNet Models
<details>
<summary><b>Evaluate IoU</b></summary>

Once trained, to evaluate the IoU with human annotations, first download [KeypointNet](https://github.com/qq456cvb/KeypointNet) data (you may only download the category that you wish to evaluate), then run
```
python eval_iou.py --kpnet_root /your/kpnet/root
```
</details>

<details>
<summary><b>Visualization</b></summary>

To test and visualize on ShapeNet models, run:
```
python visualize.py --type shapenet --nms --nms_radius 0.1
```
</details>

# Train on SMPL Models
<details>
<summary><b>Prepare Data</b></summary>

You should register on [SMPL](https://smpl.is.tue.mpg.de/) website and download the model. We follow [this repo](https://github.com/CalciferZh/SMPL#usage) to pre-process the model to generate ``model.pkl``. Place ``model.pkl`` into ``data/model.pkl``.
</details>

<details>
<summary><b>Start Training</b></summary>

The following command start training on SMPL models on the fly:
```
python train.py cat_name=smpl symmetry_factor=0
```
</details>

# Test on SMPL Models

<details>
<summary><b>Visualization</b></summary>

To test and visualize on SMPL models, run:
```
python visualize.py --type smpl --nms --nms_radius 0.2 --kp_num 10
```
</details>

# Test on Real-world Scenes

For this task, we use the model that is trained on a large collection of ShapeNet models (across 10+ categories), called ``universal``.
<details>
<summary><b>Prepare Data</b></summary>

You will need to download data from [3DMatch](http://3dmatch.cs.princeton.edu/#geometric-registration-benchmark). We also provide a demo scene for visualization.
</details>
<details>

<summary><b>Visualization</b></summary>


To test and visualize on 3DMatch, run:
```
python visualize.py --type 3dmatch --nms --nms_radius 0.05
```
</details>

# Pretrained Models
We provide pretrained models on [Google Drive](https://drive.google.com/drive/folders/1yaf8rzYvfz1T3Ii5oll7afdt66LX1UXy?usp=sharing).

# Related Projects
- [KeypointNet: A Large-scale 3D Keypoint Dataset Aggregated from Numerous Human Annotations](https://github.com/qq456cvb/KeypointNet)
- [TopNet: Structural Point Cloud Decoder](https://github.com/lynetcha/completion3d)
- [The Perfect Match: 3D Point Cloud Matching with Smoothed Densities](https://github.com/zgojcic/3DSmoothNet)
# Citation
If you find our algorithm useful in your research, please consider citing:
```
@inproceedings{you2022ukpgan,
  title={UKPGAN: A General Self-Supervised Keypoint Detector},
  author={You, Yang and Liu, Wenhai and Ze, Yanjie and Li, Yong-Lu and Wang, Weiming and Lu, Cewu},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2022}
}
```
