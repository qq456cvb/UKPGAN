# Pytorch version of UKPGAN
UKPGAN: A General Self-Supervised Keypoint Detector (CVPR2022)

# Training
We use [pybind11](https://pybind11.readthedocs.io/en/stable/) to build SHOT descriptors and [pytorch-lightning](https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html) to train the model.

To train the model, first install PCL >= 1.8, then run
```
cd src_shot
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make
cd ../..

python train.py
```

The dataloader is identical to that of Tensorflow, except the features are now computed with SHOT.
