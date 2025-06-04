# DenserRadar: A 4D millimeter-wave radar point cloud detector based on dense LiDAR point clouds

DenserRadar is a network designed to detect 4D millimeter-wave radar point clouds from raw 4D tensors, leveraging dense LiDAR point clouds. This work is published in IEEE ITSC 2024.


## Overview

This repository contains three main parts:

1. Stitched dense LiDAR point cloud ground truth;
2. Pipeline for generating dense LiDAR point cloud ground truth from raw LiDAR point clouds;
3. Network architechture and training process of DenserRadar.



## Ground Truth and Its Generation

The ground truth data is available for download, please contact the author hanzy0506@163.com for further information.

To ensure the applicability of the ground truth, we provide dense LiDAR point cloud data without projecting it into the 4D radar coordinate system. You can perform this projection by executing the final command in `run_pre_process.sh`

If you wish to generate your own ground truth, the pipeline is available in the `GT_generaiton` folder.

### Before You Start:

* Prerequisities: PyTorch 1.8+, SciPy, scikit-learn, Open3D.
* Ensure you have access to the K-Radar dataset containing raw 4D radar tensors.
* As the lidar odometry and static enhancement algorithms are proprietary, please prepare your own algorithms.
* Update the directories in ` run_pre_process.sh`

### Generating Ground Truth

To start the ground truth generation, run:

```bash
bash run_pre_process.sh
```



## DenserRadar Network

### Prerequisities

PyTorch 1.8+, EasyDict, PyYAML, SciPy, Open3D.

### Train, Validate and Transform

To train and validate the network, and transform the generated radar point clouds, execute:

```python
python train.py
```



## Citation

If you find DenserRadar useful in your research or applications, please consider giving us a star 🌟 and citing it by the following BibTeX entry.

```bash
@article{han2024denserradar,
  title={DenserRadar: A 4D millimeter-wave radar point cloud detector based on dense LiDAR point clouds},
  author={Han, Zeyu and Jiang, Junkai and Ding, Xiaokang and Meng, Qingwen and Xu, Shaobing and He, Lei and Wang, Jianqiang},
  journal={arXiv preprint arXiv:2405.05131},
  year={2024}
}
```



## Acknowledgement

We would like to acknowledge the following projects for their contributions:

* [K-Radar](https://github.com/kaist-avelab/K-Radar)
* [RPDNet](https://github.com/thucyw/RPDNet)
