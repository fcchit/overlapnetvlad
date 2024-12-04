# A Coarse-to-Fine Place Recognition Approach using Attention-guided Descriptors and Overlap Estimation (ICRA 2024)

This repository represents the official implementation of the ICRA 2024 paper:

CFPR is a coarse-to-fine framework for LiARD-based place recognition, which utilizes global descriptors to propose place candidates and employs overlap prediction to determine the final match.
[[Paper]](https://arxiv.org/abs/2303.06881)

## Instructions

This code has been tested on the following environment:

- **Python**: 3.7.16
- **PyTorch**: 1.12.1
- **CUDA**: 10.2

Pretrained models can be found [**here**](https://drive.google.com/drive/folders/1LEFAZV1Z7rlHmfk7DGtDeX7XgDGRFUzi?usp=sharing). Please download the pretrained models and place them in the `models` folder.

We use the poses file provided by [SEMANTICKITTI](https://www.semantic-kitti.org/dataset.html#download).

## Requirements

We recommend using the specific version of `spconv` to avoid compatibility issues:

- `spconv-cu102=2.1.25`

To set up your environment, you can use the following commands:

```shell
conda create -n onv python=3.7
conda activate onv
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=10.2 -c pytorch
pip install spconv-cu102==2.1.25
pip install tqdm open3d==0.9.0 tensorboardX pyyaml scikit-learn matplotlib
```

## Docker

We provide the Docker image `fuchencan/onv` for convenience.

To run the Docker container and activate the conda environment, use the following commands:

```shell
docker pull fuchencan/onv
docker run -it fuchencan/onv /bin/bash
conda activate onv
```

## Extract Features

To extract BEV features, run the following command:


```shell
python tools/utils/gen_bev_features.py
```

## Evaluation

To evaluate the model, run:

```shell
python evaluate/evaluate.py
```

## Training

For training the backbone network and overlap estimation network, please refer to [BEVNet](https://github.com/lilin-hitcrt/BEVNet). To train the global descriptor generation network, use the following command:

```shell
python train/train.py
```

The function **evaluate_vlad** is specifically for evaluating the coarse searching method using global descriptors.

## Acknowledgements

We would like to thank the authors of the following repositories for their contributions, which have greatly aided our work:

- [PointNetVlad](https://github.com/mikacuy/pointnetvlad)
- [PointNetVlad-Pytorch](https://github.com/cattaneod/PointNetVlad-Pytorch)
- [OverlapTransformer](https://github.com/haomo-ai/OverlapTransformer)

## Citation

If you find this repository helpful, please cite our work:

```
@inproceedings{fu2024coarse,
  title={A Coarse-to-Fine Place Recognition Approach using Attention-guided Descriptors and Overlap Estimation},
  author={Fu, Chencan and Li, Lin and Mei, Jianbiao and Ma, Yukai and Peng, Linpeng and Zhao, Xiangrui and Liu, Yong},
  booktitle={2024 IEEE International Conference on Robotics and Automation (ICRA)},
  pages={8493--8499},
  year={2024},
  organization={IEEE}
}
```

## TODO

- [x] Upload pretrained models
- [x] **Update the code to ICRA version**
- [ ] ...