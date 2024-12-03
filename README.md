# CFPR

This repository represents the official implementation of the paper:

**A Coarse-to-Fine Place Recognition Approach using Attention-guided Descriptors and Overlap Estimation**


CFPR is a coase-to-fine framework for LiARD-based place recognition, which use global descriptors to propose place candidates, and use overlap prediction to determine the final match.

[[Paper]](https://arxiv.org/abs/2303.06881)

## Instructions

This code has been tested on Ubuntu 18.04 (PyTorch 1.12.1, CUDA 10.2).

Pretrained models in [here](https://drive.google.com/drive/folders/1LEGhH38SB9Y7ia_ovYtQ3NzqRMfwJCt1?usp=sharing).

### Requirments

We use *spconv-cu102=2.1.25*, other version may report error. 

The rest requirments are comman and easy to handle.

```shell
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=10.2 -c pytorch
pip install spconv-cu102==2.1.25
pip install pyymal tqdm open3d tensorboardX
```

## Extract features

```shell
python tools/utils/gen_bev_features.py
```

## Train

The training of backbone network and overlap estimation network please refs to [BEVNet](https://github.com/lilin-hitcrt/BEVNet). Here is the training of global descriptor generation network.

```shell
python train/train_netvlad.py
```

## Evalute

```shell
python evaluate/evaluate.py
```

the function **evaluate_vlad** is the evaluation of the coarse seaching method using global descriptors.

## Acknowledgement

Thanks to the source code of some great works such as [pointnetvlad](https://github.com/mikacuy/pointnetvlad), [PointNetVlad-Pytorch
](https://github.com/cattaneod/PointNetVlad-Pytorch), [OverlapTransformer](https://github.com/haomo-ai/OverlapTransformer) and so on.


## Citation

If you find this repo is helpful, please cite:


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

## Todo

- [x] upload pretrained models
- [ ] add pictures
- [ ] ...