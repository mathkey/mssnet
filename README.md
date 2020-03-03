# KSSNet: Multi-Label Classification with Label Graph Superimposing [[arXiv]](https://arxiv.org/abs/1911.09243)

## Overview

The PyTorch implementation of the [KSSNet](https://arxiv.org/abs/1911.09243).

## Prerequisites

The code is built with following libraries:

- Python 3.5 or higher
- PyTorch 0.4.1 or higher
- torchvision 0.2.0 or higher
- PIL
- torchnet
- tqdm

<!-- For video data pre-processing, you may need [ffmpeg](https://www.ffmpeg.org/). -->

## Data Preparation

We have trained on [Charades](https://deepmind.com/research/open-source/open-source-datasets/kinetics/) dataset with this code. Most of data preprocessing have been done, while the remained precedure is the data preparation:

- Download tar file from [Charades](http://ai2-website.s3.amazonaws.com/data/Charades_v1_rgb.tar) and extract files  into [data/Charades_v1/Charades_v1_rgb](data/Charades_v1/Charades_v1_rgb). The preprocessed annotations have already been contained and in this repo. If necessary, The preprocessing code will be add in this repo.

## Code

This code is based on the [ML-GCN](https://github.com/Megvii-Nanjing/ML-GCN) codebase. Thanks Megvii-Nanjing for their work.

## Testing 

The scripts will test the checkpoint provided in this repo by running:

```sh
python test_i3d_charades.py
```

The superparameters are set in the '__main__' function of test_i3d_charades.py. Sorry for the rough code!

## Training 

To train I3D with this repo:
``sh
python train_i3d_charades.py
```

The superparameters are set in the '__main__' function of test_i3d_charades.py, too.

## Reference
This project is based on [ML-GCN](https://github.com/Megvii-Nanjing/ML-GCN)