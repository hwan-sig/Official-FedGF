# Official-FedGF

This repository contains the official implementation of

> Taehwan Lee and Sung Whan Yoon, [Rethinking the Flat Minima Searching in Federated Learning](https://openreview.net/pdf?id=6TM62kpI5c), International Conference on Machine Learning (ICML) 2024.

## Setup

### Environment
- install conda environment (prederred): `conda env create -f environment.yaml`

### Weights and Biases
- The code runs with WANDB. For setting up your profile, we refer you to the [quickstart documentation](https://docs.wandb.ai/quickstart). 
- WANDB MODE is set to "online" by default.
- If you set `args.wandb_project_name` as `debug`, WANDB will be 'disabled'.
- You also can switch to "offline" [Here](https://github.com/hwan-sig/Official-FedGF/master/tools/main.py#L32).

## Datasets
- Overview: Image Dataset based on [CIFAR](https://www.cs.toronto.edu/~kriz/cifar.html) and [Federated Vision Datasets](https://github.com/google-research/google-research/tree/master/federated_vision_datasets)
- We store the json file that distribute the images with Dirichlet's distribution in `tools/json_data`.
- 100 users have 500 images each. Different $\alpha$ value is possible in Dirichlet's distribution.

## Running expemiments
Example command can be found in `tools/run.sh`
```shell
cd tools/
chmod +x run.sh
./run.sh
```

## Bibtex
```
@inproceedings{
lee2024rethinking,
title={Rethinking the Flat Minima Searching in Federated Learning},
author={Taehwan Lee and Sung Whan Yoon},
booktitle={Forty-first International Conference on Machine Learning},
year={2024},
url={https://openreview.net/forum?id=6TM62kpI5c}
}
```
