# Official-FedGF

This repository contains the official implementation of

> Taehwan Lee and Sung Whan Yoon, [Rethinking the Flat Minima Searching in Federated Learning](https://openreview.net/pdf?id=6TM62kpI5c), International Conference on Machine Learning (ICML) 2024.

We refer to the [FedLab](https://github.com/SMILELab-FL/FedLab) for creating our project.

## Setup

### Environment
- install conda environment (preferred): `conda env create -f environment.yaml`

### Weights and Biases
- The code runs with WANDB. For setting up your profile, we refer you to the [quickstart documentation](https://docs.wandb.ai/quickstart). 
- WANDB MODE is set to "online" by default.
- If you set `args.wandb_project_name` as `debug`, WANDB will be 'disabled'.
- You also can switch to "offline" [Here](https://github.com/hwan-sig/Official-FedGF/blob/main/tools/main.py#L32).

## Datasets
- Overview: Image Dataset based on [CIFAR](https://www.cs.toronto.edu/~kriz/cifar.html) and [Federated Vision Datasets](https://github.com/google-research/google-research/tree/master/federated_vision_datasets)
- We store the json file that distribute the images with Dirichlet's distribution in `tools/json_data`.
- 100 users have 500 images each. Different $\alpha$ value is possible in Dirichlet's distribution.

## Running experiments
An example command can be found in `tools/experiments`
```shell
cd tools/experiments
chmod +x cifar.sh
./cifar.sh
```

## Bibtex
```
@inproceedings{
  leerethinking,
  title={Rethinking the Flat Minima Searching in Federated Learning},
  author={Lee, Taehwan and Yoon, Sung Whan},
  booktitle={Forty-first International Conference on Machine Learning}
}
```
