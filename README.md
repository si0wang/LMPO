## Overview
Lifetime-decay Model-based Policy Optimization

## Usage

MBPO:
> CUDA_VISIBLE_DEVICES=0 python main_mbpo.py --env_name 'Humanoid-v2' --num_epoch 400 --exp_name humanoid_mbpo

LMPO
> CUDA_VISIBLE_DEVICES=0 python main_mbpo.py --env_name 'Humanoid-v2' --num_epoch 400 --exp_name humanoid_lmpo --reweight_model decay --reweight_rollout decay

## code structure
During training, 'exp' folder will created aside 'lmpo' folder.

## Dependencies
MuJoCo 1.5 & MuJoCo 2.0

## Reference
This code is based on a [previous paper in the NeurIPS reproducibility challenge](https://openreview.net/forum?id=rkezvT9f6r) that reproduces the result with a tensorflow ensemble model but shows a significant drop in performance with a pytorch ensemble model. 
This code re-implements the ensemble dynamics model with pytorch and closes the gap. 
* Official tensorflow implementation: https://github.com/JannerM/mbpo
* Code to the reproducibility challenge paper: https://github.com/jxu43/replication-mbpo
