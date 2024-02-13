#!/bin/bash

sbatch training/run_training.py --task-name=Gaussians --model-type=$1 --epochs=200 --num-runs=5 --subsample=0 --label-noise=0.25 --mix-size=$2
sbatch training/run_training.py --task-name=Gaussians --model-type=$1 --epochs=200 --num-runs=5 --subsample=0 --label-noise=0.1 --mix-size=$2
sbatch training/run_training.py --task-name=Gaussians --model-type=$1 --epochs=200 --num-runs=5 --subsample=0 --label-noise=0.05 --mix-size=$2