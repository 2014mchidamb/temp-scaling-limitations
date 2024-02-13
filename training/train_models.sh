#!/bin/bash

sbatch training/run_training.py --task-name=CIFAR10 --model-type=$1 --epochs=200 --num-runs=1 --subsample=0 --label-noise=$2 --mix-size=$3 --save-model
sbatch training/run_training.py --task-name=CIFAR100 --model-type=$1 --epochs=200 --num-runs=1 --subsample=0 --label-noise=$2 --mix-size=$3 --save-model
sbatch training/run_training.py --task-name=SVHN --model-type=$1 --epochs=200 --num-runs=1 --subsample=0 --label-noise=$2 --mix-size=$3 --save-model