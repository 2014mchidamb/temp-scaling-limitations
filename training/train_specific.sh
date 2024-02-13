#!/bin/bash

sbatch training/run_training.py --task-name=Gaussians --model-type=ResNeXt50 --epochs=200 --num-runs=5 --subsample=0 --label-noise=0.01 --mix-size=2
sbatch training/run_training.py --task-name=Gaussians --model-type=ResNeXt50 --epochs=200 --num-runs=5 --subsample=0 --label-noise=0.01 --mix-size=3
sbatch training/run_training.py --task-name=Gaussians --model-type=ResNeXt50 --epochs=200 --num-runs=5 --subsample=0 --label-noise=0.01 --mix-size=5