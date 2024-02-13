#!/bin/python3
#SBATCH --job-name=mixup_plots
#SBATCH -t 48:00:00
#SBATCH --mem=30G
#SBATCH --gpus-per-node=v100:1

import argparse
import os
import pickle
import random
import sys

import numpy as np
import torch

sys.path.append(os.getcwd())

from utils.plotting_utils import *


datasets = ["CIFAR10", "CIFAR100", "SVHN"]
runs_paths = [f"analysis/{task}" for task in datasets]
logit_gap_means = []
for runs_path in runs_paths:
    logit_gap_means.append(pickle.load(open(f"{runs_path}/logit_gap_means.p", "rb")).cpu().numpy())  # Why did I store them as torch cuda tensors?

# Same settings as run_analysis.py.
nn = 10
xs = np.linspace(0, 10, nn) 

# Baseline plots.
plot_multi_dataset_metrics(
    title="ResNeXt-50 Train Logit Gaps",
    x_label="Euclidean Distance",
    y_label="Mean Logit Gap",
    fname=f"analysis/logit_gap_plot.png",
    xs=xs,
    metric_means=logit_gap_means,
    metric_stds=None,
    datasets=datasets,
)