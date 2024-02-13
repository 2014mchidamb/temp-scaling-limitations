#!/bin/python3
#SBATCH --job-name=mixup_analysis
#SBATCH -t 48:00:00
#SBATCH --mem=30G
#SBATCH --gpus-per-node=v100:1

import argparse
import copy
import os
import pickle
import random
import sys

import numpy as np
import torch
import torchvision

sys.path.append(os.getcwd())

from mlp_mixer_pytorch import MLPMixer
from netcal.presentation import ReliabilityDiagram
from pathlib import Path
from torch.utils.data import DataLoader
from utils.analysis_utils import *
from utils.data_utils import load_dataset, split_train_into_val

# Set up commandline arguments.
parser = argparse.ArgumentParser(description="Which dataset to analyze.")
parser.add_argument("--task-name", dest="task_name", default="CIFAR10", type=str)
args = parser.parse_args()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if device != "cpu":
    print("Device count: ", torch.cuda.device_count())
    print("GPU being used: {}".format(torch.cuda.get_device_name(0)))

# Need to use the same seeds as run_training.py
seed = 42
random.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)

# Set up runs path.
runs_path = f"analysis/{args.task_name}"
Path(runs_path).mkdir(parents=True, exist_ok=True)

# Load appropriate model; essentially hardcoded (sorry).
model_path = f"runs/{args.task_name}_ResNeXt50_Adam_200_epochs_1_runs_0_subsample_2_mix_0.0_noise/erm_model_gpu.p"
model = pickle.load(open(model_path, "rb"))

# Load dataset.
train_data, _, _, _ = load_dataset(
    dataset=args.task_name,
    rescale=None,
    custom_normalizer=None,
    subsample=0,
    label_noise=0,
)

# Fix a max radius of 10.
logit_gap_means, logit_gap_stds, _, _ = compute_dataset_mean_confidences(dataset=train_data, model=model, nn=10, max_radius=10, device=device)
pickle.dump(logit_gap_means, open(f"{runs_path}/logit_gap_means.p", "wb"))
pickle.dump(logit_gap_stds, open(f"{runs_path}/logit_gap_stds.p", "wb"))




