# On the Limitations of Temperature Scaling for Distributions with Overlaps
Code for the paper: https://arxiv.org/abs/2306.00740.

# Main Experiments
All of the main experiments in the paper can be recreated using `training/run_training.py` with the appropriate hyperparameters. 
The bash scripts under `training/` show examples of how to do so; they can be run directly (after modifying `training/run_training.py`) if
slurm is available.

# Logit Analysis
The logit analysis (Figure 3 in the paper) that motivated Definition 3.2 can be recreated using `training/run_analysis.py`.
