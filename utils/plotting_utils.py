import matplotlib
import numpy as np
import sys

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from typing import List


def plot_multi_dataset_metrics(
    title: str,
    x_label: str,
    y_label: str,
    fname: str,
    xs: np.ndarray,
    metric_means: np.ndarray,
    metric_stds: np.ndarray,
    datasets: List[str],
):
    """Plots metrics across several datasets.

    Args:
        title (str): Plot title.
        x_label (str): X-axis label.
        y_label (str): Y-axis label.
        fname (str): File to save plot as.
        xs (np.ndarray): X-axis.
        metric_means (np.ndarray): List of metric means.
        metric_stds (np.ndarray): List of metric stds.
        datasets (List[str]): The datasets that the metrics are associated with.
    """
    if len(datasets) != len(metric_means):
        sys.exit("Length of datasets and metrics arrays must be the same.")

    # Plot parameters.
    plt.figure(figsize=(9, 7))
    plt.rc("axes", titlesize=18, labelsize=18)
    plt.rc("xtick", labelsize=15)
    plt.rc("ytick", labelsize=15)
    plt.rc("legend", fontsize=18)
    plt.rc("figure", titlesize=18)

    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)

    for i, dataset in enumerate(datasets):
        plt.plot(xs, metric_means[i], label=dataset, color=f"C{i}")
        if metric_stds is not None:
            # One std area around each curve.
            plt.fill_between(
                xs,
                metric_means[i] - metric_stds[i],
                metric_means[i] + metric_stds[i],
                facecolor=f"C{i}",
                alpha=0.2,
            )

    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(fname)
