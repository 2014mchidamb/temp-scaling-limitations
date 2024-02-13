import numpy as np
import pickle
import torch


def point_nbhd_mean_confidence(
    model: torch.nn.Module,
    point: torch.Tensor,
    label: int,
    min_radius: int,
    max_radius: int,
    nn: int,
    n_sample: int = 500,
    use_label_argmax=False,
    device: str = "cpu",
):
    """Computes mean confidence of model's true label output in neighborhoods around a provided point.

    Args:
        model (torch.nn.Module): Model for computing predictions.
        point (torch.Tensor): Point around which to sample neighborhoods.
        label (int): True label of point - this is for assessing model confidence of correct output.
        min_radius (int): Minimum radius around point.
        max_radius (int): Maximum radius around point.
        nn (int): Number of neighborhoods to consider.
        n_sample (int): Number of points to sample in each neighborhood.
        use_label_argmax (bool): If true, ignores provided label and always considers confidence from softmax argmax.
        device (str): Device to run on.
    """
    # Need this for generating nbhds.
    flattened_shape = 1
    for dim in point.shape:
        flattened_shape *= dim

    # Linearly interpolate neighborhood radii.
    radii = np.linspace(min_radius, max_radius, nn)
    mean_logit_gaps, mean_confidences = [], []
    point = point.unsqueeze(dim=0).to(device)

    # This is for tracking logit gaps, when use_label_argmax is false.
    mask = torch.ones(model(point).detach().shape[1])
    mask[label] = 0
    mask = (mask == 1)

    # Compute confidences.
    for radius in radii:
        if radius < 1e-9:
            logits = model(point)[0].detach()
            if use_label_argmax:
                logits, _ = torch.sort(logits)
                # Gap between largest and second largest logit.
                mean_logit_gaps.append(logits[-1].item() - logits[-2].item())
                mean_confidences.append(
                    torch.nn.functional.softmax(model(point), dim=1)[0]
                    .detach()
                    .max()
                    .item()
                )
            else:
                mean_logit_gaps.append(logits[label].item() - logits[mask].max().item())
                mean_confidences.append(
                    torch.nn.functional.softmax(model(point), dim=1)[0, label]
                    .detach()
                    .item()
                )
            continue
        nbhd = torch.randn(n_sample, flattened_shape).to(device)
        nbhd = radius * (nbhd / torch.linalg.norm(nbhd, dim=1).unsqueeze(dim=1))
        nbhd = nbhd.reshape(n_sample, *point.shape[1:]) + point
        logits = model(nbhd).detach()

        if use_label_argmax:
            logits, _ = torch.sort(logits, dim=1)
            mean_logit_gaps.append((logits[:, -1] - logits[:, -2]).mean().item())
            mean_confidences.append(
                torch.nn.functional.softmax(logits, dim=1).max(dim=1)[0].mean().item()
            )
        else:
            mean_logit_gaps.append((logits[:, label] - logits[:, mask].max(dim=1)[0]).mean().item())
            mean_confidences.append(
                torch.nn.functional.softmax(logits, dim=1)[:, label].mean().item()
            )

    return torch.FloatTensor(mean_logit_gaps).to(device), torch.FloatTensor(
        mean_confidences
    ).to(device)


def compute_dataset_norm_statistics(
    dataset: torch.utils.data.Dataset, device: str = "cpu"
):
    """Computes norm statistics over a dataset.

    Args:
        dataset (torch.utils.data.Dataset): Torch dataset.
        device (str): Device to run on.
    """
    min_stat, max_stat, mean_stat = 1e6, 0, 0
    for point, _ in dataset:
        cur_norm = torch.linalg.norm(point.to(device)).item()
        min_stat = min(min_stat, cur_norm)
        max_stat = max(max_stat, cur_norm)
        mean_stat = mean_stat + cur_norm / len(dataset)

    return min_stat, max_stat, mean_stat


def compute_dataset_mean_confidences(
    dataset: torch.utils.data.Dataset,
    model: torch.nn.Module,
    nn: int,
    max_radius: int = 0,
    device: str = "cpu",
):
    """Computes mean neighborhood confidences along with standard deviations over entire dataset.

    Args:
        dataset (torch.utils.data.Dataset): Dataset to run model on.
        model (torch.nn.Module): Model for computing predictions.
        nn (int): Number of neighborhoods.
        max_radius (int): Radius to consider.
        device (str): Device to run on.
    """
    mean_logits, mean_confidences = [], []
    
    for point, label in dataset:
        logits, confidences = point_nbhd_mean_confidence(
            model,
            point,
            label,
            min_radius=0,
            max_radius=max_radius,
            nn=nn,
            use_label_argmax=False,
            device=device,
        )
        mean_logits.append(logits)
        mean_confidences.append(confidences)

    mean_logits = torch.vstack(mean_logits)
    mean_confidences = torch.vstack(mean_confidences)
    return (
        mean_logits.mean(dim=0),
        mean_logits.std(dim=0),
        mean_confidences.mean(dim=0),
        mean_confidences.std(dim=0),
    )
