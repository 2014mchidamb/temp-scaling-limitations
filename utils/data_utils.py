import numpy as np
import random
import sys
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms


from torch.utils.data import TensorDataset


def construct_transform(rescale: int = None, normalizer=None):
    """Returns a preprocessing transform.

    Args:
        rescale (int, optional): If provided, rescales images to desired resolution. Defaults to None.
        custom_normalize (optional): Normalization to use for input data. Defaults to None.
    """
    if rescale is not None:
        tfs = [transforms.Resize(rescale), transforms.ToTensor(), normalizer]
    else:
        tfs = [transforms.ToTensor(), normalizer]
    return transforms.Compose(tfs)


def load_mnist(rescale: int = None, normalizer=None):
    """Loads MNIST dataset.

    Args:
        rescale (int, optional): Input to construct_transform. Defaults to None.
        normalizer (optional): Input to construct_transform. Defaults to None.
    """
    if normalizer is None:
        transform = construct_transform(
            rescale, normalizer=transforms.Normalize((0.1306606,), (0.3081078,))
        )
    else:
        transform = construct_transform(rescale, normalizer)
    return (
        datasets.MNIST("data", train=True, download=True, transform=transform),
        datasets.MNIST("data", train=False, download=True, transform=transform),
    )


def load_fmnist(rescale: int = None, normalizer=None):
    """Loads FMNIST dataset.

    Args:
        rescale (int, optional): Input to construct_transform. Defaults to None.
        normalizer (optional): Input to construct_transform. Defaults to None.
    """
    if normalizer is None:
        transform = construct_transform(
            rescale, normalizer=transforms.Normalize((0.2860402,), (0.3530239,))
        )
    else:
        transform = construct_transform(rescale, normalizer)
    return (
        datasets.FashionMNIST("data", train=True, download=True, transform=transform),
        datasets.FashionMNIST("data", train=False, download=True, transform=transform),
    )


def load_cifar10(rescale: int = None, normalizer=None):
    """Loads CIFAR-10 dataset.

    Args:
        rescale (int, optional): Input to construct_transform. Defaults to None.
        normalizer (optional): Input to construct_transform. Defaults to None.
    """
    if normalizer is None:
        transform = construct_transform(
            rescale,
            normalizer=transforms.Normalize(
                (0.4913997, 0.48215827, 0.4465312), (0.2470323, 0.2434850, 0.2615877)
            ),
        )
    else:
        transform = construct_transform(rescale, normalizer)
    return (
        datasets.CIFAR10("data", train=True, download=True, transform=transform),
        datasets.CIFAR10("data", train=False, download=True, transform=transform),
    )


def load_cifar100(rescale: int = None, normalizer=None):
    """Loads CIFAR-100 dataset.

    Args:
        rescale (int, optional): Input to construct_transform. Defaults to None.
        normalizer (optional): Input to construct_transform. Defaults to None.
    """
    if normalizer is None:
        transform = construct_transform(
            rescale,
            normalizer=transforms.Normalize(
                (0.5070746, 0.4865490, 0.4409179), (0.2673342, 0.2564385, 0.2761506)
            ),
        )
    else:
        transform = construct_transform(rescale, normalizer)
    return (
        datasets.CIFAR100("data", train=True, download=True, transform=transform),
        datasets.CIFAR100("data", train=False, download=True, transform=transform),
    )


def load_svhn(rescale: int = None, normalizer=None):
    """Loads SVHN dataset.

    Args:
        rescale (int, optional): Input to construct_transform. Defaults to None.
        normalizer (optional): Input to construct_transform. Defaults to None.
    """
    if normalizer is None:
        transform = construct_transform(
            rescale,
            normalizer=transforms.Normalize(
                (0.4376822, 0.4437693, 0.4728043), (0.1980301, 0.201016, 0.1970359)
            ),
        )
    else:
        transform = construct_transform(rescale, normalizer)
    return datasets.SVHN(
        "data", split="train", transform=transform, download=True
    ), datasets.SVHN("data", split="test", transform=transform, download=True)


def load_k_class_gaussians(
    n: int = 5000, k: int = 2, mus: torch.FloatTensor = torch.FloatTensor([0, 1]), std: int = 1, data_shape=(1, 28, 28)
):
    """Gets a dataset of k multivariate gaussians.

    Args:
        n (int, optional): Number of totaldata points (80/20 train-test split). Defaults to 5000.
        k (int, optional): Number of classes. Defaults to 2.
        mus (torch.FloatTensor, optional): Means of classes. Must be length k.
        std (int, optional): Variance scaling. Defaults to 1.
        data_shape: Shape of data. Defaults to MNIST shape.
    """
    if len(mus) != k:
        print(f"Specified {len(mus)} means but asked for {k} classes.")
        return
    
    # Need this for broadcasting purposes.
    flattened_shape = 1
    for dim in data_shape:
        flattened_shape *= dim

    labels = torch.randint(0, k, (n,))
    data = std * torch.randn(n, flattened_shape) + mus[labels].unsqueeze(dim=1)
    data = data.reshape(n, *data_shape)

    test_cutoff = n - n // 5  # 80/20 train-test split.
    train_data = TensorDataset(data[:test_cutoff], labels[:test_cutoff])
    test_data = TensorDataset(data[test_cutoff:], labels[test_cutoff:])

    return train_data, test_data


class LabelNoiseDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, n_classes, noise_prop=0) -> None:
        super().__init__()
        self.dataset = dataset
        self.new_labels = [0] * len(dataset)
        self.class_mapping = {
            i: random.choice(list(range(i)) + list(range(i + 1, n_classes)))
            for i in range(n_classes)
        }
        for i in range(len(dataset)):
            _, y = dataset[i]
            if np.random.rand() < noise_prop:
                self.new_labels[i] = self.class_mapping[y]
            else:
                self.new_labels[i] = y

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        x, _ = self.dataset[index]
        return x, self.new_labels[index]


def load_dataset(
    dataset: str,
    rescale: int = None,
    custom_normalizer=None,
    subsample: int = 0,
    label_noise: float = 0,
):
    """Loads dataset specified by provided string.

    Args:
        dataset (str): Dataset name.
        rescale (int, optional): Input to construct_transform. Defaults to None.
        custom_normalizer (optional): Input to construct_transform. Defaults to None.
        subsample (int, optional): How much to subsample data by. Defaults to 0 (no subsampling).
        label_noise (float, optional): Label noise proportion.
    """
    out_dim = 10
    n_channels = 3  # Number of channels in input image.
    if dataset == "MNIST":
        n_channels = 1
        train_data, test_data = load_mnist(rescale, custom_normalizer)
    elif dataset == "FMNIST":
        n_channels = 1
        train_data, test_data = load_fmnist(rescale, custom_normalizer)
    elif dataset == "CIFAR10":
        train_data, test_data = load_cifar10(rescale, custom_normalizer)
    elif dataset == "CIFAR100":
        out_dim = 100
        train_data, test_data = load_cifar100(rescale, custom_normalizer)
    elif dataset == "SVHN":
        train_data, test_data = load_svhn(rescale, custom_normalizer)
    elif dataset == "Gaussians":
        # Set up 2 class gaussians dataset, label_noise in this case is equal to the mean of the second Gaussian.
        # Unfortunately kind of hacky, but saves adding some extra arguments.
        if label_noise == 0:
            sys.exit(f"Requested Gaussians dataset with zero separation. Label noise refers to mean separation in this case.")
        mus = torch.FloatTensor([0, label_noise])
        out_dim = 2
        train_data, test_data = load_k_class_gaussians(n=5000, k=2, mus=mus, std=1, data_shape=(3, 10, 10))
    else:
        sys.exit(f"Dataset {dataset} is an invalid dataset.")

    # Subsample as necessary.
    if subsample > 0:
        train_data = torch.utils.data.Subset(
            train_data,
            np.random.choice(
                list(range(len(train_data))), size=subsample, replace=False
            ),
        )
        test_data = torch.utils.data.Subset(
            test_data,
            np.random.choice(
                list(range(len(test_data))), size=int(0.2 * subsample), replace=False
            ),
        )

    # Add label noise if necessary.
    if label_noise > 0 and dataset != "Gaussians":
        train_data = LabelNoiseDataset(
            dataset=train_data, n_classes=out_dim, noise_prop=label_noise
        )

    return train_data, test_data, n_channels, out_dim


def split_train_into_val(train_data, val_prop: float = 0.1):
    """Splits training dataset into train and val.

    Args:
        train_data: Training dataset.
        val_prop: Proportion of data to use for validation.
    """
    val_len = int(val_prop * len(train_data))
    train_subset, val_subset = torch.utils.data.random_split(
        train_data, [len(train_data) - val_len, val_len]
    )
    return train_subset, val_subset
