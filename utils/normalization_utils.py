import torch


def compute_data_normalizer(data, n_channels):
    """Computes channel means and stds."""
    dl = torch.utils.data.DataLoader(data, batch_size=500, shuffle=False)
    imgs = None
    for batch, _ in dl:
        if imgs is None:
            imgs = batch
        else:
            imgs = torch.cat([imgs, batch], dim=0)
    imgs = imgs.numpy()

    for i in range(n_channels):
        print(f"Mean for channel {i}: {imgs[:, i, :, :].mean()}")
        print(f"Std for channel {i}: {imgs[:, i, :, :].std()}\n")




