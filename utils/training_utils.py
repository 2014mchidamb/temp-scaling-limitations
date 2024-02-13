import numpy as np
import torch

from models.temperature_scaled_model import TemperatureScaledModel
from netcal.metrics import ACE, ECE, MCE
from utils.mixup_utils import GeneralMixupLoss


def reset_weights(m):
    reset_parameters = getattr(m, "reset_parameters", None)
    if callable(reset_parameters):
        m.reset_parameters()


def get_grad_norm(model):
    grad_norm = 0
    for p in model.parameters():
        grad_norm += p.grad.data.norm(2).item() ** 2
    return grad_norm**0.5


def get_model_param_tensor(model):
    flattened_params = []
    for param_tensor in model.parameters():
        flattened_params.append(torch.flatten(param_tensor))
    return torch.cat(flattened_params)


def get_model_evaluations(model, data_loader, device="cpu"):
    model.eval()
    softmax = torch.nn.Softmax(dim=1)
    output = None
    with torch.no_grad():
        for data, _ in data_loader:
            data = data.to(device)
            output = softmax(model(data))
    return output


def train(
    model,
    train_loader,
    loss_fn,
    optimizer,
    epoch,
    batch_size,
    out_file,
    log_epoch_stats=False,
    device="cpu",
):
    model.train()
    use_mixup = isinstance(loss_fn, GeneralMixupLoss)  # Check if we are training with Mixup.
    avg_batch_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        if use_mixup:
            loss = loss_fn(model, data, target)
        else:
            output = model(data)
            loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()
        avg_batch_loss += loss.item() / len(train_loader)

    if log_epoch_stats:
        print(f"[Epoch {epoch}] Average Batch Loss: {avg_batch_loss}", file=out_file)


def test(model, test_loader, loss_fn, out_file, device="cpu"):
    model.eval()
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(
                dim=1, keepdim=True
            )  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            
    return 100 * (1 - (correct / len(test_loader.dataset)))


def compute_nll(model, test_loader, device="cpu"):
    criterion = torch.nn.CrossEntropyLoss(reduction="sum")
    avg_loss = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            avg_loss += criterion(output, target).item() / len(test_loader.dataset)
    
    return avg_loss


def get_confidences_and_labels(model, test_loader, device="cpu"):
    logits, softmaxes, labels = [], [], []
    with torch.no_grad():
        for data, target in test_loader:
            out = model(data.to(device))
            logits.append(out)
            softmaxes.append(torch.nn.functional.softmax(out, dim=1))
            labels.append(target)
    logits = torch.cat(logits)
    softmaxes = torch.cat(softmaxes)
    labels = torch.cat(labels)
    return logits.cpu().numpy(), softmaxes.cpu().numpy(), labels.cpu().numpy()


def get_confidence_metrics(confidences, labels, n_bins=15):
    ace, ece, mce = ACE(n_bins), ECE(n_bins), MCE(n_bins)
    return [ace.measure(confidences, labels), ece.measure(confidences, labels), mce.measure(confidences, labels)]


def full_train_test_loop(
    model,
    test_loader,
    test_loss_fn,
    train_loader,
    train_loss_fn,
    cal_loader,
    optimizer,
    num_epochs,
    batch_size,
    model_name,
    out_file,
    num_runs=10,
    log_epoch_stats=False,
    n_bins=15,
    device="cpu",
):
    cal_metrics, ts_cal_metrics, nll, ts_nll = [], [], [], []
    print("{} model results: ".format(model_name), file=out_file)
    for i in range(num_runs):
        model.apply(reset_weights)
        for j in range(1, num_epochs + 1):
            train(
                model,
                train_loader,
                train_loss_fn,
                optimizer,
                j,
                batch_size,
                out_file,
                log_epoch_stats,
                device,
            )
        # First compute non-temp-scaled outputs and metrics.
        _, confidences, labels = get_confidences_and_labels(model, test_loader, device)
        cal_metrics.append(get_confidence_metrics(confidences, labels, n_bins))
        nll.append(compute_nll(model, test_loader, device))

        # Perform temperature scaling.
        ts_model = TemperatureScaledModel(model, device=device)
        ts_model.fit(cal_loader)

        # Compute temp-scaled outputs and metrics.
        _, ts_confidences, _ = get_confidences_and_labels(ts_model, test_loader, device)
        ts_cal_metrics.append(get_confidence_metrics(ts_confidences, labels, n_bins))
        ts_nll.append(compute_nll(ts_model, test_loader, device))

    # For analyzing training-time behavior.
    train_logits, _, train_labels = get_confidences_and_labels(model, train_loader, device)

    cal_metrics = np.array(cal_metrics)
    nll = np.array(nll)
    ts_cal_metrics = np.array(ts_cal_metrics)
    ts_nll = np.array(ts_nll)

    # Train and test error.
    train_error = test(model, train_loader, test_loss_fn, out_file, device)
    test_error = test(model, test_loader, test_loss_fn, out_file, device)
    print(f"Last Train Error: {train_error:.4f}", file=out_file)
    print(f"Last Test Error: {test_error:.4f}", file=out_file)
    print("-------------------------------------------------\n", file=out_file)

    # Calibration errors.
    print(f"Average ACE: {cal_metrics[:, 0].mean():.4f}\t 1 std: {cal_metrics[:, 0].std():.4f}", file=out_file)
    print(f"Average ECE: {cal_metrics[:, 1].mean():.4f}\t 1 std: {cal_metrics[:, 1].std():.4f}", file=out_file)
    print(f"Average MCE: {cal_metrics[:, 2].mean():.4f}\t 1 std: {cal_metrics[:, 2].std():.4f}", file=out_file)
    print(f"Average NLL: {nll.mean():.4f}\t 1 std: {nll.std():.4f}", file=out_file)
    print("-------------------------------------------------\n", file=out_file)

    print(f"Average Post-TS ACE: {ts_cal_metrics[:, 0].mean():.4f}\t 1 std: {ts_cal_metrics[:, 0].std():.4f}", file=out_file)
    print(f"Average Post-TS ECE: {ts_cal_metrics[:, 1].mean():.4f}\t 1 std: {ts_cal_metrics[:, 1].std():.4f}", file=out_file)
    print(f"Average Post-TS MCE: {ts_cal_metrics[:, 2].mean():.4f}\t 1 std: {ts_cal_metrics[:, 2].std():.4f}", file=out_file)  
    print(f"Average Post-TS NLL: {ts_nll.mean():.4f}\t 1 std: {ts_nll.std():.4f}", file=out_file)
    print("-------------------------------------------------\n", file=out_file)

    return train_logits, train_labels, confidences, ts_confidences, labels  # Should really only be returning logits but too lazy to refactor.
