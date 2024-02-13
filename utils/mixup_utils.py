import numpy as np
import torch


class GeneralMixupLoss(torch.nn.Module):

    def __init__(self, criterion, alpha=1, mix_size=2):
        super().__init__()
        self.criterion = criterion  # Reduce should be set to none
        self.alpha = alpha
        self.mix_size = mix_size

    def forward(self, model, data, target):
        weights = np.random.dirichlet([self.alpha] * self.mix_size)
        mix_data, all_targets = weights[0] * data, [target]
        for i in range(1, self.mix_size):
            shuffle = torch.randperm(len(target))
            mix_data += weights[i] * data[shuffle]
            all_targets.append(target[shuffle].clone())

        out = model(mix_data)
        total_loss = 0
        for i, weight in enumerate(weights):
            total_loss += weight * self.criterion(out, all_targets[i]).mean()
        return total_loss