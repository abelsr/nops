import torch.nn as nn


class RelativeL2Loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, target):
        diff_norm = torch.linalg.vector_norm(pred - target, ord=2, dim=(2, 3, 4))
        target_norm = torch.linalg.vector_norm(target, ord=2, dim=(2, 3, 4))
        relative_error = diff_norm / (target_norm + 1e-8)
        return torch.mean(relative_error)