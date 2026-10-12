import torch
import torch.nn as nn
from torchvision.ops import sigmoid_focal_loss

# https://docs.pytorch.org/vision/stable/_modules/torchvision/ops/focal_loss.html
# https://discuss.pytorch.org/t/is-this-a-correct-implementation-for-focal-loss-in-pytorch/43327/4
class SigmoidFocalLoss(nn.Module):
    def __init__(
        self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = "mean"
    ):
        """
        A class-based wrapper for torchvision's sigmoid_focal_loss.
        Args:
            alpha (float): balancing factor between classes (0 ≤ alpha ≤ 1).
            gamma (float): focusing parameter (gamma ≥ 0).
            reduction (str): 'none' | 'mean' | 'sum'
        """
        super(SigmoidFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor):
        return sigmoid_focal_loss(
            inputs,
            targets,
            alpha=self.alpha,
            gamma=self.gamma,
            reduction=self.reduction,
        )
