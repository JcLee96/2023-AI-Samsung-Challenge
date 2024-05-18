import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class CEOLoss(nn.Module):
    """
    Args:
        num_classes (int): number of classes.
    """
    def __init__(self, num_classes=21):
        super(CEOLoss, self).__init__()
        self.num_classes = num_classes
        self.level = torch.arange(self.num_classes)

    def forward(self, x, y):
        """"
        Args:
            x (tensor): Regression/ordinal output, size (B), type: float
            y (tensor): Ground truth,  size (B), type: int/long

        Returns:
            CEOLoss: Cross-Entropy Ordinal loss
        """
        levels = self.level.repeat(len(y), 1).cuda()
        logit = x.repeat(self.num_classes, 1).permute(1, 0)
        logit = torch.abs((logit - levels))
        return F.cross_entropy(-logit, y, reduction='mean')