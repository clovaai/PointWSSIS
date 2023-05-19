"""
PointWSSIS (https://arxiv.org/abs/2303.15062)
Copyright (c) 2023-present NAVER Cloud Corp.
Apache-2.0
"""

import torch.nn as nn
import torch.nn.functional as F
import torch

class CELoss(nn.Module):
    """
    pixel-wise cross-entropy loss
    """

    def __init__(
        self,
        ignore_label=255,
    ):
        super(CELoss, self).__init__()
        self.criterion = nn.CrossEntropyLoss(ignore_index=255, reduction="mean")

    def forward(self, logits, labels):
        loss = self.criterion(logits, labels)

        return loss

    
class DiceLoss(nn.Module):
    def __init__(
        self,
    ):
        super(DiceLoss, self).__init__()

    def forward(self, logits, true, eps=1e-7):
        """Computes the Sørensen–Dice loss.
        Note that PyTorch optimizers minimize a loss. In this
        case, we would like to maximize the dice loss so we
        return the negated dice loss.
        Args:
            labels: a tensor of shape [B, 1, H, W].
            logits: a tensor of shape [B, C, H, W]. Corresponds to
                the raw output or logits of the model.
            eps: added to the denominator for numerical stability.
        Returns:
            dice_loss: the Sørensen–Dice loss.
        """
        num_classes = logits.shape[1]
        if num_classes == 1:
            true = torch.where(true == 255, torch.zeros_like(true), true)
            true_1_hot = F.one_hot(true, num_classes + 1)
            true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
            true_1_hot_f = true_1_hot[:, 0:1, :, :]
            true_1_hot_s = true_1_hot[:, 1:2, :, :]
            true_1_hot = torch.cat([true_1_hot_s, true_1_hot_f], dim=1)
            pos_prob = torch.sigmoid(logits)
            neg_prob = 1 - pos_prob
            probas = torch.cat([pos_prob, neg_prob], dim=1)
        else:
            true = torch.where(true == 255, torch.zeros_like(true), true)
            true_1_hot = F.one_hot(true, num_classes)
            true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
            probas = F.softmax(logits, dim=1)
            #probas = torch.sigmoid(logits)
        true_1_hot = true_1_hot.type(logits.type())
        dims = (0,) + tuple(range(2, true.ndimension()))
        intersection = torch.sum(probas * true_1_hot, dims)
        cardinality = torch.sum(probas + true_1_hot, dims)
        dice_loss = (2.0 * intersection / (cardinality + eps)).mean()
        dice_loss = 1.0 - dice_loss

        return dice_loss
