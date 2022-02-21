import torch
import torch.nn as nn
import torch.nn.functional as F
from .lovasz_losses import lovasz_softmax
from ..registry import LOSSES

@LOSSES.register_module
class SegLoss(nn.Module):
  '''Semantic Segmentation loss for an output tensor
    Arguments:
      output (batch x dim x h x w)
      labels (batch x h x w)
  '''
  def __init__(self, ignore=0):
      super(SegLoss, self).__init__()
      self.ignore = ignore
      self.loss_func = nn.CrossEntropyLoss(ignore_index=ignore)
  
  def forward(self, outputs, labels):
      loss = lovasz_softmax(F.softmax(outputs, dim=1), labels, ignore=self.ignore) + self.loss_func(
        outputs, labels)
      return loss

