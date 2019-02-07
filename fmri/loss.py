import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_loss(logits, targets, regression=False):
    if regression:
        loss = F.mse_loss(logits.view(-1), targets, reduction='none')
    else:
        loss = F.cross_entropy(logits, targets, reduction='none')
    return loss.mean()

def compute_preds(logits):
    preds = logits.cpu().data.numpy()
    return preds.reshape(-1)

# returns only probs of the class 1
def compute_probs(logits):
    probs = F.softmax(logits, dim=-1).cpu().data.numpy()
    return probs[:, 1]