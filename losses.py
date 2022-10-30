import torch
from torch.nn import functional as F



EPSILON = 1e-6

class DiceLoss(torch.nn.Module):
    def __init__(self,):
        super().__init__()
    
    def forward(self, pred, mask):
        pred = pred.flatten()
        mask = mask.flatten()
        
        intersect = (mask * pred).sum()
        dice_score = 2*intersect / (pred.sum() + mask.sum() + EPSILON)
        dice_loss = 1 - dice_score
        return dice_loss

    
class DiceLossWithLogtis(torch.nn.Module):
    def __init__(self,):
        super().__init__()
    
    def forward(self, pred, mask):
        prob = F.softmax(pred, dim=1)
        true_1_hot = mask.type(prob.type())
        
        dims = (0,) + tuple(range(2, true_1_hot.ndimension()))
        intersection = torch.sum(prob * true_1_hot, dims)
        cardinality = torch.sum(prob + true_1_hot, dims)
        dice_loss = (2. * intersection / (cardinality + EPSILON)).mean()
        return (1 - dice_loss)
