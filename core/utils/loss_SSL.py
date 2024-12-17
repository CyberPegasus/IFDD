import torch
import torch.nn as nn
from math import log

class SimCLRLoss(nn.Module):
    def __init__(self,t=0.1) -> None:
        super().__init__()
        # about temperature: https://www.reddit.com/r/MachineLearning/comments/n1qk8w/comment/gweu1y9/?utm_source=share&utm_medium=web2x&context=3/
        self.temperature = t # Temperature
        self.eps = 1e-5
        
    def forward(self, projections, targets)->torch.Tensor:
        device = projections.device
        dot_product_tempered = torch.mm(projections, projections.T) / self.temperature
        exp_dot_tempered = (
            torch.exp(dot_product_tempered - torch.max(dot_product_tempered, dim=1, keepdim=True)[0]) + self.eps
        ) #       
        mask_similar_class = (targets.unsqueeze(1).repeat(1, targets.shape[0]) == targets).to(device) #        torch.diag  ，  batch  
        mask_anchor_out = (1 - torch.eye(exp_dot_tempered.shape[0])).to(device) #     i   j   
        mask_combined = mask_similar_class * mask_anchor_out #                   
        cardinality_per_samples = torch.sum(mask_combined, dim=1) #                 （     ）       
        
        # infoNCE
        log_prob = -torch.log(exp_dot_tempered / (torch.sum(exp_dot_tempered * mask_anchor_out, dim=1, keepdim=True)))
        SimCLR_loss_per_sample = torch.sum(log_prob * mask_combined, dim=1) / cardinality_per_samples
        SimCLR_loss_per_sample = SimCLR_loss_per_sample.mean()
        # if SimCLR_loss_per_sample.isnan().any():
        #     _x = (torch.sum(exp_dot_tempered * mask_anchor_out, dim=1, keepdim=True))
        #     print(_x)
        #     print(cardinality_per_samples)
        return SimCLR_loss_per_sample