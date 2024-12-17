from typing import List, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

class LiftingLoss(nn.Module):
    def __init__(self,delta:float = 1.0,lambda1:float=0.1):
        super().__init__()
        self.delta = delta
        self.lambda1 =lambda1
    def huberloss(self,x:List[torch.Tensor]):
        loss = 0
        for _x in x:
            if _x.requires_grad:
                _tmp_zero = torch.zeros_like(_x).to(device=_x.device)
                _loss = F.smooth_l1_loss(_x,_tmp_zero)
                loss += _loss.mean()
        return loss

    def forward(self,pool_a:List[torch.Tensor], x_d:List[torch.Tensor])->torch.Tensor:
        _loss = self.huberloss(pool_a) # + self.huberloss(x_d) 
        return self.lambda1*_loss
        
    

def convert_to_one_hot(
    targets: torch.Tensor,
    num_class: int,
    label_smooth: float = 0.0,
) -> torch.Tensor:
    """
    This function converts target class indices to one-hot vectors,
    given the number of classes.

    Args:
        targets (torch.Tensor): Index labels to be converted.
        num_class (int): Total number of classes.
        label_smooth (float): Label smooth value for non-target classes. Label smooth
            is disabled by default (0).
    """
    assert (
        torch.max(targets).item() < num_class
    ), "Class Index must be less than number of classes"
    assert 0 <= label_smooth < 1.0, "Label smooth value needs to be between 0 and 1."

    non_target_value = label_smooth / num_class
    target_value = 1.0 - label_smooth + non_target_value
    one_hot_targets = torch.full(
        (targets.shape[0], num_class),
        non_target_value,
        dtype=torch.long if label_smooth == 0.0 else None,
        device=targets.device,
    )
    one_hot_targets.scatter_(1, targets.long().view(-1, 1), target_value)
    return one_hot_targets

class SoftTargetCrossEntropyLoss(nn.Module):
    """
    Copy from https://github.com/facebookresearch/pytorchvideo/blob/main/pytorchvideo/losses/soft_target_cross_entropy.py
    This allows the targets for the cross entropy loss to be multi-label.
    """

    def __init__(
        self,
        ignore_index: int = -100,
        reduction: str = "mean",
        normalize_targets: bool = True,
    ) -> None:
        """
        Args:
            ignore_index (int): sample should be ignored for loss if the class is this value.
            reduction (str): specifies reduction to apply to the output.
            normalize_targets (bool): whether the targets should be normalized to a sum of 1
                based on the total count of positive targets for a given sample.
        """
        super().__init__()
        self.normalize_targets = normalize_targets
        self.reduction = reduction
        self.ignore_index = ignore_index
        
        assert isinstance(self.normalize_targets, bool)
        if self.reduction not in ["mean", "none"]:
            raise NotImplementedError(
                'reduction type "{}" not implemented'.format(self.reduction)
            )
        self.eps = torch.finfo(torch.float32).eps

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input (torch.Tensor): the shape of the tensor is N x C, where N is the number of
                samples and C is the number of classes. The tensor is raw input without
                softmax/sigmoid.
            target (torch.Tensor): the shape of the tensor is N x C or N. If the shape is N, we
                will convert the target to one hot vectors.
        """
        # Check if targets are inputted as class integers
        if target.ndim == 1:
            assert (
                input.shape[0] == target.shape[0]
            ), "SoftTargetCrossEntropyLoss requires input and target to have same batch size!"
            target = convert_to_one_hot(target.view(-1, 1), input.shape[1])

        assert input.shape == target.shape, (
            "SoftTargetCrossEntropyLoss requires input and target to be same "
            f"shape: {input.shape} != {target.shape}"
        )

        # Samples where the targets are ignore_index do not contribute to the loss
        N, C = target.shape
        valid_mask = torch.ones((N, 1), dtype=torch.float).to(input.device)
        if 0 <= self.ignore_index <= C - 1:
            drop_idx = target[:, self.ignore_index] > 0
            valid_mask[drop_idx] = 0

        valid_targets = target.float() * valid_mask
        if self.normalize_targets:
            valid_targets /= self.eps + valid_targets.sum(dim=1, keepdim=True)
        per_sample_per_target_loss = -valid_targets * F.log_softmax(input, -1)

        per_sample_loss = torch.sum(per_sample_per_target_loss, -1)
        # Perform reduction
        if self.reduction == "mean":
            # Normalize based on the number of samples with > 0 non-ignored targets
            loss = per_sample_loss.sum() / torch.sum(
                (torch.sum(valid_mask, -1) > 0)
            ).clamp(min=1)
        elif self.reduction == "none":
            loss = per_sample_loss

        return loss
    
class FocalLoss(nn.Module):
    def __init__(self, gamma=2, size_average=True, alpha:list=None):
        super().__init__()
        print('Focal Loss need former network output to be not activated by sigmoid or softmax.')
        self.size_average = size_average
        if alpha:
            self.alpha = torch.Tensor(alpha)
            self.use_alpha = True
        else:
            self.use_alpha = False

        self.gamma = gamma

    def forward(self, pred, target):
        if self.use_alpha:
            alpha = self.alpha[target]
        else:
            alpha = 1.0
        log_softmax = torch.log_softmax(pred, dim=1) # log_softmax calculate p_t
        logpt = torch.gather(log_softmax, dim=1, index=target.view(-1, 1))
        logpt = logpt.view(-1)
        ce_loss = -logpt
        pt = torch.exp(logpt)
        focal_loss = alpha * (1 - pt) ** self.gamma * ce_loss
        if self.size_average:
            return torch.mean(focal_loss)
        else:
            return torch.sum(focal_loss)
        
class DecoupleLoss(nn.Module):
    """
        Decouple Neutral Expression
    """
    def __init__(self,neutral_id:int,neg_emo:bool=False):
        super().__init__()
        from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss, L1Loss
        self.neutral_index = neutral_id
        self.emotion_loss = CrossEntropyLoss() # SoftTargetCrossEntropyLoss(ignore_index=neutral_id) # CrossEntropyLoss(ignore_index=neutral_id)
        self.neutral_loss = BCEWithLogitsLoss()
        self.neg_emotion = True # DEBUG neg_emo
        if self.neg_emotion:
            self.neg_emotion_loss = L1Loss()
        

    def forward(self, pred_neutral, pred_emotion, target):
        target_neutral = torch.where(target==self.neutral_index,1,0)
        target_neutral = convert_to_one_hot(target_neutral,num_class=2).float()
        loss_neutral = self.neutral_loss(pred_neutral,target_neutral)
        
        if torch.any(target!=self.neutral_index):
            tmp_target = target[target!=self.neutral_index]
            _target = torch.where(tmp_target > 3,tmp_target-1,tmp_target)
            _pred_emotion = pred_emotion[target!=self.neutral_index]
            loss_emotion = self.emotion_loss(_pred_emotion, _target)
            if self.neg_emotion:
                _pred_neutral = pred_emotion[target==self.neutral_index]
                neg_target = torch.zeros_like(_pred_neutral).to(target.device)
                loss_neg_emotion = self.neg_emotion_loss(_pred_neutral, neg_target)
                loss_emotion += loss_neg_emotion
            return loss_neutral + loss_emotion
        else:
            return loss_neutral
        
class LabelSmoothingCrossEntropy(nn.Module):
    """ NLL loss with label smoothing.
    """
    def __init__(self, smoothing=0.1):
        super(LabelSmoothingCrossEntropy, self).__init__()
        assert smoothing < 1.0
        self.smoothing = smoothing
        self.confidence = 1. - smoothing

    def forward(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        logprobs = F.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()
    
class SoftTargetCrossEntropy(nn.Module):
    # from timm
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss = torch.sum(-target * F.log_softmax(x, dim=-1), dim=-1)
        return loss.mean()
    
    
class SoftTargetCrossEntropy_andReg(nn.Module):
    # from timm
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss = torch.sum(-target * F.log_softmax(x, dim=-1), dim=-1).mean()
        x = F.sigmoid(x)
        loss += F.smooth_l1_loss(x,target,beta=0.1)
        return loss
        