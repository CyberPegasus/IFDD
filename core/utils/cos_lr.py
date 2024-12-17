import math
from torch.optim import Optimizer

class lr_updater(object):
    def __init__(self, max_epoch:int, warmup_epoch:int, base_lr:float=1e-4,end_lr:float = 1e-6) -> None:
        self.max_epoch = max_epoch
        self.warmup_epoch = warmup_epoch
        self.base_lr = base_lr
        self.end_lr = end_lr
        
    def set_lr(self, optimizer:Optimizer, cur_epoch:int):
        _lr = self.get_lr_at_epoch(cur_epoch=cur_epoch)
        for param_group in optimizer.param_groups:
            param_group['lr'] = _lr
        return _lr
            
    def get_lr_at_epoch(self, cur_epoch:int, warmup_start_lr:float=1e-6):
        """
        Retrieve the learning rate of the current epoch with the option to perform
        warm up in the beginning of the training stage.
        """
        max_epoch = self.max_epoch
        warmup_epoch = self.warmup_epoch
        lr = self.lr_func_cosine(cur_epoch=cur_epoch,max_epoch=max_epoch,warmup_epoch=warmup_epoch,base_lr=self.base_lr,end_lr=self.end_lr)
        # Perform warm up.
        if cur_epoch < warmup_epoch:
            lr_start = warmup_start_lr
            lr_end = self.lr_func_cosine(cur_epoch=warmup_epoch,max_epoch=max_epoch)
            alpha = (lr_end - lr_start) / warmup_epoch
            lr = cur_epoch * alpha + lr_start
        return lr
    
    def lr_func_cosine(self, cur_epoch:int, max_epoch:int, warmup_epoch:float=1.0, base_lr:float = 1e-4, end_lr:float = 1e-6):
        """
        Retrieve the learning rate to specified values at specified epoch with the
        cosine learning rate schedule. Details can be found in:
        Ilya Loshchilov, and  Frank Hutter
        SGDR: Stochastic Gradient Descent With Warm Restarts.
        """
        offset = warmup_epoch if warmup_epoch > 0 else 0.0
        assert end_lr < base_lr
        return (
            end_lr
            + (base_lr - end_lr)
            * (
                math.cos(
                    math.pi * (cur_epoch - offset) / (max_epoch - offset)
                )
                + 1.0
            )
            * 0.5
        )