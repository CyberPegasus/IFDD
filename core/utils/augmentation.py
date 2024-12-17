from functools import partial
from typing import Any, Tuple
import torch
from torchvision import transforms as T

class identity(object):
    def __init__(self) -> None:
        pass
    def __call__(self, img):
        return img
    
class SSL_DataAug(object):
    def __init__(self,train_mode:bool,target_size:Tuple[int]) -> None:
        if train_mode:
            if int(torch.__version__.split('.')[0])>1:
                createRandomResizedCrop = partial(T.RandomResizedCrop,antialias=True)
            else:
                createRandomResizedCrop = partial(T.RandomResizedCrop)
            self.trans = T.Compose([
                T.RandomHorizontalFlip(0.5),
                createRandomResizedCrop(size=target_size,scale=(0.8,1.0)),
                T.RandomApply([T.ColorJitter(0.4,0.4,0.4,0.2),],p=0.8),
                T.RandomGrayscale(p=0.1),
            ])          
        else:
            # self.trans = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            self.trans = identity()
            
    def __call__(self, x:torch.Tensor) -> Any:
        # X: C,T,H,W or N,C,T,H,W
        test_flag = False
        if len(x.shape)==4:
            c,t,h,w = x.shape
            x = x.transpose(0,1) # to b,t,c,h,w
        elif len(x.shape)==5:
            test_flag = True
            n,c,t,h,w = x.shape
            x = x.transpose(1,2).view(n*t,c,h,w) # to N*T,C,H,W
        else:
            raise NotImplementedError()
        
        x = self.trans(x)
        
        if test_flag:
            x = x.view(n,t,c,h,w).transpose(1,2)
        else:
            x = x.transpose(0,1)
            
        x = torch.clamp(x,min=0.0,max=1.0)
        return x # c,t,h,w or n,c,t,h,w
        

