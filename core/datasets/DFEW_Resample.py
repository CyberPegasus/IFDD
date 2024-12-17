import numpy as np
import torch

from .DFEW import DFEW_CLS

class DFEW_resample(DFEW_CLS):
    def __init__(self, root_dir: str, mode: str, input_resolution: int, SSL_train: bool = False, soft_label: bool = False, split_num: int = 0, clip_duration: int = 16, clip_stride: int = 1, flip_p: float = 0.5) -> None:
        super().__init__(root_dir, mode, input_resolution, SSL_train, soft_label, split_num, clip_duration, clip_stride, flip_p)
        
    def _get_sampleWeight(self):
        _labels = np.array([i[1]-1 for i in self.anno_list])
        _cls = np.unique(_labels)
        _cls_count = np.array([len(np.where(_labels==t)[0]) for t in _cls])
        _cls_weight = 1. / _cls_count
        _sample_weight = np.array([_cls_weight[t] for t in _labels])
        _sample_weight = torch.from_numpy(_sample_weight).double()
        return _sample_weight
            
        