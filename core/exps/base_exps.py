import ast
import pprint
from abc import ABCMeta, abstractmethod
from typing import Dict
from tabulate import tabulate
import os
import torch
from torch.nn import Module
from torch.optim.optimizer import Optimizer as Optimizer

class BaseExp(metaclass=ABCMeta):
    """Basic class for any experiment."""

    def __init__(self):
        self.seed = None

    @abstractmethod
    def get_model(self) -> Module:
        pass

    @abstractmethod
    def get_data_loader(
        self, mode:str, is_distributed: bool
    ) -> Dict[str, torch.utils.data.DataLoader]:
        pass

    @abstractmethod
    def get_optimizer(self) -> torch.optim.Optimizer:
        pass

    @abstractmethod
    def get_evaluator(self):
        pass

    # @abstractmethod
    # def eval(self, model, evaluator, weights):
    #     pass

    def __repr__(self):
        table_header = ["keys", "values"]
        exp_table = [
            (str(k), pprint.pformat(v))
            for k, v in vars(self).items()
            if not k.startswith("_")
        ]
        return tabulate(exp_table, headers=table_header, tablefmt="fancy_grid")

    def merge(self, cfg_list:dict):
        for k, v in cfg_list.items():
            # only update value with same key
            if hasattr(self, k):
                src_value = getattr(self, k)
                src_type = type(src_value)
                if src_value is not None and src_type != type(v):
                    try:
                        v = src_type(v)
                    except Exception:
                        v = ast.literal_eval(v)
                setattr(self, k, v)
                
class mainExp(BaseExp):
    def __init__(self):
        super().__init__()
        
        self.SSL = False
        
        # ---------------- model config ---------------- #
        self.num_classes = 0
        self.dropout = 0.5
        self.attention_dropout = 0.0
        self.use_LADM = False # enable by config
        self.embed_net = 'CNN' # CNN or Conv
        
        # ---------------- path config ---------------- #
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]
        self.dataset_dir = ''
        self.output_dir = "./exp_results"
        
        # ---------------- dataloader config ---------------- #
        self.batch_size = 8
        self.num_workers = 8
        self.shuffle = True
        self.pin_memory = True
        self.occupy = False
        
        # --------------- video and transform ----------------- #
        self.clip_duration = 32
        self.clip_stride = 2
        self.flip_prob = 0.5
        
        # --------------  training config --------------------- #
        self.max_epoch = 100
        self.basic_lr_per_img = 1e-4 / 16 # normal batch 16
        self.optimizer = 'AdamW'
        self.loss = 'SoftCE' # 'SoftCE'
        self.addtion_loss = ['LiftLoss']
        self.print_inter = 100
        self.val_inter = 3
        self.grad_clip = 1.0
        self.warmup_epoch = 1.0
        self.weight_decay = 5e-4
        
        # -----------------  testing config ------------------ #

        
    def get_model(self) -> Module:
        from core.model import MFViT, MFCNN, embedNet, embedNet_light, backboneNet, headNet
        if self.embed_net == 'CNN':
            embed = embedNet(input_channel=3,embed_dim=96,out_resolution=56)
        elif self.embed_net == 'Conv':
            embed = embedNet_light(input_channel=3,embed_dim=96,out_resolution=56)
        else:
            raise NotImplementedError()
        backbone = backboneNet(type='MFViT_XS',
                               attention_dropout=self.attention_dropout,use_WaveSF=self.use_LADM
                               )
        head = headNet(num_classes=self.num_classes,dropout=self.dropout,sigmoid=True)
        model = MFViT(embedNet=embed, backbone=backbone, head=head, use_WaveSF=self.use_LADM, SSL=self.SSL)
        model.embed.load_pretrain()
        model.backbone.backbone.load_pretrain()
        # model = MFCNN(num_class=self.num_classes)
        # model.backbone.load_pretrain()
        return model
    
    def get_evaluator(self):
        return super().get_evaluator()
    
    def get_data_loader(
        self, mode:str, is_distributed: bool
    ) -> Dict[str, torch.utils.data.DataLoader]:
        return super().get_data_loader()
    
    def get_optimizer(self, model:Module) -> Optimizer:
        if self.optimizer == 'AdamW':
            from torch.optim import AdamW
            _lr = self.basic_lr_per_img*self.batch_size
            # optim = AdamW(params=[
            #     {'params':model.embed.parameters()},
            #     {'params':model.backbone.parameters()},
            #     {'params':model.head.parameters()},
            #     ], lr=_lr, weight_decay=self.weight_decay)
            optim = AdamW(model.parameters(), lr=_lr, weight_decay=self.weight_decay)
        elif self.optimizer == 'SGD':
            from torch.optim import SGD
            _lr = self.basic_lr_per_img*self.batch_size
            optim = SGD(model.parameters(), lr=_lr, weight_decay=self.weight_decay)
        else:
            raise NotImplementedError()
        
        return optim
    
    def get_loss(self)->Module:
        from core.utils import SoftTargetCrossEntropyLoss
        from torch.nn import CrossEntropyLoss,BCEWithLogitsLoss
        from core.utils import FocalLoss
        
        if self.loss=='SoftCE':
            lossf = SoftTargetCrossEntropyLoss()
        elif self.loss=='CE':
            lossf = CrossEntropyLoss()
        elif self.loss=='focal':
            lossf = FocalLoss()
        else:
            raise NotImplementedError()
        
        return lossf
    
    def get_addtional_loss(self)->Module:
        from core.utils import LiftingLoss
        # outputs: (x,x_a,x_d)
        loss_f = None
        if 'LiftLoss' in self.addtion_loss:
            # Lifting Scheme
            loss_f = LiftingLoss()
        return loss_f