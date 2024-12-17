# -*- coding:utf-8 -*-

import os
from typing import Dict
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.optim.optimizer import Optimizer as Optimizer
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
from torch.nn import Module
# from torch.utils.data.distributed import DistributedSampler

from core.exps import mainExp


class Exp(mainExp):
    def __init__(self):
        super().__init__()
        # ----------------- SSL config: Not Used,Deprecated ----------------- #
        self.SSL = False
        self.SSL_Network = False
        self.backbone_type = 'MFViT_XS'
        
        # -------prompt config: Not Used,Deprecated------------ #
        self.facial_project = False # use facial landmark features
        self.use_prompt = False
        self.prompt_load_offline = False
        self.prompt_offline_save = False
        self.resolution = 224
        
        # -------IFDD: Wavelet Lifting config------- #
        self.use_stage = [2,3]
        self.use_LADM = False # enable by config
        self.WaveSF_preUpdate = False # enable by config
        self.use_ISSM = False # enable by config
        self.use_lowFreq = False # enable by config
        self.WaveNorm = False
        self.WavePool = False # qkv pool
        
        # ------------decouple loss: Not Used,Deprecated---------------#
        self.use_decouple_neutral = False  # enable by config
        self.neutral_id = 4 # id for neutral class
        
        # ---------------- model config ---------------- #
        self.num_classes = 11
        self.split_num = 0 # [1,5], but not split for SSL
        self.dropout = 0.5
        self.attention_dropout = 0.0
        self.optimize_WaveSF = False # enable by config
        # embed
        self.embed_net = 'Conv' # CNN or Conv
        # head
        self.ssl_out_dim = 96
        self.enable_cosHead = False
        
        # ---------------- path config ---------------- #
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]
        self.dataset_dir = 'datasets/MAFW'
        self.output_dir = "./exp_results"
        
        # ---------------- dataloader config ---------------- #
        self.batch_size = 8
        self.num_workers = 1
        self.shuffle = True
        self.pin_memory = True
        self.occupy = False
        # read soft label for training
        self.soft_label = False
        # augmentation
        self.strong_aug = False
        # ---- MixUp ----
        self.mixup = False
        self.label_smoothing = 0.1
        # augmentation
        self.strong_aug = False
        # resample for small-amount samples
        self.resample = False
        self.resample_initHead = False
        
        # --------------- video and transform ----------------- #
        self.clip_duration = 16
        self.clip_stride = 1
        self.flip_prob = 0.5
        self.single_val_view = True
        
        # --------------  training config --------------------- #
        self.max_epoch = 100
        self.lr = 1e-4
        self.basic_lr_per_img = self.lr / 16
        self.optimizer = 'AdamW'
        self.loss = 'CE' # 'SoftCE'
        self.addtion_loss = ['LiftLoss']
        self.print_inter = 50
        self.val_inter = 1
        self.grad_clip = 1.0
        self.warmup_epoch = 1.0
        self.weight_decay = 1e-3
        self.freeze_list = [] # ['embed','backbone','norm','wave']
        self.resume = False
        self.resume_optimizer = False
        self.ckpt = None

        self.use_ema = False
        self.ema_decay = 0.995
        self.ema_update_every_step = 50

        # ---- FP16 ----
        self.fp16 = False
        
        # -----------------  testing config ------------------ #
    
    def _dynamic_update(self):
        # dynamic setting
        self.enable_waveSF = self.use_LADM
        self.use_LADM = self.use_LADM or self.use_ISSM
        
    def get_model(self) -> Module:
        from core.model.embedNet import embedNet,embedNet_light
        from core.model.backbone import MFViT_FPN
        from core.model.head import headNet
        from core.model.net_main import MFViT_Main
        if self.embed_net == 'CNN':
            embed = embedNet(input_channel=3,embed_dim=96,out_resolution=56)
        elif self.embed_net == 'Conv':
            embed = embedNet_light(input_channel=3,input_time=self.clip_duration,embed_dim=96,out_resolution=56,facial_project=self.facial_project)
        else:
            raise NotImplementedError()
        backbone = MFViT_FPN(type=self.backbone_type,
                             use_stage=self.use_stage,
                             attention_dropout=self.attention_dropout,
                             use_WaveSF=self.enable_waveSF,
                             SSL=self.SSL_Network,  # not SSL train but need SSL network
                             use_prompt=self.use_prompt,
                             WaveSF_preUpdate=self.WaveSF_preUpdate,
                             AdaptSplit=self.use_ISSM,
                             WaveNorm = self.WaveNorm,
                             WavePool=self.WavePool,
                             facial=self.facial_project,
                             )
        head = headNet(num_classes=self.num_classes,embed_dim=backbone.output_dim,dropout=self.dropout)
        model = MFViT_Main(embedNet=embed,
                           backbone=backbone,
                           head=head,
                           use_WaveSF=self.use_LADM,
                           SSL=self.SSL_Network,
                           facial=self.facial_project,
                           )
        # freeze: Not Used,Deprecated
        model._frozen(self.freeze_list) 
        
        return model

    def get_loss(self)->Module:
        if self.mixup:
            from core.utils.loss import SoftTargetCrossEntropy
            return SoftTargetCrossEntropy()
        elif self.soft_label:
            from core.utils.loss import SoftTargetCrossEntropy_andReg
            return SoftTargetCrossEntropy_andReg()
        else:
            return super().get_loss()
    
    def get_data_loader(self, mode:str, is_distributed: bool):
        from core.utils import (
            get_local_rank,
            wait_for_the_master,
        )
        from core.datasets import MAFW_CLS
        
        local_rank = get_local_rank()
        with wait_for_the_master(local_rank):
            dataset = MAFW_CLS(root_dir=self.dataset_dir,
                            mode=mode,
                            input_resolution=self.resolution,
                            SSL_train=False,
                            split_num=self.split_num,
                            clip_duration=self.clip_duration,
                            clip_stride=self.clip_stride,
                            flip_p=self.flip_prob,
                            strong_aug = self.strong_aug,
                            single_val_view = self.single_val_view
                            )
        if is_distributed:
            batch_size = self.batch_size // dist.get_world_size()
        else:
            batch_size = self.batch_size
            
        dataloader = None
        if mode == "train":
            if self.resample:
                from torch.utils.data import WeightedRandomSampler
                sample_weights = dataset._get_sampleWeight()
                resampler = WeightedRandomSampler(sample_weights,len(sample_weights),replacement=False)
                dataloader = DataLoader(dataset,
                                        batch_size=batch_size,
                                        num_workers=self.num_workers,
                                        pin_memory=self.pin_memory,
                                        drop_last=True,
                                        sampler=resampler,
                                        )
            else:
                dataloader = DataLoader(dataset,
                                        batch_size=batch_size,
                                        num_workers=self.num_workers,
                                        shuffle=self.shuffle, # ddp mode should set to false
                                        pin_memory=self.pin_memory,
                                        drop_last=True,
                                        )
        else:
            if self.single_val_view:
                dataloader = DataLoader(dataset,
                                        batch_size=self.batch_size,
                                        num_workers=self.num_workers,
                                        shuffle=False, # ddp mode should set to false
                                        pin_memory=False,
                                        )
            else:
                dataloader = DataLoader(dataset,
                                        batch_size=1,
                                        num_workers=2,
                                        shuffle=False, # ddp mode should set to false
                                        pin_memory=False,
                                        )                
        return dataloader
    
    
    
        
        