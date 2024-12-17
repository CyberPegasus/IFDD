# -*- coding:utf-8 -*-

# Pre-setting for visible devices
cfg_file = 'configs/IFDD_DFEW.json'
# for other datasets, please use:
# 'configs/IFDD_FERV39k.json'
# 'configs/IFDD_MAFW.json'


import json
from easydict import EasyDict as edict
def load_cfg(cfg_path:str)->edict:
    with open(cfg_path,'r') as f:
        cfg = json.load(f)
    cfg = edict(cfg)
    return cfg
cfg = load_cfg(cfg_file)

import os
os.environ["CUDA_VISIBLE_DEVICES"]=cfg.gpuid
import sys
sys.path.append(os.getcwd())
import torch.backends.cudnn as cudnn
from core.exps import load_cfg, get_exp_by_file
from core.utils import get_num_devices
from core.train import Trainer

def main(exp, cfg):
    cudnn.benchmark = True
    trainer = Trainer(exp,cfg)
    trainer.train()
    
if __name__ == "__main__":
    exp = get_exp_by_file(cfg.exp_file)
    exp.merge(cfg)
    #           
    if hasattr(exp,'_dynamic_update'):
        exp._dynamic_update()
    if 'DFEW' or 'MAFW' in exp.exp_name:
        exp.exp_name += f'_split{exp.split_num}'
    num_gpu = get_num_devices()
    visible_gpu = cfg.gpuid.split(',')
    assert num_gpu == len(visible_gpu)
    
    main(exp,cfg)