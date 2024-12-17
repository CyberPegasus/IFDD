# -*- coding:utf-8 -*-

# Pre-setting for visible devices
cfg_file = 'configs/DFEW/MFViT_DFEW_SSLCLS_ViT.json'
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

import torch
import torch.backends.cudnn as cudnn
from core.exps import load_cfg, get_exp_by_file
from core.utils import get_num_devices,setup_logger
from loguru import logger
from tqdm import tqdm
import torchmetrics

def main(exp, cfg):
    cudnn.benchmark = True
    logger.info("configs:\n {}".format(cfg))
    logger.info("exp settings:\n {}".format(exp))
    device = 'cuda:0'
    test_loader = exp.get_data_loader(mode='test',is_distributed=False)
    logger.info("test_loader done.")
    model = exp.get_model()
    model:torch.nn.Module = model.to(device)
    exp.exp_name += f'_split{exp.split_num}'
    if exp.resume:
        weights = torch.load(exp.ckpt)
    else:
        weights = torch.load(exp.output_dir+'/'+exp.exp_name+'/'+exp.exp_name+'_best_train_loss_test.pth') # '_best_acc.pth')
    model.load_state_dict(weights['model'],strict=True)
    model = model.eval()
    logger.info("Model construct done.")
    logger.info("Model structure as following:\n {}".format(model))
    logger.info("Testing start...")
    
    _prediction_list = []
    _label_list = []
    with torch.no_grad():
        t = tqdm(test_loader)
        for iter, _data in enumerate(t):
            if hasattr(exp,'prompt_load_offline') and exp.prompt_load_offline:
                multi_clips, labels, prompts = _data
            else:
                multi_clips, labels = _data
            multi_clips = multi_clips.to(device)
            labels = labels.to(device)
            if exp.prompt_load_offline:
                prompts:torch.Tensor = prompts.to(device)
            num_clip = multi_clips.shape[1]
            b = multi_clips.shape[0]
            preds = torch.zeros((b,exp.num_classes),device=device)
            for i in range(num_clip):
                if exp.prompt_load_offline:
                    pred = model(multi_clips[:,i,...],prompts[:,i,...])
                else:
                    pred = model(multi_clips[:,i,...])
                preds += pred
            preds /= num_clip
            _prediction_list.append(preds.detach().cpu())
            _label_list.append(labels.detach().cpu())
    predictions = torch.cat(_prediction_list,dim=0).argmax(dim=1)
    labels = torch.cat(_label_list,dim=0)
    precision_cal = torchmetrics.Precision(task="multiclass", average='micro', num_classes=exp.num_classes)
    uar_call = torchmetrics.Recall(task="multiclass",average='macro', num_classes=exp.num_classes)
    per_class_call = torchmetrics.Precision(task="multiclass",average='none', num_classes=exp.num_classes)
    _precision = precision_cal(predictions, labels).float()
    uar = uar_call(predictions, labels).mean().float()
    per_class:torch.Tensor = per_class_call(predictions, labels).float()*100
    per_class = per_class.tolist()
                
    logger.info(f'Test Result: Acc(WAR) {100*_precision:.3f}%, UAR {100*uar:.3f}%')
    logger.info(f'Per Class: {per_class}')
        
if __name__ == "__main__":
    exp = get_exp_by_file(cfg.exp_file)
    exp.merge(cfg)
    
    num_gpu = get_num_devices()
    visible_gpu = cfg.gpuid.split(',')
    assert num_gpu == len(visible_gpu)

    file_name = os.path.join(exp.output_dir, exp.exp_name)
    setup_logger(
            file_name,
            filename="test_log.txt",
            mode="a",
        )
        
    main(exp,cfg)