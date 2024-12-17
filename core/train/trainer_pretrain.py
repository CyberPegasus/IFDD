import os
from loguru import logger

import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from core.exps import BaseExp
from core.utils import (
    get_local_rank,
    get_rank,
    setup_logger,
    lr_updater,
    occupy_mem
)
from tqdm import tqdm
from .trainer import Trainer

import torchmetrics

class Trainer_pretrain(Trainer):
    def __init__(self, *args, **kargs) -> None:
        super().__init__(*args, **kargs)
        self.exp.device = self.device

    def train(self):
        try:
            if hasattr(torch.cuda, 'empty_cache'):
                torch.cuda.empty_cache()
            self.before_train()
            self.train_epochs() # self.train_loop()
            
        except RuntimeError as exception:
            logger.info(str(exception))
            raise exception
    
    def before_train(self):
        logger.info("configs:\n {}".format(self.cfg))
        logger.info("exp settings:\n {}".format(self.exp))
        torch.cuda.set_device(self.local_rank)
        if self.exp.occupy:
            occupy_mem(self.cfg.gpuid,mem_ratio=0.9)
        # model
        self.model = self.exp.get_model()
        logger.info("Model Structure as following:\n {}".format(self.model))
        if self.exp.resume:
            if self.exp.ckpt:
                data = torch.load(self.exp.ckpt, map_location='cpu')
            else:
                data = torch.load(self.exp.output_dir+'/'+self.exp.exp_name+'/'+self.exp.exp_name+'_best_train_loss.pth', map_location='cpu')
            self.model.load_state_dict(data['model'], strict=True)
            
        # training loader
        _resolution = self.model.embed.image_resolution if self.exp.finetune else self.model.embed.model.visual.input_resolution
        self.train_loader = self.exp.get_data_loader(mode='train',is_distributed=False,resolution=_resolution)
        self.val_loader = self.exp.get_data_loader(mode='test',is_distributed=False,resolution=_resolution)
        
        logger.info("train_loader done.")
        self.max_iter = len(self.train_loader)
        
        # model distributed
        self.model = self.model.to(self.device)
        logger.info(f"model to {self.device} done.")
        # optimizer and lr scheduer
        self.optimizer = self.exp.get_optimizer(self.model)
        if self.exp.resume_optimizer:
            self.optimizer.load_state_dict(data['opt'])
            # start_epoch = data['epoch']
        # self.model.train()
        # # frozen clip's normalization layer
        # self.model.embed.model.eval() 
        logger.info("Training start...")
        self.lossf = self.exp.get_loss().to(self.device)
        _base_lr = self.exp.lr / 32 * self.exp.batch_size
        self.lr_schduler = lr_updater(max_epoch=self.exp.max_epoch,warmup_epoch=self.exp.warmup_epoch,base_lr=_base_lr,end_lr=_base_lr*0.01)
    
    def train_epochs(self):
        self.max_train_p = 0
        self.max_val_p = 0
        self.min_train_loss = 1e10
        for self.epoch in range(self.exp.max_epoch):
            logger.info(f"---> start train epoch {self.epoch + 1} / {self.exp.max_epoch}")            
            _lr = self.lr_schduler.set_lr(self.optimizer,cur_epoch=self.epoch)
            logger.info(f'lr set to {_lr} for epoch {self.epoch}.')
            if self.epoch < self.exp.warmup_epoch:
                logger.info(f'Warmup with lr {_lr} at epoch {self.epoch}.')
            
            self.train_in_one_iter_pretrain()
            self.train_after_one_iter_pretrain()
            if self.max_val_p > 0.95:
                # avoid overfitting for val set which is splited from training set
                logger.info(f'Training early stopped with val Acc {self.max_val_p} at epoch {self.epoch}.')
                break
            
    def train_in_one_iter_pretrain(self):
        self.model.train()
        if not self.exp.finetune:
            self.model.embed.model.eval() 
        _mean_loss = 0
        
        _prediction_list = []
        _label_list = []
        for iter, (features, labels) in enumerate(tqdm(self.train_loader)):
            features = features.to(self.device)
            labels = labels.to(self.device)
            
            outputs = self.model(features)
            loss = self.lossf(outputs,labels)
            loss.backward()
            
            _prediction_list.append(outputs.detach().cpu())
            _label_list.append(labels.detach().cpu())
            
            if self.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(parameters=self.model.parameters(),max_norm=self.grad_clip)
            self.optimizer.step()
            _mean_loss += loss.item()
            if (iter+1)%self.loss_inter_print==0:
                loop_lenth = len(labels)*iter
                logger.info(f'iter {iter+1}/epoch {self.epoch}: mean_loss {_mean_loss/loop_lenth:.8f}')
        _mean_loss /= len(self.train_loader)*self.exp.batch_size*2
        logger.info(f'epoch {self.epoch}: Mean loss of total iters {_mean_loss:.8f}%')
        
        predictions = torch.cat(_prediction_list,dim=0).argmax(dim=1)
        labels = torch.cat(_label_list,dim=0)
        _precision = torchmetrics.Precision(task="multiclass", average='micro', num_classes=self.exp.num_classes)
        _recall = torchmetrics.Recall(task="multiclass",average='none', num_classes=self.exp.num_classes)
        recall = _recall(predictions, labels).mean().float()
        precision = _precision(predictions,labels).float()
        
        logger.info(f'epoch {self.epoch}: Train UAR {100*precision:.3f}%, WR {100*recall:.3f}%')
        if _mean_loss <= self.min_train_loss:
            self.min_train_loss = _mean_loss
            data = {
                'epoch':self.epoch,
                'model':self.model.state_dict(),
                'opt':self.optimizer.state_dict(),}
            logger.info(f'Saving best training loss {self.min_train_loss:.8f} at epoch {self.epoch}.')
            # del outputs
            torch.save(data, self.exp.output_dir+'/'+self.exp.exp_name+'/'+self.exp.exp_name+'_best_train_loss.pth')
        if precision >= self.max_train_p:
            self.max_train_p = precision
            data = {
                'epoch':self.epoch,
                'model':self.model.state_dict(),
                'opt':self.optimizer.state_dict(),
            }
            logger.info(f'Saving best training UAR {self.max_train_p} at epoch {self.epoch}.')
            # del outputs
            torch.save(data, self.exp.output_dir+'/'+self.exp.exp_name+'/'+self.exp.exp_name+'_best_train_acc.pth')
                       
    def train_after_one_iter_pretrain(self):
        if (self.epoch+1)%self.exp.val_inter==0:
            self.model.eval()
            logger.info(f"Valing starting at epoch {self.epoch}")
            _prediction_list = []
            _label_list = []
            with torch.no_grad():
                for iter, (features, labels) in enumerate(tqdm(self.val_loader)):
                    features = features.to(self.device)
                    labels = labels.to(self.device)
                    num_clip = features.shape[1]
                    b=features.shape[0]
                    preds = torch.zeros((b,self.exp.num_classes),device=self.device)
                                        
                    for i in range(num_clip):
                        pred = self.model(features[:,i,...])
                        preds += pred
                    preds /= num_clip                    
                    _prediction_list.append(preds.detach().cpu())
                    _label_list.append(labels.detach().cpu())
            
            predictions = torch.cat(_prediction_list,dim=0).argmax(dim=1)
            labels = torch.cat(_label_list,dim=0)
            uar_recall = torchmetrics.Recall(task="multiclass",average='none', num_classes=self.exp.num_classes)
            war_recall = torchmetrics.Recall(task="multiclass",average='micro', num_classes=self.exp.num_classes)
            uar = uar_recall(predictions, labels).mean().float()
            war = war_recall(predictions, labels).float()
            
            logger.info(f'epoch {self.epoch}: Val UAR {100*uar:.3f}%, WAR {100*war:.3f}%')
            if uar >= self.max_val_p:
                self.max_val_p = uar
                data = {
                    'epoch':self.epoch,
                    'model':self.model.state_dict(),
                    'opt':self.optimizer.state_dict(),
                }
                logger.info(f'Saving best Val UAR {self.max_val_p} at epoch {self.epoch}.')
                # del outputs
                torch.save(data, self.exp.output_dir+'/'+self.exp.exp_name+'/'+self.exp.exp_name+'_best_val_acc.pth')
                
        
        