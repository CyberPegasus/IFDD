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
    occupy_mem,
    Mixup
)
from tqdm import tqdm
import torchmetrics
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from ema_pytorch import EMA
from torch.cuda.amp import autocast,GradScaler,grad_scaler

class Trainer:
    def __init__(self, exp:BaseExp, cfg):
        self.exp = exp
        self.cfg = cfg
        # training
        self.max_epoch = exp.max_epoch
        self.grad_clip = self.exp.grad_clip
        self.use_ema = exp.use_ema
        # distributed
        self.rank = get_rank()
        self.local_rank = get_local_rank()
        self.device = "cuda:{}".format(self.local_rank)
        # log
        self.file_name = os.path.join(exp.output_dir, exp.exp_name)
        if self.rank == 0:
            os.makedirs(self.file_name, exist_ok=True)
        
        self.loss_inter_print = self.exp.print_inter
        setup_logger(
            self.file_name,
            distributed_rank=self.rank,
            filename="train_log.txt",
            mode="a",
        )
        self.writer = SummaryWriter(self.file_name+'/tb_logs')
        
    def train(self):
        try:
            if hasattr(torch.cuda, 'empty_cache'):
                torch.cuda.empty_cache()
            self.before_train()
            self.train_epochs() # self.train_loop()
            if self.writer is not None:
                self.writer.close()
            
        except RuntimeError as exception:
            logger.info(str(exception))
            # if 'CUDA out of memory' in str(exception):
            #     if hasattr(torch.cuda, 'empty_cache'):
            #         torch.cuda.empty_cache()
            #     logger.info("\nWarning!!! Batch size is too huge.")
            # else:
            raise exception
    
    def before_train(self):
        logger.info("configs:\n {}".format(self.cfg))
        logger.info("exp settings:\n {}".format(self.exp))
        torch.cuda.set_device(self.local_rank)
        if self.exp.occupy:
            occupy_mem(self.cfg.gpuid,mem_ratio=0.9)
        # model
        model = self.exp.get_model()
        logger.info("Model Structure as following:\n {}".format(model))
        # model efficiency calculation # TODO
        from thop import profile
        input_tmp = torch.randn(2,3,self.exp.clip_duration,self.exp.resolution,self.exp.resolution) # b,c,t,h,w
        flops, params = profile(model,inputs=(input_tmp,))
        logger.info(f"Model Efficiency as following:\n FLOPs {flops/(2*1000**3):.2f} G, Params {params/(1000**2):.2f} M.")
        # resume
        if self.exp.resume:
            _resume_path = self.exp.ckpt if self.exp.ckpt else self.exp.output_dir+'/'+self.exp.exp_name+'/'+self.exp.exp_name+'_best_val_acc.pth'
            data = torch.load(_resume_path, map_location='cpu')
            logger.info(f"Model resume from {_resume_path}")
            if 'ema' in _resume_path:
                data = data
                del data['initted']
                del data['step']
                for key in list(data.keys()):
                    _key = key.replace('ema_model.','')
                    data[_key] = data.pop(key)
            else:
                data = data['model']
            _model_dict = model.state_dict()
            def check_in(x:list,y:str):
                for i in x:
                    if i in y:
                        return True
                return False
            pretrained_dict = {}
            count = 0
            for k,v in data.items():
                count += 1
                if k in _model_dict and v.shape == _model_dict[k].shape:
                    pretrained_dict[k] = v
                else:
                    logger.info(f"{count}: missing {k}")
            _model_dict.update(pretrained_dict)
            missing, unexpected = model.load_state_dict(_model_dict,strict=False)
            logger.info(f"Missing keys:{missing}\nUnexpected Keys:{unexpected}")
            del data
        # training loader
        self.train_loader = self.exp.get_data_loader(mode='train',is_distributed=False)
        self.val_loader = self.exp.get_data_loader(mode='val',is_distributed=False)
        logger.info("train_loader done.")
        self.max_iter = len(self.train_loader)
        # model distributed
        self.model = model.to(self.device)
        del model
        logger.info(f"model to {self.device} done.")
        if hasattr(self.exp,'resample') and self.exp.resample:
            if hasattr(self.exp,'resample_initHead') and self.exp.resample_initHead:
                self.model.head._init_weights()
                logger.info(f"model head has reinited for resample, self.exp.freeze_list:{self.exp.freeze_list}.")
            else:
                logger.info(f"model head reinit enable for resample but not activated since self.exp.freeze_list:{self.exp.freeze_list} is blank.")
        # ema
        if self.use_ema:
            self.ema = EMA(self.model,beta=self.exp.ema_decay,update_every=self.exp.ema_update_every_step,include_online_model=False)
            self.ema = self.ema.to(device=self.device)
            logger.info(f"EMA has been built.")
        # optimizer and lr scheduer
        self.optimizer = self.exp.get_optimizer(self.model)
        if self.exp.resume_optimizer:
            self.optimizer.load_state_dict(data['opt'])
            logger.info(f"resume_optimizer done.")
        self.model.train()
        logger.info("Training start...")
        self.lossf = self.exp.get_loss().to(self.device)
        if self.exp.use_WaveSF and self.exp.optimize_WaveSF:
            self.loss_addition = self.exp.get_addtional_loss().to(self.device)
        self.exp.basic_lr_per_img = self.exp.lr / 32 # refresh lr
        _base_lr=self.exp.basic_lr_per_img*self.exp.batch_size
        self.lr_schduler = lr_updater(max_epoch=self.exp.max_epoch,warmup_epoch=self.exp.warmup_epoch,base_lr=_base_lr,end_lr=_base_lr*0.01)
        
        self.use_prompt = False
        self.load_prompt_offline = False
        if hasattr(self.exp,'use_prompt') and self.exp.use_prompt:
            self.use_prompt = True
            if hasattr(self.exp,'prompt_load_offline') and not self.exp.prompt_load_offline:
                import face_alignment
                self.prompt_engine = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, flip_input=False, device=self.device)
            else:
                self.load_prompt_offline = True
                
        if self.exp.fp16:
            self.scaler = GradScaler()
            
        self.mixup = None
        if self.exp.mixup:
            self.mixup = Mixup(
                mixup_alpha=0.8,
                cutmix_alpha=1.0,
                cutmix_minmax=None,
                prob=1.0,
                switch_prob=0.5,
                mode='batch',
                label_smoothing=self.exp.label_smoothing,
                num_classes=self.exp.num_classes
                               )
            
    
    def train_epochs(self):
        self.max_train_p = 0
        self.max_val_p = 0
        self.min_train_loss = 1e10
        self.min_ema_loss = 1e10
        for self.epoch in range(self.exp.max_epoch):
            logger.info(f"---> start train epoch {self.epoch + 1} / {self.exp.max_epoch}")
            _lr = self.lr_schduler.set_lr(self.optimizer,cur_epoch=self.epoch)
            logger.info(f'lr set to {_lr} for epoch {self.epoch}.')
            if self.epoch < self.exp.warmup_epoch:
                logger.info(f'Warmup with lr {_lr} at epoch {self.epoch}.')
            
            if not self.exp.SSL:
                # with torch.autograd.set_detect_anomaly(True): # DEBUG
                self.train_in_one_iter()
                self.train_after_one_iter()
                
            else:
                self.train_in_one_iter_SSL()
                self.train_after_one_iter_SSL()
                
            if self.max_val_p > 0.95:
                # avoid overfitting for val set which is splited from training set
                logger.info(f'Training early stopped with val Acc {self.max_val_p} at epoch {self.epoch}.')
                break
            logger.info(f'Result for now: Min train Loss: {self.min_train_loss}')
            logger.info(f'Max train Acc: {self.max_train_p}')
            logger.info(f'Max val Acc: {self.max_val_p}')
            
    def train_in_one_iter(self):
        self.model.train()
        self.model._frozen(self.exp.freeze_list)
        _mean_loss = 0
        _prediction_list = []
        _label_list = []
        t = tqdm(self.train_loader)
        for iter, _data in enumerate(t):
            
            if not self.load_prompt_offline:
                if not self.exp.soft_label:
                    clips, labels = _data
                else:
                    clips, labels, soft_labels = _data
            else:
                clips, labels, prompts = _data
            del _data
            
            clips:torch.Tensor = clips.to(self.device)
            labels:torch.Tensor  = labels.to(self.device)
            if self.exp.soft_label:
                labels:torch.Tensor = soft_labels.to(self.device)
            if self.load_prompt_offline:
                prompts:torch.Tensor = prompts.to(self.device)
            
            # mixup or softlabel
            assert not((self.mixup is not None) and self.exp.soft_label), f'not support to use mixup and soft_label together'
            if self.mixup is not None:
                clips, labels = self.mixup(clips, labels)
            
            # face detect
            if self.use_prompt and not self.load_prompt_offline:
                clips_detect = clips.detach() # b,c,T,H,W
                b,c,T,H,W = clips_detect.shape
                clips_detect = clips_detect.transpose(1,2).contiguous().view(b*T,c,H,W)*255.0 # b*T,H,W,c
                # r = self.exp.resolution    # ,detected_faces=[np.array([[0,0,r,r]])]*b
                preds_batch = self.prompt_engine.get_landmarks_from_batch(clips_detect) # preds_batch, b*T,N,68; N = 1 by default
                neg_edge = -10
                preds_batch = max(0,preds_batch - neg_edge)
                for i, pred in enumerate(preds_batch):
                    if len(pred)>68:
                        preds_batch[i] = pred[0:68]
                    elif len(pred)==0:
                        # print(i)
                        _tmp = np.zeros((68,2),dtype=np.float32)
                        preds_batch[i] = _tmp
                prompts = preds_batch.to(self.device)
            
            with autocast(enabled=self.exp.fp16):
                if self.exp.use_WaveSF and not self.use_prompt:
                    outputs,pool_a,x_d = self.model(clips)
                elif self.exp.use_WaveSF and self.use_prompt:
                    outputs,pool_a,x_d = self.model(clips,prompt=prompts)
                else:
                    outputs = self.model(clips)

            with autocast(enabled=self.exp.fp16):
                if not self.exp.use_decouple_neutral:
                    loss = self.lossf(outputs,labels)
                else:
                    loss = self.lossf(xn,xe,labels)
                    
                if self.exp.use_WaveSF and self.exp.optimize_WaveSF:
                    loss += self.loss_addition(pool_a,x_d)
                
            _mean_loss += loss.item()
            if self.exp.fp16:
                self.scaler.scale(loss).backward()
                if self.grad_clip > 0:
                    self.scaler.unscale_(self.optimizer)
                    grad_norm = torch.nn.utils.clip_grad_norm_(parameters=self.model.parameters(),max_norm=self.grad_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                if self.grad_clip > 0:
                    grad_norm = torch.nn.utils.clip_grad_norm_(parameters=self.model.parameters(),max_norm=self.grad_clip)
                self.optimizer.step()
            self.optimizer.zero_grad()
            
            if self.use_ema:
                self.ema.update()
            if (iter+1)%self.loss_inter_print==0:
                loop_lenth = len(labels)*(iter+1)
                logger.info(f'iter {iter+1}/epoch {self.epoch}: mean_loss {_mean_loss/loop_lenth:.3f}')
            if self.grad_clip > 0 and not self.exp.fp16:
                t.set_description(f'Cur Loss:{_mean_loss/(len(labels)*(iter+1))}, Grad Norm:{grad_norm}') 
            else:
                t.set_description(f'Cur Loss:{_mean_loss/(len(labels)*(iter+1))}') 
                
            if self.mixup is None:
                if self.exp.use_decouple_neutral:
                    xn, xe=outputs
                    prediction = torch.argmax(xe,dim=1)
                    prediction = torch.where(prediction>3,prediction+1,prediction)
                    neutral_pred = torch.argmax(xn,dim=1)
                    prediction = torch.where(neutral_pred==1,self.exp.neutral_id,prediction)
                else:
                    prediction = torch.argmax(outputs,dim=1)
                    if self.exp.soft_label:
                        labels = torch.argmax(labels,dim=1)

                # training accuracy
                _prediction_list.append(prediction.detach().cpu())
                _label_list.append(labels.detach().cpu())
        
        # get a min loss
        _mean_loss /= self.exp.batch_size*len(self.train_loader)
        if _mean_loss <= self.min_train_loss:
            self.min_train_loss = _mean_loss
            _ema_save = self.ema.state_dict() if self.use_ema else None
            data = {
                'epoch': self.epoch,
                'model': self.model.state_dict(),
                'opt':   self.optimizer.state_dict(),
                'ema':   _ema_save,
            }
            logger.info(f'Saving best training loss {self.min_train_loss:.5f} at epoch {self.epoch}.')
            # del outputs
            torch.save(data, self.exp.output_dir+'/'+self.exp.exp_name+'/'+self.exp.exp_name+'_best_train_loss.pth')
            
        if self.mixup is None:
            epoch_predictions = torch.cat(_prediction_list,dim=0)
            epoch_labels = torch.cat(_label_list,dim=0)
            precision_cal = torchmetrics.Precision(task="multiclass", average='micro', num_classes=self.exp.num_classes)
            uar_call = torchmetrics.Recall(task="multiclass",average='macro', num_classes=self.exp.num_classes)
            confusion_cal = torchmetrics.ConfusionMatrix(task="multiclass", num_classes=self.exp.num_classes)
            per_class_call = torchmetrics.Precision(task="multiclass",average='none', num_classes=self.exp.num_classes)
            _precision = precision_cal(epoch_predictions, epoch_labels).float()
            _uar = uar_call(epoch_predictions, epoch_labels).float()
            _confmat:torch.Tensor = confusion_cal(epoch_predictions, epoch_labels).float()
            per_class:torch.Tensor = per_class_call(epoch_predictions, epoch_labels).float()*100
            per_class = per_class.tolist()
            _mean_loss /= len(labels)*(iter+1)
            logger.info(f'epoch {self.epoch}: Train Acc {100*_precision:.3f}%, UAR {100*_uar:.3f}%, Loss {_mean_loss:.5f}.\nConfusion Matrix:\n{_confmat}\nClass Acc:\n{per_class}')
            self.writer.add_scalar("Loss",_mean_loss,self.epoch+1)
            self.writer.add_scalar("Train Acc",_precision*100,self.epoch+1)
            self.writer.add_scalar("Train UAR",_uar*100,self.epoch+1)
            if _precision >= self.max_train_p:
                self.max_train_p = _precision
                logger.info(f'best training Acc {self.max_train_p} at epoch {self.epoch}.')
                
    def train_after_one_iter(self):
        # begin validation
        if (self.epoch+1)%self.exp.val_inter==0 and not self.exp.SSL:
            # if self.use_ema:
            #     _model = self.ema.ema_model
            #     _model.eval()
            self.model.eval()
            logger.info(f"Valing starting at epoch {self.epoch}")
            if not self.exp.use_decouple_neutral:
                _prediction_list = []
            else:
                _xn_list = []
                _xe_list = []
            _label_list = []
            with torch.no_grad():
                t = tqdm(self.val_loader)
                for iter, _data in enumerate(t):
                    if not self.load_prompt_offline:
                        if not self.exp.soft_label:
                            multi_clips, labels = _data
                        else:
                            multi_clips, labels, soft_labels = _data
                    else:
                        multi_clips, labels, multi_prompts = _data
                    del _data
                    multi_clips = multi_clips.to(self.device)
                    labels = labels.to(self.device)
                    if self.load_prompt_offline:
                        multi_prompts = multi_prompts.to(self.device)
                    
                    num_clip = multi_clips.shape[1]
                    b = multi_clips.shape[0]
                    if not self.exp.use_decouple_neutral:
                        preds = torch.zeros((b,self.exp.num_classes),device=self.device)
                    else:
                        xns = torch.zeros((b,2),device=self.device)
                        xes = torch.zeros((b,self.exp.num_classes-1),device=self.device)
                    for i in range(num_clip):
                        if not self.exp.use_decouple_neutral:
                            if not self.use_prompt:
                                pred = self.model(multi_clips[:,i,...])
                            else:
                                pred = self.model(multi_clips[:,i,...],prompt = multi_prompts[:,i,...])
                            preds += pred
                        else:
                            if not self.use_prompt:
                                xn, xe = self.model(multi_clips[:,i,...])
                            else:
                                xn, xe = self.model(multi_clips[:,i,...],prompt = multi_prompts[:,i,...])                            
                            xns += xn
                            xes += xe
                    if not self.exp.use_decouple_neutral:
                        preds /= num_clip
                    else:
                        xns /= num_clip
                        xes /= num_clip
                    if not self.exp.use_decouple_neutral:
                        _prediction_list.append(preds.detach().cpu())
                    else:
                        _xn_list.append(xns.detach().cpu())
                        _xe_list.append(xes.detach().cpu())
                    _label_list.append(labels.detach().cpu())
            # xn, xe=outputs
            # prediction = torch.argmax(xe,dim=1)
            # neutral_pred = torch.argmax(xn,dim=1)
            # prediction = torch.where(neutral_pred==1,self.exp.neutral_id,prediction)
            if not self.exp.use_decouple_neutral:
                predictions = torch.cat(_prediction_list,dim=0).argmax(dim=1)
            else:
                xns = torch.cat(_xn_list,dim=0)
                xes = torch.cat(_xe_list,dim=0)
                predictions = torch.argmax(xes,dim=1)
                predictions = torch.where(predictions>3,predictions+1,predictions)
                neutral_preds = torch.argmax(xns,dim=1)
                predictions = torch.where(neutral_preds==1,self.exp.neutral_id,predictions)
            labels = torch.cat(_label_list,dim=0)
            precision_cal = torchmetrics.Precision(task="multiclass", average='micro', num_classes=self.exp.num_classes)
            uar_call = torchmetrics.Recall(task="multiclass",average='macro', num_classes=self.exp.num_classes)
            war_call = torchmetrics.Recall(task="multiclass",average='micro', num_classes=self.exp.num_classes)
            confusion_cal = torchmetrics.ConfusionMatrix(task="multiclass", num_classes=self.exp.num_classes)
            per_class_call = torchmetrics.Precision(task="multiclass",average='none', num_classes=self.exp.num_classes)
            _precision = precision_cal(predictions, labels).float()
            uar = uar_call(predictions, labels).float()
            war = war_call(predictions, labels).float()
            _confmat = confusion_cal(predictions, labels).float()
            per_class:torch.Tensor = per_class_call(predictions, labels).float()*100
            per_class = per_class.tolist()
            logger.info(f'epoch {self.epoch}: Val Acc {100*_precision:.3f}%, UAR {100*uar:.3f}%, WAR {100*war:.3f}%\nConfusion Matrix:\n{_confmat}\nClass Acc:\n{per_class}')
            self.writer.add_scalar("Val UAR",uar*100,self.epoch+1)
            self.writer.add_scalar("Val WAR",war*100,self.epoch+1)
            if _precision >= self.max_val_p:
                self.max_val_p = _precision
                data = {
                    'epoch':self.epoch,
                    'model':self.model.state_dict(),
                    'opt':self.optimizer.state_dict(),
                }
                logger.info(f'Saving best val Acc {self.max_val_p} at epoch {self.epoch}.')
                # del outputs
                torch.save(data, self.exp.output_dir+'/'+self.exp.exp_name+'/'+self.exp.exp_name+'_best_val_acc.pth')
            
    def train_in_one_iter_SSL(self): # SimCLR contrastive learning
        self.model.train()
        _mean_loss = 0
        # torch.autograd.set_detect_anomaly(True) # DEBUG only
        t = tqdm(self.train_loader)
        for iter, (clips1, clips2, labels) in enumerate(t):
            clips = torch.cat((clips1,clips2),dim=0).to(self.device)
            labels = labels.repeat(2).to(self.device) # B,1
            self.optimizer.zero_grad()

            if self.exp.use_WaveSF:
                outputs,pool_a,x_d = self.model(clips)
            else: # val mode for contrastive learning
                outputs = self.model(clips)
            
            loss = self.lossf(outputs,labels)
            if self.exp.use_WaveSF and self.exp.optimize_WaveSF: # DEBUG
                loss += self.loss_addition(pool_a,x_d)
            # with torch.autograd.detect_anomaly(True): # DEBUG
            loss.backward()
            if self.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(parameters=self.model.parameters(),max_norm=self.grad_clip)
            self.optimizer.step()  
            if self.use_ema:
                self.ema.update() 
            # logger.info(f'iter {iter+1}/epoch {self.epoch}: {loss1} / {loss2}') # DEBUG
            _mean_loss += loss.item()
            if (iter+1)%self.loss_inter_print==0:
                loop_lenth = len(labels)*iter
                logger.info(f'iter {iter+1}/epoch {self.epoch}: mean_loss {_mean_loss/loop_lenth:.8f}')
            t.set_description(f'Cur Loss:{_mean_loss/(len(labels)*(iter+1))}') 
        _mean_loss /= len(self.train_loader)*self.exp.batch_size*2
        logger.info(f'epoch {self.epoch}: Mean loss of total iters {_mean_loss:.8f}%')
        if _mean_loss <= self.min_train_loss:
            self.min_train_loss = _mean_loss
            _ema_save = self.ema.state_dict() if self.use_ema else None
            data = {
                'epoch': self.epoch,
                'model': self.model.state_dict(),
                'opt':   self.optimizer.state_dict(),
                'ema':   _ema_save,
            }
            logger.info(f'Saving best training loss {self.min_train_loss:.8f} at epoch {self.epoch}.')
            # del outputs
            torch.save(data, self.exp.output_dir+'/'+self.exp.exp_name+'/'+self.exp.exp_name+'_best_train_loss.pth')
            
    def train_after_one_iter_SSL(self):
        if (self.epoch+1)%self.exp.val_inter==0 and self.use_ema:
            self.model.eval()
            logger.info(f"Evaluating for SSL Loss on training set starting at epoch {self.epoch}")
            _mean_loss = 0
            with torch.no_grad():
                t = tqdm(self.train_loader)
                _model = self.ema.ema_model
                _model.eval()
                for iter, (clips1, clips2, labels) in enumerate(t):
                    clips = torch.cat((clips1,clips2),dim=0).to(self.device)
                    labels = labels.repeat(2).to(self.device) # B,1
                    outputs = _model(clips)
                    
                    loss = self.lossf(outputs,labels)
                    loss = loss.item()
                    _mean_loss += loss
                    t.set_description(f'Cur Loss:{_mean_loss/(len(labels)*(iter+1))}')
            _mean_loss /= len(self.train_loader)*self.exp.batch_size*2
            logger.info(f'epoch {self.epoch}: Mean loss for EMA model of total iters {_mean_loss:.8f}%')
            if _mean_loss <= self.min_ema_loss:
                self.min_ema_loss = _mean_loss
                logger.info(f'Saving best training loss {self.min_ema_loss:.8f} for EMA model at epoch {self.epoch}.')
                # del outputs
                torch.save(self.ema.state_dict(), self.exp.output_dir+'/'+self.exp.exp_name+'/'+self.exp.exp_name+'_best_ema_loss.pth')
                
            
                    
            
        
        