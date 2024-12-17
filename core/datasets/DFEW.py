import csv
from functools import partial
from typing import Any, Dict, List
from loguru import logger
import numpy as np
import glob
import random
import jpeg4py as jpeg

import torch
from torch.utils.data.dataset import Dataset
import clip

class DFEW(Dataset):
    def __init__(self,
                 root_dir:str,
                 mode:str, # train for training, val for validation, test for evaluation
                 SSL_train:bool=False,
                 split_num:int=0, # 1,2,3 splits for 3-fold evaluation
                 clip_duration:int=16, # video clip with fixed number of frames
                 clip_stride:int = 1, # Sampling Interval Length for each frame in the clip
                 flip_p:float = 0.5, # random Horizontal Flip
                 ) -> None:
        super().__init__()
        self.mode = mode
        self.split_num = split_num
        #-----reading annotation, video path and class id--------#
        annotation_path = root_dir+'/Annotation'
        annnotation_file = annotation_path + '/annotation.csv'
        self.classes = ['multiLabel','happy','sad','neutral','angry','surprise','disgust','fear']
        self.anno_list_all = []
        with open(annnotation_file,'r') as f:
            f_csv = csv.reader(f)
            header = next(f_csv)
            for row in f_csv:
                if int(row[-1]) != 0:
                    _tuple = (row[-2],row[-1])
                    self.anno_list_all.append(_tuple)
                else:
                    continue
        #------video path and preload-----#
        self.video_list_all = [f'{int(i[0]):0>5d}' for i in self.anno_list_all]
        self.video_list = []
        self.anno_list = []
        self.frame_path = root_dir+'/Clip/clip_224x224_16f/'
        for i,j in zip(self.video_list_all,self.anno_list_all):
            _tmp = self.frame_path+i
            _tmp = glob.glob(_tmp+'/*.jpg')
            if len(_tmp) < 16:
                pass
            else:
                self.video_list.append(i)
                self.anno_list.append(j)
        logger.info(f'Reading {len(self.anno_list)} videos and annotations for {mode} mode / {split_num} split')
        self.width = 224
        self.height = 224
        self.crop_size = (224,224)
        self.clip_duration = clip_duration # Clip length
        self.clip_stride = clip_stride # Sampling Interval Length for each frame in the clip
        self.flip_p = flip_p

        
    def __getitem__(self, index):
        _video = self.video_list[index]
        # rescale to 0~class_num-1,filter out 'multiLabel'
        _label = int(self.anno_list[index][1]) - 1
        assert int(_video) == int(self.anno_list[index][0]), f'{int(_video)} != {int(self.anno_list[index][0])}'
        
        # get frames
        _frame_path = self.frame_path +'/'+ _video
        _frames = glob.glob(_frame_path+'/*.jpg')
        # DEBUG
        _debug = True
        if _debug:
            _frames = sorted(_frames,key=lambda x: int(x.split('.')[0].split('/')[-1]))
        else:
            _frames = sorted(_frames)
        # _frames = sorted(_frames,key=lambda x: int(x.split('.')[0].split('/')[-1]))
        
        # load video or images
        if self.mode == 'train':
            _frame_list = self.get_clip(_frames)
            _frames = self.load_frames(_frame_list)
            if not self.SSL_train:
            # spatial transform
                _frames = self.RandomCrop(_frames)
                _frames = self.HorizontalFlip(_frames)
            _frames = self.Normalize(_frames)    
            # T,H,W,C -> C,T,H,W
            _frames = np.ascontiguousarray(_frames.transpose(3,0,1,2))
        else:
            _frame_list = self.get_clip(_frames)
            _frames = self.load_frames(_frame_list)
            _frames = self.CenterCrop(_frames)
            _frames = self.Normalize(_frames) 
            # N,T,H,W,C -> N,C,T,H,W
            _frames = np.ascontiguousarray(_frames.transpose(0,4,1,2,3))

        _frames = torch.from_numpy(_frames)
        _frames = torch.clamp(_frames,min=0.0,max=1.0)
        return _frames, _label
    
    def __len__(self):
        return len(self.video_list)
    
    def load_frames(self,frame_list:list):
        if self.mode == 'train':
            buffer = np.empty((self.clip_duration, self.height, self.width, 3), np.dtype('float32'))
            for _frame,buffer_count in zip(frame_list,range(len(frame_list))):
                _img = jpeg.JPEG(_frame).decode() # H,W,3
                buffer[buffer_count] = _img            
        else:
            if not isinstance(frame_list[0],list):
                raise ValueError("The frame_list for test mode should be a list of clips.")
            clip_num = len(frame_list)
            buffer = np.empty((clip_num,self.clip_duration, self.height, self.width, 3), np.dtype('float32'))
            for frames,clip_count in zip(frame_list,range(clip_num)):
                for _frame,buffer_count in zip(frames,range(len(frames))):
                    _img = jpeg.JPEG(_frame).decode() # T,H,W,3
                    buffer[clip_count][buffer_count] = _img
        
        buffer[np.isnan(buffer)]=0.0 # filtered out nan 
        return buffer # test:N,T,H,W,C or train:T,H,W,C
    
    def get_clip(self,frames:list)->list:
        if self.mode == 'train': #     split  ，          
            frame_len = len(frames)
            # FIXME：         1？
            _clip_duration = self.clip_duration*self.clip_stride
            if frame_len>=_clip_duration:
                #      frame_len-_clip_duration
                start_index = random.randrange(0,frame_len-_clip_duration+1)
            else:
                start_index = 0
            end_index = start_index+_clip_duration
            try:
                clip = frames[start_index:end_index:self.clip_stride]
            except IndexError:
                # FIXME          ，         _clip_duration，       self.clip_duration
                logger.info(f'Not enough length for frame of {frame_len} to get clip of {_clip_duration}')
                if frame_len < self.clip_duration:
                    clip = frames
                    _copy_frame = clip[-1]
                    while len(clip) < self.clip_duration:
                        clip.append(_copy_frame)
                else:
                    clip = frames[0:self.clip_duration]

            return clip
                
        else: # testing or validation mode
            frame_len = len(frames)
            # FIXME：         1？
            _clip_duration = self.clip_duration*self.clip_stride
            clips = []
            if frame_len >= _clip_duration:
                start_index = 0
                end_index =start_index+_clip_duration #   
                while end_index <= frame_len:
                    clips.append(frames[start_index:end_index:self.clip_stride])
                    start_index = end_index
                    end_index += _clip_duration
            else:
                if frame_len < self.clip_duration:
                    clip = frames
                    _copy_frame = clip[-1]
                    while len(clip) < self.clip_duration:
                        clip.append(_copy_frame)
                else:
                    clip = frames[0:self.clip_duration]

                clips.append(clip)
                
            return clips
    
    def HorizontalFlip(self, frames:np.ndarray):
        # frames decoded from jpeg4py as T,H,W,C
        if random.random() < self.flip_p:
            return np.ascontiguousarray(frames[:, :, ::-1, ...])
        else:
            return frames
    
    def Normalize(self, frames:np.ndarray, positive:bool = True):
        _max = 255.0
        if positive:
            # Scale to [0,1]
            frames = frames/_max 
        else:
            # Scale to [-1,1]
            frames = (frames-_max/2)/_max*2
        return frames
    
    def RandomCrop(self, frames:np.ndarray):
        # only for training mode
        t,h,w,c = frames.shape
        ch,cw = self.crop_size
        h_start = np.random.randint(h - ch)
        w_start = np.random.randint(w - cw)
        
        return np.ascontiguousarray(frames[:,h_start:h_start+ch,w_start:w_start+cw,:])
    
    def CenterCrop(self, frames:np.ndarray):
        # only for test mode
        h,w = frames.shape[-3],frames.shape[-2]
        ch,cw = self.crop_size
        h_start = (h - ch)//2
        w_start = (w - cw)//2
        
        return np.ascontiguousarray(frames[...,h_start:h_start+ch,w_start:w_start+cw,:])
        
class DFEW_SSL(DFEW):
    def __init__(self, *args, **kargs) -> None:
        super().__init__(*args, **kargs)
        from core.utils import SSL_DataAug
        if self.SSL_train and self.mode=='train':
            self.SSL_data_aug = SSL_DataAug(train_mode=True,target_size=self.crop_size)
        else:
            self.SSL_data_aug = SSL_DataAug(train_mode=False,target_size=self.crop_size)
            
    def __getitem__(self, index):
        if self.mode=='train' and self.SSL_train:
            x, label = super().__getitem__(index)
            return self.SSL_data_aug(x), self.SSL_data_aug(x), label
        else:
            x, label = super().__getitem__(index)
            return self.SSL_data_aug(x), label

class DFEW_pretrain_simple(DFEW):
    def __init__(self, *args, **kargs) -> None:
        super().__init__(*args, **kargs)
        from core.frontEnd import clip_engine
        self.clip_engine = clip_engine(device=self.device,clip_duration=self.clip_duration,height=self.height,width=self.width)
        if self.mode != 'train':
            raise NotImplementedError()
        
    def __getitem__(self, index):
        _video = self.video_list[index]
        _label = int(self.anno_list[index][1])
        assert int(_video) == int(self.anno_list[index][0]), f'{int(_video)} != {int(self.anno_list[index][0])}'
        
        # get frames
        _frame_path = self.frame_path +'/'+ _video
        _frames = glob.glob(_frame_path+'/*.jpg')
        _frames = sorted(_frames)
        
        # load path list of video or images
        _frame_list = self.get_clip(_frames)
        frame_features = self.clip_engine(_frame_list)
        return frame_features
    
class DFEW_pretrain(Dataset):
    def __init__(self,
                 root_dir:str,
                 mode:str, # train for training, val for validation, test for evaluation
                 input_resolution:int,
                 SSL_train:bool=False,
                 split_num:int=0, # 1,2,3 splits for 3-fold evaluation
                 clip_duration:int=16, # video clip with fixed number of frames
                 clip_stride:int = 1, # Sampling Interval Length for each frame in the clip
                 flip_p:float = 0.5, # random Horizontal Flip
                 ) -> None:
        super().__init__()
        self.mode = mode
        self.split_num = split_num
        self.SSL_train = SSL_train
        self.resolution = input_resolution
        #-----reading annotation, video path and class id--------#
        annotation_path = root_dir+'/EmoLabel_DataSplit'
        annnotation_file = annotation_path + f'/{mode}_single-labeled/set_{split_num}.csv'
        self.classes = ['multiLabel','happy','sad','neutral','angry','surprise','disgust','fear']
        self.anno_list_all = []
        with open(annnotation_file,'r') as f:
            f_csv = csv.reader(f)
            header = next(f_csv)
            for row in f_csv:
                if int(row[-1]) != 0:
                    _tuple = (row[-2],row[-1])
                    self.anno_list_all.append(_tuple)
                else:
                    continue
        #------video path and preload-----#
        self.video_list_all = [f'{int(i[0]):0>5d}' for i in self.anno_list_all]
        self.video_list = []
        self.anno_list = []
        self.frame_path = root_dir+'/Clip/clip_224x224_16f/'
        for i,j in zip(self.video_list_all,self.anno_list_all):
            _tmp = self.frame_path+i
            _tmp = glob.glob(_tmp+'/*.jpg')
            if len(_tmp) < 16:
                pass
            else:
                self.video_list.append(i)
                self.anno_list.append(j)
        logger.info(f'Reading {len(self.anno_list)} videos and annotations for {mode} mode / {split_num} split')
        self.width = 224
        self.height = 224
        self.crop_size = (224,224)
        self.clip_duration = clip_duration # Clip length
        self.clip_stride = clip_stride # Sampling Interval Length for each frame in the clip
        self.flip_p = flip_p
        from torchvision.transforms import Compose, Resize, CenterCrop, Normalize, InterpolationMode, RandomHorizontalFlip, RandomResizedCrop
        self.tensor_transform = Compose([
            Resize(self.resolution, interpolation=InterpolationMode.BICUBIC,antialias=True),
            CenterCrop(self.resolution),
            Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])

    def load_frames(self, frame_list: list, scale:bool = True):
        # special case
        if len(frame_list) == 0:
            return None
        # type check
        if isinstance(frame_list[0],str):
            _mode = 'singleView'
        elif isinstance(frame_list[0], list) and isinstance(frame_list[0][0], str):
            _mode = 'multiView'
            # raise NotImplementedError(f'{_mode} has not been supported. We will add it in the future. ')
        else:
            raise NotImplementedError()
        # load frame using jpeg, pre-occupy memory for faster process
        if _mode == 'singleView':
            buffer = np.empty((self.clip_duration, self.height, self.width, 3), np.dtype('float32'))
            for _frame,buffer_count in zip(frame_list,range(len(frame_list))):
                _img = jpeg.JPEG(_frame).decode() # H,W,3, in RGB format
                buffer[buffer_count] = _img  
            buffer = np.ascontiguousarray(buffer.transpose((0,3,1,2)))
        else:
            if not isinstance(frame_list[0],list):
                raise ValueError("The frame_list for test mode should be a list of clips.")
            clip_num = len(frame_list)
            buffer = np.empty((clip_num,self.clip_duration, self.height, self.width, 3), np.dtype('float32'))
            for frames,clip_count in zip(frame_list,range(clip_num)):
                for _frame,buffer_count in zip(frames,range(len(frames))):
                    _img = jpeg.JPEG(_frame).decode() # T,H,W,3
                    buffer[clip_count][buffer_count] = _img
            buffer = np.ascontiguousarray(buffer.transpose((0,1,4,2,3)))
        # filtered out nan 
        if scale:
            buffer[np.isnan(buffer)]=0.0
            buffer = buffer/255.0
            buffer = torch.from_numpy(buffer)
        else:
            buffer[np.isnan(buffer)]=0
            buffer = torch.from_numpy(buffer)
            buffer = buffer.to(torch.uint8)
        
        return buffer # singleView:T,C,H,W or multiView:N,T,C,H,W
    
    def get_clip(self,frames:list)->list:
        if self.mode == 'train': #     split  ，          
            frame_len = len(frames)
            _clip_duration = self.clip_duration*self.clip_stride
            if frame_len>=_clip_duration:
                #      frame_len-_clip_duration
                start_index = random.randrange(0,frame_len-_clip_duration+1)
            else:
                start_index = 0
            end_index = start_index+_clip_duration
            try:
                clip = frames[start_index:end_index:self.clip_stride]
            except IndexError:
                logger.info(f'Not enough length for frame of {frame_len} to get clip of {_clip_duration}')
                if frame_len < self.clip_duration:
                    clip = frames
                    _copy_frame = clip[-1]
                    while len(clip) < self.clip_duration:
                        clip.append(_copy_frame)
                else:
                    clip = frames[0:self.clip_duration]

            return clip
                
        else: # testing or validation mode
            frame_len = len(frames)
            _clip_duration = self.clip_duration*self.clip_stride
            clips = []
            if frame_len >= _clip_duration:
                start_index = 0
                end_index =start_index+_clip_duration #   
                while end_index <= frame_len:
                    clips.append(frames[start_index:end_index:self.clip_stride])
                    start_index = end_index
                    end_index += _clip_duration
            else:
                if frame_len < self.clip_duration:
                    clip = frames
                    _copy_frame = clip[-1]
                    while len(clip) < self.clip_duration:
                        clip.append(_copy_frame)
                else:
                    clip = frames[0:self.clip_duration]

                clips.append(clip)
                
            return clips
         
    def __getitem__(self, index):
        _video = self.video_list[index]
        # rescale to 0~class_num-1
        _label = int(self.anno_list[index][1]) - 1
        assert int(_video) == int(self.anno_list[index][0]), f'{int(_video)} != {int(self.anno_list[index][0])}'
        
        # get frames
        _frame_path = self.frame_path +'/'+ _video
        _frames = glob.glob(_frame_path+'/*.jpg')
        # DEBUG
        _debug = True
        if _debug:
            _frames = sorted(_frames,key=lambda x: int(x.split('.')[0].split('/')[-1]))
        else:
            _frames = sorted(_frames)
        
        # load path list of video or images
        _frame_list = self.get_clip(_frames)
        _frames_tensor = self.load_frames(_frame_list)
        if self.mode!='train':
            n,t,c,h,w = _frames_tensor.shape
            _frames_tensor = _frames_tensor.view(-1,c,h,w)
        _frames_tensor = self.tensor_transform(_frames_tensor)
        if self.mode!='train':
            _,c,h,w= _frames_tensor.shape
            _frames_tensor = _frames_tensor.view(n,t,c,h,w)
        return _frames_tensor, _label
    
    def __len__(self):
        return len(self.video_list)
    
if int(torch.__version__.split('.')[0])>1:
    from torchvision.transforms.v2 import Transform
    class FloatScale(Transform):
        def _transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
            return inpt/255.0
else:
    class FloatScale():
        def __init__(self) -> None:
            pass
        def __call__(self, inpt: Any) -> Any:
            return inpt/255.0  
        
class DFEW_CLS(DFEW_pretrain):
    def __init__(self, root_dir: str, mode: str, input_resolution: int, SSL_train: bool = False, soft_label:bool = False, split_num: int = 0, clip_duration: int = 16, clip_stride: int = 1, flip_p: float = 0.5, strong_aug:bool=False) -> None:
        super().__init__(root_dir, mode, input_resolution, SSL_train, split_num, clip_duration, clip_stride, flip_p)
        
        # antialias only for torch version >= 2
        if int(torch.__version__.split('.')[0])>1:
            from torchvision.transforms.v2 import Compose, Resize, CenterCrop, Normalize, InterpolationMode, RandomHorizontalFlip, RandomResizedCrop, ToDtype,RandomPerspective,ColorJitter
            createRandomResizedCrop = partial(RandomResizedCrop,antialias=True)
        else:
            from torchvision.transforms import Compose, Resize, CenterCrop, Normalize, InterpolationMode, RandomHorizontalFlip, RandomResizedCrop, RandomPerspective,ColorJitter
            createRandomResizedCrop = partial(RandomResizedCrop)
            class ToDtype():
                def __init__(self,dtype:torch.dtype) -> None:
                    self.dtype = dtype
                def __call__(self, inpt: Any) -> Any:
                    return inpt.to(dtype=self.dtype)
        # antialias only for torch version >= 2
        if int(torch.__version__.split('.')[0])>1:
            createRandomResizedCrop = partial(RandomResizedCrop,antialias=True)
        else:
            createRandomResizedCrop = partial(RandomResizedCrop)
        self.strong_aug = strong_aug
        if self.mode == 'train':
            if not strong_aug:
                self.tensor_transform = Compose([
                    RandomHorizontalFlip(0.5),
                    createRandomResizedCrop(size=self.crop_size,scale=(0.5,1.0),interpolation=InterpolationMode.BICUBIC),
                ])
            else:
                self.tensor_transform = Compose([
                    # size of FERV39k's img is 224*224, don't need resize
                    RandomHorizontalFlip(0.5),
                    createRandomResizedCrop(size=self.crop_size,scale=(0.5,1.0),interpolation=InterpolationMode.BILINEAR),
                    ColorJitter(brightness=0.4,contrast=0.4,saturation=0.4,hue=0.4),
                    # RandomPerspective(distortion_scale=0.2,p=0.25),
                    ToDtype(torch.float32),
                    FloatScale(),
                ])                
        else:
            self.tensor_transform = Compose([
                Resize(self.resolution, interpolation=InterpolationMode.BICUBIC),
            ])
        self.soft_label = soft_label
        if soft_label:
            soft_label_path = root_dir+'/Annotation/annotation.csv'
            self.soft_anno_dict = {}
            with open(soft_label_path,'r') as f:
                f_csv = csv.reader(f)
                # 1happy,2sad,3neutral,4angry,5surprise,6disgust,7fear,order,label
                header = next(f_csv)
                for row in f_csv:
                    key = int(row[-2])
                    _soft_label = [int(row[i]) for i in range(7)] 
                    self.soft_anno_dict[key] = _soft_label  
                     
    def _get_sampleWeight(self):
        _labels = np.array([int(i[1])-1 for i in self.anno_list])
        _cls_count = np.unique(_labels,return_counts=True)[1]
        # _cls_count = np.array([len(np.where(_labels==t)[0]) for t in _cls])
        _cls_weight = 1. / _cls_count
        #        
        _T = _cls_weight.mean()+_cls_weight.std()
        _anomaly_index = _cls_weight > _T
        _cls_weight[_anomaly_index] = _T
        # get weight expanded as label
        _sample_weight = _cls_weight[_labels]
        _sample_weight = torch.from_numpy(_sample_weight)
        return _sample_weight
    
    def __getitem__(self, index):
        _video = self.video_list[index]
        # rescale to 0~class_num-1
        _label = int(self.anno_list[index][1]) - 1
        if self.soft_label:
            # 10 is the max value of soft_label
            _soft_label = np.array(self.soft_anno_dict[int(_video)])/10.0
        assert int(_video) == int(self.anno_list[index][0]), f'{int(_video)} != {int(self.anno_list[index][0])}'
        
        # get frames
        _frame_path = self.frame_path +'/'+ _video
        _frames = glob.glob(_frame_path+'/*.jpg')
         # DEBUG
        _debug = True
        if _debug:
            _frames = sorted(_frames,key=lambda x: int(x.split('.')[0].split('/')[-1]))
        else:
            _frames = sorted(_frames)
        
        # load path list of video or images
        _frame_list = self.get_clip(_frames)
        not_scale = self.strong_aug and (self.mode == 'train')
        _frames_tensor = self.load_frames(_frame_list,scale=not not_scale)
        if self.mode!='train':
            n,t,c,h,w = _frames_tensor.shape
            _frames_tensor = _frames_tensor.view(-1,c,h,w)
        _frames_tensor = self.tensor_transform(_frames_tensor)
        if self.mode!='train':
            _,c,h,w= _frames_tensor.shape
            _frames_tensor = _frames_tensor.view(n,t,c,h,w)
            _frames_tensor = _frames_tensor.transpose(1,2)
        else:
            _frames_tensor = _frames_tensor.transpose(0,1)
        # clamp
        _frames_tensor:torch.Tensor = _frames_tensor.clamp(0,1)
        # b,c,t,h,w
        if not self.soft_label:
            return _frames_tensor, _label
        else:
            return _frames_tensor, _label, _soft_label
    
    def __len__(self):
        return len(self.video_list)

class DFEW_CLS_vit(DFEW_pretrain):
    def __init__(self, root_dir: str, mode: str, input_resolution: int, SSL_train: bool = False, split_num: int = 0, clip_duration: int = 16, clip_stride: int = 1, flip_p: float = 0.5) -> None:
        super().__init__(root_dir, mode, input_resolution, SSL_train, split_num, clip_duration, clip_stride, flip_p)
        from torchvision.transforms import Compose, Resize, CenterCrop, Normalize, InterpolationMode, RandomHorizontalFlip, RandomResizedCrop
        # antialias only for torch version >= 2
        if int(torch.__version__.split('.')[0])>1:
            createRandomResizedCrop = partial(RandomResizedCrop,antialias=True)
        else:
            createRandomResizedCrop = partial(RandomResizedCrop)
        if self.mode == 'train':
            self.tensor_transform = Compose([
                RandomHorizontalFlip(0.5),
                createRandomResizedCrop(size=self.crop_size,scale=(0.8,1.0),interpolation=InterpolationMode.BICUBIC),
                CenterCrop(self.resolution),
                Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
            ])
        else:
            self.tensor_transform = Compose([
                Resize(self.resolution, interpolation=InterpolationMode.BICUBIC,antialias=True),
                CenterCrop(self.resolution),
                Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
            ])
        
    def __getitem__(self, index):
        _video = self.video_list[index]
        # rescale to 0~class_num-1
        _label = int(self.anno_list[index][1]) - 1
        assert int(_video) == int(self.anno_list[index][0]), f'{int(_video)} != {int(self.anno_list[index][0])}'
        
        # get frames
        _frame_path = self.frame_path +'/'+ _video
        _frames = glob.glob(_frame_path+'/*.jpg')
         # DEBUG
        _debug = True
        if _debug:
            _frames = sorted(_frames,key=lambda x: int(x.split('.')[0].split('/')[-1]))
        else:
            _frames = sorted(_frames)
        
        # load path list of video or images
        _frame_list = self.get_clip(_frames)
        _frames_tensor = self.load_frames(_frame_list)
        if self.mode!='train':
            n,t,c,h,w = _frames_tensor.shape
            _frames_tensor = _frames_tensor.view(-1,c,h,w)
        _frames_tensor = self.tensor_transform(_frames_tensor)
        if self.mode!='train':
            _,c,h,w= _frames_tensor.shape
            _frames_tensor = _frames_tensor.view(n,t,c,h,w)
            _frames_tensor = _frames_tensor.transpose(1,2)
        else:
            _frames_tensor = _frames_tensor.transpose(0,1)
        # b,c,t,h,w
        return _frames_tensor, _label
    
    def __len__(self):
        return len(self.video_list)

if int(torch.__version__.split('.')[0])>1:
    from torchvision.transforms.v2 import Transform
    class FloatScale(Transform):
        def _transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
            return inpt/255.0
else:
    class FloatScale():
        def __init__(self) -> None:
            pass
        def __call__(self, inpt: Any) -> Any:
            return inpt/255.0  
        
class DFEW_CLS_prompt(DFEW_CLS):
    def __init__(self, root_dir: str, mode: str, input_resolution: int, prompt_load_offline:bool=False, SSL_train: bool = False, soft_label:bool=False, strong_aug:bool=False, split_num: int = 0, clip_duration: int = 16, clip_stride: int = 1, flip_p: float = 0.5) -> None:
        super().__init__(root_dir, mode, input_resolution, SSL_train, soft_label, split_num, clip_duration, clip_stride, flip_p, strong_aug)
        if int(torch.__version__.split('.')[0])>1:
            from torchvision.transforms.v2 import Compose, Resize, CenterCrop, Normalize, InterpolationMode, RandomHorizontalFlip, RandomResizedCrop, ToDtype,RandomPerspective,ColorJitter
            createRandomResizedCrop = partial(RandomResizedCrop,antialias=True)
        else:
            from torchvision.transforms import Compose, Resize, CenterCrop, Normalize, InterpolationMode, RandomHorizontalFlip, RandomResizedCrop, RandomPerspective,ColorJitter
            createRandomResizedCrop = partial(RandomResizedCrop)
            class ToDtype():
                def __init__(self,dtype:torch.dtype) -> None:
                    self.dtype = dtype
                def __call__(self, inpt: Any) -> Any:
                    return inpt.to(dtype=self.dtype)
        # antialias only for torch version >= 2
        if int(torch.__version__.split('.')[0])>1:
            createRandomResizedCrop = partial(RandomResizedCrop,antialias=True)
        else:
            createRandomResizedCrop = partial(RandomResizedCrop)
        self.strong_aug = strong_aug
        if self.mode == 'train':
            if not strong_aug:
                self.tensor_transform = Compose([
                    RandomHorizontalFlip(0.5),
                    createRandomResizedCrop(size=self.crop_size,scale=(0.8,1.0),interpolation=InterpolationMode.BICUBIC),
                    CenterCrop(self.resolution),
                ])
            else:
                self.tensor_transform = Compose([
                    # size of FERV39k's img is 224*224, don't need resize
                    RandomHorizontalFlip(0.5),
                    createRandomResizedCrop(size=self.crop_size,scale=(0.8,1.0),interpolation=InterpolationMode.BILINEAR),
                    ColorJitter(brightness=0.3,contrast=0.3,saturation=0.3),
                    RandomPerspective(distortion_scale=0.2,p=0.5),
                    ToDtype(torch.float32),
                    FloatScale(),
                ])                
        else:
            self.tensor_transform = Compose([
                Resize(self.resolution, interpolation=InterpolationMode.BICUBIC,antialias=True),
                CenterCrop(self.resolution),
            ])
        self.prompt_load_offline = prompt_load_offline
        self.neg_edge = -10
        
        self.soft_label = soft_label
        if soft_label:
            soft_label_path = root_dir+'/Annotation/annotation.csv'
            self.soft_anno_dict = {}
            with open(soft_label_path,'r') as f:
                f_csv = csv.reader(f)
                # 1happy,2sad,3neutral,4angry,5surprise,6disgust,7fear,order,label
                header = next(f_csv)
                for row in f_csv:
                    key = int(row[-2])
                    _soft_label = [int(row[i]) for i in range(7)] 
                    self.soft_anno_dict[key] = _soft_label    
            # self.soft_anno_list = []
            # for i,j in zip(self.video_list_all,self.anno_list_all):
            #     _tmp = self.frame_path+i
            #     _tmp = glob.glob(_tmp+'/*.jpg')
            #     if len(_tmp) < 16:
            #         pass
            #     else:
            #         video_i = int(i)
            #         self.soft_anno_list.append(self.soft_anno_dict[video_i])
            
    def _align_points_process(self, x:np.ndarray):
        x = x - self.neg_edge
        return np.maximum(x, 0)
    
    def __getitem__(self, index):
        _video = self.video_list[index]
        # rescale to 0~class_num-1
        _label = int(self.anno_list[index][1]) - 1
        if self.soft_label:
            # 10 is the max value of soft_label
            _soft_label = np.array(self.soft_anno_dict[int(_video)])/10.0
        assert int(_video) == int(self.anno_list[index][0]), f'{int(_video)} != {int(self.anno_list[index][0])}'
        
        # get frames
        _frame_path = self.frame_path +'/'+ _video
        _frames = glob.glob(_frame_path+'/*.jpg')
         # DEBUG
        _debug = True
        if _debug:
            _frames = sorted(_frames,key=lambda x: int(x.split('.')[0].split('/')[-1]))
        else:
            _frames = sorted(_frames)
        
        # load path list of video or images
        _frame_list = self.get_clip(_frames)
        _frames_tensor = self.load_frames(_frame_list)
        # load face points
        if self.prompt_load_offline:
            # -1 to get index start from 0
            _array0 = np.load(self.frame_path+_video+'/face_preds.npy') # T,68,2
            if self.mode=='train':
                _frame_id = [int(i.split('.')[0].split('/')[-1])-1 for i in _frame_list]
                _array = _array0[_frame_id,:,:]
                assert _array.shape[0] == self.clip_duration*self.clip_stride, f'{_array.shape[0]} != {self.clip_duration*self.clip_stride} !'
            else:
                _array = []
                for l in _frame_list:
                    _frame_id = [int(i.split('.')[0].split('/')[-1])-1 for i in l]
                    _array.append(_array0[_frame_id,:,:])
                    assert len(_frame_id) == self.clip_duration*self.clip_stride, f'{len(_frame_id)} != {self.clip_duration*self.clip_stride} !'
                _array = np.stack(_array,axis=0) # n,T,68,2
            _array = self._align_points_process(_array)
            _array = torch.from_numpy(_array) # (n),T,68,2
        if self.mode!='train':
            n,t,c,h,w = _frames_tensor.shape
            _frames_tensor = _frames_tensor.view(-1,c,h,w)
        _frames_tensor = self.tensor_transform(_frames_tensor)
        if self.mode!='train':
            _,c,h,w= _frames_tensor.shape
            _frames_tensor = _frames_tensor.view(n,t,c,h,w)
            _frames_tensor = _frames_tensor.transpose(1,2)
        else:
            _frames_tensor = _frames_tensor.transpose(0,1)
        # b,c,t,h,w
        if self.prompt_load_offline:
            return _frames_tensor, _label, _array
        else:
            if self.soft_label:
                return _frames_tensor, _label, _soft_label
            else:
                return _frames_tensor, _label
    
class DFEW_CLS_prompt_save(DFEW_CLS_prompt):
    """
        Add facial keypoints to prompts.
    """
    def __init__(self, root_dir: str, mode: str , input_resolution: int,device:str=None, SSL_train: bool = False, split_num: int = 0, clip_duration: int = 16, clip_stride: int = 1, flip_p: float = 0.5) -> None:
        super().__init__(root_dir, mode, input_resolution, SSL_train, split_num, clip_duration, clip_stride, flip_p)
        from torchvision.transforms import Normalize
        import face_alignment
        if device==None:
            self.device = 'cuda'
        else:
            self.device = device
        # if prompt_offline_save:
        self.prompt_offline_save = True
        self.prompt_engine = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, flip_input=False, device=self.device)
    def __getitem__(self, index):
        _video = self.video_list[index]
        # rescale to 0~class_num-1
        _label = int(self.anno_list[index][1]) - 1
        assert int(_video) == int(self.anno_list[index][0]), f'{int(_video)} != {int(self.anno_list[index][0])}'
        
        # get frames
        _frame_path = self.frame_path +'/'+ _video
        _frames = glob.glob(_frame_path+'/*.jpg')
        _frames = sorted(_frames)
        
        # load path list of video or images
        self.clip_duration = len(_frames)
        _frames_tensor = self.load_frames(_frames) # T,C,H,W
        count_A = 0
        count_B = 0
        count = 0
        clips_detect = _frames_tensor.clone().to(self.device)
        clips_detect = clips_detect*255.0
        preds_batch = self.prompt_engine.get_landmarks_from_batch(clips_detect)
        for i, pred in enumerate(preds_batch):
            if len(pred)>68:
                preds_batch[i] = pred[0:68]
                count_A+=1
            elif len(pred)==0:
                # print(i)
                _tmp = np.empty((68,2),dtype=np.float32)
                _tmp.fill(-1000)
                preds_batch[i] = _tmp
                count_B+=1
            else:
                count += 1
        # save
        np.save(_frame_path+'/face_preds.npy',preds_batch)
        print(f'nolmal:{count}, no face {count_B}, more than 1 face {count_A}')
        return _frames_tensor, _label