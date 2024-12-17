import csv
from functools import partial
from typing import Any, Dict, List
import cv2
from loguru import logger
import numpy as np
import glob
import random
import jpeg4py as jpeg

import torch
from torch.utils.data.dataset import Dataset
import clip
    
class DFEW_ViT(Dataset):
    def __init__(self,
                 root_dir:str,
                 mode:str, # train for training, val for validation, test for evaluation
                 input_resolution:int=160,
                 SSL_train:bool=False,
                 split_num:int=0, # 1,2,3 splits for 3-fold evaluation
                 clip_duration:int=16, # video clip with fixed number of frames
                 clip_stride:int = 1, # Sampling Interval Length for each frame in the clip
                 flip_p:float = 0.5, # random Horizontal Flip
                 test_num:int = 0,
                 ) -> None:
        super().__init__()
        self.mode = mode
        self.split_num = split_num
        self.SSL_train = SSL_train
        self.resolution = input_resolution
        if test_num == 0:
            self.test_seg = False
        else:
            self.test_seg = True
            self.test_num = test_num
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
        self.width = 160
        self.height = 160
        self.crop_size = (160,160)
        self.clip_duration = clip_duration # Clip length
        self.clip_stride = clip_stride # Sampling Interval Length for each frame in the clip
        self.flip_p = flip_p
        from torchvision.transforms import Compose, Resize, CenterCrop, Normalize, InterpolationMode
        self.tensor_transform = Compose([
            Resize(self.resolution, interpolation=InterpolationMode.BICUBIC,antialias=True),
            CenterCrop(self.resolution),
            Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

    def load_frames(self, frame_list: list):
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
            buffer = np.empty((self.clip_duration, self.height, self.width, 3), np.dtype('uint8'))
            for _frame,buffer_count in zip(frame_list,range(len(frame_list))):
                # _img = jpeg.JPEG(_frame).decode() # H,W,3, in RGB format
                _img = cv2.imread(_frame,cv2.IMREAD_COLOR)
                _img = cv2.resize(_img,dsize=self.crop_size)
                _img = cv2.cvtColor(_img,cv2.COLOR_BGR2RGB)
                buffer[buffer_count] = _img  
            buffer = np.ascontiguousarray(buffer.transpose((0,3,1,2)))
        else:
            if not isinstance(frame_list[0],list):
                raise ValueError("The frame_list for test mode should be a list of clips.")
            clip_num = len(frame_list)
            buffer = np.empty((clip_num,self.clip_duration, self.height, self.width, 3), np.dtype('uint8'))
            for frames,clip_count in zip(frame_list,range(clip_num)):
                for _frame,buffer_count in zip(frames,range(len(frames))):
                    # _img = jpeg.JPEG(_frame).decode() # T,H,W,3
                    _img = cv2.imread(_frame,cv2.IMREAD_COLOR)
                    _img = cv2.resize(_img,dsize=self.crop_size)
                    _img = cv2.cvtColor(_img,cv2.COLOR_BGR2RGB)
                    buffer[clip_count][buffer_count] = _img
            buffer = np.ascontiguousarray(buffer.transpose((0,1,4,2,3)))
        # filtered out nan 
        buffer[np.isnan(buffer)]=0
        buffer = torch.from_numpy(buffer).to(torch.uint8)
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
            if self.test_seg:
                seg_num = self.test_num
                clips = []
                frame_len = len(frames)
                _clip_duration = self.clip_duration*self.clip_stride
                temporal_step = max((frame_len - _clip_duration) / (seg_num - 1), 0)
                for clip_i in range(seg_num):
                    start_index = int(clip_i * temporal_step)
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
                    clips.append(clip)
                
            else:
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
        _label = np.int64(_label)
        assert int(_video) == int(self.anno_list[index][0]), f'{int(_video)} != {int(self.anno_list[index][0])}'
        
        # get frames
        _frame_path = self.frame_path +'/'+ _video
        _frames = glob.glob(_frame_path+'/*.jpg')
        _frames = sorted(_frames,key=lambda x: int(x.split('.')[0].split('/')[-1]))
        
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
    
class DFEW_ViT_CLS(DFEW_ViT):
    def __init__(self, root_dir: str, mode: str, input_resolution: int, SSL_train: bool = False, split_num: int = 0, clip_duration: int = 16, clip_stride: int = 1, flip_p: float = 0.5,test_num:int=0) -> None:
        super().__init__(root_dir, mode, input_resolution, SSL_train, split_num, clip_duration, clip_stride, flip_p, test_num)
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
        if self.mode == 'train':
            self.tensor_transform = Compose([
                # size of FERV39k's img is 224*224, don't need resize
                RandomHorizontalFlip(0.5),
                createRandomResizedCrop(size=self.crop_size,scale=(0.75,1.0),interpolation=InterpolationMode.BILINEAR),
                ColorJitter(brightness=0.3,contrast=0.3,saturation=0.3),
                RandomPerspective(distortion_scale=0.2,p=0.5),
                ToDtype(torch.float32),
                FloatScale(),
                Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]) 
        else:
            self.tensor_transform = Compose([
                # Resize(self.resolution, interpolation=InterpolationMode.BICUBIC,antialias=True),
                # CenterCrop(self.resolution),
                ToDtype(torch.float32),
                FloatScale(),
                Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ])
        
    def __getitem__(self, index):
        _video = self.video_list[index]
        # rescale to 0~class_num-1
        _label = int(self.anno_list[index][1]) - 1
        _label = np.int64(_label)
        assert int(_video) == int(self.anno_list[index][0]), f'{int(_video)} != {int(self.anno_list[index][0])}'
        
        # get frames
        _frame_path = self.frame_path +'/'+ _video
        _frames = glob.glob(_frame_path+'/*.jpg')
        _frames = sorted(_frames,key=lambda x: int(x.split('.')[0].split('/')[-1]))
        
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
