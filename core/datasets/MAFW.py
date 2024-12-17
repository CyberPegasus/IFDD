import csv
from functools import partial
from typing import Any, Dict, List
import cv2
from loguru import logger
import numpy as np
import glob
import random
# import jpeg4py as jpeg
import os
import torch
from torch.utils.data.dataset import Dataset
import clip
    
class MAFW_pretrain(Dataset):
    def __init__(self,
                 root_dir:str,
                 mode:str, # train for training, val for validation, test for evaluation
                 input_resolution:int,
                 SSL_train:bool=False,
                 split_num:int=0, # 1,2,3 splits for 3-fold evaluation
                 clip_duration:int=16, # video clip with fixed number of frames
                 clip_stride:int = 2, # Sampling Interval Length for each frame in the clip
                 flip_p:float = 0.5, # random Horizontal Flip
                 single_val_view:bool = False # only one clip for val process
                 ) -> None:
        super().__init__()
        self.mode = mode
        self.SSL_train = SSL_train
        self.resolution = input_resolution
        self.width = 224
        self.height = 224
        self.crop_size = (224,224)
        self.clip_duration = clip_duration # Clip length
        self.clip_stride = clip_stride # Sampling Interval Length for each frame in the clip
        self.sample_num = self.clip_duration*self.clip_stride
        self.flip_p = flip_p
        self.len_filter = False
        self.single_val_view = single_val_view
        #-----reading annotation, video path and class id--------#
        annotation_path = root_dir+f'/label/no_caption/set_{split_num}'
        annnotation_file = annotation_path + f'/{mode}.txt'
        self.video_root = root_dir + '/frames/'
        self.classes = {'anger':0,'disgust':1,'fear':2,'happiness':3,'neutral':4,'sadness':5,'surprise':6,'contempt':7,'anxiety':8,'helplessness':9,'disappointment':10}
        self.anno_name = []
        self.anno_list = []
        self.video_list = []
        with open(annnotation_file,'r') as f:
            _list = f.readlines()
            for i in _list:
                _tmp = i
                _tmp = _tmp.strip('\n')
                _anno = _tmp.split()[-1]
                _anno_clsnum = self.classes[_anno]
                _video = self.video_root + _tmp.split()[0].split('.')[0]
                self.anno_name.append(_anno)
                self.anno_list.append(_anno_clsnum)
                self.video_list.append(_video)
        
        logger.info(f'Reading {len(self.anno_list)} videos and annotations for {mode} mode/{split_num} split of MAFW')
        from collections import Counter
        logger.info(f'Class Num: {Counter(self.anno_name)}')
        
        from torchvision.transforms import Compose, Resize, CenterCrop, Normalize, InterpolationMode, RandomHorizontalFlip, RandomResizedCrop
        self.tensor_transform = Compose([
            Resize(self.resolution, interpolation=InterpolationMode.BICUBIC,antialias=True),
            CenterCrop(self.resolution),
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
            if scale:
                buffer = np.empty((self.clip_duration, self.height, self.width, 3), np.dtype('float32'))
            else:
                buffer = np.empty((self.clip_duration, self.height, self.width, 3), np.dtype('uint8'))
            for _frame,buffer_count in zip(frame_list,range(len(frame_list))):
                _img = cv2.imread(_frame,cv2.IMREAD_COLOR)
                _img = cv2.resize(_img,dsize=self.crop_size)
                _img = cv2.cvtColor(_img,cv2.COLOR_BGR2RGB)
                # _img = jpeg.JPEG(_frame).decode() # H,W,3, in RGB format
                buffer[buffer_count] = _img  
            buffer = np.ascontiguousarray(buffer.transpose((0,3,1,2)))
        else:
            if not isinstance(frame_list[0],list):
                raise ValueError("The frame_list for test mode should be a list of clips.")
            clip_num = len(frame_list)
            if scale:
                buffer = np.empty((clip_num,self.clip_duration, self.height, self.width, 3), np.dtype('float32'))
            else:
                buffer = np.empty((clip_num,self.clip_duration, self.height, self.width, 3), np.dtype('uint8'))
            for frames,clip_count in zip(frame_list,range(clip_num)):
                for _frame,buffer_count in zip(frames,range(len(frames))):
                    _img = cv2.imread(_frame,cv2.IMREAD_COLOR)
                    _img = cv2.resize(_img,dsize=self.crop_size)
                    _img = cv2.cvtColor(_img,cv2.COLOR_BGR2RGB)
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
        frame_len = len(frames)
        _clip_duration = (self.clip_duration-1)*self.clip_stride + 1 #     clip   stride
        # get_clip_index
        if self.mode == 'train' or self.single_val_view: #     split  ，          
            if frame_len>=_clip_duration:
                start_index = random.randrange(0,frame_len-_clip_duration + 1) # +1        frame_len-_clip_duration
                end_index = start_index+_clip_duration #     end index 1，             
                clip = frames[start_index:end_index:self.clip_stride]
            else:
                # logger.info(f'Not enough length for frame of {frame_len} to get clip of {_clip_duration}. \n Expanding.')
                #      +              
                _expand_times = _clip_duration//frame_len
                _last_len = _clip_duration%frame_len
                _count = 0
                _expand_frames = []
                for s in range(len(frames)-1,-1,-1):
                    _it = [frames[s]]*_expand_times
                    if _count < _last_len:
                        _it += [frames[s]]
                        _count+=1
                    _expand_frames += _it
                _expand_frames = list(reversed(_expand_frames))
                start_index = 0
                end_index = _clip_duration
                clip = _expand_frames[start_index:end_index:self.clip_stride]
            
            if self.single_val_view and self.mode != 'train':
                return [clip,]
            else:
                return clip
                
        else: # testing or validation mode
            clips = []
            if frame_len >= _clip_duration:
                start_index = 0
                end_index =start_index+_clip_duration #   
                while end_index <= frame_len:
                    clips.append(frames[start_index:end_index:self.clip_stride])
                    start_index = end_index
                    end_index += _clip_duration
            else:
                # logger.info(f'Not enough length for frame of {frame_len} to get clip of {_clip_duration}. \n Expanding.')
                _expand_times = _clip_duration//frame_len
                _last_len = _clip_duration%frame_len
                _count = 0
                _expand_frames = []
                for s in range(len(frames)-1,-1,-1):
                    _it = [frames[s]]*_expand_times
                    if _count < _last_len:
                        _it += [frames[s]]
                        _count+=1
                    _expand_frames += _it
                _expand_frames = list(reversed(_expand_frames))
                start_index = 0
                end_index = _clip_duration
                clip = _expand_frames[start_index:end_index:self.clip_stride]
                clips.append(clip)
                
            return clips
         
    def __getitem__(self, index):
        _video = self.video_list[index]
        _label = self.anno_list[index]
        
        # get frames
        _frames = glob.glob(_video+'/*.png')
        _frames = sorted(_frames,key=lambda x: int(x.split('.')[0].split('/')[-1]))
        
        # load path list of video or images
        _frame_list = self.get_clip(_frames)
        _frames_tensor = self.load_frames(_frame_list) # 0~1, singleView:T,C,H,W or multiView:N,T,C,H,W
        if self.mode!='train':
            n,t,c,h,w = _frames_tensor.shape
            _frames_tensor = _frames_tensor.view(-1,c,h,w)
        _frames_tensor = self.tensor_transform(_frames_tensor)
        if self.mode!='train':
            _,c,h,w = _frames_tensor.shape
            _frames_tensor = _frames_tensor.view(n,t,c,h,w)
            _frames_tensor = _frames_tensor.transpose(1,2) # n,c,t,h,w
        else:
            _frames_tensor = _frames_tensor.transpose(0,1) # c,t,h,w
        return _frames_tensor, _label
    
    def __len__(self):
        return len(self.video_list)
    
class MAFW_SSL(MAFW_pretrain):
    def __init__(self, *args, **kargs) -> None:
        super().__init__(*args, **kargs)
        from torchvision.transforms import Compose, Resize, CenterCrop, Normalize, InterpolationMode, RandomHorizontalFlip, RandomResizedCrop
        self.tensor_transform = Compose([
            Resize(self.resolution, interpolation=InterpolationMode.BICUBIC,antialias=True),
        ])
        from core.utils import SSL_DataAug
        if self.SSL_train and self.mode=='train':
            self.SSL_data_aug = SSL_DataAug(train_mode=True,target_size=self.crop_size)
        else:
            self.SSL_data_aug = SSL_DataAug(train_mode=False,target_size=self.crop_size)
        
    def __getitem__(self, index):
        if self.mode=='train' and self.SSL_train:
            x, label = super().__getitem__(index) # c,t,h,w or n,c,t,h,w
            return self.SSL_data_aug(x), self.SSL_data_aug(x), label
        else:
            x, label = super().__getitem__(index)
            return x, label
        
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

class MAFW_CLS(MAFW_pretrain):
    def __init__(self, root_dir: str, mode: str, input_resolution: int, SSL_train: bool = False, split_num:int=0, clip_duration: int = 16, clip_stride: int = 2, flip_p: float = 0.5,strong_aug:bool=False, single_val_view:bool = False) -> None:
        super().__init__(root_dir, mode, input_resolution, SSL_train,split_num, clip_duration, clip_stride, flip_p, single_val_view)
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

    def _get_sampleWeight(self):
        _labels = np.array([int(i) for i in self.anno_list])
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
        _label = self.anno_list[index]
        
        # get frames
        _frames = glob.glob(_video+'/*.png')
        _frames = sorted(_frames,key=lambda x: int(x.split('.')[0].split('/')[-1]))
        
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
        return _frames_tensor, _label
    
    def __len__(self):
        return len(self.video_list)

class MAFW_CLS_prompt(MAFW_CLS):
    def __init__(self, root_dir: str, mode: str, input_resolution: int, prompt_load_offline:bool=False, SSL_train: bool = False, split_num:int=0, clip_duration: int = 16, clip_stride: int = 2, flip_p: float = 0.5, multi_view: bool = False, strong_aug: bool = False, single_val_view:bool = False) -> None:
        super().__init__(root_dir, mode, input_resolution, SSL_train, split_num, clip_duration, clip_stride, flip_p , single_val_view)
        self.multi_view = multi_view
        self.view_num = 4
        self.prompt_load_offline = prompt_load_offline
        self.neg_edge = -10
        self.use_scale = not strong_aug
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
            if not strong_aug:
                self.tensor_transform = Compose([
                    # size of FERV39k's img is 224*224, don't need resize
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
            if not self.multi_view and not strong_aug:
                self.tensor_transform = Compose([
                    Resize(self.resolution, interpolation=InterpolationMode.BICUBIC,antialias=True),
                    CenterCrop(self.resolution),
                ])
            elif not self.multi_view and strong_aug:
                self.tensor_transform = Compose([
                    CenterCrop(self.resolution), # FIXME or Resize ?
                    ToDtype(torch.float32),
                    FloatScale(),
                ])                
            else:
                self.tensor_transform = Compose([
                    # size of FERV39k's img is 224*224, don't need resize
                    RandomHorizontalFlip(0.5),
                    createRandomResizedCrop(size=self.crop_size,scale=(0.8,1.0),interpolation=InterpolationMode.BICUBIC),
                    CenterCrop(self.resolution),
                ])

        
    def _align_points_process(self, x:np.ndarray):
        x = x - self.neg_edge
        return np.maximum(x, 0)
    
    def __getitem__(self, index):
        _video = self.video_list[index]
        _label = self.anno_list[index]
        
        # get frames
        _frames = glob.glob(_video+'/*.png')
        _frames = sorted(_frames,key=lambda x: int(x.split('.')[0].split('/')[-1]))
        
        # DEBUG         
        _lost_flag = False
        _id_map = {}
        if int(_frames[-1].split('.')[0].split('/')[-1])+1 != len(_frames):
            _lost_flag = True
            for i in range(len(_frames)):
                _name_id = int(_frames[i].split('.')[0].split('/')[-1])
                _id_map[_name_id] = i
                
        # load path list of video or images
        _frame_list = self.get_clip(_frames)
        if self.mode!='train' and self.multi_view:
            while len(_frame_list) > 2:
                _frame_list.pop()
        _frames_tensor = self.load_frames(_frame_list,scale=self.use_scale) # 0~1, singleView:T,C,H,W or multiView:N,T,C,H,W
        # load face points
        if self.prompt_load_offline:
            # -1 to get index start from 0
            _array0 = np.load(_video+'/face_preds.npy') # T,68,2
            if self.mode=='train':
                if not _lost_flag:
                    _frame_id = [int(i.split('.')[0].split('/')[-1]) for i in _frame_list]
                else:
                    _frame_id = [_id_map[int(i.split('.')[0].split('/')[-1])] for i in _frame_list]
                try:
                    _array = _array0[_frame_id,:,:]
                except:
                    raise IndexError(f'{_frame_id} for {_array0.shape}')
                try:
                    _test_len = _array[self.clip_duration-1]
                except:
                    raise IndexError(f'{_array.shape[0]} != {self.clip_duration} !')
            else:
                _array = []
                for l in _frame_list:
                    if not _lost_flag:
                        _frame_id = [int(i.split('.')[0].split('/')[-1]) for i in l]
                    else:
                        _frame_id = [_id_map[int(i.split('.')[0].split('/')[-1])] for i in l]
                    try:
                        _array.append(_array0[_frame_id,:,:])
                    except:
                        raise IndexError(f'{_frame_id} for {_array0.shape}')
                    try:
                        _test_len = _frame_id[self.clip_duration-1]
                    except:
                        raise IndexError(f'{_array[-1].shape[0]} != {self.clip_duration} !')
                _array = np.stack(_array,axis=0) # n,T,68,2
            _array = self._align_points_process(_array)
            _array = torch.from_numpy(_array) # (n),T,68,2
        if self.mode!='train':
            n,t,c,h,w = _frames_tensor.shape
            _frames_tensor = _frames_tensor.view(-1,c,h,w)
            
        if self.multi_view and self.mode!='train':
            _view = self.view_num//n
            if self.prompt_load_offline:
                _array = _array.repeat(_view,1,1,1)
            _tmp = []
            for _ in range(_view):
                tmp_frames_tensor = self.tensor_transform(_frames_tensor)
                tmp_frames_tensor = tmp_frames_tensor.view(n,t,c,h,w)
                _tmp.append(tmp_frames_tensor)
            _frames_tensor = torch.cat(_tmp,dim=0)
            _frames_tensor = _frames_tensor.transpose(1,2)
        else:
            _frames_tensor = self.tensor_transform(_frames_tensor)
            
        if not self.multi_view:
            if self.mode!='train':
                _,c,h,w = _frames_tensor.shape
                _frames_tensor = _frames_tensor.view(n,t,c,h,w)
                _frames_tensor = _frames_tensor.transpose(1,2)
            else:
                _frames_tensor = _frames_tensor.transpose(0,1)
        # b,c,t,h,w
        if self.prompt_load_offline:
            return _frames_tensor, _label, _array
        else:
            return _frames_tensor, _label

class MAFW_CLS_prompt_save(MAFW_CLS_prompt):
    """
        Add facial keypoints to prompts.
        Replace (bool): if exist face_preds.npy, whether detect+replace or skip
    """
    def __init__(self, root_dir: str, mode: str , input_resolution: int,prompt_load_offline:bool=False, device:str=None, SSL_train: bool = False, split_num:int=0, clip_duration: int = 16, clip_stride: int = 2, flip_p: float = 0.5, replace:bool=True) -> None:
        super().__init__(root_dir, mode, input_resolution, prompt_load_offline, SSL_train, split_num, clip_duration, clip_stride, flip_p)
        from torchvision.transforms import Normalize
        import face_alignment
        if device==None:
            self.device = 'cuda'
        else:
            self.device = device
        self.replace = replace
        # if prompt_offline_save:
        self.prompt_offline_save = True
        self.prompt_engine = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, flip_input=False, device=self.device)
        # search for exist npy
        self.exist_video_set = set()
        for _video in self.video_list:
            if os.path.exists(_video+'/face_preds.npy'):
                self.exist_video_set.add(_video)
            else:
                continue
        
    def __getitem__(self, index):
        _video = self.video_list[index]
        _label = self.anno_list[index]
        if (not self.replace) and (_video in self.exist_video_set):
            return -1
        # get frames
        _frames = glob.glob(_video+'/*.png')
        _frames = sorted(_frames,key=lambda x: int(x.split('.')[0].split('/')[-1]))
        
        # load path list of video or images
        self.clip_duration = len(_frames)
        _frames_tensor = self.load_frames(_frames) # T,C,H,W
        count_A = 0
        count_B = 0
        count = 0
        clips_detect = _frames_tensor.clone().to(self.device)
        clips_detect = clips_detect*255.0 # from 0~1 to 0~255
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
        np.save(_video+'/face_preds.npy',preds_batch)
        print(f'nolmal:{count}, no face {count_B}, more than 1 face {count_A}')
        return _label
    
if __name__ == '__main__':
    from torch.utils.data import DataLoader
    from tqdm import tqdm
    dataset = MAFW_CLS_prompt_save(root_dir='/home/syx/workspace/Repos/priorkernel/datasets/MAFW',
                               mode='test', # train or test
                               device='cuda:5',
                               input_resolution=224,
                               SSL_train=False,
                               split_num=1,
                               clip_duration=16,
                               clip_stride=2,
                               replace=True)
    _dataloader = DataLoader(dataset,batch_size=16,shuffle=False,num_workers=0)
    for batch in tqdm(_dataloader):
        i  = batch
        pass