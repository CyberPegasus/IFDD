import csv
from functools import partial
from typing import Any, Dict, List
from loguru import logger
import numpy as np
import glob
import random
import jpeg4py as jpeg
import os
import torch
from torch.utils.data.dataset import Dataset
import clip
    
class FERV39k_pretrain(Dataset):
    def __init__(self,
                 root_dir:str,
                 mode:str, # train for training, val for validation, test for evaluation
                 input_resolution:int,
                 SSL_train:bool=False,
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
        annotation_path = root_dir+'/All_scenes'
        annnotation_file = annotation_path + f'/{mode}_All.csv'
        self.video_root = root_dir + '/2_ClipsforFaceCrop/'
        self.classes = {'Angry':0,'Disgust':1,'Fear':2,'Happy':3,'Neutral':4,'Sad':5,'Surprise':6}
        self.classes_DFEW = {'Happy':0,'Sad':1,'Neutral':2,'Angry':3,'Surprise':4,'Disgust':5,'Fear':6}
        self.anno_list_all = []
        with open(annnotation_file,'r') as f:
            f_csv = csv.reader(f, delimiter=' ')
            for row in f_csv:
                _class = self.classes[row[-1]]
                _tuple = (self.video_root+row[-2],_class)
                self.anno_list_all.append(_tuple)
                
        #------video path and preload-----#
        self.video_list = []
        self.anno_list = []
        
        # FERV39k has (31088 - 27831) frames less than 16, the max length is 47 and the min length is 8. mean length is 28
        # So the load_frames need to repeat some frames for required self.clip_duration
        # the proper clip_duration is 32
        for i,j in self.anno_list_all:
            if self.len_filter:
                _tmp = glob.glob(i+'/*.jpg')
                if len(_tmp) < self.clip_duration:
                    continue
            self.video_list.append(i)
            self.anno_list.append(j)
        logger.info(f'Reading {len(self.anno_list)} videos and annotations for {mode} mode split')

        from torchvision.transforms import Compose, Resize, CenterCrop, Normalize, InterpolationMode, RandomHorizontalFlip, RandomResizedCrop
        self.tensor_transform = Compose([
            Resize(self.resolution, interpolation=InterpolationMode.BICUBIC,antialias=True),
            CenterCrop(self.resolution),
            # Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
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
                _img = jpeg.JPEG(_frame).decode() # H,W,3, in RGB format
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
                for s in range(len(frames)):
                    _it = [frames[s]]*_expand_times
                    if _count < _last_len:
                        _it += [frames[s]] #          1
                        _count+=1
                    _expand_frames += _it
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
                for s in range(len(frames)):
                    _it = [frames[s]]*_expand_times
                    if _count < _last_len:
                        _it += [frames[s]]
                        _count+=1
                    _expand_frames += _it
                start_index = 0
                end_index = _clip_duration
                clip = _expand_frames[start_index:end_index:self.clip_stride]
                clips.append(clip)
                
            return clips
         
    def __getitem__(self, index):
        _video = self.video_list[index]
        _label = self.anno_list[index]
        
        # get frames
        _frames = glob.glob(_video+'/*.jpg')
        _frames = sorted(_frames)
        
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
    
class FERV39k_SSL(FERV39k_pretrain):
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
        
    
class FERV39k_CLS(FERV39k_pretrain):
    def __init__(self, root_dir: str, mode: str, input_resolution: int, SSL_train: bool = False, clip_duration: int = 16, clip_stride: int = 2, flip_p: float = 0.5, single_val_view:bool = False) -> None:
        super().__init__(root_dir, mode, input_resolution, SSL_train, clip_duration, clip_stride, flip_p, single_val_view)
        from torchvision.transforms import Compose, Resize, CenterCrop, Normalize, InterpolationMode, RandomHorizontalFlip, RandomResizedCrop
        # antialias only for torch version >= 2
        if int(torch.__version__.split('.')[0])>1:
            createRandomResizedCrop = partial(RandomResizedCrop,antialias=True)
        else:
            createRandomResizedCrop = partial(RandomResizedCrop)
        if self.mode == 'train':
            self.tensor_transform = Compose([
                # size of FERV39k's img is 224*224, don't need resize
                RandomHorizontalFlip(0.5),
                createRandomResizedCrop(size=self.crop_size,scale=(0.8,1.0),interpolation=InterpolationMode.BICUBIC),
                # CenterCrop(self.resolution),
                # Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
            ])
        else:
            self.tensor_transform = Compose([
                Resize(self.resolution, interpolation=InterpolationMode.BICUBIC,antialias=True),
                # CenterCrop(self.resolution),
                # Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
            ])
    
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

class FERV39k_CLS_prompt(FERV39k_CLS):
    def __init__(self, root_dir: str, mode: str, input_resolution: int, prompt_load_offline:bool=False, SSL_train: bool = False, clip_duration: int = 16, clip_stride: int = 2, flip_p: float = 0.5, multi_view: bool = False, strong_aug: bool = False, single_val_view:bool = False) -> None:
        super().__init__(root_dir, mode, input_resolution, SSL_train, clip_duration, clip_stride, flip_p , single_val_view)
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
                    # Normalize(mean=[0.45,0.45,0.45],std=[0.225,0.225,0.225])
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
        _frames = glob.glob(_video+'/*.jpg')
        _frames = sorted(_frames)
        
        #         
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

class FERV39k_CLS_prompt_save(FERV39k_CLS_prompt):
    """
        Add facial keypoints to prompts.
        Replace (bool): if exist face_preds.npy, whether detect+replace or skip
    """
    def __init__(self, root_dir: str, mode: str , input_resolution: int,prompt_load_offline:bool=False, device:str=None, SSL_train: bool = False, clip_duration: int = 16, clip_stride: int = 2, flip_p: float = 0.5, replace:bool=True) -> None:
        super().__init__(root_dir, mode, input_resolution, prompt_load_offline, SSL_train, clip_duration, clip_stride, flip_p)
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
        _frames = glob.glob(_video+'/*.jpg')
        _frames = sorted(_frames)
        
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
    dataset = FERV39k_CLS_prompt_save(root_dir='/home/syx/workspace/Repos/priorkernel/datasets/FERV39k',
                               mode='test',
                               device='cuda:4',
                               input_resolution=224,
                               SSL_train=False,
                               clip_duration=16,
                               clip_stride=2,
                               replace=False)
    _dataloader = DataLoader(dataset,batch_size=16,shuffle=False,num_workers=0)
    for batch in tqdm(_dataloader):
        i  = batch
        pass