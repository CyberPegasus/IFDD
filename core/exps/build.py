import json
from easydict import EasyDict as edict
import importlib
import os
import sys

def load_cfg(cfg_path:str)->edict:
    with open(cfg_path,'r') as f:
        cfg = json.load(f)
    cfg = edict(cfg)
    
    return cfg
    
    
def get_exp_by_file(exp_file):
    try:
        sys.path.append(os.path.dirname(exp_file))
        current_exp = importlib.import_module(os.path.basename(exp_file).split(".")[0])
        exp = current_exp.Exp()
    except Exception:
        raise ImportError("{} doesn't contains class named 'Exp'".format(exp_file))
    return exp