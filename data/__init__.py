from constants import *
from data.base import DataInstance
from data.helpers import tokenize
from data.i2b2 import load_i2b2_dataset
from data.e3c import load_e3c_dataset
import torch

def load_data(split_nb, tokenizer, configs):
    base_path = 'resources'
    if configs['dataset'] == E3C:
        return load_e3c_dataset(base_path+"/e3c", tokenizer, split_nb, configs) 
    elif configs['dataset'] == I2B2:
        return load_i2b2_dataset(base_path+"/i2b2", tokenizer, split_nb, configs) 
